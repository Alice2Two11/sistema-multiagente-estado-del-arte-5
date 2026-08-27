"""Ensamblador factual: une Bloque 5A (numérico) y Bloque 5B (claims/citas)
sin duplicar cálculos, y reproduce ``factual_consistency_ok`` (celda 19,
final).

Hallazgo importante (corrige una nota de una ronda anterior): existen DOS
puertas de validación distintas e independientes en el notebook, no una:

1. ``factual_consistency_ok`` (celda 19, aquí) — NO está gobernada por
   ningún flag de ``EVALUATION_POLICY``. Su condición de bloqueo/no-bloqueo
   es ``source_stage == "AGENT07" and upstream_runtime_status == "PARTIAL"``.
2. ``evaluation_validation_ok`` (celda 23, fuera de este módulo) — esa SÍ
   está gobernada por ``EVALUATION_POLICY["fail_on_invalid_evaluation"]``.

No confundir ambas: se implementa aquí únicamente la primera.
"""

from __future__ import annotations

from typing import Any

from src.tools.evaluation.claim_citation_audit import (
    build_citation_rows,
    build_claim_audit_rows,
    build_factual_metric_rows,
    build_valid_source_chunk_pairs,
    compute_citation_metrics,
    compute_claim_factual_metrics,
    count_removed_claims,
    select_active_claims,
)
from src.tools.evaluation.numeric_validation import (
    aggregate_numeric_metrics,
    build_chunk_text_by_pair,
    build_section_text_by_id,
    extract_numeric_rows,
)


def build_factual_audit(
    *,
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    traceability_rows: list[dict[str, Any]],
    generated_content_text: str,
) -> dict[str, Any]:
    """Construye toda la auditoría factual en memoria: numérica (5A),
    claims/citas (5B), y las 12 filas de ``factual_metric_rows`` combinadas
    — sin recalcular nada que ya exista en 5A/5B."""

    section_text_by_id = build_section_text_by_id(sections)
    chunk_text_by_pair = build_chunk_text_by_pair(chunks)
    valid_source_chunk_pairs = build_valid_source_chunk_pairs(chunks)

    numeric_rows = extract_numeric_rows(
        section_text_by_id=section_text_by_id, chunk_text_by_pair=chunk_text_by_pair
    )
    numeric_metrics = aggregate_numeric_metrics(numeric_rows)

    claim_audit_rows = build_claim_audit_rows(
        traceability_rows=traceability_rows,
        generated_content_text=generated_content_text,
        valid_source_chunk_pairs=valid_source_chunk_pairs,
    )
    active_claims = select_active_claims(claim_audit_rows)
    claim_metrics = compute_claim_factual_metrics(active_claims)
    removed_claims = count_removed_claims(claim_audit_rows)

    citation_rows = build_citation_rows(
        section_text_by_id=section_text_by_id, valid_source_chunk_pairs=valid_source_chunk_pairs
    )
    citation_metrics = compute_citation_metrics(citation_rows)

    factual_metric_rows = build_factual_metric_rows(
        total_active_claims=claim_metrics["total_active_claims"],
        supported_claims=claim_metrics["supported_claims"],
        factual_precision=claim_metrics["factual_precision"],
        hallucination_rate=claim_metrics["hallucination_rate"],
        evidence_coverage=claim_metrics["evidence_coverage"],
        traceability_text_coverage=claim_metrics["traceability_text_coverage"],
        citation_error_rate=citation_metrics["citation_error_rate"],
        numeric_error_rate=numeric_metrics["numeric_error_rate"],
        invalid_traceability_pairs=claim_metrics["invalid_traceability_pairs"],
        removed_claims=removed_claims,
        total_final_citations=citation_metrics["total_citations"],
        total_numeric_values_checked=numeric_metrics["total_numeric_values"],
    )

    return {
        "numeric_rows": numeric_rows,
        "numeric_metrics": numeric_metrics,
        "claim_audit_rows": claim_audit_rows,
        "active_claims": active_claims,
        "claim_metrics": claim_metrics,
        "removed_claims": removed_claims,
        "citation_rows": citation_rows,
        "citation_metrics": citation_metrics,
        "factual_metric_rows": factual_metric_rows,
    }


def evaluate_factual_consistency(factual_audit: dict[str, Any]) -> dict[str, Any]:
    """Reproduce EXACTAMENTE la condición real de ``factual_consistency_ok``
    (celda 19, final) — los 9 términos, con ``and`` estricto, incluida la
    exigencia de que ``citation_error_rate``/``numeric_error_rate`` no sean
    ``None`` (si no hubo citas o valores numéricos comprobables, la
    evaluación NUNCA puede quedar `factual_consistency_ok=True`, aunque
    todo lo demás sea perfecto — comportamiento real, no un defecto de esta
    migración)."""

    claim_metrics = factual_audit["claim_metrics"]
    citation_error_rate = factual_audit["citation_metrics"]["citation_error_rate"]
    numeric_error_rate = factual_audit["numeric_metrics"]["numeric_error_rate"]

    factual_consistency_ok = (
        claim_metrics["factual_precision"] == 1.0
        and claim_metrics["hallucination_rate"] == 0.0
        and claim_metrics["evidence_coverage"] == 1.0
        and claim_metrics["traceability_text_coverage"] == 1.0
        and citation_error_rate is not None
        and citation_error_rate == 0.0
        and numeric_error_rate is not None
        and numeric_error_rate == 0.0
        and claim_metrics["invalid_traceability_pairs"] == 0
    )

    factual_consistency_status = "APPROVED" if factual_consistency_ok else "MEASURED_WITH_PENDING_ISSUES"

    return {
        "factual_consistency_ok": factual_consistency_ok,
        "factual_consistency_status": factual_consistency_status,
    }


def resolve_factual_gate(
    *, factual_consistency_result: dict[str, Any], source_stage: str, upstream_runtime_status: str
) -> None:
    """Reproduce EXACTAMENTE la puerta de bloqueo/no-bloqueo real (celda 19,
    final): si ``factual_consistency_ok`` es False, solo se permite
    continuar (sin lanzar) cuando ``source_stage == "AGENT07" and
    upstream_runtime_status == "PARTIAL"`` — en cualquier otro caso, lanza
    ``ValueError`` con el mensaje literal real. **No está gobernada por
    ningún flag de EVALUATION_POLICY** — es una condición fija sobre el
    origen/estado del upstream, distinta de ``FAIL_ON_INVALID_EVALUATION``
    (que gobierna una puerta DIFERENTE, ``evaluation_validation_ok`` en la
    celda 23, fuera de este módulo)."""

    if factual_consistency_result["factual_consistency_ok"]:
        return

    if source_stage == "AGENT07" and upstream_runtime_status == "PARTIAL":
        return  # no bloqueante: se registra el resultado con pendientes

    raise ValueError(
        "La entrada upstream declara una aprobación incompatible "
        "con las métricas factuales calculadas. "
        "Revisa factual_metrics.csv, final_claim_audit.csv "
        "y final_citation_check.csv."
    )
