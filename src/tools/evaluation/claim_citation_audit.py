"""Bloque 5B de la migración de la etapa 08: auditoría de claims, citas y trazabilidad.

Copias LITERALES de ``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb``,
celda 19 (resto del bloque "MÉTRICAS FACTUALES Y DE TRAZABILIDAD FINAL",
la parte que no es validación numérica — ver Bloque 5A). No incluye LLM
Judge, la validación factual final (``factual_consistency_ok``/
``APPROVED``/``MEASURED_WITH_PENDING_ISSUES``/excepción bloqueante — eso se
ensambla DESPUÉS de unir 5A+5B, por instrucción explícita, para no
duplicar reglas), persistencia, contrato transaccional ni ``StageSpec``.

Reutilización explícita (no se duplica nada)
-----------------------------------------------
- ``safe_str``, ``normalize_claim_text``, ``citation_pattern`` — Bloque 1
  (``src/tools/evaluation/text_normalization.py``).
- ``to_bool``, ``build_section_text_by_id`` — Bloque 5A
  (``src/tools/evaluation/numeric_validation.py``). En el notebook real,
  ``section_text_by_id`` se construye UNA sola vez en la celda 19 y lo
  reutilizan tanto el bucle de citas como el bucle numérico — aquí quedó
  dividido entre dos módulos por el orden de extracción por bloques, así
  que este módulo IMPORTA la función de Bloque 5A en vez de reescribirla.

Mapa función → celda original
------------------------------
| Función | Celda | Dependencias | Entradas | Salidas | Excepciones | Archivos |
|---|---|---|---|---|---|---|
| ``first_non_empty`` | 19 | ``safe_str`` (Bloque 1) | ``values: Iterable`` | primer valor no vacío, o ``""`` | ninguna | ninguno |
| ``build_valid_source_chunk_pairs`` | 11 (``valid_source_chunk_pairs``, código de módulo) | ``safe_str`` | ``chunks: list[dict]`` | ``set[tuple[str,str]]`` | ninguna | ninguno |
| ``validate_required_traceability_columns`` | 19 (``required_traceability_columns``, código de módulo) | ninguna | ``traceability_rows: list[dict]`` | ``None`` | ``ValueError`` si faltan `claim_id`/`verdict` | ninguno |
| ``build_claim_audit_rows`` | 19 (bucle ``for claim_id, group in df_traceability.groupby(...)``) | ``safe_str``, ``normalize_claim_text``, ``to_bool`` | ``traceability_rows``, ``generated_content_text``, ``valid_source_chunk_pairs`` | ``list[dict]`` (equivalente en memoria a ``final_claim_audit.csv``) | ``ValueError`` si columnas faltan o no hay ningún claim final | ninguno |
| ``select_active_claims`` | 19 (``active_claims = ...``) | ``to_bool`` | ``claim_audit_rows`` | ``list[dict]`` (subconjunto activo) | ``ValueError`` si no hay ningún claim activo | ninguno |
| ``compute_claim_factual_metrics`` | 19 (``supported_claims``...``invalid_traceability_pairs``) | ``to_bool`` | ``active_claims`` | ``dict`` con 8 valores agregados | ninguna | ninguno |
| ``count_removed_claims`` | 19 (``removed_claims``) | ninguna | ``claim_audit_rows`` (TODOS, no solo activos) | ``int`` | ninguna | ninguno |
| ``build_citation_rows`` | 19 (bucle ``for section in sections`` + ``citation_rows.append``) | ``citation_pattern`` (Bloque 1) | ``section_text_by_id`` (de Bloque 5A), ``valid_source_chunk_pairs`` | ``list[dict]`` (equivalente en memoria a ``final_citation_check.csv``) | ninguna | ninguno |
| ``compute_citation_metrics`` | 19 (``total_citations``/``invalid_citations``/``citation_error_rate``) | ``to_bool`` | ``citation_rows`` | ``dict`` con 3 valores | ninguna | ninguno |
| ``build_factual_metric_rows`` | 19 (``factual_metric_rows``, código de módulo) | ninguna | los agregados de claims/citas + ``numeric_error_rate``/``total_numeric_values_checked`` del Bloque 5A (recibidos, NO recalculados) | ``list[dict]`` de 12 filas, orden real exacto | ninguna | ninguno |

Compatibilidad histórica con 07C (punto 9 del pedido)
-----------------------------------------------------------
El mensaje de error de ``validate_required_traceability_columns`` conserva
literalmente el texto real del notebook, que menciona
``"post_correction_traceability_matrix.csv"`` — el nombre de archivo real
de la ruta histórica 07C. **La ruta activa del orquestador viene de 07,
no de 07C**; ese nombre de archivo queda en el mensaje solo porque es
texto literal del notebook, no porque 07C vuelva a formar parte del flujo
activo. No se agrega ninguna lógica que dependa de 07C.

Agrupación por ``claim_id`` (nota de fidelidad frente a pandas)
-----------------------------------------------------------------
``df_traceability.groupby(df_traceability["claim_id"].astype(str),
dropna=False)`` en pandas real: (a) usa ``str(valor)`` como clave de
agrupación (no ``safe_str`` — por eso un ``claim_id`` nulo agrupa bajo la
clave literal ``"None"``, que NO es una cadena vacía y por lo tanto NO se
descarta por el ``if not claim_id: continue`` posterior); (b) itera los
grupos en orden alfabético ascendente de la clave (comportamiento por
defecto ``sort=True`` de pandas, nunca desactivado en el notebook). Ambos
comportamientos se reproducen aquí exactamente con
``sorted(groups)``/``str(row.get("claim_id"))`` — no es una elección de
diseño de esta migración, es el comportamiento real que produce
``df_traceability.groupby(...)`` sin argumentos adicionales.

Veredictos y niveles de riesgo problemáticos (preservados literales)
-----------------------------------------------------------------------
``PROBLEM_VERDICTS = {"partially_supported", "unclear", "unsupported"}``,
``PROBLEM_RISK_LEVELS = {"medium", "high"}`` — un claim activo cuenta como
"problema" si su veredicto está en la primera lista, O su riesgo está en
la segunda, O ``correction_needed`` es verdadero (OR lógico de las tres
condiciones, exactamente como en el notebook).

Importar este módulo no lee archivos, no llama a OpenAI y no tiene
efectos secundarios. Las funciones puras reciben ``list[dict]`` ya
cargados — el cargador de artefactos (que leería
``post_correction_traceability_matrix.csv``/el bundle de 07/
``chunks_clean_for_rag.csv`` de disco) queda fuera de este bloque.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

from src.tools.evaluation.numeric_validation import build_section_text_by_id, to_bool
from src.tools.evaluation.text_normalization import (
    citation_pattern,
    normalize_claim_text,
    safe_str,
)

REQUIRED_TRACEABILITY_COLUMNS = {"claim_id", "verdict"}
PROBLEM_VERDICTS = {"partially_supported", "unclear", "unsupported"}
PROBLEM_RISK_LEVELS = {"medium", "high"}

CLAIM_AUDIT_COLUMNS = [
    "claim_id",
    "claim",
    "verdict",
    "hallucination_risk",
    "correction_needed",
    "active_claim",
    "claim_in_final_text",
    "evidence_pair_count",
    "evidence_present",
    "invalid_evidence_pair_count",
    "invalid_evidence_pairs",
]

CITATION_CHECK_COLUMNS = [
    "section_id",
    "citation_index",
    "source_filename",
    "chunk_id",
    "citation",
    "exists_in_clean_chunks",
]

# re-exportada para que el llamador no tenga que importar de dos módulos
# distintos solo para reproducir la construcción de section_text_by_id.
__all__ = [
    "REQUIRED_TRACEABILITY_COLUMNS",
    "PROBLEM_VERDICTS",
    "PROBLEM_RISK_LEVELS",
    "CLAIM_AUDIT_COLUMNS",
    "CITATION_CHECK_COLUMNS",
    "first_non_empty",
    "build_valid_source_chunk_pairs",
    "validate_required_traceability_columns",
    "build_claim_audit_rows",
    "select_active_claims",
    "compute_claim_factual_metrics",
    "count_removed_claims",
    "build_citation_rows",
    "compute_citation_metrics",
    "build_factual_metric_rows",
    "build_section_text_by_id",
]


def first_non_empty(values: Iterable[Any]) -> str:
    for value in values:
        text = safe_str(value)
        if text:
            return text
    return ""


def build_valid_source_chunk_pairs(chunks: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (safe_str(chunk["source_filename"]).strip(), safe_str(chunk["chunk_id"]).strip())
        for chunk in chunks
    }


def validate_required_traceability_columns(
    traceability_rows: list[dict[str, Any]],
    *,
    required: set[str] = REQUIRED_TRACEABILITY_COLUMNS,
) -> None:
    present_columns: set[str] = set()
    for row in traceability_rows:
        present_columns.update(row.keys())

    missing = sorted(required - present_columns)
    if missing:
        raise ValueError(
            "post_correction_traceability_matrix.csv está incompleto. "
            f"Faltan: {missing}"
        )


def build_claim_audit_rows(
    *,
    traceability_rows: list[dict[str, Any]],
    generated_content_text: str,
    valid_source_chunk_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    validate_required_traceability_columns(traceability_rows)

    all_columns: set[str] = set()
    for row in traceability_rows:
        all_columns.update(row.keys())

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in traceability_rows:
        groups[str(row.get("claim_id"))].append(row)

    claim_audit_rows: list[dict[str, Any]] = []

    for claim_id_key in sorted(groups):
        group = groups[claim_id_key]
        claim_id = safe_str(claim_id_key)

        if not claim_id:
            continue

        claim_text = (
            first_non_empty(row.get("claim") for row in group) if "claim" in all_columns else ""
        )
        verdict = first_non_empty(row.get("verdict") for row in group).casefold()
        risk = (
            first_non_empty(row.get("hallucination_risk") for row in group).casefold()
            if "hallucination_risk" in all_columns
            else ""
        )

        correction_needed = False
        if "correction_needed" in all_columns:
            correction_needed = any(to_bool(row.get("correction_needed")) for row in group)

        evidence_pairs: set[tuple[str, str]] = set()
        if {"source_filename", "chunk_id"}.issubset(all_columns):
            for row in group:
                source = safe_str(row.get("source_filename"))
                chunk_id = safe_str(row.get("chunk_id"))
                if source and chunk_id:
                    evidence_pairs.add((source, chunk_id))

        active_claim = verdict != "removed" and bool(normalize_claim_text(claim_text))
        claim_in_final_text = (
            normalize_claim_text(claim_text) in normalize_claim_text(generated_content_text)
            if active_claim
            else True
        )
        invalid_evidence_pairs = sorted(
            pair for pair in evidence_pairs if pair not in valid_source_chunk_pairs
        )

        claim_audit_rows.append(
            {
                "claim_id": claim_id,
                "claim": claim_text,
                "verdict": verdict,
                "hallucination_risk": risk,
                "correction_needed": correction_needed,
                "active_claim": active_claim,
                "claim_in_final_text": claim_in_final_text,
                "evidence_pair_count": len(evidence_pairs),
                "evidence_present": bool(evidence_pairs),
                "invalid_evidence_pair_count": len(invalid_evidence_pairs),
                "invalid_evidence_pairs": json.dumps(
                    invalid_evidence_pairs, ensure_ascii=False
                ),
            }
        )

    if not claim_audit_rows:
        raise ValueError("No se encontraron claims finales para evaluar.")

    return claim_audit_rows


def select_active_claims(claim_audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_claims = [row for row in claim_audit_rows if to_bool(row["active_claim"])]
    if not active_claims:
        raise ValueError(
            "No existen claims activos en la entrada seleccionada para el Agente 08."
        )
    return active_claims


def compute_claim_factual_metrics(active_claims: list[dict[str, Any]]) -> dict[str, Any]:
    total_active_claims = len(active_claims)

    supported_claims = int(sum(1 for row in active_claims if row["verdict"] == "supported"))

    problem_claim_count = int(
        sum(
            1
            for row in active_claims
            if row["verdict"] in PROBLEM_VERDICTS
            or row["hallucination_risk"] in PROBLEM_RISK_LEVELS
            or to_bool(row["correction_needed"])
        )
    )

    factual_precision = supported_claims / total_active_claims
    hallucination_rate = problem_claim_count / total_active_claims
    evidence_coverage = float(
        sum(1 for row in active_claims if to_bool(row["evidence_present"])) / total_active_claims
    )
    traceability_text_coverage = float(
        sum(1 for row in active_claims if to_bool(row["claim_in_final_text"]))
        / total_active_claims
    )
    invalid_traceability_pairs = int(
        sum(row["invalid_evidence_pair_count"] for row in active_claims)
    )

    return {
        "total_active_claims": total_active_claims,
        "supported_claims": supported_claims,
        "problem_claims": problem_claim_count,
        "factual_precision": factual_precision,
        "hallucination_rate": hallucination_rate,
        "evidence_coverage": evidence_coverage,
        "traceability_text_coverage": traceability_text_coverage,
        "invalid_traceability_pairs": invalid_traceability_pairs,
    }


def count_removed_claims(claim_audit_rows: list[dict[str, Any]]) -> int:
    return int(sum(1 for row in claim_audit_rows if row["verdict"] == "removed"))


def build_citation_rows(
    *,
    section_text_by_id: dict[str, str],
    valid_source_chunk_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    citation_rows: list[dict[str, Any]] = []

    for section_id, section_text in section_text_by_id.items():
        for citation_index, (source_filename, chunk_id) in enumerate(
            citation_pattern.findall(section_text), start=1
        ):
            pair = (source_filename.strip(), chunk_id.strip())
            citation_rows.append(
                {
                    "section_id": section_id,
                    "citation_index": citation_index,
                    "source_filename": pair[0],
                    "chunk_id": pair[1],
                    "citation": f"[{pair[0]} | {pair[1]}]",
                    "exists_in_clean_chunks": pair in valid_source_chunk_pairs,
                }
            )

    return citation_rows


def compute_citation_metrics(citation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_citations = len(citation_rows)
    invalid_citations = (
        int(sum(1 for row in citation_rows if not to_bool(row["exists_in_clean_chunks"])))
        if total_citations
        else 0
    )
    citation_error_rate = invalid_citations / total_citations if total_citations else None

    return {
        "total_citations": total_citations,
        "invalid_citations": invalid_citations,
        "citation_error_rate": citation_error_rate,
    }


def build_factual_metric_rows(
    *,
    total_active_claims: int,
    supported_claims: int,
    factual_precision: float,
    hallucination_rate: float,
    evidence_coverage: float,
    traceability_text_coverage: float,
    citation_error_rate: float | None,
    numeric_error_rate: float | None,
    invalid_traceability_pairs: int,
    removed_claims: int,
    total_final_citations: int,
    total_numeric_values_checked: int,
) -> list[dict[str, Any]]:
    return [
        {
            "metric": "total_active_claims",
            "value": int(total_active_claims),
            "description": (
                "Claims presentes en el texto final, excluyendo "
                "los fragmentos eliminados."
            ),
        },
        {
            "metric": "supported_claims",
            "value": supported_claims,
            "description": "Claims finales con veredicto supported.",
        },
        {
            "metric": "factual_precision",
            "value": factual_precision,
            "description": "Claims supported dividido para claims activos.",
        },
        {
            "metric": "hallucination_rate",
            "value": hallucination_rate,
            "description": (
                "Claims parcialmente soportados, ambiguos, "
                "no soportados, de riesgo medio/alto o pendientes."
            ),
        },
        {
            "metric": "evidence_coverage",
            "value": evidence_coverage,
            "description": "Claims activos con al menos un par fuente-chunk.",
        },
        {
            "metric": "traceability_text_coverage",
            "value": traceability_text_coverage,
            "description": "Claims activos localizados en el texto final.",
        },
        {
            "metric": "citation_error_rate",
            "value": citation_error_rate,
            "description": (
                "Proporción de citas del texto final que no existen "
                "en chunks_clean_for_rag.csv. Es null si no hay citas detectadas."
            ),
        },
        {
            "metric": "numeric_error_rate",
            "value": numeric_error_rate,
            "description": (
                "Proporción de valores numéricos que no aparecen "
                "en los chunks citados. Es null si no hubo valores comprobables."
            ),
        },
        {
            "metric": "invalid_traceability_pairs",
            "value": invalid_traceability_pairs,
            "description": (
                "Pares fuente-chunk de la matriz final "
                "que no existen en los chunks limpios."
            ),
        },
        {
            "metric": "removed_claims_after_correction",
            "value": removed_claims,
            "description": "Claims problemáticos eliminados antes de la evaluación.",
        },
        {
            "metric": "total_final_citations",
            "value": total_final_citations,
            "description": "Citas internas revisadas en el texto final.",
        },
        {
            "metric": "total_numeric_values_checked",
            "value": total_numeric_values_checked,
            "description": (
                "Valores numéricos disponibles en la ruta upstream seleccionada."
            ),
        },
    ]
