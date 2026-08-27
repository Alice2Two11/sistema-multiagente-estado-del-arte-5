"""Resolve the authoritative upstream input consumed by Agent 08.

Agent 08 accepts either:
1. Agent 07C evaluation-ready artifacts, when accepted corrections were
   applied and reverified; or
2. committed Agent 07 artifacts plus the unchanged committed Agent 06 draft,
   when Agent 07 proves that no claim is eligible for 07C.

The adapter never fabricates a 07C execution and never suppresses manual-review
claims from Agent 07.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

AGENT08_UPSTREAM_SCHEMA_VERSION = "AGENT08_UPSTREAM_INPUT_V1"
SOURCE_STAGE_AGENT07 = "AGENT07"
SOURCE_STAGE_AGENT07C = "AGENT07C"
NO_ACCEPTED_CORRECTIONS = "NO_ACCEPTED_CORRECTIONS"
ACCEPTED_CORRECTIONS_REVERIFIED = "ACCEPTED_CORRECTIONS_REVERIFIED"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"AGENT08_UPSTREAM_ARTIFACT_MISSING:{path.name}") from None
    except json.JSONDecodeError:
        raise ValueError(f"AGENT08_UPSTREAM_JSON_INVALID:{path.name}") from None
    if not isinstance(value, dict):
        raise ValueError(f"AGENT08_UPSTREAM_JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        raise ValueError(f"AGENT08_UPSTREAM_ARTIFACT_MISSING:{path.name}") from None
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _nonempty_text(value: Any, code: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(code)
    return text


def _bool(value: Any) -> bool:
    return value is True


def _manual_review_claims(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    claims = {
        _nonempty_text(row.get("claim_id"), "AGENT08_CLAIM_ID_MISSING")
        for row in rows
        if _bool(row.get("manual_review_required"))
    }
    return tuple(sorted(claims))


@dataclass(frozen=True, slots=True)
class Agent08UpstreamInput:
    schema_version: str
    source_stage: str
    reverification_performed: bool
    reverification_reason: str
    upstream_runtime_status: str
    claims_verified: int
    claims_requiring_manual_review: int
    manual_review_claim_ids: tuple[str, ...]
    generated_state_of_art_json_path: str
    generated_state_of_art_markdown_path: str
    traceability_rows: tuple[Mapping[str, Any], ...]
    numeric_check_rows: tuple[Mapping[str, Any], ...]
    claim_report_rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_stage": self.source_stage,
            "reverification_performed": self.reverification_performed,
            "reverification_reason": self.reverification_reason,
            "upstream_runtime_status": self.upstream_runtime_status,
            "claims_verified": self.claims_verified,
            "claims_requiring_manual_review": self.claims_requiring_manual_review,
            "manual_review_claim_ids": list(self.manual_review_claim_ids),
            "generated_state_of_art_json_path": self.generated_state_of_art_json_path,
            "generated_state_of_art_markdown_path": self.generated_state_of_art_markdown_path,
            "traceability_rows": [dict(row) for row in self.traceability_rows],
            "numeric_check_rows": [dict(row) for row in self.numeric_check_rows],
            "claim_report_rows": [dict(row) for row in self.claim_report_rows],
            "provenance": dict(self.provenance),
        }


def _validate_generated_draft(
    json_path: Path, markdown_path: Path, expected_fingerprint: str | None = None
) -> tuple[dict[str, Any], str]:
    draft = _read_json(json_path)
    _sha256_file(markdown_path)
    sections = draft.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("AGENT08_GENERATED_DRAFT_SECTIONS_INVALID")
    for section in sections:
        if not isinstance(section, Mapping):
            raise ValueError("AGENT08_GENERATED_DRAFT_SECTION_INVALID")
        _nonempty_text(section.get("section_id"), "AGENT08_GENERATED_SECTION_ID_MISSING")
        text = section.get("draft_text")
        if text is None:
            text = section.get("verified_text")
        _nonempty_text(text, "AGENT08_GENERATED_SECTION_TEXT_MISSING")
    fingerprint = _sha256_json(draft)
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError("AGENT08_SOURCE_DRAFT_FINGERPRINT_MISMATCH")
    return draft, fingerprint


def build_agent08_input_from_committed_agent07(
    *,
    agent07_directory: str | Path,
    draft_json_path: str | Path,
    draft_markdown_path: str | Path,
) -> Agent08UpstreamInput:
    """Build direct 07→08 input when no accepted correction requires 07C."""
    directory = Path(agent07_directory)
    manifest_path = directory / "agent07_artifact_manifest.json"
    runtime_path = directory / "agent07_runtime_report.json"
    resolution_path = directory / "multi_proposal_resolution_result.json"
    bundle_path = directory / "provisional_verification_traceability_bundle.json"

    manifest = _read_json(manifest_path)
    runtime = _read_json(runtime_path)
    resolution = _read_json(resolution_path)
    bundle = _read_json(bundle_path)

    if manifest.get("stage") != "07_agente_verificador":
        raise ValueError("AGENT08_AGENT07_MANIFEST_STAGE_INVALID")
    runtime_status = _nonempty_text(
        runtime.get("runtime_status"), "AGENT08_AGENT07_RUNTIME_STATUS_MISSING"
    )
    if runtime_status not in {"COMPLETED", "PARTIAL"}:
        raise ValueError("AGENT08_AGENT07_RUNTIME_NOT_EVALUABLE")
    if runtime.get("provisional_bundle") is None:
        raise ValueError("AGENT08_AGENT07_PROVISIONAL_BUNDLE_MISSING")
    if runtime.get("multi_proposal_resolution_result") is None:
        raise ValueError("AGENT08_AGENT07_RESOLUTION_MISSING")
    if bundle.get("aggregation_status") not in {"VALID", "PARTIAL"}:
        raise ValueError("AGENT08_AGENT07_AGGREGATION_NOT_EVALUABLE")
    if resolution.get("resolution_status") not in {"COMPLETED", "PARTIAL"}:
        raise ValueError("AGENT08_AGENT07_RESOLUTION_NOT_EVALUABLE")
    if _bool(resolution.get("eligible_for_07c")):
        raise ValueError("AGENT08_DIRECT_PATH_FORBIDDEN_07C_REQUIRED")

    metrics = bundle.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("AGENT08_AGENT07_METRICS_MISSING")
    claims_verified = int(metrics.get("claims_verified", -1))
    if claims_verified < 1:
        raise ValueError("AGENT08_AGENT07_CLAIM_COVERAGE_INVALID")
    accepted = int(metrics.get("corrections_accepted_for_07c", 0) or 0)
    accepted_claims = int(metrics.get("claims_with_accepted_proposals", 0) or 0)
    if accepted != 0 or accepted_claims != 0:
        raise ValueError("AGENT08_DIRECT_PATH_FORBIDDEN_ACCEPTED_CORRECTIONS")

    claim_rows = bundle.get("claim_traceability_rows")
    evidence_rows = bundle.get("claim_evidence_traceability_rows")
    correction_rows = bundle.get("correction_traceability_rows")
    if (
        not isinstance(claim_rows, list)
        or not isinstance(evidence_rows, list)
        or not isinstance(correction_rows, list)
    ):
        raise ValueError("AGENT08_AGENT07_TRACEABILITY_MISSING")
    if len(claim_rows) != claims_verified:
        raise ValueError("AGENT08_AGENT07_CLAIM_COVERAGE_MISMATCH")

    correction_by_claim: dict[str, list[Mapping[str, Any]]] = {}
    for row in correction_rows:
        if not isinstance(row, Mapping):
            raise ValueError("AGENT08_AGENT07_CORRECTION_TRACEABILITY_INVALID")
        claim_id = _nonempty_text(
            row.get("claim_id"), "AGENT08_CLAIM_ID_MISSING"
        )
        correction_by_claim.setdefault(claim_id, []).append(row)

    # claim_evidence_traceability_rows es el ARTEFACTO REAL de 07 que
    # contiene los pares claim -> evidencia -> source_filename -> chunk_id
    # (ClaimEvidenceTraceabilityRow, src/tools/verification/traceability.py)
    # -- ya acotado por 07 a la evidencia efectivamente considerada durante
    # la verificación (used_in_original_verification distingue "used" de
    # "rejected" dentro de ese conjunto, nunca el pool completo elegible;
    # ver validation.py, "for eid in sorted(used|rejected)"). Unión por
    # claim_id -- seguro DENTRO de un mismo bundle/ejecución de 07 (ambas
    # colecciones las produce la MISMA corrida sobre los MISMOS claims;
    # la inestabilidad de claim_id es un problema de comparar ENTRE
    # rondas, no dentro de un artefacto). Solo se incluye la evidencia
    # genuinamente usada como soporte (used_in_original_verification=True)
    # -- la rechazada no cuenta como "evidencia presente" para el claim.
    evidence_by_claim: dict[str, list[Mapping[str, Any]]] = {}
    for row in evidence_rows:
        if not isinstance(row, Mapping):
            raise ValueError("AGENT08_AGENT07_EVIDENCE_TRACEABILITY_INVALID")
        claim_id = _nonempty_text(
            row.get("claim_id"), "AGENT08_CLAIM_ID_MISSING"
        )
        if row.get("used_in_original_verification") is not True:
            continue
        source_filename = str(row.get("source_filename") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        # Fail-closed (requisito 8): un par con source/chunk vacío o
        # inválido nunca se propaga como si fuera una evidencia real --
        # se descarta explícitamente, nunca se infiere ni se rellena.
        if not source_filename or not chunk_id:
            continue
        evidence_by_claim.setdefault(claim_id, []).append(row)

    compatibility_rows: list[dict[str, Any]] = []
    for row in claim_rows:
        if not isinstance(row, Mapping):
            raise ValueError("AGENT08_AGENT07_CLAIM_TRACEABILITY_INVALID")
        claim_id = _nonempty_text(
            row.get("claim_id"), "AGENT08_CLAIM_ID_MISSING"
        )
        linked = correction_by_claim.get(claim_id, [])
        proposal_statuses = sorted({
            str(item.get("proposal_status") or "").strip()
            for item in linked
            if str(item.get("proposal_status") or "").strip()
        })
        comparison_availability = sorted({
            str(item.get("comparison_stage_availability") or "").strip()
            for item in linked
            if str(item.get("comparison_stage_availability") or "").strip()
        })
        base_row = {
            "claim_id": claim_id,
            "section_id": str(row.get("section_id") or "").strip(),
            "claim": str(row.get("original_claim_text") or "").strip(),
            "verdict": str(row.get("source_verdict") or "").strip(),
            "hallucination_risk": str(
                row.get("source_hallucination_risk") or ""
            ).strip(),
            "manual_review_required": bool(
                row.get("manual_review_required") is True
            ),
            "proposal_status": "|".join(proposal_statuses),
            "comparison_stage_availability": "|".join(
                comparison_availability
            ),
            "correction_needed": False,
            "correction_applied": False,
            "source_stage": SOURCE_STAGE_AGENT07,
        }
        claim_evidence = evidence_by_claim.get(claim_id, ())
        if claim_evidence:
            # build_claim_audit_rows agrupa por claim_id y lee source_
            # filename/chunk_id de CADA fila -- una fila por par de
            # evidencia, con los campos a nivel de claim repetidos (la
            # misma forma que ya produce build_claim_audit_rows para
            # cualquier otro campo multi-fila, ej. "verdict" via
            # first_non_empty).
            for evidence_row in sorted(
                claim_evidence, key=lambda e: (str(e.get("source_filename")), str(e.get("chunk_id")))
            ):
                compatibility_rows.append({
                    **base_row,
                    "source_filename": str(evidence_row.get("source_filename") or "").strip(),
                    "chunk_id": str(evidence_row.get("chunk_id") or "").strip(),
                })
        else:
            # Sin evidencia usada real -- nunca se inventa un par; la fila
            # queda sin source_filename/chunk_id, exactamente como antes.
            compatibility_rows.append(dict(base_row))

    manual_ids = _manual_review_claims(claim_rows)
    declared_manual = int(metrics.get("claims_requiring_manual_review", -1))
    if declared_manual != len(manual_ids):
        raise ValueError("AGENT08_AGENT07_MANUAL_REVIEW_COUNT_MISMATCH")

    draft_json = Path(draft_json_path)
    draft_md = Path(draft_markdown_path)
    _, draft_fingerprint = _validate_generated_draft(
        draft_json,
        draft_md,
        str(manifest.get("source_draft_fingerprint") or "") or None,
    )

    # Agent 07 did not apply corrections, so Agent 08 evaluates the committed
    # Agent 06 draft while preserving Agent 07's terminal traceability.
    if manifest.get("correction_applied") is not False:
        raise ValueError("AGENT08_AGENT07_CORRECTION_APPLIED_INCONSISTENT")
    if manifest.get("evaluation_ready_emitted") is not False:
        raise ValueError("AGENT08_AGENT07_EVALUATION_READY_MUST_NOT_BE_EMITTED")

    return Agent08UpstreamInput(
        schema_version=AGENT08_UPSTREAM_SCHEMA_VERSION,
        source_stage=SOURCE_STAGE_AGENT07,
        reverification_performed=False,
        reverification_reason=NO_ACCEPTED_CORRECTIONS,
        upstream_runtime_status=runtime_status,
        claims_verified=claims_verified,
        claims_requiring_manual_review=len(manual_ids),
        manual_review_claim_ids=manual_ids,
        generated_state_of_art_json_path=str(draft_json),
        generated_state_of_art_markdown_path=str(draft_md),
        traceability_rows=tuple(compatibility_rows),
        numeric_check_rows=(),
        claim_report_rows=tuple(compatibility_rows),
        provenance={
            "agent07_manifest_path": str(manifest_path),
            "agent07_manifest_sha256": _sha256_file(manifest_path),
            "agent07_runtime_report_path": str(runtime_path),
            "agent07_runtime_report_sha256": _sha256_file(runtime_path),
            "agent07_bundle_path": str(bundle_path),
            "agent07_bundle_sha256": _sha256_file(bundle_path),
            "agent07_resolution_path": str(resolution_path),
            "agent07_resolution_sha256": _sha256_file(resolution_path),
            "source_draft_fingerprint": draft_fingerprint,
            "correction_applied": False,
            "evaluation_ready_emitted": False,
            "claims_eligible_for_07c": 0,
        },
    )


def build_agent08_input_from_agent07c(
    *,
    agent07c_directory: str | Path,
) -> Agent08UpstreamInput:
    """Build the historical 07→07C→08 input from approved 07C outputs."""
    directory = Path(agent07c_directory)
    manifest_path = directory / "post_correction_recheck_manifest.json"
    validation_path = directory / "post_correction_recheck_validation_report.json"
    json_path = directory / "verified_state_of_art_EVALUATION_READY.json"
    md_path = directory / "verified_state_of_art_EVALUATION_READY.md"

    manifest = _read_json(manifest_path)
    validation = _read_json(validation_path)
    evaluation_ready = _read_json(json_path)
    workflow = manifest.get("workflow_state")
    if not isinstance(workflow, Mapping):
        raise ValueError("AGENT08_AGENT07C_WORKFLOW_STATE_MISSING")
    if not _bool(workflow.get("post_correction_recheck_completed")):
        raise ValueError("AGENT08_AGENT07C_REVERIFICATION_INCOMPLETE")
    if not _bool(workflow.get("all_applied_corrections_rechecked")):
        raise ValueError("AGENT08_AGENT07C_CORRECTIONS_NOT_RECHECKED")
    if not _bool(workflow.get("approved_for_final_evaluation")):
        raise ValueError("AGENT08_AGENT07C_NOT_APPROVED_FOR_EVALUATION")
    if not _bool(workflow.get("evaluation_ready_copy_created")):
        raise ValueError("AGENT08_AGENT07C_EVALUATION_READY_MISSING")
    if not _bool(validation.get("validation_ok")):
        raise ValueError("AGENT08_AGENT07C_VALIDATION_FAILED")

    _validate_generated_draft(json_path, md_path)
    summary = evaluation_ready.get("verification_summary", {})
    claims_verified = int(summary.get("total_claims", 0) or 0)
    pending = int(manifest.get("counts", {}).get("pending_manual_review", 0) or 0)

    return Agent08UpstreamInput(
        schema_version=AGENT08_UPSTREAM_SCHEMA_VERSION,
        source_stage=SOURCE_STAGE_AGENT07C,
        reverification_performed=True,
        reverification_reason=ACCEPTED_CORRECTIONS_REVERIFIED,
        upstream_runtime_status=str(workflow.get("final_status") or "COMPLETED"),
        claims_verified=claims_verified,
        claims_requiring_manual_review=pending,
        manual_review_claim_ids=(),
        generated_state_of_art_json_path=str(json_path),
        generated_state_of_art_markdown_path=str(md_path),
        traceability_rows=(),
        numeric_check_rows=(),
        claim_report_rows=(),
        provenance={
            "agent07c_manifest_path": str(manifest_path),
            "agent07c_manifest_sha256": _sha256_file(manifest_path),
            "agent07c_validation_path": str(validation_path),
            "agent07c_validation_sha256": _sha256_file(validation_path),
        },
    )


def resolve_agent08_upstream_input(
    *,
    agent07_directory: str | Path,
    draft_json_path: str | Path,
    draft_markdown_path: str | Path,
    agent07c_directory: str | Path | None = None,
) -> Agent08UpstreamInput:
    """Prefer a valid 07C result; otherwise use the strictly gated direct path."""
    if agent07c_directory is not None:
        directory = Path(agent07c_directory)
        if (directory / "post_correction_recheck_manifest.json").exists():
            return build_agent08_input_from_agent07c(agent07c_directory=directory)
    return build_agent08_input_from_committed_agent07(
        agent07_directory=agent07_directory,
        draft_json_path=draft_json_path,
        draft_markdown_path=draft_markdown_path,
    )
