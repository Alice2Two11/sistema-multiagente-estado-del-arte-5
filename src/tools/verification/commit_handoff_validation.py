"""Validación de handoff/commit del Agente 06 hacia el Agente 07.

Extraído mecánicamente de src.tools.verification.validation (Bloque C,
C1) -- sin cambios de comportamiento, lógica, firmas ni mensajes de
error respecto al código original. validation.py reexporta estos
símbolos para preservar el contrato de importación existente.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from src.config.verification_policy_config import (
    CLAIM_TYPES,
    RESOLUTION_ISSUE_CODES,
    RESOLUTION_STATUSES,
    RETRIEVAL_REASON_CODES,
    TEXT_MATCH_STATUSES,
    VERIFICATION_INTENSITIES,
    get_verification_input_policy,
)
try:
    from src.contracts.agent_input import ArtifactReference
    from src.contracts.agent_result import ExecutionStatus, TransitionAction
    from src.state.pipeline_state import DecisionLogEntry, PipelineState
    from .evidence import provisional_evidence_schema, resolve_committed_artifact_reference
except ModuleNotFoundError:  # paquete mínimo auditable sin runtime contractual
    ArtifactReference = Any  # type: ignore
    DecisionLogEntry = Any  # type: ignore
    PipelineState = Any  # type: ignore
    class ExecutionStatus:
        COMPLETED = "COMPLETED"
    class TransitionAction:
        ADVANCE = "ADVANCE"
    def provisional_evidence_schema(): return ()
    def resolve_committed_artifact_reference(*args, **kwargs):
        raise RuntimeError("CONTRACT_RUNTIME_NOT_INCLUDED_IN_MINIMAL_PACKAGE")

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_sha256_hex(value: Any, *, field: str = "fingerprint", allow_none: bool = False) -> str | None:
    """Central formal validator for lowercase SHA-256 contract values."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"SHA256_INVALID:{field}")
    return value

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CommittedAgent06Input:
    stage_name: str
    experiment_id: str
    official_output_directory: str
    stage_execution_status: str
    stage_quality_status: str
    decision_code: str
    transition_action: str
    decision_id: str
    fingerprint: str | None
    artifacts: Mapping[str, ArtifactReference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "experiment_id": self.experiment_id,
            "official_output_directory": self.official_output_directory,
            "stage_execution_status": self.stage_execution_status,
            "stage_quality_status": self.stage_quality_status,
            "decision_code": self.decision_code,
            "transition_action": self.transition_action,
            "decision_id": self.decision_id,
            "fingerprint": self.fingerprint,
            "artifacts": {name: ref.to_dict() for name, ref in self.artifacts.items()},
        }


def validate_provisional_evidence_output(frame: pd.DataFrame) -> None:
    expected = provisional_evidence_schema()
    if tuple(frame.columns) != expected:
        raise ValueError("INVALID_PROVISIONAL_EVIDENCE_SCHEMA")
    if frame["claim_id"].astype(str).duplicated().any():
        raise ValueError("DUPLICATE_PROVISIONAL_CLAIM_ID")
    for row in frame.to_dict("records"):
        if row["claim_type"] not in CLAIM_TYPES:
            raise ValueError(f"UNKNOWN_CLAIM_TYPE:{row['claim_type']}")
        if row["verification_intensity"] not in VERIFICATION_INTENSITIES:
            raise ValueError(f"UNKNOWN_VERIFICATION_INTENSITY:{row['verification_intensity']}")
        if row["resolution_status"] not in RESOLUTION_STATUSES:
            raise ValueError(f"INVALID_RESOLUTION_STATUS:{row['resolution_status']}")
        issues = tuple(row["resolution_issue_codes"])
        invalid_issues = [code for code in issues if code not in RESOLUTION_ISSUE_CODES]
        if invalid_issues:
            raise ValueError(f"INVALID_RESOLUTION_ISSUE_CODE:{invalid_issues[0]}")
        expected_status = issues[0] if issues else ("INHERITED_EVIDENCE_EMPTY" if row["unique_evidence_pair_count"] == 0 else "RESOLVED")
        if row["resolution_status"] != expected_status:
            raise ValueError(f"RESOLUTION_STATUS_PRECEDENCE_INCONSISTENT:{row['claim_id']}")
        if row["text_match_status"] not in TEXT_MATCH_STATUSES:
            raise ValueError(f"INVALID_TEXT_MATCH_STATUS:{row['text_match_status']}")
        reasons = tuple(row["retrieval_reason_codes"])
        invalid = [reason for reason in reasons if reason not in RETRIEVAL_REASON_CODES]
        if invalid:
            raise ValueError(f"INVALID_RETRIEVAL_REASON_CODE:{invalid[0]}")
        if bool(reasons) != bool(row["retrieval_required"]):
            raise ValueError(f"RETRIEVAL_REQUIREMENT_INCONSISTENT:{row['claim_id']}")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _latest_stage_decision(state: PipelineState, stage_name: str) -> DecisionLogEntry:
    for entry in reversed(state.decision_log):
        if entry.stage == stage_name or entry.agent == stage_name:
            return entry
    raise ValueError("AGENT06_COMMITTED_DECISION_NOT_FOUND")


def _require_result_field(result: Mapping[str, Any], key: str, *, allow_legacy_incomplete: bool) -> Any:
    if key not in result or result.get(key) is None:
        if allow_legacy_incomplete:
            return None
        raise ValueError(f"AGENT06_COMMITTED_RESULT_FIELD_MISSING:{key}")
    return result[key]


def _validate_committed_result(
    entry: DecisionLogEntry,
    *,
    accepted_quality_statuses: tuple[str, ...],
    required_decision_code: str,
    required_transition_action: str,
    allow_legacy_incomplete: bool,
) -> Mapping[str, Any]:
    result = entry.result
    if not isinstance(result, Mapping):
        raise ValueError("AGENT06_COMMITTED_RESULT_INVALID")
    execution_status = _require_result_field(result, "execution_status", allow_legacy_incomplete=allow_legacy_incomplete)
    quality_status = _require_result_field(result, "quality_status", allow_legacy_incomplete=allow_legacy_incomplete)
    decision = _require_result_field(result, "decision", allow_legacy_incomplete=allow_legacy_incomplete)
    transition = _require_result_field(result, "requested_transition", allow_legacy_incomplete=allow_legacy_incomplete)
    output_artifacts = _require_result_field(result, "output_artifacts", allow_legacy_incomplete=allow_legacy_incomplete)
    if execution_status is not None and str(execution_status) != ExecutionStatus.COMPLETED.value:
        raise ValueError("AGENT06_DECISION_RESULT_NOT_COMPLETED")
    if quality_status is not None and str(quality_status) not in accepted_quality_statuses:
        raise ValueError("AGENT06_DECISION_RESULT_NOT_APPROVED")
    if decision is not None:
        if not isinstance(decision, Mapping):
            raise ValueError("AGENT06_COMMITTED_RESULT_DECISION_INVALID")
        if str(decision.get("code", "")).strip() != required_decision_code:
            raise ValueError("AGENT06_COMMITTED_RESULT_DECISION_NOT_APPROVED")
    if transition is not None:
        if not isinstance(transition, Mapping):
            raise ValueError("AGENT06_COMMITTED_RESULT_TRANSITION_INVALID")
        if str(transition.get("action", "")).strip() != required_transition_action:
            raise ValueError("AGENT06_COMMITTED_RESULT_TRANSITION_NOT_ADVANCE")
    if output_artifacts is not None and not isinstance(output_artifacts, Mapping):
        raise ValueError("AGENT06_COMMITTED_RESULT_OUTPUT_ARTIFACTS_INVALID")
    return result


def _validate_manifest(path: Path, expected_fingerprint: str | None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("AGENT06_MANIFEST_INVALID") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("AGENT06_MANIFEST_INVALID")
    manifest_fingerprint = payload.get("fingerprint")
    if expected_fingerprint:
        if not isinstance(manifest_fingerprint, str) or not manifest_fingerprint.strip():
            raise ValueError("AGENT06_MANIFEST_FINGERPRINT_MISSING")
        if manifest_fingerprint.strip() != expected_fingerprint:
            raise ValueError("AGENT06_MANIFEST_FINGERPRINT_MISMATCH")
    return dict(payload)


def validate_committed_agent06_input(
    state: PipelineState,
    *,
    experiment_directory: str | Path,
    policy: Mapping[str, Any] | None = None,
) -> CommittedAgent06Input:
    if not isinstance(state, PipelineState):
        raise TypeError("state must be PipelineState")
    effective = get_verification_input_policy(policy)
    stage_name = effective["agent06_stage_name"]
    stage = state.stages.get(stage_name)
    if stage is None:
        raise ValueError("AGENT06_STAGE_NOT_COMMITTED")
    if stage.execution_status is not ExecutionStatus.COMPLETED:
        raise ValueError("AGENT06_EXECUTION_NOT_COMPLETED")
    if stage.quality_status is None or stage.quality_status.value not in effective["accepted_agent06_quality_statuses"]:
        raise ValueError("AGENT06_QUALITY_NOT_APPROVED")
    transition = stage.requested_transition
    if transition is None or transition.action is not TransitionAction.ADVANCE:
        raise ValueError("AGENT06_TRANSITION_NOT_ADVANCE")
    if state.pending_execution is not None:
        raise ValueError("PIPELINE_PENDING_EXECUTION_INCOMPATIBLE")

    decision_entry = _latest_stage_decision(state, stage_name)
    decision_code = str(decision_entry.decision.get("code", "")).strip()
    if decision_code != effective["required_agent06_decision_code"]:
        raise ValueError("AGENT06_DECISION_NOT_APPROVED")
    if decision_entry.requested_transition is None or decision_entry.requested_transition.action is not TransitionAction.ADVANCE:
        raise ValueError("AGENT06_DECISION_TRANSITION_NOT_ADVANCE")

    committed_result = _validate_committed_result(
        decision_entry,
        accepted_quality_statuses=effective["accepted_agent06_quality_statuses"],
        required_decision_code=effective["required_agent06_decision_code"],
        required_transition_action=effective["required_agent06_transition_action"],
        allow_legacy_incomplete=effective["allow_legacy_incomplete_committed_result"],
    )
    committed_output_artifacts = committed_result.get("output_artifacts", {})

    experiment_dir = Path(experiment_directory).resolve()
    official_dir = (experiment_dir / "05_outputs" / effective["official_draft_directory_name"]).resolve()
    forbidden_dirs = {(experiment_dir / "05_outputs" / name).resolve() for name in effective["forbidden_draft_directory_names"]}
    artifacts: dict[str, ArtifactReference] = {}
    for name in effective["required_agent06_artifacts"]:
        reference = resolve_committed_artifact_reference(
            state,
            committed_output_artifacts,
            name,
            producer_stage=stage_name,
            allow_basename_fallback=effective["allow_artifact_basename_fallback"],
        )
        path = Path(reference.path).resolve()
        if any(_is_within(path, forbidden) for forbidden in forbidden_dirs):
            raise ValueError(f"AGENT06_STAGING_ARTIFACT_REJECTED:{name}")
        if not _is_within(path, official_dir):
            raise ValueError(f"AGENT06_ARTIFACT_OUTSIDE_OFFICIAL_OUTPUT:{name}")
        if not path.is_file():
            raise ValueError(f"AGENT06_COMMITTED_ARTIFACT_FILE_MISSING:{name}")
        if _SHA256_RE.fullmatch(reference.hash) and sha256_file(path) != reference.hash.lower():
            raise ValueError(f"AGENT06_COMMITTED_ARTIFACT_HASH_MISMATCH:{name}")
        artifacts[name] = reference

    fingerprint = stage.fingerprints.composite
    _validate_manifest(Path(artifacts["draft_generation_manifest.json"].path), fingerprint)
    validation_report = json.loads(Path(artifacts["draft_validation_report.json"].path).read_text(encoding="utf-8"))
    if not bool(validation_report.get("validation_ok", False)):
        raise ValueError("AGENT06_VALIDATION_REPORT_NOT_APPROVED")

    return CommittedAgent06Input(
        stage_name=stage_name,
        experiment_id=state.identity.experiment_id,
        official_output_directory=str(official_dir),
        stage_execution_status=stage.execution_status.value,
        stage_quality_status=stage.quality_status.value,
        decision_code=decision_code,
        transition_action=transition.action.value,
        decision_id=decision_entry.decision_id,
        fingerprint=fingerprint,
        artifacts=artifacts,
    )
