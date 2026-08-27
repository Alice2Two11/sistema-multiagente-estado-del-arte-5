from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd

from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import (
    AgentResult,
    AgentWarning,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
    WarningSeverity,
)
from src.state.fingerprints import sha256_file
from src.tools.thematic_analysis.artifacts import (
    ARTIFACT_FILENAMES,
    thematic_table_counts,
    write_deterministic_thematic_repair_artifacts,
)
from src.tools.thematic_analysis.coverage_validation import calculate_diagnostic_metrics
from src.tools.thematic_analysis.quality import classify_quality
from src.tools.thematic_analysis.reference_validation import validate_references
from src.tools.thematic_analysis.schema_validation import (
    THEMATIC_ALIAS_VERSION,
    inspect_thematic_payload,
    normalize_thematic_output,
    validate_json_to_tables,
)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def _existing_artifact_refs(output_dir: Path) -> dict[str, ArtifactReference]:
    refs: dict[str, ArtifactReference] = {}
    for name, filename in ARTIFACT_FILENAMES.items():
        path = output_dir / filename
        if path.is_file():
            refs[name] = ArtifactReference(str(path), sha256_file(path))
    return refs


def execute_deterministic_thematic_repair(*, output_dir: str | Path, attempt_number: int = 2) -> AgentResult:
    """Repair flattening from the persisted thematic JSON without invoking an LLM."""
    started = datetime.now(timezone.utc).isoformat()
    directory = Path(output_dir)
    original_json = directory / ARTIFACT_FILENAMES["analysis"]
    original_raw = directory / ARTIFACT_FILENAMES["raw"]
    final_kb = directory / ARTIFACT_FILENAMES["kb_final"]
    excluded_kb = directory / ARTIFACT_FILENAMES["kb_excluded"]
    original_json_hash = sha256_file(original_json)
    original_raw_hash = sha256_file(original_raw) if original_raw.is_file() else None

    payload = json.loads(original_json.read_text(encoding="utf-8"))
    raw_counts = inspect_thematic_payload(payload)
    normalized, schema_issues, alias_repairs = normalize_thematic_output(payload, return_repairs=True)

    df_final = _load_csv(final_kb)
    _ = _load_csv(excluded_kb)
    ref_codes, ref_counts, _ = validate_references(normalized, df_final)
    table_counts = thematic_table_counts(normalized)
    flattening_codes, consistency = validate_json_to_tables(raw_counts, table_counts)

    codes = [item["code"] for item in schema_issues] + ref_codes + flattening_codes
    if not normalized["themes"]:
        codes.append("EMPTY_THEMATIC_OUTPUT")
    if not normalized["research_gaps"]:
        codes.append("EMPTY_THEMATIC_OUTPUT")
    if not normalized["comparative_dimensions"]:
        codes.append("EMPTY_THEMATIC_OUTPUT")

    metrics = calculate_diagnostic_metrics(normalized, df_final, ref_counts)
    codes = tuple(dict.fromkeys(codes))
    quality, action = classify_quality(codes, attempt_number, manual_allowed=False)

    repair_plan = [
        {"reason_code": "ALIAS_MAPPING_REQUIRED", "strategy": "DETERMINISTIC_ALIAS_MAPPING"},
        {"reason_code": "INVALID_THEMATIC_SCHEMA", "strategy": "REBUILD_DERIVED_TABLES_FROM_PERSISTED_JSON"},
    ]
    validation = {
        "validation_ok": not codes,
        "failure_reason_codes": list(codes),
        "metrics": metrics,
        "repair_plan": repair_plan,
        "repairs": alias_repairs,
        "json_to_tables_consistency": consistency,
        "deterministic_thematic_repair": True,
        "openai_called": False,
        "original_json_preserved": True,
        "original_json_sha256_before": original_json_hash,
        "original_raw_sha256_before": original_raw_hash,
        "alias_version": THEMATIC_ALIAS_VERSION,
    }
    manifest = {}
    manifest_path = directory / ARTIFACT_FILENAMES["manifest"]
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest.update({
        "attempt_number": attempt_number,
        "quality_status": quality.value,
        "diagnostic_metrics": metrics,
        "deterministic_thematic_repair": True,
        "openai_called": False,
        "original_json_preserved": True,
        "original_json_sha256": original_json_hash,
        "original_raw_sha256": original_raw_hash,
        "alias_version": THEMATIC_ALIAS_VERSION,
        "repair_count": len(alias_repairs),
        "json_to_tables_consistency": consistency,
    })
    repaired_refs = write_deterministic_thematic_repair_artifacts(directory, normalized, validation, manifest)

    if sha256_file(original_json) != original_json_hash:
        raise RuntimeError("ORIGINAL_THEMATIC_JSON_MODIFIED")
    if original_raw_hash is not None and sha256_file(original_raw) != original_raw_hash:
        raise RuntimeError("ORIGINAL_THEMATIC_RAW_MODIFIED")

    all_refs = _existing_artifact_refs(directory)
    all_refs.update(repaired_refs)
    warnings = tuple(
        AgentWarning(code=code, severity=WarningSeverity.WARNING, blocking=True, message=code)
        for code in codes
    )
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=quality,
        decision=DecisionInfo(
            code="THEMATIC_DETERMINISTIC_REPAIR_EVALUATED",
            rationale="Se reparó el flattening a partir del JSON temático persistido sin invocar OpenAI.",
        ),
        quality_metrics={"scientific": metrics, "technical": consistency},
        warnings=warnings,
        failure_reason_codes=codes,
        requested_transition=RequestedTransition(
            action=TransitionAction(action),
            target_stage=None,
            reason_code=quality.value,
            requires_human_confirmation=False,
        ),
        output_artifacts=all_refs,
        tool_usage=ToolUsage(llm_calls=0, validation_calls=1),
        attempt_number=attempt_number,
        started_at=started,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
