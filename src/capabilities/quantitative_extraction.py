# ============================================================
# 03B - EXTRACCIÓN CUANTITATIVA DE EVIDENCIA CIENTÍFICA
# Identifica y estructura métricas, valores numéricos, datasets
# y resultados cuantitativos respaldados por los papers.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from src.contracts.agent_input import AgentInput, ArtifactReference
from src.contracts.agent_result import (
    AgentResult, AgentWarning, DecisionInfo, ExecutionStatus, QualityStatus,
    RequestedTransition, ToolUsage, TransitionAction, WarningSeverity,
)
from src.state.fingerprints import build_stage_fingerprints, sha256_file
from src.config.quantitative_extraction_policy_config import (
    STAGE_NAME, QUANT_PROMPT_VERSION, QUANT_SCHEMA_VERSION,
    QUANT_FLATTENING_VERSION, ARTIFACT_FILENAMES, validate_quantitative_policy,
)
from src.tools.quantitative_extraction.input_validation import load_and_validate_inputs, reject_ground_truth_payload
from src.tools.quantitative_extraction.extraction import extract_quantitative_records
from src.tools.quantitative_extraction.normalization import flatten_results
from src.tools.quantitative_extraction.evidence_verification import verify_quantitative_rows
from src.tools.quantitative_extraction.quality import calculate_diagnostic_metrics, diagnostic_quality_status
from src.tools.quantitative_extraction.artifacts import write_quantitative_artifacts

# Define las dependencias externas que necesita la extracción cuantitativa:
# el LLM, la fábrica de mensajes y el parser que interpreta la respuesta JSON.
@dataclass(frozen=True)
class QuantitativeExtractionDependencies:
    llm: Any
    human_message_factory: Callable[..., Any]
    json_parser: Callable[[str], Any]

#crea una huella reproducible de la ejecución de 03B para detectar si cambiaron las entradas, 
#las reglas, el modelo o alguna versión importante del procesamiento.
def build_quantitative_composite_fingerprint(agent_input: AgentInput, policy):
    return build_stage_fingerprints(
        input_data={"experiment_id": agent_input.experiment_id, "run_id": agent_input.run_id},
        config_data={
            "policy": policy,
            "model": agent_input.agent_context.runtime_resources.get("model"),
            "prompt_version": QUANT_PROMPT_VERSION,
            "schema_version": QUANT_SCHEMA_VERSION,
            "flattening_version": QUANT_FLATTENING_VERSION,
        },
        dependencies_data={k: v.to_dict() for k, v in agent_input.dependencies.items()},
    )

# Lee un archivo JSONL línea por línea y convierte cada registro
# válido en un diccionario de Python.
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records

# Ejecuta toda la capacidad de extracción cuantitativa:
# valida entradas, extrae o reutiliza resultados, verifica los datos,
# genera métricas, guarda artefactos y devuelve el resultado de 03B.
class QuantitativeExtractionCapability:
    def __init__(self, dependencies: QuantitativeExtractionDependencies):
        self.dependencies = dependencies

    def execute(self, agent_input: AgentInput) -> AgentResult:
        started = datetime.now(timezone.utc).isoformat()
        try:
            if agent_input.stage_name != STAGE_NAME:
                raise ValueError("INVALID_CONFIGURATION: stage_name")
            if agent_input.attempt_number != 1:
                raise ValueError("INVALID_CONFIGURATION: 03B solo admite attempt_number=1")
            reject_ground_truth_payload(agent_input.to_dict())
            policy = validate_quantitative_policy(agent_input.policy)
            df, chunks, _source_manifest = load_and_validate_inputs(
                experiment_id=agent_input.experiment_id,
                run_id=agent_input.run_id,
                dependencies=agent_input.dependencies,
                policy=policy,
            )
            output_dir = Path(agent_input.agent_context.output_directory)
            fingerprints = build_quantitative_composite_fingerprint(agent_input, policy)
            manifest_path = output_dir / "quantitative_extraction_manifest.json"

            if not policy["force_rebuild"] and not policy["deterministic_flattening_repair"] and manifest_path.is_file():
                old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing = {name: output_dir / name for name in ARTIFACT_FILENAMES}
                if old_manifest.get("fingerprint") == fingerprints.composite and all(path.is_file() for path in existing.values()):
                    return AgentResult(
                        execution_status=ExecutionStatus.COMPLETED,
                        quality_status=QualityStatus.APPROVED_WITH_WARNINGS,
                        decision=DecisionInfo(code="QUANTITATIVE_EXTRACTION_REUSED", rationale="Se reutilizaron los nueve artefactos porque el fingerprint no cambió."),
                        quality_metrics=old_manifest.get("metrics", {}),
                        warnings=(AgentWarning(code="REUSED_BY_FINGERPRINT", severity=WarningSeverity.INFO, blocking=False, message="Artefactos 03B reutilizados."),),
                        failure_reason_codes=(),
                        requested_transition=RequestedTransition(action=TransitionAction.ADVANCE, target_stage=None, reason_code="QUANTITATIVE_EXTRACTION_REUSED", requires_human_confirmation=False),
                        output_artifacts={name: ArtifactReference(path=str(path), hash=sha256_file(path)) for name, path in existing.items()},
                        tool_usage=ToolUsage(), attempt_number=1, started_at=started,
                        completed_at=datetime.now(timezone.utc).isoformat(), error=None,
                    )

            preserve_sources = bool(policy["deterministic_flattening_repair"])
            if preserve_sources:
                structured_path = output_dir / ARTIFACT_FILENAMES[0]
                raw_path = output_dir / ARTIFACT_FILENAMES[1]
                if not structured_path.is_file() or not raw_path.is_file():
                    raise FileNotFoundError("DEPENDENCY_NOT_FOUND: faltan JSON estructurados para reparación determinista")
                before_hashes = {structured_path.name: sha256_file(structured_path), raw_path.name: sha256_file(raw_path)}
                results = json.loads(structured_path.read_text(encoding="utf-8"))
                raw = _read_jsonl(raw_path)
                errors = []
                calls = 0
            else:
                before_hashes = {}
                results, raw, errors, calls = extract_quantitative_records(
                    df,
                    llm=self.dependencies.llm,
                    human_message_factory=self.dependencies.human_message_factory,
                    json_parser=self.dependencies.json_parser,
                )

            flattened = flatten_results(results)
            errors = list(errors) + list(flattened.issues)
            kb_rows = {str(row["source_filename"]): row for _, row in df.iterrows()}
            quantitative = verify_quantitative_rows(
                flattened.quantitative,
                kb_rows_by_source=kb_rows,
                chunks=chunks,
                allow_all_clean_chunks_fallback=policy["allow_all_clean_chunks_fallback"],
            )
            metrics = calculate_diagnostic_metrics(
                papers_processed=len(df), quantitative_rows=quantitative,
                dataset_rows=flattened.datasets, technique_rows=flattened.techniques,
                error_rows=errors, raw_summary=flattened.raw_summary,
                flattened_summary=flattened.flattened_summary,
            )
            fallback_used = any(r.get("source_chunk_scope") == "all_clean_chunks_fallback" for r in quantitative)
            status_text, reasons = diagnostic_quality_status(metrics, fallback_used=fallback_used, error_count=0 if preserve_sources else len([e for e in errors if e.get("error_type") == "LLM_EXTRACTION_ERROR"]))
            written, manifest = write_quantitative_artifacts(
                output_dir=output_dir, results=results, raw_records=raw, errors=errors,
                quantitative_rows=quantitative, dataset_rows=flattened.datasets,
                technique_rows=flattened.techniques, metrics=metrics,
                manifest_base={
                    "stage": STAGE_NAME, "experiment_id": agent_input.experiment_id,
                    "run_id": agent_input.run_id, "fingerprint": fingerprints.composite,
                    "input_dependency": {k: v.to_dict() for k, v in agent_input.dependencies.items()},
                    "policy_status": "PROVISIONAL_DIAGNOSTIC",
                    "flattening_repair_source_hashes": before_hashes,
                },
                preserve_structured_sources=preserve_sources,
            )
            if preserve_sources:
                after_hashes = {name: sha256_file(output_dir / name) for name in ARTIFACT_FILENAMES[:2]}
                if before_hashes != after_hashes:
                    raise RuntimeError("ATOMIC_WRITE_FAILED: los JSON originales fueron modificados")

            quality = QualityStatus(status_text)
            warnings = tuple(AgentWarning(code=r, severity=WarningSeverity.WARNING, blocking=False, message=r) for r in reasons)
            decision_code = "QUANTITATIVE_FLATTENING_REPAIRED" if preserve_sources else "QUANTITATIVE_EXTRACTION_COMPLETED"
            return AgentResult(
                execution_status=ExecutionStatus.COMPLETED, quality_status=quality,
                decision=DecisionInfo(code=decision_code, rationale="La capacidad 03B produjo sus nueve artefactos con validación JSON↔tablas."),
                quality_metrics={"technical": {"llm_calls": calls}, "scientific": metrics},
                warnings=warnings, failure_reason_codes=tuple(reasons),
                requested_transition=RequestedTransition(action=TransitionAction.ADVANCE if quality in {QualityStatus.APPROVED, QualityStatus.APPROVED_WITH_WARNINGS} else TransitionAction.HALT_STAGE, target_stage=None, reason_code=decision_code, requires_human_confirmation=False),
                output_artifacts={name: ArtifactReference(path=result.path, hash=result.hash) for name, result in written.items()},
                tool_usage=ToolUsage(llm_calls=calls), attempt_number=1, started_at=started,
                completed_at=datetime.now(timezone.utc).isoformat(), error=None,
            )
        except Exception as exc:
            text = str(exc)
            known = (
                "GROUND_TRUTH_POLICY_VIOLATION", "KB_MANIFEST_MISMATCH",
                "DEPENDENCY_HASH_MISMATCH", "DEPENDENCY_NOT_FOUND",
                "INVALID_KB_SCHEMA", "DUPLICATE_KB_SOURCE", "NO_ELIGIBLE_PAPERS",
                "SOURCE_CHUNKS_SCHEMA_INVALID", "INVALID_CONFIGURATION", "ATOMIC_WRITE_FAILED",
            )
            code = next((item for item in known if text.startswith(item)), "DEPENDENCY_NOT_FOUND" if isinstance(exc, FileNotFoundError) else "RUNTIME_DEPENDENCY_FAILED")
            safe = "Error sanitizado durante la capacidad 03B." if any(x in text for x in ("sk-", "OPENAI_API_KEY", "openai_api_key")) else text
            return AgentResult(
                execution_status=ExecutionStatus.FAILED, quality_status=QualityStatus.REJECTED,
                decision=DecisionInfo(code="QUANTITATIVE_EXTRACTION_FAILED", rationale="La capacidad 03B no pudo completar la ejecución."),
                quality_metrics={"technical": {}, "scientific": {}},
                warnings=(AgentWarning(code=code, severity=WarningSeverity.ERROR, blocking=True, message=safe),),
                failure_reason_codes=(code,),
                requested_transition=RequestedTransition(action=TransitionAction.HALT_STAGE, target_stage=None, reason_code=code, requires_human_confirmation=False),
                output_artifacts={}, tool_usage=ToolUsage(), attempt_number=1,
                started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
                error={"type": type(exc).__name__, "message": safe, "stage": STAGE_NAME},
            )
