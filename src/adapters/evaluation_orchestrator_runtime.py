# ============================================================
# 08 - ORQUESTADOR DE EVALUACIÓN DEL ESTADO DEL ARTE
# Coordina las métricas y evaluaciones del resultado final,
# consolida los resultados y genera los reportes de calidad.
# ============================================================

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from src.state.fingerprints import build_stage_fingerprints
from src.tools.evaluation.evaluation_pipeline import run_evaluation_pipeline

EVALUATION_STAGE_NAME = "08_evaluacion_experimental"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

# Define la función principal que prepara y coordina toda la evaluación
# experimental del estado del arte generado.
#reúne las entradas de Stage08
def build_experimental_evaluation_execution(
    *,
    generated_plain_text: str,
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    traceability_rows: list[dict[str, Any]],
    source_stage: str,
    upstream_runtime_status: str,
    reverification_performed: bool,
    reverification_reason: str | None,
    claims_verified: int,
    claims_requiring_manual_review: int,
    manual_review_claim_ids: list[str],
    generated_status: str | None,
    evaluation_ready_json_path: str,
    experiment_id: str,
    topic_name: str,
    ground_truth_dir: str,
    evaluation_policy: dict[str, Any],
    translation_llm_factory: Callable[[], Any],
    embedding_model_factory: Callable[[str], Any] | None,
    bertscore_score_fn: Callable[..., Any] | None,
    judge_llm_factory: Callable[[], Any],
    upstream_fingerprint: str | None = None,
) -> dict[str, Any]:
    
    return {
        "generated_plain_text": generated_plain_text,
        "sections": sections,
        "chunks": chunks,
        "traceability_rows": traceability_rows,
        "source_stage": source_stage,
        "upstream_runtime_status": upstream_runtime_status,
        "reverification_performed": reverification_performed,
        "reverification_reason": reverification_reason,
        "claims_verified": claims_verified,
        "claims_requiring_manual_review": claims_requiring_manual_review,
        "manual_review_claim_ids": manual_review_claim_ids,
        "generated_status": generated_status,
        "evaluation_ready_json_path": evaluation_ready_json_path,
        "experiment_id": experiment_id,
        "topic_name": topic_name,
        "ground_truth_dir": ground_truth_dir,
        "evaluation_policy": evaluation_policy,
        "translation_llm_factory": translation_llm_factory,
        "embedding_model_factory": embedding_model_factory,
        "bertscore_score_fn": bertscore_score_fn,
        "judge_llm_factory": judge_llm_factory,
        "upstream_fingerprint": upstream_fingerprint,
    }

# Construye el resultado final de Stage08 a partir de la ejecución
# de la evaluación, registrando si terminó correctamente o con error.
def _build_evaluation_agent_result(
    *,
    attempt_number: int,
    started_at: str,
    exception: Exception | None,
    pipeline_result: dict[str, Any] | None,
    output_artifacts: dict[str, Any] | None = None,
    pipeline_outcome_metadata: dict[str, Any] | None = None,
) -> AgentResult:
    now = _now()
    # Prepara los metadatos del resultado del pipeline;
    # si no se proporcionan, asume inicialmente una ejecución exitosa.
    outcome_metadata = dict(pipeline_outcome_metadata or {"pipeline_outcome": "SUCCESS"})

   # Si Stage08 falla durante la evaluación, devuelve un resultado rechazado,
   # registra el error ocurrido y detiene completamente la etapa.
    if exception is not None:
        return AgentResult(
            execution_status=ExecutionStatus.FAILED,
            quality_status=QualityStatus.REJECTED,
            decision=DecisionInfo(code="EVALUATION_FAILED", rationale=str(exception)),
            quality_metrics={"technical": dict(outcome_metadata), "scientific": {}},
            warnings=(
                AgentWarning(
                    code=type(exception).__name__,
                    severity=WarningSeverity.ERROR,
                    blocking=True,
                    message=str(exception),
                ),
            ),
            failure_reason_codes=(type(exception).__name__,),
            requested_transition=RequestedTransition(
                action=TransitionAction.HALT_STAGE,
                target_stage=None,
                reason_code=type(exception).__name__,
                requires_human_confirmation=False,
            ),
            output_artifacts={},
            tool_usage=ToolUsage(),
            attempt_number=attempt_number,
            started_at=started_at,
            completed_at=now,
            error={"type": type(exception).__name__, "message": str(exception), "stage": EVALUATION_STAGE_NAME},
        )

    final_validation = pipeline_result["final_validation"]
    validation_errors = final_validation["validation_errors"]
    validation_warnings = final_validation["validation_warnings"]

   # Asigna el estado de calidad de Stage08 según los resultados
   # de la validación: rechazado, aprobado con advertencias o aprobado.
    if validation_errors:
        quality_status = QualityStatus.REJECTED
    elif validation_warnings:
        quality_status = QualityStatus.APPROVED_WITH_WARNINGS
    else:
        quality_status = QualityStatus.APPROVED

   # Convierte los avisos y errores de validación en objetos AgentWarning
   # para registrarlos de forma uniforme dentro del resultado de Stage08.
    warnings = tuple(
        AgentWarning(code=code, severity=WarningSeverity.WARNING, blocking=False, message=code)
        for code in validation_warnings
    ) + tuple(
        AgentWarning(code=code, severity=WarningSeverity.ERROR, blocking=False, message=code)
        for code in validation_errors
    )

   # Devuelve el resultado final de Stage08 cuando la evaluación terminó,
   # incluyendo el estado de calidad, métricas, advertencias y artefactos generados.
    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=quality_status,
        decision=DecisionInfo(
            code="EVALUATION_COMPLETED",
            rationale=final_validation["evaluation_validation_report"]["factual_consistency_status"],
        ),
        quality_metrics={
            "technical": dict(outcome_metadata),
            "scientific": {
                row["metric"]: row["value"] for row in pipeline_result["final_selected_metrics"]
            },
        },
        warnings=warnings,
        requested_transition=RequestedTransition(
            action=TransitionAction.ADVANCE,
            target_stage=None,  # última etapa de CANONICAL_STAGE_ORDER
            reason_code="EVALUATION_COMPLETED",
            requires_human_confirmation=False,
        ),
        output_artifacts=output_artifacts or {},
        tool_usage=ToolUsage(),
        attempt_number=attempt_number,
        started_at=started_at,
        completed_at=now,
    )

# Ejecuta Stage08 desde el orquestador, preparando el contexto
# de evaluación y controlando el intento actual y posibles reejecuciones.
def _run_evaluation_stage(
    *,
    store,
    project_dir,
    spec,
    attempt_number: int = 1,
    observations: dict[str, Any] | None = None,
    force_rerun: bool = False,
):

   # Importa las herramientas necesarias para controlar la ejecución de Stage08:
   # reproducibilidad, persistencia, estado del pipeline, ground truth y LLM Judge.
    from src.adapters.evaluation_fingerprint import (
        build_evaluation_signature,
        compute_evaluation_fingerprint,
    )
    from src.adapters.evaluation_persistence import find_missing_outputs
    from src.orchestration.pipeline_orchestrator import (
        _outcome_from_committed_stage,
        _outcome_from_result,
    )
    from src.tools.evaluation.ground_truth import resolve_ground_truth_comparable_text
    from src.tools.evaluation.llm_judge import PROMPT_VERSION as JUDGE_PROMPT_VERSION

   # Prepara las rutas, parámetros y metadatos necesarios
   # para ejecutar Stage08 con la configuración del experimento actual.
    project_dir = Path(project_dir)
    kwargs = dict(spec.build_execution(project_dir, attempt_number))
    output_dir = Path(kwargs.pop("output_dir"))
    numeric_check_output_dir = Path(kwargs.pop("numeric_check_output_dir"))
    backup_root = Path(kwargs.pop("backup_root"))
    openai_model_for_signature = kwargs.pop("_openai_model", "")
    upstream_fingerprint = kwargs.pop("upstream_fingerprint", None)
    pipeline_outcome_metadata = kwargs.pop("_pipeline_outcome_metadata", {"pipeline_outcome": "SUCCESS"})

    started_at = _now()

   # Carga y valida el ground truth comparable y, con esa información,
   # construye la firma completa de la evaluación para garantizar trazabilidad y reproducibilidad.
    try:
        ground_truth_plain_text, _gt_metadata, _gt_source_path = resolve_ground_truth_comparable_text( #busca y prepara el estado del arte real que se usará como referencia.
            ground_truth_dir=kwargs["ground_truth_dir"],
            minimum_words=kwargs["evaluation_policy"]["minimum_ground_truth_words"], #exige que ese ground truth tenga una longitud mínima válida
            require_explicit_end_heading=kwargs["evaluation_policy"][
                "require_explicit_ground_truth_end_heading"
            ],
        )
       # Construye una firma reproducible de la evaluación usando
       # la configuración, las entradas, la trazabilidad y las versiones utilizadas.
        evaluation_signature = build_evaluation_signature(
            experiment_id=kwargs["experiment_id"],
            evaluation_policy=kwargs["evaluation_policy"],
            openai_model=openai_model_for_signature,
            evaluation_ready_json_path=kwargs["evaluation_ready_json_path"],
            upstream_fingerprint=upstream_fingerprint,  # propagado real de 07 (state.stages[
            ground_truth_text=ground_truth_plain_text,
            chunks=kwargs["chunks"],
            traceability_rows=kwargs["traceability_rows"],
            llm_judge_prompt_version=JUDGE_PROMPT_VERSION,
        )
   # Si falla la preparación de la evaluación, registra el error en el estado
   # del pipeline, guarda fingerprints de respaldo y detiene Stage08.
    except Exception as exc:  
        result = _build_evaluation_agent_result(
            attempt_number=attempt_number, started_at=started_at, exception=exc, pipeline_result=None,
            pipeline_outcome_metadata=pipeline_outcome_metadata,
        )
        prepared = store.prepare_execution(
            target_stage=EVALUATION_STAGE_NAME, intended_action="EXECUTE_EVALUATION", attempt_number=attempt_number
        )
        store.persist_agent_result(prepared.decision_id, result)
        fallback_fingerprints = build_stage_fingerprints(
            input_data={"experiment_id": kwargs["experiment_id"], "error": type(exc).__name__},
            config_data=kwargs["evaluation_policy"],
            dependencies_data={},
        )
        store.commit_execution(
            decision_id=prepared.decision_id,
            result=result,
            stage_name=EVALUATION_STAGE_NAME,
            fingerprints=fallback_fingerprints,
            observations=dict(observations or {}),
        )
        state = store.load()
        attempts_used = state.stages[EVALUATION_STAGE_NAME].attempts_used
        return _outcome_from_result(spec, result, "FAILED", attempts_used=attempts_used)

    evaluation_fingerprint_hash = compute_evaluation_fingerprint(evaluation_signature)
    fingerprints = build_stage_fingerprints(
        input_data={"evaluation_signature_hash": evaluation_fingerprint_hash},
        config_data=kwargs["evaluation_policy"],
        dependencies_data={},
    )

    state = store.load()
    committed = state.stages.get(EVALUATION_STAGE_NAME)
    missing_outputs = find_missing_outputs(output_dir=output_dir)

   # Si Stage08 ya fue ejecutado con exactamente las mismas entradas y configuración,
   # y sus archivos de salida siguen completos, reutiliza el resultado sin volver a evaluar.
    if (
        committed is not None
        and committed.execution_status == ExecutionStatus.COMPLETED
        and not force_rerun
        and committed.fingerprints.composite == fingerprints.composite
        and not missing_outputs
    ):
        return _outcome_from_committed_stage(spec, committed, status="SKIPPED_FRESH")

   # Si existe una ejecución pendiente de Stage08, intenta resolverla
   # antes de iniciar una nueva evaluación.
    if state.pending_execution is not None and state.pending_execution.target_stage == EVALUATION_STAGE_NAME:
        resume = store.resolve_resume(
            stage_name=EVALUATION_STAGE_NAME, fingerprints=fingerprints, observations=observations
        )
        if resume.action == "COMMITTED":
            state = store.load()
            attempts_used = state.stages[EVALUATION_STAGE_NAME].attempts_used
            return _outcome_from_result(spec, resume.committed_result, "COMMITTED", attempts_used=attempts_used)
   
   # Importa las funciones encargadas de guardar, respaldar y presentar
   # los resultados producidos durante la evaluación de Stage08.
    from src.adapters.evaluation_persistence import (
        backup_existing_outputs,
        build_final_evaluation_report_markdown,
        persist_evaluation_outputs,
        persist_intermediate_numeric_check,
    )

   # Registra formalmente que Stage08 va a comenzar su ejecución
   # y prepara el contenedor donde se guardarán los artefactos generados.
    prepared = store.prepare_execution(
        target_stage=EVALUATION_STAGE_NAME, intended_action="EXECUTE_EVALUATION", attempt_number=attempt_number
    )
    output_artifacts: dict[str, Any] = {}

   # Ejecuta el pipeline de evaluación y, si termina correctamente,
   # respalda resultados anteriores y guarda la validación numérica intermedia.
    try:
        pipeline_result = run_evaluation_pipeline(**kwargs)

        backup_existing_outputs(output_dir=output_dir, backup_root=backup_root)
        persist_intermediate_numeric_check(
            output_dir=numeric_check_output_dir,
            numeric_rows=pipeline_result["factual_audit"]["numeric_rows"],
        )

       # Construye el reporte final de Stage08 reuniendo los resultados
       # automáticos, el LLM Judge, la auditoría factual y la información de trazabilidad.
        report_markdown = build_final_evaluation_report_markdown(
            experiment_id=kwargs["experiment_id"],
            topic_name=kwargs["topic_name"],
            source_stage=kwargs["source_stage"],
            reverification_performed=kwargs["reverification_performed"],
            reverification_reason=kwargs["reverification_reason"],
            upstream_runtime_status=kwargs["upstream_runtime_status"],
            claims_verified=kwargs["claims_verified"],
            claims_requiring_manual_review=kwargs["claims_requiring_manual_review"],
            manual_review_claim_ids=kwargs["manual_review_claim_ids"],
            evaluation_ready_json_path=kwargs["evaluation_ready_json_path"],
            ground_truth_source_path=str(pipeline_result["ground_truth_source_path"]),
            automatic_metric_rows=pipeline_result["automatic_metrics_result"].automatic_metric_rows,
            judge_score_rows=pipeline_result["judge_score_rows"],
            overall_assessment=pipeline_result["llm_judge_result"]["overall_assessment"],
            factual_metric_rows=pipeline_result["factual_audit"]["factual_metric_rows"],
            final_selected_metrics=pipeline_result["final_selected_metrics"],
            corpus_gap_rows=pipeline_result["corpus_gap_rows"],
        )

       # Construye el manifest de Stage08 para dejar registrado exactamente
       # qué se evaluó, de dónde vino, con qué modelos/configuración y con qué huellas.
        evaluation_manifest = {
            "stage": EVALUATION_STAGE_NAME,
            "created_at": _now(),
            "pipeline_outcome": pipeline_outcome_metadata,
            "experiment_id": kwargs["experiment_id"],
            "source_stage": kwargs["source_stage"],
            "reverification_performed": kwargs["reverification_performed"],
            "reverification_reason": kwargs["reverification_reason"],
            "upstream_runtime_status": kwargs["upstream_runtime_status"],
            "claims_verified": kwargs["claims_verified"],
            "claims_requiring_manual_review": kwargs["claims_requiring_manual_review"],
            "manual_review_claim_ids": kwargs["manual_review_claim_ids"],
            "input_dependencies": {
                "generated_state_of_art_path": kwargs["evaluation_ready_json_path"],
                "generated_state_of_art_sha256": evaluation_signature["generated_sha256"],
                "ground_truth_source_path": str(pipeline_result["ground_truth_source_path"]),
                "ground_truth_sha256": evaluation_signature["ground_truth_sha256"],
                "chunks_source": "chunks_clean_for_rag.csv",
                "chunks_fingerprint": evaluation_signature["chunks_fingerprint"],
                "traceability_row_count": len(kwargs["traceability_rows"]),
                "traceability_fingerprint": evaluation_signature["traceability_fingerprint"],
                "upstream_fingerprint": evaluation_signature["upstream_fingerprint"],
                "openai_model": evaluation_signature["openai_model"],
                "evaluation_embedding_model": kwargs["evaluation_policy"]["evaluation_embedding_model"],
                "bertscore_model": kwargs["evaluation_policy"]["bertscore_model"],
                "llm_judge_prompt_version": evaluation_signature["llm_judge_prompt_version"],
            },
            "fingerprint": evaluation_fingerprint_hash,
            "signature": evaluation_signature,
        }

       # Guarda todos los resultados finales de Stage08 y luego importa
      # las utilidades necesarias para registrar referencias y hashes de los archivos.
        written_paths = persist_evaluation_outputs(
            output_dir=output_dir,
            automatic_metric_rows=pipeline_result["automatic_metrics_result"].automatic_metric_rows,
            semantic_alignment_rows=pipeline_result["automatic_metrics_result"].semantic_alignment_rows, #resultados de alineación/similitud semántica.
            bertscore_pair_metadata=pipeline_result["automatic_metrics_result"].bertscore_pair_metadata, #metadatos usados para BERTScore.
            factual_metric_rows=pipeline_result["factual_audit"]["factual_metric_rows"], #métricas de consistencia factual.
            citation_rows=pipeline_result["factual_audit"]["citation_rows"], #resultados de la revisión de citas.
            claim_audit_rows=pipeline_result["factual_audit"]["claim_audit_rows"], #auditoría de los claims.
            llm_judge_result=pipeline_result["llm_judge_result"], #resultado y puntuaciones del LLM Judge.
            judge_score_rows=pipeline_result["judge_score_rows"],#brechas detectadas entre el texto generado y el corpus
            corpus_gap_rows=pipeline_result["corpus_gap_rows"],
            corpus_gap_markdown=pipeline_result["corpus_gap_markdown"],
            final_selected_metrics=pipeline_result["final_selected_metrics"], #conjunto final de métricas que se reportará.
            evaluation_summary=pipeline_result["evaluation_summary"], #resumen global de la evaluación.
            final_evaluation_report_markdown=report_markdown,
            evaluation_validation_report=pipeline_result["final_validation"]["evaluation_validation_report"], #resultado de la validación final de Stage08.
            evaluation_manifest=evaluation_manifest, #manifest de trazabilidad y reproducibilidad.
        )
        from src.contracts.agent_input import ArtifactReference
        from src.state.fingerprints import sha256_file

         # Registra cada archivo generado como un artefacto trazable
         # y construye el resultado final de Stage08 con esos artefactos.
        output_artifacts = {
            name: ArtifactReference(str(path), sha256_file(path)) for name, path in written_paths.items()
        }
        result = _build_evaluation_agent_result(
            attempt_number=attempt_number,
            started_at=started_at,
            exception=None,
            pipeline_result=pipeline_result,
            output_artifacts=output_artifacts,
            pipeline_outcome_metadata=pipeline_outcome_metadata,
        )

   # Si ocurre un error durante la ejecución principal de Stage08,
   # construye un resultado de fallo; después guarda y confirma el resultado final.
    except Exception as exc:  
        result = _build_evaluation_agent_result(
            attempt_number=attempt_number, started_at=started_at, exception=exc, pipeline_result=None,
            pipeline_outcome_metadata=pipeline_outcome_metadata,
        )

    store.persist_agent_result(prepared.decision_id, result)
    store.commit_execution(
        decision_id=prepared.decision_id,
        result=result,
        stage_name=EVALUATION_STAGE_NAME,
        fingerprints=fingerprints,
        observations=dict(observations or {}),
    )

    state = store.load()
    attempts_used = state.stages[EVALUATION_STAGE_NAME].attempts_used
    status = "COMMITTED" if result.execution_status == ExecutionStatus.COMPLETED else "FAILED"
    return _outcome_from_result(spec, result, status, attempts_used=attempts_used)

