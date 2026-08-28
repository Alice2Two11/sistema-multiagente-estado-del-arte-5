"""Contrato transaccional de la etapa 08: PREPARE/EXECUTE/COMMIT/RESUME
sobre ``StateStore``, produciendo un ``AgentResult`` — 08 no es un agente
conceptual, pero debe devolver un resultado compatible con el orquestador.

Mapeo de resultado (5 categorías, sin convertir una excepción real en un
commit exitoso):

1. **Excepción técnica** (GT ausente, chunks ausentes, etc.) →
   ``execution_status=FAILED``.
2. **Inconsistencia factual bloqueante** (``resolve_factual_gate`` real
   lanza ``ValueError``) → ``execution_status=FAILED`` — la excepción real
   del notebook se propaga hasta aquí sin capturarse antes; solo en ESTE
   punto se envuelve en un ``AgentResult`` para poder comprometerlo al
   ``StateStore`` (igual que hacen 02-06 con sus fallos de preparación).
3. **``evaluation_validation_ok=False`` con
   ``fail_on_invalid_evaluation=True``** (``resolve_final_validation_gate``
   real lanza) → ``execution_status=FAILED``, mismo tratamiento que 2.
4. **Evaluación inválida PERMITIDA** (``evaluation_validation_ok=False`` con
   ``fail_on_invalid_evaluation=False``, o pendiente factual no bloqueante
   sin otros errores) → ``execution_status=COMPLETED``,
   ``quality_status=REJECTED`` si hay ``validation_errors``, o
   ``APPROVED_WITH_WARNINGS`` si solo hay ``validation_warnings``.
5. **Evaluación válida sin advertencias** →
   ``execution_status=COMPLETED``, ``quality_status=APPROVED``.

Los casos 1-3 nunca llegan a escribir los 15 outputs (el notebook real
tampoco lo hace: la excepción interrumpe antes). Los casos 4-5 sí.
"""

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
    """Empaqueta los argumentos de ``run_evaluation_pipeline`` — separado en
    su propia función para que ``_run_evaluation_stage`` pueda construir
    fingerprints de entrada ANTES de ejecutar, sin ejecutar dos veces.

    ``upstream_fingerprint``: el fingerprint COMPUESTO real de la ejecución
    comprometida de 07 (``state.stages["07_agente_verificador"].
    fingerprints.composite``, tal como lo deja ``StateStore.commit_execution``
    tras el COMMIT real de 07) — no se crea un fingerprint paralelo; se
    propaga el mismo que ya usa el contrato transaccional. El llamador
    (``evaluation_stagespec_wiring.build_execution_for_stagespec``) lo lee
    directamente de ``store.load()`` y lo pasa aquí. Queda en ``None`` solo
    si 07 todavía no se comprometió (o en llamadas manuales/pruebas que no
    lo necesitan).

    NO incluye ``output_dir``/``numeric_check_output_dir``/``backup_root``/
    ``_openai_model``: esas 4 claves de infraestructura las agrega el
    LLAMADOR (ver ``evaluation_stagespec_wiring.build_execution_for_stagespec``,
    que hace ``{**build_experimental_evaluation_execution(...), "output_dir":
    ..., ...}``) — no son argumentos de ``run_evaluation_pipeline`` y no
    pertenecen a esta función."""

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
    outcome_metadata = dict(pipeline_outcome_metadata or {"pipeline_outcome": "SUCCESS"})

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

    if validation_errors:
        quality_status = QualityStatus.REJECTED
    elif validation_warnings:
        quality_status = QualityStatus.APPROVED_WITH_WARNINGS
    else:
        quality_status = QualityStatus.APPROVED

    warnings = tuple(
        AgentWarning(code=code, severity=WarningSeverity.WARNING, blocking=False, message=code)
        for code in validation_warnings
    ) + tuple(
        AgentWarning(code=code, severity=WarningSeverity.ERROR, blocking=False, message=code)
        for code in validation_errors
    )

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


def _run_evaluation_stage(
    *,
    store,
    project_dir,
    spec,
    attempt_number: int = 1,
    observations: dict[str, Any] | None = None,
    force_rerun: bool = False,
):
    """``custom_run`` de ``StageSpec`` para 08 — mismo patrón que 07
    (PREPARE/EXECUTE/COMMIT propios vía ``StateStore`` genérico, no un
    protocolo de 3 funciones separado como 07).

    Fingerprint único (A1): ``evaluation_signature`` se calcula UNA vez,
    antes de EXECUTE, y esa MISMA estructura alimenta tanto el fingerprint
    transaccional de ``StateStore`` (``SKIPPED_FRESH``) como
    ``evaluation_manifest.json`` — no hay una segunda representación
    "débil" en ningún punto.
    """

    from src.adapters.evaluation_fingerprint import (
        build_evaluation_signature,
        compute_evaluation_fingerprint,
    )
    from src.adapters.evaluation_persistence import find_missing_outputs
    from src.orchestration.stage_execution import (
        _outcome_from_committed_stage,
        _outcome_from_result,
    )
    from src.tools.evaluation.ground_truth import resolve_ground_truth_comparable_text
    from src.tools.evaluation.llm_judge import PROMPT_VERSION as JUDGE_PROMPT_VERSION

    project_dir = Path(project_dir)
    kwargs = dict(spec.build_execution(project_dir, attempt_number))
    output_dir = Path(kwargs.pop("output_dir"))
    numeric_check_output_dir = Path(kwargs.pop("numeric_check_output_dir"))
    backup_root = Path(kwargs.pop("backup_root"))
    openai_model_for_signature = kwargs.pop("_openai_model", "")
    upstream_fingerprint = kwargs.pop("upstream_fingerprint", None)
    pipeline_outcome_metadata = kwargs.pop("_pipeline_outcome_metadata", {"pipeline_outcome": "SUCCESS"})

    started_at = _now()

    # --- Fingerprint único (A1): se calcula ANTES de EXECUTE. Requiere
    # resolver el Ground Truth (para su hash) — si eso falla (GT ausente,
    # demasiado corto, etc.), es la MISMA "excepción técnica" que fallaría
    # dentro de run_evaluation_pipeline; se trata igual (FAILED), sin
    # ejecutar el resto del pipeline dos veces por accidente.
    try:
        ground_truth_plain_text, _gt_metadata, _gt_source_path = resolve_ground_truth_comparable_text(
            ground_truth_dir=kwargs["ground_truth_dir"],
            minimum_words=kwargs["evaluation_policy"]["minimum_ground_truth_words"],
            require_explicit_end_heading=kwargs["evaluation_policy"][
                "require_explicit_ground_truth_end_heading"
            ],
        )
        evaluation_signature = build_evaluation_signature(
            experiment_id=kwargs["experiment_id"],
            evaluation_policy=kwargs["evaluation_policy"],
            openai_model=openai_model_for_signature,
            evaluation_ready_json_path=kwargs["evaluation_ready_json_path"],
            upstream_fingerprint=upstream_fingerprint,  # propagado real de 07 (state.stages[
            # "07_agente_verificador"].fingerprints.composite), leído por
            # evaluation_stagespec_wiring.build_execution_for_stagespec y
            # pasado hasta aquí. Ningún fingerprint paralelo: es el mismo
            # compuesto que ya usa el contrato transaccional de 07.
            ground_truth_text=ground_truth_plain_text,
            chunks=kwargs["chunks"],
            traceability_rows=kwargs["traceability_rows"],
            llm_judge_prompt_version=JUDGE_PROMPT_VERSION,
        )
    except Exception as exc:  # noqa: BLE001 - excepción técnica real, no se silencia
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

    # --- A2: fingerprint vigente por sí solo NO basta -- también deben
    # estar los 15 outputs. fingerprint vigente + outputs completos ->
    # SKIPPED_FRESH; fingerprint vigente + outputs faltantes -> reconstruir.
    state = store.load()
    committed = state.stages.get(EVALUATION_STAGE_NAME)
    missing_outputs = find_missing_outputs(output_dir=output_dir)

    if (
        committed is not None
        and committed.execution_status == ExecutionStatus.COMPLETED
        and not force_rerun
        and committed.fingerprints.composite == fingerprints.composite
        and not missing_outputs
    ):
        return _outcome_from_committed_stage(spec, committed, status="SKIPPED_FRESH")

    if state.pending_execution is not None and state.pending_execution.target_stage == EVALUATION_STAGE_NAME:
        resume = store.resolve_resume(
            stage_name=EVALUATION_STAGE_NAME, fingerprints=fingerprints, observations=observations
        )
        if resume.action == "COMMITTED":
            state = store.load()
            attempts_used = state.stages[EVALUATION_STAGE_NAME].attempts_used
            return _outcome_from_result(spec, resume.committed_result, "COMMITTED", attempts_used=attempts_used)

    from src.adapters.evaluation_persistence import (
        backup_existing_outputs,
        build_final_evaluation_report_markdown,
        persist_evaluation_outputs,
        persist_intermediate_numeric_check,
    )

    prepared = store.prepare_execution(
        target_stage=EVALUATION_STAGE_NAME, intended_action="EXECUTE_EVALUATION", attempt_number=attempt_number
    )
    output_artifacts: dict[str, Any] = {}

    try:
        pipeline_result = run_evaluation_pipeline(**kwargs)

        backup_existing_outputs(output_dir=output_dir, backup_root=backup_root)
        persist_intermediate_numeric_check(
            output_dir=numeric_check_output_dir,
            numeric_rows=pipeline_result["factual_audit"]["numeric_rows"],
        )

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

        # A4: input_dependencies completo para la ruta activa (07 directo).
        # Sin campos exclusivos de 07C (post_correction_recheck_manifest,
        # etc.) -- ninguno de esos existe en esta ruta.
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

        written_paths = persist_evaluation_outputs(
            output_dir=output_dir,
            automatic_metric_rows=pipeline_result["automatic_metrics_result"].automatic_metric_rows,
            semantic_alignment_rows=pipeline_result["automatic_metrics_result"].semantic_alignment_rows,
            bertscore_pair_metadata=pipeline_result["automatic_metrics_result"].bertscore_pair_metadata,
            factual_metric_rows=pipeline_result["factual_audit"]["factual_metric_rows"],
            citation_rows=pipeline_result["factual_audit"]["citation_rows"],
            claim_audit_rows=pipeline_result["factual_audit"]["claim_audit_rows"],
            llm_judge_result=pipeline_result["llm_judge_result"],
            judge_score_rows=pipeline_result["judge_score_rows"],
            corpus_gap_rows=pipeline_result["corpus_gap_rows"],
            corpus_gap_markdown=pipeline_result["corpus_gap_markdown"],
            final_selected_metrics=pipeline_result["final_selected_metrics"],
            evaluation_summary=pipeline_result["evaluation_summary"],
            final_evaluation_report_markdown=report_markdown,
            evaluation_validation_report=pipeline_result["final_validation"]["evaluation_validation_report"],
            evaluation_manifest=evaluation_manifest,
        )
        from src.contracts.agent_input import ArtifactReference
        from src.state.fingerprints import sha256_file

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
    except Exception as exc:  # noqa: BLE001 - ver docstring del módulo: no se silencia, se registra
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

