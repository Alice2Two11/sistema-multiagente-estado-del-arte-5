"""Constructores concretos de ``(agente, AgentInput)`` y transacciones por
etapa, extraídos de ``pipeline_orchestrator.py`` (Bloque 1 de la migración a
LangGraph, MAIN 5) sin cambiar una sola línea de lógica -- solo la
ubicación.

Cada función de aquí sabe construir/ejecutar UNA etapa concreta, delegando
en su ``adapters/*_runtime.py`` real -- cero conocimiento de qué etapa
viene antes o después, ni de qué motor las está orquestando.

Importa de ``src.orchestration.stage_execution`` a nivel de módulo
(``DRAFT_STAGE_NAME``, ``StageOutcome``, ``StageSpec``, ``_outcome_from_result``)
porque esa dirección es segura: ``stage_execution.py`` solo importa de aquí
de forma diferida, dentro de ``_stage_registry()`` -- ver la docstring de
ese módulo para el porqué de esa asimetría (evita un ciclo de imports).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping

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
from src.orchestration.stage_execution import (
    DRAFT_STAGE_NAME,
    StageOutcome,
    StageSpec,
    _outcome_from_result,
)
from src.state.fingerprints import build_stage_fingerprints
from src.state.state_store import StateStore


def _real_extraction_execution(project_dir, attempt_number: int):
    from src.adapters.extraction_runtime import (
        build_agent_input,
        build_extraction_runtime,
        load_runtime_configuration,
        resolve_openai_api_key,
    )
    from src.agents.extraction_agent import ExtractionAgent

    api_key = resolve_openai_api_key(project_dir=project_dir, required=True)
    configuration = load_runtime_configuration(project_dir)
    runtime = build_extraction_runtime(configuration, api_key=api_key)
    agent_input = build_agent_input(
        configuration,
        attempt_number=attempt_number,
        runtime_resources={
            "df_chunks_clean": runtime.dataframe,
            "collection": runtime.collection,
        },
    )
    return ExtractionAgent(runtime.dependencies), agent_input


def _real_quantitative_execution(project_dir, attempt_number: int):
    from src.adapters.quantitative_extraction_runtime import (
        build_quantitative_agent_input,
        build_quantitative_capability,
        load_quantitative_configuration,
    )

    # El Agente 03B sólo admite attempt_number=1 en el wiring actual de src/.
    configuration = load_quantitative_configuration(project_dir)
    capability = build_quantitative_capability(configuration)
    agent_input = build_quantitative_agent_input(configuration)
    return capability, agent_input


def _real_thematic_execution(project_dir, attempt_number: int):
    from src.adapters.thematic_analysis_runtime import build_real_thematic_execution

    agent, agent_input, _configuration = build_real_thematic_execution(
        project_dir, attempt_number
    )
    return agent, agent_input


def _real_outline_execution(project_dir, attempt_number: int):
    from src.adapters.outline_generation_runtime import build_real_outline_execution

    agent, agent_input, _configuration = build_real_outline_execution(
        project_dir, attempt_number
    )
    return agent, agent_input


def _resolve_draft_execution_mode(project_dir, store) -> dict[str, Any] | None:
    """Detecta si 06 debe ejecutarse en modo REVISION: hay un ciclo
    ``writer_verifier`` ACTIVE con al menos una ronda usada. Si es así,
    lee la ÚLTIMA ronda persistida por 07 (``writer_verifier_cycle/
    round_NN/writer_revision_request.json``) y el borrador comprometido
    más reciente de 06, y devuelve los ``policy_overrides`` para
    construir el ``AgentInput`` de revisión. Devuelve ``None`` si debe
    ejecutarse en modo INITIAL_DRAFT (sin ciclo activo, o ciclo resuelto)."""

    state = store.load()
    cycle = state.cycles.get("writer_verifier")
    if cycle is None or cycle.status != "ACTIVE" or cycle.rounds_used == 0:
        return None

    import json as _json

    from src.tools.verification.cycle_round_persistence import (
        list_persisted_rounds,
        read_round_artifact,
        read_round_status,
        round_is_persisted,
    )

    active_experiment = _json.loads((project_dir / "active_experiment.json").read_text(encoding="utf-8"))
    experiment_id = active_experiment["active_experiment_id"]

    persisted_rounds = list_persisted_rounds(project_dir=project_dir, experiment_id=experiment_id)
    if not persisted_rounds:
        raise RuntimeError(
            "DRAFT_REVISION_ROUND_NOT_PERSISTED: el ciclo writer_verifier está "
            "ACTIVE pero no hay ninguna ronda persistida en writer_verifier_cycle/."
        )

    round_number = persisted_rounds[-1]
    if not round_is_persisted(project_dir=project_dir, experiment_id=experiment_id, round_number=round_number):
        raise RuntimeError(f"DRAFT_REVISION_ROUND_NOT_PERSISTED: round_{round_number:02d}")

    status = read_round_status(project_dir=project_dir, experiment_id=experiment_id, round_number=round_number)
    if status is None:
        raise RuntimeError(f"DRAFT_REVISION_ROUND_NOT_PERSISTED: round_{round_number:02d}")
    if status["status"] not in {"AWAITING_REVISION", "REVISION_COMPLETED"}:
        raise RuntimeError(
            f"DRAFT_REVISION_ROUND_UNEXPECTED_STATUS: round_{round_number:02d} está en "
            f"{status['status']!r}, se esperaba 'AWAITING_REVISION' o 'REVISION_COMPLETED'."
        )
    # 'REVISION_COMPLETED' significa que 06 YA completó esta ronda en una
    # ejecución previa -- no hay nada nuevo que escribir. Esta función NO
    # decide si 06 se reinvoca: solo reconstruye el MISMO AgentInput con el
    # que se comprometió esa revisión, para que su fingerprint coincida con
    # el ya comprometido y run_stage() lo reconozca como SKIPPED_FRESH sin
    # tocar la ronda. writer_revision_request.json y el borrador previo son
    # los MISMOS archivos persistidos en ambos estados (AWAITING_REVISION y
    # REVISION_COMPLETED) -- solo cambia el campo de estado de la ronda, que
    # no forma parte de este AgentInput. Si, pese a la coincidencia de
    # fingerprint, algo más forzara una reinvocación real de 06 sobre esta
    # misma ronda, complete_round_revision() ya rechaza explícitamente un
    # segundo intento de completarla (ver su docstring) -- esa red de
    # seguridad no se toca aquí.

    writer_revision_request = read_round_artifact(
        project_dir=project_dir, experiment_id=experiment_id, round_number=round_number,
        filename="writer_revision_request.json",
    )

    # Consistencia (punto 5): la ronda debe corresponder al MISMO experimento
    # y ronda activos -- si algo no coincide, no se arma un AgentInput de
    # revisión con datos inconsistentes.
    if writer_revision_request.get("experiment_id") != experiment_id:
        raise RuntimeError(
            "DRAFT_REVISION_EXPERIMENT_MISMATCH: writer_revision_request "
            f"pertenece a {writer_revision_request.get('experiment_id')!r}, "
            f"se esperaba {experiment_id!r}."
        )
    if int(writer_revision_request.get("round_number", -1)) != round_number:
        raise RuntimeError(
            "DRAFT_REVISION_ROUND_MISMATCH: writer_revision_request declara ronda "
            f"{writer_revision_request.get('round_number')!r}, se esperaba {round_number}."
        )

    experiment_dir = project_dir / experiment_id
    draft_json_path = experiment_dir / "05_outputs" / "05_draft" / "state_of_art_draft.json"
    if not draft_json_path.is_file():
        raise RuntimeError("DRAFT_REVISION_PREVIOUS_DRAFT_NOT_FOUND")
    previous_draft = _json.loads(draft_json_path.read_text(encoding="utf-8"))

    if previous_draft.get("source_draft_fingerprint") not in (
        None,
        writer_revision_request["source_draft_fingerprint"],
    ):
        raise RuntimeError(
            "DRAFT_REVISION_FINGERPRINT_MISMATCH: el borrador en disco no coincide "
            "con source_draft_fingerprint del writer_revision_request."
        )

    return {
        "mode": "REVISION",
        "writer_revision_request": writer_revision_request,
        "previous_draft": previous_draft,
        "round_number": round_number,
        "cycle_project_dir": str(project_dir),
        "experiment_id": experiment_id,
    }


def _real_draft_execution(project_dir, attempt_number: int):
    from pathlib import Path

    from src.adapters.draft_writing_runtime import build_real_draft_execution
    from src.orchestration.stage_execution import ensure_pipeline_state

    project_dir = Path(project_dir)
    store = ensure_pipeline_state(project_dir)
    revision_overrides = _resolve_draft_execution_mode(project_dir, store)

    agent, agent_input, _configuration = build_real_draft_execution(
        project_dir, attempt_number, policy_overrides=revision_overrides
    )
    return agent, agent_input


def _experimental_verification_execution(project_dir, attempt_number: int):
    from src.adapters.verification_orchestrator_runtime import (
        build_experimental_verification_execution,
    )

    return build_experimental_verification_execution(project_dir, attempt_number)


def _experimental_evaluation_execution(project_dir, attempt_number: int):
    from src.adapters.evaluation_stagespec_wiring import build_execution_for_stagespec

    return build_execution_for_stagespec(project_dir, attempt_number)


def _run_evaluation_stage(**kwargs):
    from src.adapters.evaluation_orchestrator_runtime import (
        _run_evaluation_stage as _real_run_evaluation_stage,
    )

    return _real_run_evaluation_stage(**kwargs)


def _draft_runtime_transaction(
    *,
    store: StateStore,
    build_execution: Callable[[], tuple[Any, Any]],
    attempt_number: int,
    observations: Mapping[str, Any] | None = None,
):
    from src.runtime.draft_writing_protocol import (
        DraftWritingTransactionResult,
        build_draft_fingerprints,
    )

    prepared = store.prepare_execution(
        target_stage=DRAFT_STAGE_NAME,
        intended_action="EXECUTE_DRAFT_WRITING",
        attempt_number=attempt_number,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        agent, agent_input = build_execution()
        result = agent.execute(agent_input)
        fingerprints = build_draft_fingerprints(agent_input)
    except Exception as exc:  # noqa: BLE001 - convertido a AgentResult FAILED
        text = str(exc)
        if isinstance(exc, FileNotFoundError):
            code = "DEPENDENCY_NOT_FOUND"
        elif "OPENAI_API_KEY" in text:
            code = "CREDENTIAL_NOT_FOUND"
        else:
            code = "RUNTIME_DEPENDENCY_FAILED"
        now = datetime.now(timezone.utc).isoformat()
        result = AgentResult(
            execution_status=ExecutionStatus.FAILED,
            quality_status=QualityStatus.REJECTED,
            decision=DecisionInfo(
                code="DRAFT_RUNTIME_FAILED",
                rationale="Falló la preparación de la etapa 06.",
            ),
            quality_metrics={"technical": {}, "scientific": {}},
            warnings=(
                AgentWarning(
                    code=code,
                    severity=WarningSeverity.ERROR,
                    blocking=True,
                    message=text,
                ),
            ),
            failure_reason_codes=(code,),
            requested_transition=RequestedTransition(
                action=TransitionAction.HALT_STAGE,
                target_stage=None,
                reason_code=code,
                requires_human_confirmation=False,
            ),
            output_artifacts={},
            tool_usage=ToolUsage(),
            attempt_number=attempt_number,
            started_at=started_at,
            completed_at=now,
            error={"type": type(exc).__name__, "message": text, "stage": DRAFT_STAGE_NAME},
        )
        fingerprints = build_stage_fingerprints(
            input_data={"stage_name": DRAFT_STAGE_NAME, "attempt_number": attempt_number},
            config_data={"runtime_resolution": "FAILED"},
            dependencies_data={},
        )

    persisted_path = store.persist_agent_result(prepared.decision_id, result)
    committed_state = store.commit_execution(
        decision_id=prepared.decision_id,
        result=result,
        stage_name=DRAFT_STAGE_NAME,
        fingerprints=fingerprints,
        observations=dict(observations or {}),
    )
    return DraftWritingTransactionResult(
        prepared, result, str(persisted_path), committed_state
    )


def _quantitative_runtime_transaction(
    *,
    store: StateStore,
    build_execution: Callable[[], tuple[Any, Any]],
    attempt_number: int,
    observations: Mapping[str, Any] | None = None,
):
    # execute_quantitative_runtime_transaction no admite attempt_number: la
    # etapa 03B siempre corre como attempt_number=1 en el wiring actual.
    from src.runtime.quantitative_extraction_protocol import (
        execute_quantitative_runtime_transaction,
    )

    return execute_quantitative_runtime_transaction(
        store=store, build_execution=build_execution, observations=observations
    )


# ---------------------------------------------------------------------------
# Ejecución dedicada de la etapa 07 (no encaja en build_execution +
# runtime_transaction + resolve_resume genéricos — ver StageSpec.custom_run)
# ---------------------------------------------------------------------------
#
# A diferencia de 02-06, la etapa 07 ya trae su propia semántica de RESUME
# más rica que {NO_PENDING, COMMITTED, REEXECUTE}
# (resume_agent07_execution devuelve COMMITTED, EXECUTED_NOT_COMMITTED,
# REEXECUTE, NO_COMMIT, FINGERPRINT_MISMATCH, ARTIFACT_MISMATCH o
# MANIFEST_INCOMPLETE — ver src/adapters/verification_notebook.py). Esa misma
# función YA decide internamente si el resultado comprometido sigue vigente
# (compara fingerprints), así que aquí no se duplica ese chequeo: se llama
# siempre, incluso sin pending_execution, y se interpreta su resultado.


def _run_verification_stage(
    *,
    store: StateStore,
    project_dir,
    spec: StageSpec,
    attempt_number: int = 1,
    observations: Mapping[str, Any] | None = None,
    force_rerun: bool = False,
) -> StageOutcome:
    from pathlib import Path

    from src.adapters.verification_notebook import (
        commit_executed_agent07,
        execute_prepared_agent07,
        prepare_agent07_execution,
        resume_agent07_execution,
    )

    project_dir = Path(project_dir)
    had_pending = store.load().pending_execution is not None

    try:
        dependencies, runtime_input = spec.build_execution(project_dir, attempt_number)
    except Exception as exc:  # noqa: BLE001
        # Ocurre ANTES de cualquier PREPARE (build_execution no toca
        # StateStore) — a diferencia de 02-06, aquí no se fabrica un
        # AgentResult FAILED sintético para no inventar comportamiento que
        # verification_notebook.py no tiene. Se deja como excepción real,
        # capturada por el try/except genérico de run_stage.
        raise

    def _do_fresh_execution() -> AgentResult:
        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        executed = execute_prepared_agent07(
            store=store, prepared=prepared, dependencies=dependencies
        )
        commit_executed_agent07(store=store, executed=executed)
        return executed.agent_result

    if force_rerun and not had_pending:
        result = _do_fresh_execution()
        status = (
            "COMMITTED"
            if result.execution_status == ExecutionStatus.COMPLETED
            else "FAILED"
        )
    else:
        resume = resume_agent07_execution(store=store, runtime_input=runtime_input)
        if resume.action == "COMMITTED":
            result = resume.committed_result
            status = "COMMITTED" if had_pending else "SKIPPED_FRESH"
        elif resume.action == "EXECUTED_NOT_COMMITTED":
            commit_executed_agent07(store=store, executed=resume.executed)
            result = resume.executed.agent_result
            status = "COMMITTED"
        elif resume.action in {
            "NO_COMMIT",
            "REEXECUTE",
            "FINGERPRINT_MISMATCH",
            "ARTIFACT_MISMATCH",
            "MANIFEST_INCOMPLETE",
        }:
            result = _do_fresh_execution()
            status = (
                "COMMITTED"
                if result.execution_status == ExecutionStatus.COMPLETED
                else "FAILED"
            )
        else:  # pragma: no cover - RESUME_ACTIONS es cerrado en el repo real
            raise RuntimeError(
                f"resume_agent07_execution devolvió una acción inesperada: {resume.action}"
            )

    state = store.load()
    attempts_used = state.stages[spec.key].attempts_used
    return _outcome_from_result(spec, result, status, attempts_used=attempts_used)
