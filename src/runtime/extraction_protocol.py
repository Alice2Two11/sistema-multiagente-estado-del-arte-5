"""External PREPARE → EXECUTE → persist → COMMIT protocol for stage 03.

The ExtractionAgent remains unaware of StateStore. This module only binds the
approved transaction store to an already constructed agent and AgentInput.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.contracts.agent_input import (
    AgentInput,
)
from src.contracts.agent_result import (
    AgentResult,
)
from src.state.fingerprints import (
    build_stage_fingerprints,
)
from src.state.pipeline_state import (
    PipelineState,
    StageFingerprints,
)
from src.state.state_store import (
    PrepareResult,
    ResumeResolution,
    StateStore,
)


@dataclass(frozen=True, slots=True)
class ExtractionTransactionResult:
    prepare: PrepareResult
    agent_result: AgentResult
    persisted_result_path: str
    committed_state: PipelineState


def build_agent_input_fingerprints(
    agent_input: AgentInput,
) -> StageFingerprints:
    if not isinstance(
        agent_input,
        AgentInput,
    ):
        raise TypeError(
            "agent_input debe ser AgentInput."
        )

    return build_stage_fingerprints(
        input_data={
            "experiment_id": (
                agent_input.experiment_id
            ),
            "run_id": agent_input.run_id,
            "stage_name": (
                agent_input.stage_name
            ),
            "attempt_number": (
                agent_input.attempt_number
            ),
            "mode": (
                agent_input.mode.value
            ),
        },
        config_data=dict(
            agent_input.policy
        ),
        dependencies_data={
            name: reference.to_dict()
            for name, reference
            in agent_input.dependencies.items()
        },
    )


def execute_extraction_transaction(
    *,
    store: StateStore,
    agent: Any,
    agent_input: AgentInput,
    fingerprints: (
        StageFingerprints | None
    ) = None,
    observations: (
        Mapping[str, Any] | None
    ) = None,
    intended_action: str = (
        "EXECUTE_EXTRACTION_STAGE"
    ),
) -> ExtractionTransactionResult:
    """Execute the complete external transaction without graph decisions."""

    if not isinstance(
        store,
        StateStore,
    ):
        raise TypeError(
            "store debe ser StateStore."
        )
    if not isinstance(
        agent_input,
        AgentInput,
    ):
        raise TypeError(
            "agent_input debe ser AgentInput."
        )
    execute = getattr(
        agent,
        "execute",
        None,
    )
    if not callable(execute):
        raise TypeError(
            "agent debe exponer execute()."
        )

    resolved_fingerprints = (
        fingerprints
        or build_agent_input_fingerprints(
            agent_input
        )
    )

    prepared = store.prepare_execution(
        target_stage=(
            agent_input.stage_name
        ),
        intended_action=(
            intended_action
        ),
        attempt_number=(
            agent_input.attempt_number
        ),
    )

    result = execute(agent_input)
    if not isinstance(
        result,
        AgentResult,
    ):
        raise TypeError(
            "agent.execute() debe devolver "
            "AgentResult."
        )

    persisted_path = (
        store.persist_agent_result(
            prepared.decision_id,
            result,
        )
    )
    committed_state = (
        store.commit_execution(
            decision_id=(
                prepared.decision_id
            ),
            result=result,
            stage_name=(
                agent_input.stage_name
            ),
            fingerprints=(
                resolved_fingerprints
            ),
            observations=dict(
                observations or {}
            ),
        )
    )

    return ExtractionTransactionResult(
        prepare=prepared,
        agent_result=result,
        persisted_result_path=str(
            persisted_path
        ),
        committed_state=(
            committed_state
        ),
    )


def resolve_extraction_resume(
    *,
    store: StateStore,
    agent_input: AgentInput,
    fingerprints: (
        StageFingerprints | None
    ) = None,
    observations: (
        Mapping[str, Any] | None
    ) = None,
) -> ResumeResolution:
    """Resolve a pending stage-03 transaction without executing the agent."""

    resolved_fingerprints = (
        fingerprints
        or build_agent_input_fingerprints(
            agent_input
        )
    )
    return store.resolve_resume(
        stage_name=(
            agent_input.stage_name
        ),
        fingerprints=(
            resolved_fingerprints
        ),
        observations=dict(
            observations or {}
        ),
    )


def _runtime_failure_result(
    *,
    attempt_number: int,
    started_at: str,
    error: Exception,
) -> AgentResult:
    from datetime import datetime, timezone
    from src.contracts.agent_result import (
        AgentWarning,
        DecisionInfo,
        ExecutionStatus,
        QualityStatus,
        RequestedTransition,
        ToolUsage,
        TransitionAction,
        WarningSeverity,
    )

    text = str(error)
    lowered = text.casefold()
    if isinstance(error, FileNotFoundError) or any(
        token in lowered for token in (
            "no existe", "not found", "inexistente",
            "couldn't connect", "offline mode", "name resolution",
        )
    ):
        code = "DEPENDENCY_NOT_FOUND"
    elif any(
        token in lowered
        for token in ("no coincide", "mismatch", "desaline", "collection", "colección", "hash")
    ):
        code = "DEPENDENCY_MISMATCH"
    else:
        code = "RUNTIME_CONFIGURATION_ERROR"

    if any(
        marker in text
        for marker in ("sk-", "OPENAI_API_KEY", "openai_api_key.key", "openai_api_key.enc")
    ):
        safe_message = "Error de runtime sanitizado durante la preparación de la etapa 03."
    else:
        safe_message = text

    return AgentResult(
        execution_status=ExecutionStatus.FAILED,
        quality_status=QualityStatus.REJECTED,
        decision=DecisionInfo(
            code="EXTRACTION_RUNTIME_FAILED",
            rationale="La etapa 03 no pudo resolver o validar sus dependencias.",
        ),
        quality_metrics={"technical": {}, "scientific": {}},
        warnings=(
            AgentWarning(
                code=code,
                severity=WarningSeverity.ERROR,
                blocking=True,
                message=safe_message,
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
        completed_at=datetime.now(timezone.utc).isoformat(),
        error={
            "type": type(error).__name__,
            "message": safe_message,
            "stage": "03_agente_extraccion_kb",
        },
    )


def execute_extraction_runtime_transaction(
    *,
    store: StateStore,
    build_execution: Any,
    attempt_number: int,
    stage_name: str = "03_agente_extraccion_kb",
    observations: Mapping[str, Any] | None = None,
) -> ExtractionTransactionResult:
    """Prepare before runtime resolution and persist any early failure.

    ``build_execution`` must return ``(agent, agent_input)``. Configuration,
    chunks, manifest, Chroma, and credential errors are converted into a
    FAILED AgentResult and committed instead of escaping without state.
    """
    from datetime import datetime, timezone

    if not isinstance(store, StateStore):
        raise TypeError("store debe ser StateStore.")
    if not callable(build_execution):
        raise TypeError("build_execution debe ser callable.")
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
        raise TypeError("attempt_number debe ser entero.")
    if attempt_number not in {1, 2}:
        raise ValueError("El Agente 03 admite únicamente attempt_number 1 o 2.")

    started_at = datetime.now(timezone.utc).isoformat()
    prepared = store.prepare_execution(
        target_stage=stage_name,
        intended_action="EXECUTE_EXTRACTION_STAGE",
        attempt_number=attempt_number,
    )

    try:
        agent, agent_input = build_execution()
        if not isinstance(agent_input, AgentInput):
            raise TypeError("build_execution debe producir AgentInput.")
        if agent_input.attempt_number != attempt_number:
            raise ValueError("AgentInput.attempt_number no coincide con PREPARE.")
        execute = getattr(agent, "execute", None)
        if not callable(execute):
            raise TypeError("El agente construido no expone execute().")
        fingerprints = build_agent_input_fingerprints(agent_input)
        result = execute(agent_input)
        if not isinstance(result, AgentResult):
            raise TypeError("ExtractionAgent.execute() debe devolver AgentResult.")
    except Exception as error:
        result = _runtime_failure_result(
            attempt_number=attempt_number,
            started_at=started_at,
            error=error,
        )
        fingerprints = build_stage_fingerprints(
            input_data={
                "stage_name": stage_name,
                "attempt_number": attempt_number,
            },
            config_data={"runtime_resolution": "FAILED"},
            dependencies_data={},
        )

    persisted_path = store.persist_agent_result(
        prepared.decision_id,
        result,
    )
    committed_state = store.commit_execution(
        decision_id=prepared.decision_id,
        result=result,
        stage_name=stage_name,
        fingerprints=fingerprints,
        observations=dict(observations or {}),
    )
    return ExtractionTransactionResult(
        prepare=prepared,
        agent_result=result,
        persisted_result_path=str(persisted_path),
        committed_state=committed_state,
    )
