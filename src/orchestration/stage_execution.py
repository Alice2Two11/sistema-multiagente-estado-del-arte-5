"""Infraestructura de ejecución de UNA etapa del pipeline, independiente de
qué motor recorre las etapas entre sí.

Extraído de ``pipeline_orchestrator.py`` (Bloque 1 de la migración a
LangGraph, MAIN 5) sin cambiar una sola línea de lógica -- solo la
ubicación. Contiene todo lo que responde a "¿cómo ejecuto/resuelvo/salto
una etapa dado su ``StageSpec``?", nunca "¿qué etapa sigue después?" (eso
sigue siendo responsabilidad exclusiva del motor de orquestación que
consuma este módulo -- hoy ``pipeline_orchestrator.run_pipeline``, más
adelante también ``src/orchestration_langgraph``).

No importa nada de ``pipeline_orchestrator.py`` -- la dependencia va en un
solo sentido: los motores de orquestación importan de aquí, nunca al
revés. La única referencia a los constructores concretos por etapa
(``src.orchestration.stage_constructors``) es un import diferido dentro de
``_stage_registry()``, exactamente el mismo patrón que ya usaba esta
función para los protocolos de ``src.runtime.*`` antes de esta extracción
-- evita un ciclo de imports con ``stage_constructors.py``, que sí importa
de este módulo a nivel de módulo (``StageOutcome``/``StageSpec``/
``_outcome_from_result``/``DRAFT_STAGE_NAME``/``ensure_pipeline_state``).

La reconciliación de ``pending_execution`` de una etapa distinta a la
actual (``_reconcile_pending_execution_for_other_stage``) vive desde el
Bloque 2 en su propio módulo, ``pending_reconciliation.py`` -- no aquí.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.contracts.agent_result import (
    AgentResult,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    TransitionAction,
)
from src.orchestration.decision_engine import (
    CANONICAL_STAGE_ORDER,
    MAX_ATTEMPTS_DEFAULT,
    ValidatedTransition,
    is_stage_fresh,
    validate_transition,
)
from src.orchestration.decision_log_frontier import _reconstruct_authoritative_frontier
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore

DRAFT_STAGE_NAME = "06_agente_redactor"


# ---------------------------------------------------------------------------
# Resolución de rutas del experimento activo
# ---------------------------------------------------------------------------


def load_active_experiment(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    active_path = root / "active_experiment.json"
    if not active_path.is_file():
        raise FileNotFoundError(
            "No existe active_experiment.json en "
            f"{root}. Ejecuta primero la etapa 00 (bootstrap del proyecto)."
        )
    return json.loads(active_path.read_text(encoding="utf-8"))


def resolve_state_path(project_dir: str | Path) -> tuple[Path, str, str]:
    """Devuelve (state_path, experiment_id, run_id) del experimento activo."""

    root = Path(project_dir).resolve()
    active = load_active_experiment(root)
    experiment_id = active["active_experiment_id"]
    run_id = active.get("run_id", experiment_id)
    experiment_dir = root / experiment_id
    state_path = (
        experiment_dir / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
    )
    return state_path, experiment_id, run_id


def ensure_pipeline_state(project_dir: str | Path) -> StateStore:
    """Abre el ``StateStore`` canónico, inicializándolo si es la primera vez."""

    state_path, experiment_id, run_id = resolve_state_path(project_dir)
    store = StateStore(state_path)
    if not state_path.is_file():
        now = datetime.now(timezone.utc).isoformat()
        store.initialize(
            PipelineState(
                identity=PipelineIdentity(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    created_at=now,
                    updated_at=now,
                    schema_version="1.0",
                )
            )
        )
    return store


# ---------------------------------------------------------------------------
# Registro de etapas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    build_execution: Callable[[Path, int], tuple[Any, Any]]
    runtime_transaction: Callable[..., Any]
    resolve_resume: Callable[..., Any]
    # None cuando la etapa no expone (todavía) una función pública de
    # fingerprints con la misma firma que build_thematic_fingerprints/etc.
    # (caso de 07: solo existe una versión privada, _stage_fingerprints, no
    # se importa aquí — ver verification_orchestrator_runtime.py). Cuando es
    # None, run_stage usa el chequeo antiguo (solo COMPLETED, sin comparar
    # vigencia) en vez del chequeo de vigencia por fingerprints.
    build_fingerprints: Callable[[Any], Any] | None = None
    max_attempt_number: int | None = None
    # Punto 7 del pedido: APPROVED_PENDING_MANUAL_REVIEW no se trata como
    # ADVANCE automático salvo que la etapa lo permita explícitamente. Hoy
    # ninguna etapa lo permite (False en las 6); queda aquí como el punto de
    # extensión previsto, no como una regla de negocio ya decidida.
    bypass_manual_review: bool = False
    # Escape hatch: cuando una etapa no encaja en el patrón genérico
    # build_execution+runtime_transaction+resolve_resume (caso de 07, que
    # tiene PREPARE/EXECUTE/COMMIT como 3 llamadas separadas con firmas
    # propias, y semántica de RESUME más rica que {NO_PENDING,COMMITTED,
    # REEXECUTE}), custom_run reemplaza por completo la lógica de run_stage
    # para esa etapa. Firma: (*, store, project_dir, spec, attempt_number,
    # observations, force_rerun) -> StageOutcome.
    custom_run: Callable[..., "StageOutcome"] | None = None


def _stage_registry() -> list[StageSpec]:
    # Los imports quedan diferidos a la primera llamada para no forzar
    # dependencias pesadas (langchain, chromadb) sólo por importar este
    # módulo o inspeccionar el registro. El import de stage_constructors
    # también es diferido específicamente para evitar un ciclo de imports
    # con ese módulo (ver docstring del archivo).
    from src.runtime.extraction_protocol import (
        build_agent_input_fingerprints,
        execute_extraction_runtime_transaction,
        resolve_extraction_resume,
    )
    from src.runtime.outline_generation_protocol import (
        build_outline_fingerprints,
        execute_outline_runtime_transaction,
        resolve_outline_resume,
    )
    from src.runtime.quantitative_extraction_protocol import (
        build_quantitative_fingerprints,
        resolve_quantitative_resume,
    )
    from src.runtime.thematic_analysis_protocol import (
        build_thematic_fingerprints,
        execute_thematic_runtime_transaction,
        resolve_thematic_resume,
    )
    from src.runtime.draft_writing_protocol import (
        build_draft_fingerprints,
        resolve_draft_resume,
    )
    from src.orchestration.stage_constructors import (
        _draft_runtime_transaction,
        _experimental_evaluation_execution,
        _experimental_verification_execution,
        _quantitative_runtime_transaction,
        _real_draft_execution,
        _real_extraction_execution,
        _real_outline_execution,
        _real_quantitative_execution,
        _real_thematic_execution,
        _run_evaluation_stage,
        _run_verification_stage,
    )

    return [
        StageSpec(
            key="03_agente_extraccion_kb",
            label="02 · Extracción de información científica",
            build_execution=_real_extraction_execution,
            runtime_transaction=execute_extraction_runtime_transaction,
            resolve_resume=resolve_extraction_resume,
            build_fingerprints=build_agent_input_fingerprints,
            max_attempt_number=2,
        ),
        StageSpec(
            key="03B_extraccion_cuantitativa_kb",
            label="03 · Extracción y normalización cuantitativa",
            build_execution=_real_quantitative_execution,
            runtime_transaction=_quantitative_runtime_transaction,
            resolve_resume=resolve_quantitative_resume,
            build_fingerprints=build_quantitative_fingerprints,
            max_attempt_number=1,
        ),
        StageSpec(
            key="04_agente_analisis_tematico",
            label="04 · Análisis temático",
            build_execution=_real_thematic_execution,
            runtime_transaction=execute_thematic_runtime_transaction,
            resolve_resume=resolve_thematic_resume,
            build_fingerprints=build_thematic_fingerprints,
        ),
        StageSpec(
            key="05_generador_esquema",
            label="05 · Generación del esquema",
            build_execution=_real_outline_execution,
            runtime_transaction=execute_outline_runtime_transaction,
            resolve_resume=resolve_outline_resume,
            build_fingerprints=build_outline_fingerprints,
        ),
        StageSpec(
            key=DRAFT_STAGE_NAME,
            label="06 · Redacción del borrador",
            build_execution=_real_draft_execution,
            runtime_transaction=_draft_runtime_transaction,
            resolve_resume=resolve_draft_resume,
            build_fingerprints=build_draft_fingerprints,
        ),
        StageSpec(
            key="07_agente_verificador",
            label="07 · Verificación y trazabilidad",
            build_execution=_experimental_verification_execution,
            runtime_transaction=None,  # ver custom_run
            resolve_resume=None,  # ver custom_run
            build_fingerprints=None,  # ver comentario en el campo del dataclass
            custom_run=_run_verification_stage,
        ),
        StageSpec(
            key="08_evaluacion_experimental",
            label="08 · Evaluación experimental",
            build_execution=_experimental_evaluation_execution,
            runtime_transaction=None,  # ver custom_run
            resolve_resume=None,  # ver custom_run
            build_fingerprints=None,  # ver comentario en el campo del dataclass
            custom_run=_run_evaluation_stage,
        ),
    ]


# ---------------------------------------------------------------------------
# Ejecución de una etapa
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageOutcome:
    key: str
    label: str
    # SKIPPED_FRESH | COMMITTED (via resume) | COMMITTED | FAILED
    status: str
    execution_status: str | None
    quality_status: str | None
    warnings: tuple[str, ...]
    error: Mapping[str, Any] | None
    attempt_number: int
    # Transición ya validada por decision_engine.validate_transition:
    # ADVANCE | RETRY | RETURN | HALT_STAGE | STOP_PIPELINE
    next_action: str
    target_stage: str | None
    reason_code: str


def _validated_transition_for(
    spec: StageSpec,
    *,
    requested_transition: RequestedTransition,
    quality_status: QualityStatus,
    attempts_used: int,
) -> ValidatedTransition:
    return validate_transition(
        current_stage=spec.key,
        requested_transition=requested_transition,
        quality_status=quality_status,
        attempts_used=attempts_used,
        max_attempts=spec.max_attempt_number or MAX_ATTEMPTS_DEFAULT,
        known_stages=frozenset(CANONICAL_STAGE_ORDER),
        bypass_manual_review=spec.bypass_manual_review,
    )


def _outcome_from_result(
    spec: StageSpec, result: AgentResult, status: str, *, attempts_used: int
) -> StageOutcome:
    validated = _validated_transition_for(
        spec,
        requested_transition=result.requested_transition,
        quality_status=result.quality_status,
        attempts_used=attempts_used,
    )
    return StageOutcome(
        key=spec.key,
        label=spec.label,
        status=status,
        execution_status=result.execution_status.value,
        quality_status=result.quality_status.value,
        warnings=tuple(w.message for w in result.warnings),
        error=result.error,
        attempt_number=result.attempt_number,
        next_action=validated.action,
        target_stage=validated.target_stage,
        reason_code=validated.reason_code,
    )


def _outcome_from_committed_stage(spec: StageSpec, committed, *, status: str) -> StageOutcome:
    """Construye un StageOutcome a partir de un StageState ya comprometido
    (usado para SKIPPED_FRESH), reutilizando su ``requested_transition``
    histórica en vez de asumir ADVANCE por defecto."""

    requested = committed.requested_transition or RequestedTransition(
        action=TransitionAction.ADVANCE, reason_code="ASSUMED_ADVANCE_NO_HISTORY"
    )
    validated = _validated_transition_for(
        spec,
        requested_transition=requested,
        quality_status=committed.quality_status,
        attempts_used=committed.attempts_used,
    )
    return StageOutcome(
        key=spec.key,
        label=spec.label,
        status=status,
        execution_status=committed.execution_status.value,
        quality_status=(
            committed.quality_status.value if committed.quality_status else None
        ),
        warnings=tuple(w.get("message", "") for w in committed.warnings),
        error=committed.last_error,
        attempt_number=committed.attempts_used,
        next_action=validated.action,
        target_stage=validated.target_stage,
        reason_code=validated.reason_code,
    )


def run_stage(
    *,
    store: StateStore,
    project_dir: str | Path,
    spec: StageSpec,
    attempt_number: int = 1,
    observations: Mapping[str, Any] | None = None,
    force_rerun: bool = False,
) -> StageOutcome:
    """Ejecuta (o resuelve/salta) una única etapa y devuelve su resultado.

    Si ``spec.custom_run`` está definido (caso de 07), delega por completo en
    él y el resto de esta función no se ejecuta — ver el comentario en el
    campo ``custom_run`` de ``StageSpec`` para el porqué.

    Para las demás etapas, orden de decisiones:
    1. Si hay una ``pending_execution`` de esta etapa (interrupción previa),
       se resuelve primero (COMMIT del resultado ya persistido, o liberación
       para reejecutar) — igual que antes.
    2. Si la etapa ya quedó COMPLETED, y ``spec.build_fingerprints`` existe,
       se reconstruye su AgentInput actual y se comparan sus fingerprints
       contra los comprometidos: si coinciden, se salta (SKIPPED_FRESH); si
       no, se considera obsoleta y se reejecuta aunque no se haya pedido
       force_rerun explícitamente. Si ``spec.build_fingerprints`` es None
       (ninguna etapa hoy), se usa el chequeo antiguo (solo COMPLETED).
    3. En cualquier otro caso, se ejecuta la transacción real de la etapa.

    Cualquier excepción no capturada por la propia etapa (solo puede ocurrir
    hoy en 07, que no envuelve fallos de preparación en un AgentResult — ver
    ``_run_verification_stage``) se convierte aquí en un StageOutcome
    ``status="FAILED"``/``next_action="HALT_STAGE"`` para que el llamador
    nunca termine con una excepción sin registrar, en vez de dejarla
    propagar sin control.
    """

    project_dir = Path(project_dir)
    if spec.max_attempt_number is not None and attempt_number > spec.max_attempt_number:
        raise ValueError(
            f"{spec.key} admite como máximo attempt_number={spec.max_attempt_number}."
        )

    try:
        if spec.custom_run is not None:
            return spec.custom_run(
                store=store,
                project_dir=project_dir,
                spec=spec,
                attempt_number=attempt_number,
                observations=observations,
                force_rerun=force_rerun,
            )

        def build_execution() -> tuple[Any, Any]:
            return spec.build_execution(project_dir, attempt_number)

        state = store.load()

        if (
            state.pending_execution is not None
            and state.pending_execution.target_stage == spec.key
        ):
            _agent, agent_input = build_execution()
            resume = spec.resolve_resume(
                store=store, agent_input=agent_input, observations=observations
            )
            if resume.action == "COMMITTED":
                state = store.load()
                attempts_used = state.stages[spec.key].attempts_used
                return _outcome_from_result(
                    spec, resume.committed_result, "COMMITTED", attempts_used=attempts_used
                )
            # REEXECUTE o NO_PENDING: el pending quedó liberado; se sigue abajo.
            state = store.load()

        committed = state.stages.get(spec.key)
        if (
            committed is not None
            and committed.execution_status == ExecutionStatus.COMPLETED
            and not force_rerun
        ):
            if spec.build_fingerprints is not None:
                _agent, agent_input = build_execution()
                current_fingerprints = spec.build_fingerprints(agent_input)
                if is_stage_fresh(committed, current_fingerprints):
                    return _outcome_from_committed_stage(
                        spec, committed, status="SKIPPED_FRESH"
                    )
                # Fingerprints obsoletos: no se salta aunque no se haya pedido
                # force_rerun explícito.
                observations = dict(observations or {})
                observations["orchestrator_note"] = (
                    "stage_stale_fingerprint_mismatch_reexecuting"
                )
            else:
                # Sin build_fingerprints disponible: se conserva el chequeo
                # antiguo (solo COMPLETED, sin comparar vigencia).
                return _outcome_from_committed_stage(
                    spec, committed, status="SKIPPED_FRESH"
                )

        transaction = spec.runtime_transaction(
            store=store,
            build_execution=build_execution,
            attempt_number=attempt_number,
            observations=observations,
        )
        status = (
            "COMMITTED"
            if transaction.agent_result.execution_status == ExecutionStatus.COMPLETED
            else "FAILED"
        )
        state = store.load()
        attempts_used = state.stages[spec.key].attempts_used
        return _outcome_from_result(
            spec, transaction.agent_result, status, attempts_used=attempts_used
        )
    except Exception as exc:  # noqa: BLE001 - red de seguridad genérica, ver docstring
        return StageOutcome(
            key=spec.key,
            label=spec.label,
            status="FAILED",
            execution_status=None,
            quality_status=None,
            warnings=(),
            error={"type": type(exc).__name__, "message": str(exc)},
            attempt_number=attempt_number,
            next_action="HALT_STAGE",
            target_stage=None,
            reason_code=f"UNCAUGHT_EXCEPTION:{type(exc).__name__}",
        )


def _check_already_terminal_state(
    *, store, registry: Mapping[str, "StageSpec"], start_stage: str | None, force_rerun: bool
) -> "StageOutcome | None":
    """Si la decisión AUTORITATIVA del ``decision_log`` (ver
    ``_reconstruct_authoritative_frontier`` -- ni la última entrada
    cronológica ni asumir un único tramo desde el principio) pidió
    explícitamente ``HALT_STAGE`` o ``STOP_PIPELINE``, el pipeline ya
    está en un estado TERMINAL -- un restart sin ``start_stage``
    explícito ni ``--force-rerun`` no debe recorrer las etapas de nuevo
    asumiendo que hay trabajo pendiente. Devuelve el ``StageOutcome``
    terminal a reportar tal cual (sin tocar ningún estado), o ``None``
    si no aplica.

    El ``StageOutcome`` se construye SIEMPRE a partir del propio
    ``frontier_entry.result`` (el ``AgentResult`` persistido en esa
    entrada exacta del log, vía ``AgentResult.from_dict`` +
    ``_outcome_from_result`` -- la misma función que ya usa el resto del
    módulo para construir un ``StageOutcome`` desde un ``AgentResult``
    real) -- nunca desde ``state.stages[stage]`` (el estado COMPROMETIDO
    VIGENTE de esa etapa), que puede corresponder a una ejecución
    POSTERIOR y distinta de la entrada histórica que este chequeo
    determinó como terminal. Mezclar ambas fuentes es exactamente lo
    que producía ``ALREADY_TERMINAL`` con un ``next_action=ADVANCE`` --
    una contradicción de contrato que nunca debe poder ocurrir: se
    afirma explícitamente como invariante antes de devolver."""

    if start_stage is not None or force_rerun:
        return None

    state = store.load()
    frontier_entry = _reconstruct_authoritative_frontier(state.decision_log)
    if frontier_entry is None:
        return None

    frontier_transition = frontier_entry.requested_transition
    if (
        frontier_transition is None
        or frontier_transition.action not in (TransitionAction.HALT_STAGE, TransitionAction.STOP_PIPELINE)
        or frontier_entry.stage not in registry
    ):
        return None

    frontier_result = AgentResult.from_dict(frontier_entry.result)
    outcome = _outcome_from_result(
        registry[frontier_entry.stage], frontier_result, "ALREADY_TERMINAL", attempts_used=frontier_entry.attempt
    )

    # Invariante obligatoria: ALREADY_TERMINAL nunca puede coexistir con
    # una transición no terminal. Si por cualquier motivo no se cumple
    # (no debería, dado el chequeo de frontier_transition.action arriba,
    # pero se verifica explícitamente en vez de confiar en eso
    # implícitamente), no se afirma un estado terminal que el propio
    # outcome contradice -- se deja que el flujo normal decida.
    if outcome.next_action not in ("HALT_STAGE", "STOP_PIPELINE"):
        return None

    return outcome
