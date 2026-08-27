"""Motor de decisiones del orquestador: interpreta ``RequestedTransition``.

Este módulo es deliberadamente independiente de la ejecución de agentes. Solo
sabe interpretar y validar una transición ya solicitada (``RequestedTransition``,
del contrato ``AgentResult`` aprobado en ``src/contracts``), decidir si una
etapa comprometida sigue vigente según sus fingerprints, e invalidar etapas
descendientes cuando corresponde. No importa nada de ``src/adapters`` ni
``src/agents``: solo ``src/contracts`` y ``src/state``.

Alcance deliberado
-------------------
``RETURN`` no lo emite ningún agente real hoy (verificado por inspección de
``src/agents/*``): todos usan ``ADVANCE``/``RETRY``/``HALT_STAGE`` con
``target_stage=None``. Este módulo igual implementa y prueba ``RETURN`` con
dobles, para el ciclo ``writer_verifier`` (06↔07) que ``CycleState`` ya
reserva en ``pipeline_state.json`` pero que todavía no está conectado a
ningún runtime.

``08`` (evaluación) sí aparece en ``CANONICAL_STAGE_ORDER`` con el nombre
``"08_evaluacion_experimental"`` — se encontró ese literal dentro del propio
notebook 08 (usado allí como valor de un campo `"stage"` en su diccionario de
fingerprint, no como un `stage_name` de `StateStore` ya conectado). Incluirlo
aquí permite que RETURN/ADVANCE lo validen como destino válido y que la
invalidación en cascada lo alcance, pero **no implica que exista un
``StageSpec`` ejecutable para 08** — ver ``pipeline_orchestrator.py`` y el
informe de la iteración para el detalle de por qué 08 se registra sin
adaptador todavía.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Mapping

from src.contracts.agent_result import QualityStatus, RequestedTransition
from src.state.fingerprints import fingerprints_match
from src.state.pipeline_state import (
    CycleState,
    ExecutionStatus,
    PipelineState,
    StageFingerprints,
    StageState,
)
from src.state.state_store import StateStore

# Orden canónico completo conocido, incluida la etapa 07 aunque todavía no
# tenga StageSpec ejecutable en pipeline_orchestrator.py. Se usa únicamente
# para validar existencia/orden de destinos de transición y para calcular el
# alcance de una invalidación en cascada — no implica que el orquestador
# pueda ejecutar esas etapas todavía.
CANONICAL_STAGE_ORDER: tuple[str, ...] = (
    "03_agente_extraccion_kb",
    "03B_extraccion_cuantitativa_kb",
    "04_agente_analisis_tematico",
    "05_generador_esquema",
    "06_agente_redactor",
    "07_agente_verificador",
    "08_evaluacion_experimental",
)

# Nombre y umbrales del ciclo writer_verifier (07 -> 06) ya reservado en
# PipelineState.cycles, aunque hasta ahora ningún runtime lo poblaba.
WRITER_VERIFIER_CYCLE_NAME = "writer_verifier"
WRITER_VERIFIER_TRIGGER_STAGE = "07_agente_verificador"
WRITER_VERIFIER_RETURN_TARGET = "06_agente_redactor"
WRITER_VERIFIER_DEFAULT_MAX_ROUNDS = 3  # mismo valor por defecto que CycleState.max_rounds


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CycleReturnOutcome:
    new_state: PipelineState
    invalidated_stages: tuple[str, ...]
    cycle_exhausted: bool


def apply_return_with_cycle(
    store: StateStore,
    *,
    from_stage: str,
    target_stage: str,
    reason: str,
    max_rounds: int = WRITER_VERIFIER_DEFAULT_MAX_ROUNDS,
    order: tuple[str, ...] = CANONICAL_STAGE_ORDER,
) -> CycleReturnOutcome:
    """Aplica un RETURN ya validado, gestionando ``CycleState`` cuando corresponde.

    Si ``from_stage``/``target_stage`` son exactamente el par 07→06 (el único
    ciclo ``writer_verifier`` reservado hoy en ``pipeline_state.json``),
    incrementa ``CycleState.rounds_used`` antes de invalidar, o — si ya se
    alcanzó ``max_rounds`` — NO invalida ni avanza: marca el ciclo
    ``EXHAUSTED`` y deja que el llamador decida detener el pipeline
    (``cycle_exhausted=True``).

    Para cualquier otro par etapa-origen/etapa-destino, no existe ciclo
    declarado y esta función solo invalida (delega en ``invalidate_from``),
    sin tocar ``PipelineState.cycles``.
    """

    is_writer_verifier_cycle = (
        from_stage == WRITER_VERIFIER_TRIGGER_STAGE
        and target_stage == WRITER_VERIFIER_RETURN_TARGET
    )

    if not is_writer_verifier_cycle:
        new_state, invalidated = invalidate_from(
            store, from_stage_inclusive=target_stage, reason=reason, order=order
        )
        return CycleReturnOutcome(new_state, invalidated, False)

    state = store.load()
    cycle = state.cycles.get(WRITER_VERIFIER_CYCLE_NAME) or CycleState(
        max_rounds=max_rounds
    )

    if cycle.rounds_used >= cycle.max_rounds:
        exhausted_cycle = replace(
            cycle, status="EXHAUSTED", last_return_reason=reason
        )
        updated_cycles = dict(state.cycles)
        updated_cycles[WRITER_VERIFIER_CYCLE_NAME] = exhausted_cycle
        now = _now_iso()
        new_state = replace(
            state,
            cycles=updated_cycles,
            identity=replace(state.identity, updated_at=now),
        )
        store.save(new_state)
        return CycleReturnOutcome(new_state, (), True)

    updated_cycle = replace(
        cycle,
        rounds_used=cycle.rounds_used + 1,
        status="ACTIVE",
        last_return_reason=reason,
    )
    updated_cycles = dict(state.cycles)
    updated_cycles[WRITER_VERIFIER_CYCLE_NAME] = updated_cycle
    now = _now_iso()
    state = replace(
        state, cycles=updated_cycles, identity=replace(state.identity, updated_at=now)
    )
    store.save(state)

    new_state, invalidated = invalidate_from(
        store, from_stage_inclusive=target_stage, reason=reason, order=order
    )
    return CycleReturnOutcome(new_state, invalidated, False)


def resolve_cycle_if_active(
    store: StateStore, *, cycle_name: str = WRITER_VERIFIER_CYCLE_NAME
) -> PipelineState:
    """Marca un ciclo como RESOLVED si estaba ACTIVE (etapa disparadora avanzó sin RETURN)."""

    state = store.load()
    cycle = state.cycles.get(cycle_name)
    if cycle is None or cycle.status != "ACTIVE":
        return state
    updated_cycles = dict(state.cycles)
    updated_cycles[cycle_name] = replace(cycle, status="RESOLVED")
    now = _now_iso()
    new_state = replace(
        state, cycles=updated_cycles, identity=replace(state.identity, updated_at=now)
    )
    store.save(new_state)
    return new_state


# CycleState.max_rounds usa 3 como presupuesto por defecto del ciclo
# writer_verifier; se reutiliza el mismo número como límite por defecto de
# reintentos por etapa para no introducir una segunda convención de "cuántos
# intentos son razonables" sin justificación.
MAX_ATTEMPTS_DEFAULT = 3


class TransitionValidationError(ValueError):
    """Una transición solicitada por un agente no es válida y se rechaza."""


@dataclass(frozen=True)
class ValidatedTransition:
    action: str  # ADVANCE | RETRY | RETURN | HALT_STAGE | STOP_PIPELINE
    target_stage: str | None
    reason_code: str


def default_next_stage(
    stage_key: str, order: tuple[str, ...] = CANONICAL_STAGE_ORDER
) -> str | None:
    if stage_key not in order:
        raise TransitionValidationError(
            f"Etapa desconocida en CANONICAL_STAGE_ORDER: {stage_key!r}"
        )
    idx = order.index(stage_key)
    if idx + 1 >= len(order):
        return None
    return order[idx + 1]


def validate_transition(
    *,
    current_stage: str,
    requested_transition: RequestedTransition,
    quality_status: QualityStatus,
    attempts_used: int,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT,
    known_stages: frozenset[str] | None = None,
    bypass_manual_review: bool = False,
    advance_allowed_skips: Mapping[str, frozenset[str]] | None = None,
    order: tuple[str, ...] = CANONICAL_STAGE_ORDER,
) -> ValidatedTransition:
    """Valida y normaliza la transición solicitada por un agente.

    No ejecuta nada; solo decide si la transición solicitada es aceptable y,
    de serlo, cuál es la acción/destino resultante. Lanza
    ``TransitionValidationError`` si el agente pidió algo inválido (destino
    inexistente, RETURN hacia adelante, ADVANCE saltando etapas no
    autorizado, etc.).
    """

    known = known_stages if known_stages is not None else frozenset(order)
    advance_allowed_skips = advance_allowed_skips or {}
    action = requested_transition.action.value

    # Punto 7: APPROVED_PENDING_MANUAL_REVIEW no se trata como aprobación
    # automática salvo bypass explícito por etapa.
    if (
        quality_status is QualityStatus.APPROVED_PENDING_MANUAL_REVIEW
        and action == "ADVANCE"
        and not bypass_manual_review
    ):
        return ValidatedTransition("HALT_STAGE", None, "MANUAL_REVIEW_REQUIRED")

    if action == "ADVANCE":
        resolved_target = requested_transition.target_stage or default_next_stage(
            current_stage, order
        )
        if resolved_target is None:
            return ValidatedTransition("STOP_PIPELINE", None, "PIPELINE_COMPLETE")
        if resolved_target not in known:
            raise TransitionValidationError(
                f"ADVANCE hacia una etapa no registrada: {resolved_target!r}"
            )
        default_target = default_next_stage(current_stage, order)
        allowed_extra = advance_allowed_skips.get(current_stage, frozenset())
        if resolved_target != default_target and resolved_target not in allowed_extra:
            raise TransitionValidationError(
                f"ADVANCE desde {current_stage!r} hacia {resolved_target!r} no está "
                f"permitido (destino por defecto: {default_target!r}; "
                f"saltos permitidos: {sorted(allowed_extra)})."
            )
        return ValidatedTransition(
            "ADVANCE", resolved_target, requested_transition.reason_code or "ADVANCE"
        )

    if action == "RETRY":
        if attempts_used >= max_attempts:
            return ValidatedTransition("HALT_STAGE", None, "RETRY_EXHAUSTED")
        return ValidatedTransition(
            "RETRY", current_stage, requested_transition.reason_code or "RETRY"
        )

    if action == "RETURN":
        target = requested_transition.target_stage
        if target is None or target not in known:
            raise TransitionValidationError(
                f"RETURN sin destino válido: {target!r}"
            )
        if current_stage not in order or target not in order:
            raise TransitionValidationError(
                "RETURN requiere que la etapa actual y el destino estén en "
                "CANONICAL_STAGE_ORDER."
            )
        if order.index(target) >= order.index(current_stage):
            raise TransitionValidationError(
                f"RETURN debe apuntar a una etapa anterior a {current_stage!r}; "
                f"se recibió {target!r}."
            )
        return ValidatedTransition(
            "RETURN", target, requested_transition.reason_code or "RETURN"
        )

    if action == "HALT_STAGE":
        return ValidatedTransition(
            "HALT_STAGE", None, requested_transition.reason_code or "HALT_STAGE"
        )

    if action == "STOP_PIPELINE":
        return ValidatedTransition(
            "STOP_PIPELINE", None, requested_transition.reason_code or "STOP_PIPELINE"
        )

    raise TransitionValidationError(f"Acción de transición desconocida: {action!r}")


def is_stage_fresh(
    committed: StageState, current_fingerprints: StageFingerprints
) -> bool:
    """True solo si la etapa terminó correctamente, quedó aprobada
    y sus fingerprints siguen vigentes.

    Una etapa INVALIDATED, FAILED, NEEDS_REVISION o REJECTED nunca
    puede reutilizarse como SKIPPED_FRESH aunque sus fingerprints
    coincidan.
    """

    execution_status = getattr(
        committed.execution_status,
        "value",
        committed.execution_status,
    )

    quality_status = getattr(
        committed.quality_status,
        "value",
        committed.quality_status,
    )

    if execution_status != "COMPLETED":
        return False

    if quality_status not in {
        "APPROVED",
        "APPROVED_WITH_WARNINGS",
    }:
        return False

    if committed.fingerprints.composite is None:
        return False

    return fingerprints_match(
        committed.fingerprints,
        current_fingerprints,
    )


def invalidate_from(
    store: StateStore,
    *,
    from_stage_inclusive: str,
    reason: str,
    order: tuple[str, ...] = CANONICAL_STAGE_ORDER,
) -> tuple[PipelineState, tuple[str, ...]]:
    """Invalida ``from_stage_inclusive`` y todas las etapas posteriores en ``order``.

    Usa ``ExecutionStatus.INVALIDATED`` (ya parte del contrato ``AgentResult``,
    sin uso previo en el repo) en vez de ``InvalidationState`` porque
    ``InvalidationState`` está deliberadamente restringido a
    ``scope_type="FULL"`` (invalidación de todo el pipeline) y no admite un
    subconjunto de etapas; modificar esa validación habría significado tocar
    un contrato ya aprobado ("Decisión 3") en vez de extenderlo.

    Solo modifica etapas que ya tienen ``StageState`` comprometido (una etapa
    que nunca corrió no tiene nada que invalidar). Si había una
    ``pending_execution`` apuntando a alguna de las etapas invalidadas, se
    cancela primero.

    Devuelve el nuevo ``PipelineState`` y las claves efectivamente invalidadas.
    """

    if from_stage_inclusive not in order:
        raise TransitionValidationError(
            f"Etapa desconocida para invalidar: {from_stage_inclusive!r}"
        )
    to_invalidate = set(order[order.index(from_stage_inclusive) :])

    state = store.load()
    if (
        state.pending_execution is not None
        and state.pending_execution.target_stage in to_invalidate
    ):
        state = store.cancel_pending_execution()

    now = datetime.now(timezone.utc).isoformat()
    updated_stages = dict(state.stages)
    changed: list[str] = []
    for stage_key in order:
        if stage_key not in to_invalidate:
            continue
        existing = updated_stages.get(stage_key)
        if existing is None or existing.execution_status == ExecutionStatus.INVALIDATED:
            continue
        existing_codes = existing.failure_reason_codes
        if reason not in existing_codes:
            existing_codes = (*existing_codes, reason)
        updated_stages[stage_key] = replace(
            existing,
            execution_status=ExecutionStatus.INVALIDATED,
            failure_reason_codes=existing_codes,
            updated_at=now,
        )
        changed.append(stage_key)

    new_state = replace(
        state,
        stages=updated_stages,
        identity=replace(state.identity, updated_at=now),
    )
    store.save(new_state)
    return new_state, tuple(changed)
