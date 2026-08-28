# ============================================================
# MOTOR DE DECISIONES Y VALIDACIÓN DE TRANSICIONES DEL PIPELINE
# Interpreta las transiciones solicitadas por los agentes,
# valida su recorrido y mantiene la coherencia del estado.
# ============================================================

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

# Define el orden oficial de ejecución de las etapas del pipeline.
# Esta secuencia actúa como referencia común para validar transiciones y destinos.
CANONICAL_STAGE_ORDER: tuple[str, ...] = (
    "03_agente_extraccion_kb",
    "03B_extraccion_cuantitativa_kb",
    "04_agente_analisis_tematico",
    "05_generador_esquema",
    "06_agente_redactor",
    "07_agente_verificador",
    "08_evaluacion_experimental",
)

# Define la configuración del ciclo controlado entre el redactor 06 y el verificador 07:
# nombre del ciclo, etapa que lo activa, etapa de retorno y máximo de rondas permitidas.
WRITER_VERIFIER_CYCLE_NAME = "writer_verifier"
WRITER_VERIFIER_TRIGGER_STAGE = "07_agente_verificador"
WRITER_VERIFIER_RETURN_TARGET = "06_agente_redactor"
WRITER_VERIFIER_DEFAULT_MAX_ROUNDS = 3  # mismo valor por defecto que CycleState.max_rounds


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# resultado de intentar realizar un retorno dentro
# del ciclo controlado entre el redactor y el verificador.
@dataclass(frozen=True)
class CycleReturnOutcome:
    new_state: PipelineState #nuevo estado del pipeline después de aplicar el retorno
    invalidated_stages: tuple[str, ...] # qué etapas quedaron invalidadas porque se volvió hacia atrás y sus resultados ya no deben considerarse vigentes.
    cycle_exhausted: bool

    # Comprueba si la transición solicitada corresponde específicamente
    # al retorno controlado del verificador (07) hacia el redactor (06).
    is_writer_verifier_cycle = (
        from_stage == WRITER_VERIFIER_TRIGGER_STAGE
        and target_stage == WRITER_VERIFIER_RETURN_TARGET
    )

    # Si el RETURN no corresponde al ciclo especial 07→06,
    # invalida desde la etapa destino hacia adelante y termina sin agotar ningún ciclo.
    if not is_writer_verifier_cycle:
        new_state, invalidated = invalidate_from(
            store, from_stage_inclusive=target_stage, reason=reason, order=order
        )
        return CycleReturnOutcome(new_state, invalidated, False)

    # Carga el estado actual del pipeline y recupera la información
    # existente del ciclo 06↔07; si aún no existe, crea un ciclo nuevo.
    state = store.load()
    cycle = state.cycles.get(WRITER_VERIFIER_CYCLE_NAME) or CycleState(
        max_rounds=max_rounds
    )

    # Comprueba si el ciclo 06↔07 ya alcanzó el máximo de rondas permitidas.
    if cycle.rounds_used >= cycle.max_rounds:
        # Marca el ciclo como agotado y registra la causa del último retorno.
        exhausted_cycle = replace(
            cycle, status="EXHAUSTED", last_return_reason=reason
        )

        # Actualiza la información de ciclos dentro del estado del pipeline.
        updated_cycles = dict(state.cycles)
        updated_cycles[WRITER_VERIFIER_CYCLE_NAME] = exhausted_cycle
        # Actualiza la fecha de modificación del estado.
        now = _now_iso()
        new_state = replace(
            state,
            cycles=updated_cycles,
            identity=replace(state.identity, updated_at=now),
        )
        store.save(new_state)     # Guarda el nuevo estado persistido con el ciclo marcado como agotado.
        return CycleReturnOutcome(new_state, (), True)     # Devuelve que no se invalidaron nuevas etapas y que el ciclo se agotó.

    # Si todavía quedan rondas disponibles, incrementa en uno el contador
    # del ciclo 06↔07, lo mantiene activo y registra la causa del retorno.
    updated_cycle = replace(
        cycle,
        rounds_used=cycle.rounds_used + 1,
        status="ACTIVE",
        last_return_reason=reason,
    )
    # Actualiza el ciclo writer-verifier dentro del estado general del pipeline,
    # registra la nueva fecha de modificación y guarda el estado persistido.
    updated_cycles = dict(state.cycles)
    updated_cycles[WRITER_VERIFIER_CYCLE_NAME] = updated_cycle
    now = _now_iso()
    state = replace(
        state, cycles=updated_cycles, identity=replace(state.identity, updated_at=now)
    )
    store.save(state)

    # Invalida la etapa destino y todas las posteriores porque sus resultados
    # pueden haber quedado obsoletos después del retorno.
    new_state, invalidated = invalidate_from(
        store, from_stage_inclusive=target_stage, reason=reason, order=order
    )
    return CycleReturnOutcome(new_state, invalidated, False)


# Marca como resuelto el ciclo 06↔07 cuando estaba activo
# y el verificador logra avanzar sin solicitar un nuevo RETURN al redactor.
def resolve_cycle_if_active(
    store: StateStore, *, cycle_name: str = WRITER_VERIFIER_CYCLE_NAME
) -> PipelineState:
    """Marca un ciclo como RESOLVED si estaba ACTIVE (etapa disparadora avanzó sin RETURN)."""

    state = store.load() # Carga el estado actual del pipeline.
    cycle = state.cycles.get(cycle_name)  # Recupera la información del ciclo solicitado.
    if cycle is None or cycle.status != "ACTIVE": # Si el ciclo no existe o ya no está activo, no hace ningún cambio.
        return state
    # Cambia el estado del ciclo de ACTIVE a RESOLVED.
    updated_cycles = dict(state.cycles)
    updated_cycles[cycle_name] = replace(cycle, status="RESOLVED")
    
    now = _now_iso()
    new_state = replace(
        state, cycles=updated_cycles, identity=replace(state.identity, updated_at=now)
    )
    store.save(new_state)
    return new_state

# Define el número máximo de intentos permitidos por defecto
# para volver a ejecutar una etapa cuando solicita RETRY. 
MAX_ATTEMPTS_DEFAULT = 3


class TransitionValidationError(ValueError):
    """Una transición solicitada por un agente no es válida y se rechaza."""

# Representa una transición ya validada por el motor de decisiones,
# indicando qué acción ejecutar, a qué etapa dirigirse y por qué motivo.
@dataclass(frozen=True)
class ValidatedTransition:
    action: str  # ADVANCE | RETRY | RETURN | HALT_STAGE | STOP_PIPELINE
    target_stage: str | None
    reason_code: str


# Devuelve la siguiente etapa según el orden canónico del pipeline.
# Si la etapa actual es la última, devuelve None.
def default_next_stage(
    stage_key: str, order: tuple[str, ...] = CANONICAL_STAGE_ORDER
) -> str | None:
    # Verifica que la etapa exista dentro del orden oficial.
    if stage_key not in order:
        raise TransitionValidationError(
            f"Etapa desconocida en CANONICAL_STAGE_ORDER: {stage_key!r}"
        )
    # Busca la posición de la etapa actual.
    idx = order.index(stage_key)
    # Si ya es la última etapa, no existe una siguiente.
    if idx + 1 >= len(order):
        return None
    # Devuelve la etapa que sigue inmediatamente.
    return order[idx + 1]


# Valida la transición solicitada por una etapa y determina
# cuál es la acción segura que realmente puede ejecutar el pipeline.
def validate_transition(
    *,
    current_stage: str, #etapa que acaba de ejecutarse
    requested_transition: RequestedTransition, #ADVANCE, RETRY, RETURN, etc.
    quality_status: QualityStatus, #estado de calidad obtenido por la etapa
    attempts_used: int, #intentos lleva esa etapa
    max_attempts: int = MAX_ATTEMPTS_DEFAULT, #máximo de intentos permitidos; por defecto 3
    known_stages: frozenset[str] | None = None, #conjunto de etapas válidas
    bypass_manual_review: bool = False,
    advance_allowed_skips: Mapping[str, frozenset[str]] | None = None,
    order: tuple[str, ...] = CANONICAL_STAGE_ORDER, #usa el orden canónico 03 → 03B → 04 → 05 → 06 → 07 → 08.
) -> ValidatedTransition:

    # Prepara los datos necesarios para validar la transición solicitada:
    # define las etapas conocidas, los saltos permitidos y la acción pedida.
    known = known_stages if known_stages is not None else frozenset(order)
    advance_allowed_skips = advance_allowed_skips or {}
    action = requested_transition.action.value

    # Si la etapa requiere revisión manual antes de avanzar y dicha revisión
    # no fue omitida explícitamente, detiene la etapa en lugar de continuar.
    if (
        quality_status is QualityStatus.APPROVED_PENDING_MANUAL_REVIEW
        and action == "ADVANCE"
        and not bypass_manual_review
    ):
        return ValidatedTransition("HALT_STAGE", None, "MANUAL_REVIEW_REQUIRED")

    
    # Valida una transición de tipo ADVANCE:
    # resuelve el destino, comprueba que exista y que el salto esté permitido.
    if action == "ADVANCE":
         # Usa el destino solicitado o, si no se indicó uno, la siguiente etapa normal.
        resolved_target = requested_transition.target_stage or default_next_stage(
            current_stage, order
        )
        # Si no existe una etapa siguiente, el pipeline terminó.
        if resolved_target is None:
            return ValidatedTransition("STOP_PIPELINE", None, "PIPELINE_COMPLETE")
            
        if resolved_target not in known:
            raise TransitionValidationError(
                f"ADVANCE hacia una etapa no registrada: {resolved_target!r}"
            )
        # Obtiene el destino normal y los saltos excepcionales permitidos.
        default_target = default_next_stage(current_stage, order)
        allowed_extra = advance_allowed_skips.get(current_stage, frozenset())
        # Rechaza cualquier salto que no sea el normal ni esté autorizado.
        if resolved_target != default_target and resolved_target not in allowed_extra:
            raise TransitionValidationError(
                f"ADVANCE desde {current_stage!r} hacia {resolved_target!r} no está "
                f"permitido (destino por defecto: {default_target!r}; "
                f"saltos permitidos: {sorted(allowed_extra)})."
            )
        # Devuelve la transición ADVANCE ya validada
        return ValidatedTransition(
            "ADVANCE", resolved_target, requested_transition.reason_code or "ADVANCE"
        )

    
    # Valida una transición RETRY:
    # permite repetir la etapa solo mientras no se alcance el máximo de intentos.
    if action == "RETRY":
        if attempts_used >= max_attempts: #Si ya se agotaron los intentos permitidos, detiene la etapa.
            return ValidatedTransition("HALT_STAGE", None, "RETRY_EXHAUSTED")
        return ValidatedTransition( # Si todavía quedan intentos, vuelve a ejecutar la misma etapa.
            "RETRY", current_stage, requested_transition.reason_code or "RETRY"
        )


    # Valida una transición RETURN:
    # exige un destino válido y comprueba que realmente sea una etapa anterior.
    if action == "RETURN":
        target = requested_transition.target_stage     # Obtiene la etapa a la que se quiere regresar.
        # El RETURN debe tener un destino existente y conocido.
        if target is None or target not in known:
            raise TransitionValidationError(
                f"RETURN sin destino válido: {target!r}"
            )
        # Tanto la etapa actual como el destino deben pertenecer
        # al orden canónico del pipeline.
        if current_stage not in order or target not in order:
            raise TransitionValidationError(
                "RETURN requiere que la etapa actual y el destino estén en "
                "CANONICAL_STAGE_ORDER."
            )
        # Impide que RETURN apunte a la misma etapa o a una posterior.
        if order.index(target) >= order.index(current_stage):
            raise TransitionValidationError(
                f"RETURN debe apuntar a una etapa anterior a {current_stage!r}; "
                f"se recibió {target!r}."
            )
        # Devuelve el RETURN ya validado.
        return ValidatedTransition(
            "RETURN", target, requested_transition.reason_code or "RETURN"
        )
        

    # Valida una transición HALT_STAGE y detiene únicamente
    # la etapa actual, sin dirigir el flujo hacia otro nodo.
    if action == "HALT_STAGE":
        return ValidatedTransition(
            "HALT_STAGE", None, requested_transition.reason_code or "HALT_STAGE"
        )


    # Valida una transición STOP_PIPELINE y detiene por completo
    # la ejecución del flujo, sin dirigirlo hacia ninguna otra etapa.
    if action == "STOP_PIPELINE":
        return ValidatedTransition(
            "STOP_PIPELINE", None, requested_transition.reason_code or "STOP_PIPELINE"
        )

    raise TransitionValidationError(f"Acción de transición desconocida: {action!r}")


# Comprueba si el resultado previamente guardado de una etapa sigue siendo válido
# comparando su estado de ejecución, calidad y fingerprints actuales.
def is_stage_fresh(
    committed: StageState, current_fingerprints: StageFingerprints
) -> bool:

    # Normaliza el estado de ejecución para compararlo como texto.
    execution_status = getattr(
        committed.execution_status,
        "value",
        committed.execution_status,
    )

    # Normaliza también el estado de calidad.
    quality_status = getattr(
        committed.quality_status,
        "value",
        committed.quality_status,
    )

    # La etapa debe haber terminado correctamente.
    if execution_status != "COMPLETED":
        return False

    # Su calidad debe haber sido aceptada.
    if quality_status not in {
        "APPROVED",
        "APPROVED_WITH_WARNINGS",
    }:
        return False

    # Debe existir un fingerprint compuesto previo para poder comparar.
    if committed.fingerprints.composite is None:
        return False

    # Comprueba si los fingerprints guardados coinciden con los actuales.
    return fingerprints_match(
        committed.fingerprints,
        current_fingerprints,
    )


# Invalida una etapa y todas las posteriores cuando sus resultados
# ya no deben considerarse vigentes por un cambio o retorno en el pipeline.
def invalidate_from(
    store: StateStore,
    *,
    from_stage_inclusive: str,
    reason: str,
    order: tuple[str, ...] = CANONICAL_STAGE_ORDER,
) -> tuple[PipelineState, tuple[str, ...]]:

    # Verifica que la etapa desde la que se invalidará exista en el orden oficial.
    if from_stage_inclusive not in order:
        raise TransitionValidationError(
            f"Etapa desconocida para invalidar: {from_stage_inclusive!r}"
        )

    # Obtiene la etapa indicada y todas las que vienen después.
    to_invalidate = set(order[order.index(from_stage_inclusive) :])

    # Carga el estado actual del pipeline.
    state = store.load()
    # Si existe una ejecución pendiente de alguna etapa que ahora será invalidada,
    # cancela también esa ejecución pendiente.
    if (
        state.pending_execution is not None
        and state.pending_execution.target_stage in to_invalidate
    ):
        state = store.cancel_pending_execution()

    # Genera la fecha de actualización y prepara una copia de los estados.
    now = datetime.now(timezone.utc).isoformat()
    updated_stages = dict(state.stages)
    changed: list[str] = []
    # Recorre las etapas y marca como INVALIDATED las que correspondan.
    for stage_key in order:
        if stage_key not in to_invalidate:
            continue
        existing = updated_stages.get(stage_key)
        # Si la etapa no existe todavía o ya estaba invalidada, no hace nada.
        if existing is None or existing.execution_status == ExecutionStatus.INVALIDATED:
            continue
        # Registra la causa de invalidación si aún no estaba registrada.
        existing_codes = existing.failure_reason_codes
        if reason not in existing_codes:
            existing_codes = (*existing_codes, reason)
        # Actualiza la etapa como INVALIDATED.
        updated_stages[stage_key] = replace(
            existing,
            execution_status=ExecutionStatus.INVALIDATED,
            failure_reason_codes=existing_codes,
            updated_at=now,
        )
        changed.append(stage_key)

    # Construye y guarda el nuevo estado general del pipeline.
    new_state = replace(
        state,
        stages=updated_stages,
        identity=replace(state.identity, updated_at=now),
    )
    store.save(new_state)
    return new_state, tuple(changed)
