# ============================================================
# ENRUTAMIENTO DE TRANSICIONES DEL PIPELINE CON LANGGRAPH
# Traduce las decisiones de cada etapa en el siguiente nodo del grafo,
# gestionando avances, reintentos, retornos, ciclos y finalización.
# ============================================================

from __future__ import annotations

from src.orchestration import decision_engine as de
from src.orchestration.decision_engine import (
    CANONICAL_STAGE_ORDER,
    apply_return_with_cycle,
    resolve_cycle_if_active,
)
from src.orchestration.stage_execution import StageOutcome
from src.orchestration_langgraph.graph_state import GraphState

from langgraph.graph import END


# Resuelve la transición que debe seguir el pipeline después de una etapa,
# actualizando intentos y devolviendo el nodo destino que usará LangGraph.
def resolve_transition(
    outcome: StageOutcome, #resultado de la etapa que acaba de ejecutarse
    *,
    store, #acceso al estado persistido del pipeline
    stage_key: str, #identifica qué etapa acaba de terminar
    attempt_number: int, #número de intento de esa ejecución
    attempt_numbers: dict[str, int], #registro de intentos de todas las etapas
    until: str | None, #etapa en la que el usuario pidió detener el pipeline
) -> tuple[str, dict[str, int], StageOutcome | None]: #nodo destino al que debe ir LangGraph; diccionario actualizado de intentos por etapa;

    attempt_numbers = dict(attempt_numbers) #toma el diccionario original de intentos y hace una copia

    # Si el usuario pidió detener el pipeline en esta etapa,
    # finaliza el grafo inmediatamente.
    if until is not None and stage_key == until:
        return END, attempt_numbers, None
        
    # Si la etapa terminó correctamente y solicita avanzar,
    # continúa hacia la etapa indicada o finaliza si ya no existe destino.
    if outcome.next_action == "ADVANCE":
        if outcome.target_stage is None:
            return END, attempt_numbers, None  # pipeline completo
        # Si se sale correctamente del ciclo redactor-verificador,
        # marca el ciclo como resuelto en el estado persistido.
        if stage_key == de.WRITER_VERIFIER_TRIGGER_STAGE:
            resolve_cycle_if_active(store)
        return outcome.target_stage, attempt_numbers, None
        
    # Si la etapa solicita un reintento, incrementa su contador
    # y vuelve a ejecutar el mismo nodo.
    if outcome.next_action == "RETRY":
        attempt_numbers[stage_key] = attempt_number + 1
        return stage_key, attempt_numbers, None
        
    # Si la etapa solicita volver a una etapa anterior,
    # aplica las reglas del ciclo controlado y actualiza el estado.
    if outcome.next_action == "RETURN":
        cycle_result = apply_return_with_cycle(# 07<->06
            store,
            from_stage=stage_key,
            target_stage=outcome.target_stage,
            reason=f"INVALIDATED_BY_RETURN_FROM_{stage_key}",
        )
        # Si el ciclo alcanzó su límite de retornos,
        # genera un resultado sintético y detiene el pipeline.
        if cycle_result.cycle_exhausted:
            synthetic = StageOutcome(
                key=stage_key,
                label=f"(ciclo {de.WRITER_VERIFIER_CYCLE_NAME} agotado)",
                status="CYCLE_EXHAUSTED",
                execution_status=None,
                quality_status=None,
                warnings=(),
                error=None,
                attempt_number=attempt_number,
                next_action="HALT_STAGE",
                target_stage=None,
                reason_code="WRITER_VERIFIER_CYCLE_EXHAUSTED",
            )
            return END, attempt_numbers, synthetic
        
        # Si el retorno es válido, elimina los contadores de intento
        # de la etapa destino y de todas las etapas posteriores.
        for stage_key_to_clear in CANONICAL_STAGE_ORDER[
            CANONICAL_STAGE_ORDER.index(outcome.target_stage):
        ]:
            attempt_numbers.pop(stage_key_to_clear, None)
        
        # Retorna a la etapa solicitada para continuar el ciclo.
        return outcome.target_stage, attempt_numbers, None

    # HALT_STAGE o STOP_PIPELINE: se detiene el grafo.
    return END, attempt_numbers, None

# Lee del estado el destino que ya fue calculado por resolve_transition()
# y se lo devuelve a LangGraph para continuar el recorrido del grafo.
def route_after_stage(state: "GraphState") -> str:
    target = state.get("route_target")
    # Si el nodo anterior no dejó definido un destino, detiene la ejecución
    # porque la transición debió resolverse previamente.
    if not target:
        raise RuntimeError(
            "route_after_stage: state['route_target'] ausente -- el nodo "
            "que acaba de correr debió resolverlo vía resolve_transition()."
        )
    return target
