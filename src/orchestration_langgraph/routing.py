"""Traduce ``StageOutcome.next_action`` (ya validado por
``decision_engine.validate_transition`` dentro de ``run_stage()``) a una
decisión de ruteo para LangGraph.

Misma semántica exacta que ``pipeline_orchestrator._apply_stage_transition``
-- de hecho, la implementación fue trasladada revisando línea por línea
contra esa función para no introducir una segunda política de transición
paralela. Diferencia de forma, no de fondo: en vez de devolver
``(nuevo_current_stage, debe_detenerse)`` para un bucle ``for`` imperativo,
devuelve el nombre del nodo destino (o ``"__end__"``) para un edge
condicional de LangGraph.

``resolve_transition()`` es la función compartida real -- se llama DESDE
CADA NODO (ver ``nodes.py``), inmediatamente después de ``run_stage()``,
porque necesita poder mutar ``StateStore`` (``resolve_cycle_if_active``/
``apply_return_with_cycle``) y devolver el ``attempt_numbers`` actualizado
como parte del estado del nodo -- LangGraph separa "los nodos mutan
estado" de "los edges condicionales solo leen estado para decidir", así
que la decisión ya viene resuelta quando el edge condicional
(``route_after_stage``) se evalúa: ese edge solo lee
``state["route_target"]``, nunca vuelve a llamar a ``decision_engine`` ni
a tocar ``StateStore``.
"""

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


def resolve_transition(
    outcome: StageOutcome,
    *,
    store,
    stage_key: str,
    attempt_number: int,
    attempt_numbers: dict[str, int],
    until: str | None,
) -> tuple[str, dict[str, int], StageOutcome | None]:
    """Devuelve ``(route_target, attempt_numbers_actualizado,
    outcome_sintetico_o_None)``.

    ``outcome_sintetico`` es no-``None`` únicamente en el caso
    ``CYCLE_EXHAUSTED`` (mismo ``StageOutcome`` sintético que
    ``_apply_stage_transition`` agrega a ``outcomes`` en ese caso) -- el
    nodo llamador debe agregarlo a ``state["outcomes"]`` si no es
    ``None``.
    """

    attempt_numbers = dict(attempt_numbers)

    if until is not None and stage_key == until:
        return END, attempt_numbers, None

    if outcome.next_action == "ADVANCE":
        if outcome.target_stage is None:
            return END, attempt_numbers, None  # pipeline completo
        if stage_key == de.WRITER_VERIFIER_TRIGGER_STAGE:
            resolve_cycle_if_active(store)
        return outcome.target_stage, attempt_numbers, None

    if outcome.next_action == "RETRY":
        attempt_numbers[stage_key] = attempt_number + 1
        return stage_key, attempt_numbers, None

    if outcome.next_action == "RETURN":
        cycle_result = apply_return_with_cycle(
            store,
            from_stage=stage_key,
            target_stage=outcome.target_stage,
            reason=f"INVALIDATED_BY_RETURN_FROM_{stage_key}",
        )
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
        for stage_key_to_clear in CANONICAL_STAGE_ORDER[
            CANONICAL_STAGE_ORDER.index(outcome.target_stage):
        ]:
            attempt_numbers.pop(stage_key_to_clear, None)
        return outcome.target_stage, attempt_numbers, None

    # HALT_STAGE o STOP_PIPELINE: se detiene el grafo.
    return END, attempt_numbers, None


def route_after_stage(state: "GraphState") -> str:
    """Callback de edge condicional de LangGraph -- SOLO lee
    ``state["route_target"]``, ya resuelto por ``resolve_transition()``
    dentro del nodo que acaba de correr. Nunca decide nada por sí mismo,
    nunca toca ``StateStore``."""

    target = state.get("route_target")
    if not target:
        raise RuntimeError(
            "route_after_stage: state['route_target'] ausente -- el nodo "
            "que acaba de correr debió resolverlo vía resolve_transition()."
        )
    return target
