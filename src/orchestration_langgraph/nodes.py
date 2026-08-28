"""Nodos del ``StateGraph`` de MAIN 5.

Cada nodo delega por completo en la infraestructura ya extraída
(``stage_execution.run_stage``, ``pending_reconciliation.
_reconcile_pending_execution_for_other_stage``) y en ``routing.
resolve_transition`` para decidir a dónde ir después -- ningún nodo
reimplementa lógica de ejecución de etapa ni de transición.

No depende de ninguna máquina de estados propia -- ``pipeline_orchestrator.py``
se retiró por completo en el Bloque 6; este módulo nunca lo importó ni lo importa.
"""

from __future__ import annotations

from typing import Any, Callable

from src.orchestration.decision_engine import CANONICAL_STAGE_ORDER
from src.orchestration.pending_reconciliation import (
    _reconcile_pending_execution_for_other_stage,
)
from src.orchestration.stage_execution import (
    StageSpec,
    _check_already_terminal_state,
    _stage_registry,
    ensure_pipeline_state,
    run_stage,
)
from src.orchestration_langgraph.graph_state import GraphState
from src.orchestration_langgraph.routing import END, resolve_transition


def reconcile_pending_node(state: GraphState) -> dict[str, Any]:
    """Nodo de arranque único, siempre el primero en correr.

    Replica exactamente el pre-flight de ``run_pipeline()`` ANTES de su
    bucle ``for``: primero ``_check_already_terminal_state`` (si el
    experimento ya está en un estado terminal comprometido, reporta ese
    outcome y termina sin tocar nada más), después
    ``_reconcile_pending_execution_for_other_stage`` (si hay una
    ``pending_execution`` de una etapa distinta a la de entrada, la
    resuelve primero). Si ninguna de las dos aplica, rutea directo a la
    etapa de entrada.
    """

    project_dir = state["project_dir"]
    store = ensure_pipeline_state(project_dir)
    registry = {spec.key: spec for spec in _stage_registry()}
    entry_stage = state.get("entry_stage") or CANONICAL_STAGE_ORDER[0]
    force_rerun = bool(state.get("force_rerun"))
    attempt_numbers = dict(state.get("attempt_numbers") or {})
    observations = state.get("observations")
    outcomes = list(state.get("outcomes") or [])

    terminal_outcome = _check_already_terminal_state(
        store=store,
        registry=registry,
        start_stage=state.get("entry_stage"),
        force_rerun=force_rerun,
    )
    if terminal_outcome is not None:
        outcomes.append(terminal_outcome)
        return {
            "outcomes": outcomes,
            "last_stage_outcome": terminal_outcome,
            "route_target": END,
        }

    reconcile_outcomes, must_stop = _reconcile_pending_execution_for_other_stage(
        store=store,
        project_dir=project_dir,
        registry=registry,
        current_stage=entry_stage,
        attempt_numbers=attempt_numbers,
        observations=observations,
    )
    if reconcile_outcomes:
        reconcile_outcome = reconcile_outcomes[0]
        outcomes.append(reconcile_outcome)
        if must_stop:
            return {
                "outcomes": outcomes,
                "last_stage_outcome": reconcile_outcome,
                "route_target": END,
            }
        reconciled_key = reconcile_outcome.key
        reconciled_attempt = attempt_numbers.get(reconciled_key, 1)
        route_target, attempt_numbers, synthetic = resolve_transition(
            reconcile_outcome,
            store=store,
            stage_key=reconciled_key,
            attempt_number=reconciled_attempt,
            attempt_numbers=attempt_numbers,
            until=state.get("until"),
        )
        if synthetic is not None:
            outcomes.append(synthetic)
        return {
            "outcomes": outcomes,
            "last_stage_outcome": reconcile_outcome,
            "attempt_numbers": attempt_numbers,
            "route_target": route_target,
        }

    # Nada que reconciliar: entra directo a la etapa de entrada.
    return {
        "attempt_numbers": attempt_numbers,
        "route_target": entry_stage,
    }


def make_stage_node(spec: StageSpec) -> Callable[[GraphState], dict[str, Any]]:
    """Fábrica de nodo genérico para UNA etapa -- mismo cuerpo para las 7,
    parametrizado únicamente por su ``StageSpec``. Cero lógica de decisión:
    ejecuta la etapa (``run_stage``) y resuelve a dónde ir después
    (``resolve_transition``), en ese orden, y nada más."""

    stage_key = spec.key

    def _node(state: GraphState) -> dict[str, Any]:
        project_dir = state["project_dir"]
        store = ensure_pipeline_state(project_dir)
        attempt_numbers = dict(state.get("attempt_numbers") or {})
        attempt_number = attempt_numbers.get(stage_key, 1)

        # force_rerun se consume en la primera etapa que realmente corre
        # (nunca en reconcile_pending) y no se vuelve a aplicar después --
        # mismo comportamiento que `force_rerun_current` en run_pipeline.
        force_rerun = bool(state.get("force_rerun"))

        outcome = run_stage(
            store=store,
            project_dir=project_dir,
            spec=spec,
            attempt_number=attempt_number,
            observations=state.get("observations"),
            force_rerun=force_rerun,
        )

        route_target, attempt_numbers, synthetic = resolve_transition(
            outcome,
            store=store,
            stage_key=stage_key,
            attempt_number=attempt_number,
            attempt_numbers=attempt_numbers,
            until=state.get("until"),
        )

        outcomes = list(state.get("outcomes") or [])
        outcomes.append(outcome)
        if synthetic is not None:
            outcomes.append(synthetic)

        return {
            "attempt_numbers": attempt_numbers,
            "last_stage_outcome": outcome,
            "outcomes": outcomes,
            "route_target": route_target,
            "force_rerun": False,
        }

    _node.__name__ = f"node_{stage_key}"
    return _node
