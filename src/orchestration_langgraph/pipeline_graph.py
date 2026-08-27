"""Ensamblado y compilación del ``StateGraph`` de MAIN 5.

Construye la topología del grafo a partir de
``decision_engine.CANONICAL_STAGE_ORDER`` -- la MISMA fuente única de
verdad del orden de etapas que usa la máquina de estados propia (nunca la
duplica a mano). No importa nada de
``src.orchestration.pipeline_orchestrator`` -- ni el módulo, ni ninguno de
sus símbolos, directa ni indirectamente.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.orchestration.decision_engine import CANONICAL_STAGE_ORDER
from src.orchestration.stage_execution import _stage_registry
from src.orchestration_langgraph.graph_state import GraphState
from src.orchestration_langgraph.nodes import make_stage_node, reconcile_pending_node
from src.orchestration_langgraph.routing import route_after_stage


def build_pipeline_graph():
    """Construye (sin compilar) el ``StateGraph`` completo: nodo de
    reconciliación de arranque + un nodo por cada etapa de
    ``CANONICAL_STAGE_ORDER``, con edges condicionales que replican
    ADVANCE/RETRY/RETURN/HALT_STAGE/STOP_PIPELINE vía
    ``routing.route_after_stage``."""

    registry = {spec.key: spec for spec in _stage_registry()}

    graph = StateGraph(GraphState)
    graph.add_node("reconcile_pending", reconcile_pending_node)
    for stage_key in CANONICAL_STAGE_ORDER:
        graph.add_node(stage_key, make_stage_node(registry[stage_key]))

    graph.set_entry_point("reconcile_pending")

    # Cualquier nodo (incluido reconcile_pending) puede rutear a
    # cualquier etapa del orden canónico (ADVANCE normal, RETRY como
    # self-loop, o RETURN hacia una etapa anterior -- hoy solo 07->06) o
    # a END (HALT_STAGE/STOP_PIPELINE/pipeline completo/`until`
    # alcanzado). El path_map es el mismo para todos los nodos: no hay
    # restricción de aristas más allá de la que ya impone
    # decision_engine.validate_transition dentro de cada StageOutcome.
    path_map = {stage_key: stage_key for stage_key in CANONICAL_STAGE_ORDER}
    path_map[END] = END

    graph.add_conditional_edges("reconcile_pending", route_after_stage, path_map)
    for stage_key in CANONICAL_STAGE_ORDER:
        graph.add_conditional_edges(stage_key, route_after_stage, path_map)

    return graph


def compile_pipeline_graph():
    """Grafo compilado, listo para ``.invoke(...)``."""

    return build_pipeline_graph().compile()


def run_pipeline_via_langgraph(
    project_dir: str,
    *,
    start_stage: str | None = None,
    until: str | None = None,
    attempt_numbers: dict[str, int] | None = None,
    force_rerun: bool = False,
    observations: dict | None = None,
    recursion_limit: int = 50,
):
    """Punto de entrada equivalente a
    ``pipeline_orchestrator.run_pipeline`` pero corriendo sobre el grafo
    compilado. Misma firma de argumentos relevantes (``start_stage``
    tiene aquí el nombre ``entry_stage`` dentro del estado, por
    consistencia con el resto de ``GraphState``)."""

    app = compile_pipeline_graph()
    initial_state: GraphState = {
        "project_dir": str(project_dir),
        "entry_stage": start_stage,
        "until": until,
        "force_rerun": force_rerun,
        "attempt_numbers": dict(attempt_numbers or {}),
        "last_stage_outcome": None,
        "outcomes": [],
        "observations": observations,
    }
    final_state = app.invoke(
        initial_state, config={"recursion_limit": recursion_limit}
    )
    return final_state["outcomes"]
