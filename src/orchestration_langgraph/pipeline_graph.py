# ============================================================
# ENSAMBLADO Y COMPILACIÓN DEL STATEGRAPH DEL PIPELINE
# Define la topología, nodos y transiciones del flujo principal
# usando LangGraph como único motor de orquestación.
# ============================================================

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.orchestration.decision_engine import CANONICAL_STAGE_ORDER
from src.orchestration.stage_execution import _stage_registry
from src.orchestration_langgraph.graph_state import GraphState
from src.orchestration_langgraph.nodes import make_stage_node, reconcile_pending_node
from src.orchestration_langgraph.routing import route_after_stage

# Construye el grafo principal de LangGraph, registra cada etapa como nodo
# y define las rutas condicionales que determinan cómo avanza el pipeline.
def build_pipeline_graph():
    registry = {
        spec.key: spec
        for spec in _stage_registry()
    }
    graph = StateGraph(GraphState) # Crea el StateGraph usando el estado compartido del pipeline.
    graph.add_node( # Añade primero el nodo encargado de ejecutar fases pendientes.
        "reconcile_pending",
        reconcile_pending_node,
    ) 
    for stage_key in CANONICAL_STAGE_ORDER:     # Registra cada etapa canónica del sistema como un nodo de LangGraph.
        graph.add_node(stage_key, 
                       make_stage_node(registry[stage_key]))
    graph.set_entry_point("reconcile_pending") # Define reconcile_pending como punto inicial del grafo.
    # Construye el mapa de destinos válidos: cualquier etapa canónica o el final del grafo.
    path_map = {
        stage_key: stage_key
        for stage_key in CANONICAL_STAGE_ORDER
    }
    path_map[END] = END
    # Después de reconciliar pendientes, decide dinámicamente cuál será el siguiente nodo.
    graph.add_conditional_edges(
        "reconcile_pending",
        route_after_stage,
        path_map,
    )
    # Después de cada etapa, vuelve a usar la misma función de routing
    # para decidir si avanza, retorna, reintenta o termina.    
    for stage_key in CANONICAL_STAGE_ORDER:
        graph.add_conditional_edges(
            stage_key,
            route_after_stage,
            path_map,
        )
    return graph # Devuelve el grafo completamente ensamblado.

# Construye el grafo y lo compila para dejarlo listo
# para ejecutar el pipeline mediante LangGraph.
def compile_pipeline_graph():
    """Grafo compilado, listo para ``.invoke(...)``."""
    return build_pipeline_graph().compile()

# Ejecuta el pipeline completo usando el grafo compilado de LangGraph,
# construye el estado inicial y devuelve los resultados de todas las etapas.
def run_pipeline_via_langgraph(
    project_dir: str,
    *,
    start_stage: str | None = None, #permite comenzar desde una etapa concreta
    until: str | None = None, #permite detener la ejecución en una etapa determinada
    attempt_numbers: dict[str, int] | None = None, #conserva cuántos intentos lleva cada etapa
    force_rerun: bool = False, #obliga a volver a ejecutar aunque existan resultados previos
    observations: dict | None = None, #permite pasar información adicional al estado del grafo
    recursion_limit: int = 50,
):

    app = compile_pipeline_graph()  # Compila el StateGraph para dejarlo listo para ejecutar.

    # Construye el estado inicial compartido por todos los nodos del grafo.
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

    # Ejecuta el grafo completo desde el estado inicial.
    final_state = app.invoke(
        initial_state, config={"recursion_limit": recursion_limit}
    )
    return final_state["outcomes"] # Devuelve los resultados acumulados durante la ejecución del pipeline.






























# Define los argumentos disponibles para ejecutar el pipeline desde la terminal,
# permitiendo elegir el proyecto, la etapa inicial, la etapa final y si se fuerza una reejecución.
def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", required=True, help="Ruta a PROJECT_DIR (contiene active_experiment.json)."
    )
    parser.add_argument(
        "--until",
        default=None,
        choices=CANONICAL_STAGE_ORDER,
        help="Detenerse tras completar esta etapa (por defecto corre hasta 08).",
    )
    parser.add_argument(
        "--start-stage",
        default=None,
        choices=CANONICAL_STAGE_ORDER,
        help=(
            "Empezar el recorrido directamente en esta etapa, en vez de "
            "CANONICAL_STAGE_ORDER[0] -- ejecuta ÚNICAMENTE esta etapa y las "
            "que resulten de sus transiciones reales (nunca las anteriores). "
            "Con start-stage explícito, el chequeo de estado ya-terminal se "
            "omite deliberadamente (se respeta la petición explícita del "
            "llamador, igual que --force-rerun) -- si la etapa ya está "
            "COMPLETED y vigente (fingerprints sin cambios), sigue "
            "reconociéndose SKIPPED_FRESH con normalidad; si su último "
            "commit fue FAILED (ej. HALT_STAGE), esto la reintenta con un "
            "decision_id nuevo, SIN --force-rerun y sin tocar ninguna etapa "
            "previa."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Reejecuta la etapa inicial aunque ya esté COMPLETED y vigente en pipeline_state.json.",
    )
    return parser.parse_args(argv)

# Ejecuta el pipeline desde la terminal, muestra el resultado de cada etapa
# y devuelve un código de salida según si la ejecución terminó correctamente.
def main(argv=None) -> int:
    # Lee los argumentos enviados desde la línea de comandos.
    args = _parse_args(argv)
    # Ejecuta el pipeline mediante LangGraph con la configuración indicada.
    outcomes = run_pipeline_via_langgraph(
        args.project_dir,
        start_stage=args.start_stage,
        until=args.until,
        force_rerun=args.force_rerun,
    )
    
    # Recorre los resultados de cada etapa ejecutada.
    for outcome in outcomes:
        print(
            f"[{outcome.status:24s}] {outcome.label:45s} "
            f"execution={outcome.execution_status} quality={outcome.quality_status} "
            f"next={outcome.next_action}->{outcome.target_stage}"
        )
        for warning in outcome.warnings:
            print(f"    warning: {warning}")
        if outcome.error:
            print(f"    error: {outcome.error}")
    # Devuelve 0 si ninguna etapa falló y 1 si ocurrió un fallo o se alcanzó una etapa no registrada.
    return 0 if all(o.status not in {"FAILED", "REACHED_UNREGISTERED_STAGE"} for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
