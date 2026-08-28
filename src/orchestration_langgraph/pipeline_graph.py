"""Ensamblado y compilación del ``StateGraph`` de MAIN 5 -- único motor de
orquestación del sistema desde el Bloque 6.

Construye la topología del grafo a partir de
``decision_engine.CANONICAL_STAGE_ORDER`` -- la MISMA fuente única de
verdad del orden de etapas que usaba la máquina de estados propia antes de
retirarse (``pipeline_orchestrator.py``, eliminado en el Bloque 6). Este
módulo nunca importó ni importa ese archivo.
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
    """Punto de entrada de orquestación global del sistema, corriendo
    sobre el grafo compilado. ``start_stage`` tiene aquí el nombre
    ``entry_stage`` dentro del estado, por consistencia con el resto de
    ``GraphState``."""

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


# ---------------------------------------------------------------------------
# CLI para uso directo en Colab: `python -m src.orchestration_langgraph.pipeline_graph`
#
# Bloque 5 (MAIN 5): mismo contrato exacto que tenía la CLI de la máquina
# de estados propia antes de retirarse (``pipeline_orchestrator.py``,
# eliminado en el Bloque 6) -- mismos flags, misma semántica de exit code
# -- para que cualquier invocador externo (scripts/run_pipeline.py,
# Corrida_03_a_08.ipynb) pueda apuntar aquí sin cambiar nada más que el
# nombre del módulo.
# ---------------------------------------------------------------------------


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


def main(argv=None) -> int:
    args = _parse_args(argv)
    outcomes = run_pipeline_via_langgraph(
        args.project_dir,
        start_stage=args.start_stage,
        until=args.until,
        force_rerun=args.force_rerun,
    )
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
    return 0 if all(o.status not in {"FAILED", "REACHED_UNREGISTERED_STAGE"} for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
