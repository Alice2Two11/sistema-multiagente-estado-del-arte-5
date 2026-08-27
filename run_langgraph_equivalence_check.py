#!/usr/bin/env python3
"""Equivalencia REAL entre ``pipeline_orchestrator.py`` (motor propio) y
``orchestration_langgraph`` (LangGraph), contra un experimento real tuyo.

El equivalente en Colab del test estructural con dobles
(``tests/orchestration_langgraph/test_equivalence_pipeline_orchestrator.py``)
-- pero corriendo contra tus artefactos reales, no un guion simulado.

Dos modos:

--mode=skipped-fresh (por defecto, estrictamente fail-closed, sin escape)
    ANTES de invocar cualquiera de los dos motores, comprueba de forma
    determinista si TODAS las etapas relevantes ya están ``COMPLETED``
    con fingerprints vigentes y no hay ninguna ``pending_execution``. El
    preflight SÍ construye runtimes/carga dependencias (``spec.
    build_execution``: conexión a Chroma, credenciales) porque eso es lo
    que necesita para comparar fingerprints -- pero NUNCA llama a
    ``runtime_transaction``/``agent.execute()``, así que no produce
    ninguna llamada nueva de generación a un LLM. Si el preflight no
    puede garantizar que todo se reutilizará, ABORTA sin excepción --
    este modo no tiene forma de forzar ejecución real. Solo si el
    preflight confirma que ambos motores únicamente van a reutilizar
    resultados existentes, procede a correr los dos SOBRE EL MISMO
    ``project_dir`` y comparar -- eso es seguro específicamente porque
    SKIPPED_FRESH no escribe nada nuevo en ``pipeline_state.json``, así
    que el primer motor no modifica lo que el segundo va a ver.

--mode=fresh-execution (la ÚNICA forma de ejecutar algo real con este script)
    Requiere DOS project_dir separados con el MISMO corpus pero
    ``active_experiment_id`` distintos (cópialos tú mismo antes de correr
    esto) -- nunca el mismo directorio para los dos motores: si
    corrieras ambos sobre el mismo ``project_dir`` con ejecución real,
    el primer motor comprometería resultados que el segundo vería como
    ya hechos (SKIPPED_FRESH), invalidando la comparación. Corre
    pipeline_orchestrator sobre uno, LangGraph sobre el otro, ambos
    DESDE CERO, y compara. Consume el doble de llamadas reales.

Uso:

    python scripts/run_langgraph_equivalence_check.py \
        --project-dir /content/proyecto_estado_arte \
        --mode skipped-fresh

    python scripts/run_langgraph_equivalence_check.py \
        --project-dir-a /content/proyecto_estado_arte_a \
        --project-dir-b /content/proyecto_estado_arte_b \
        --mode fresh-execution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _normalize(outcomes) -> list[tuple]:
    return [
        (o.key, o.status, o.next_action, o.target_stage, o.reason_code, o.execution_status, o.quality_status)
        for o in outcomes
    ]


def _print_comparison(label_a: str, outcomes_a, label_b: str, outcomes_b) -> bool:
    norm_a = _normalize(outcomes_a)
    norm_b = _normalize(outcomes_b)
    print(f"\n=== {label_a}: {len(outcomes_a)} outcomes ===")
    for o in outcomes_a:
        print(f"  [{o.status:12s}] {o.key:32s} next={o.next_action}->{o.target_stage} ({o.reason_code})")
    print(f"\n=== {label_b}: {len(outcomes_b)} outcomes ===")
    for o in outcomes_b:
        print(f"  [{o.status:12s}] {o.key:32s} next={o.next_action}->{o.target_stage} ({o.reason_code})")

    matches = norm_a == norm_b
    print(f"\n{'✅ EQUIVALENTES' if matches else '❌ DIFIEREN'} estructuralmente.")
    if not matches:
        for i, (a, b) in enumerate(zip(norm_a, norm_b)):
            if a != b:
                print(f"  primera diferencia en outcome[{i}]:")
                print(f"    {label_a}: {a}")
                print(f"    {label_b}: {b}")
                break
        if len(norm_a) != len(norm_b):
            print(f"  además, longitudes distintas: {len(norm_a)} vs {len(norm_b)}")
    return matches


def _preflight_all_stages_would_be_fresh(
    project_dir: str, *, until: str | None = None
) -> tuple[bool, list[str], bool]:
    """Determina, SIN llamar nunca a ``runtime_transaction``/``agent.
    execute()`` (es decir, sin producir ninguna llamada nueva de
    generación a un LLM), si todas las etapas relevantes ya están
    ``COMPLETED`` con fingerprints vigentes -- es decir, si los dos
    motores solo van a producir ``SKIPPED_FRESH``.

    Usa exactamente los mismos bloques que ``run_stage()`` (``spec.
    build_execution`` + ``spec.build_fingerprints`` + ``is_stage_fresh``)
    para decidir vigencia -- nunca reimplementa esa regla. ``build_execution``
    SÍ construye el agente/AgentInput real (conexión a Chroma, resolución
    de credenciales) porque eso es lo que necesita para poder comparar
    fingerprints -- no es una función "sin tocar nada". Lo que garantiza
    esta función es más específico y más importante: nunca invoca al
    LLM, porque eso ocurre exclusivamente dentro de
    ``runtime_transaction``/``agent.execute()``, que esta función jamás
    llama.

    Devuelve ``(todas_frescas, etapas_problematicas, hay_pending_execution)``.
    """
    from src.orchestration.decision_engine import CANONICAL_STAGE_ORDER, is_stage_fresh
    from src.orchestration.stage_execution import _stage_registry, ensure_pipeline_state
    from src.state.pipeline_state import ExecutionStatus

    store = ensure_pipeline_state(project_dir)
    state = store.load()
    registry = {spec.key: spec for spec in _stage_registry()}

    stages_to_check = CANONICAL_STAGE_ORDER
    if until is not None:
        idx = CANONICAL_STAGE_ORDER.index(until)
        stages_to_check = CANONICAL_STAGE_ORDER[: idx + 1]

    problematic: list[str] = []
    for stage_key in stages_to_check:
        spec = registry.get(stage_key)
        if spec is None:
            problematic.append(f"{stage_key} (sin StageSpec registrado)")
            continue

        committed = state.stages.get(stage_key)
        if committed is None or committed.execution_status != ExecutionStatus.COMPLETED:
            problematic.append(f"{stage_key} (no está COMPLETED todavía)")
            continue

        if spec.build_fingerprints is None:
            # Sin forma de comparar vigencia -- run_stage() la trataría
            # igual como SKIPPED_FRESH (chequeo antiguo, solo COMPLETED),
            # así que bajo esa MISMA regla cuenta como fresca aquí.
            continue

        try:
            _agent, agent_input = spec.build_execution(project_dir, committed.attempts_used or 1)
            current_fingerprints = spec.build_fingerprints(agent_input)
        except Exception as exc:  # noqa: BLE001
            problematic.append(f"{stage_key} (no se pudo reconstruir AgentInput para comparar: {exc})")
            continue

        if not is_stage_fresh(committed, current_fingerprints):
            problematic.append(f"{stage_key} (fingerprint desactualizado)")

    has_pending = state.pending_execution is not None
    return (len(problematic) == 0 and not has_pending), problematic, has_pending


def run_skipped_fresh_mode(project_dir: str) -> bool:
    """Estrictamente fail-closed: no existe ninguna forma de forzar
    ejecución real a través de este modo. Si el preflight no garantiza
    que ambos motores solo van a reutilizar resultados existentes,
    aborta -- para ejecutar algo real, usa exclusivamente
    ``--mode=fresh-execution``."""

    print("Pre-flight fail-closed: comprobando vigencia SIN ejecutar nada...")
    all_fresh, problematic, has_pending = _preflight_all_stages_would_be_fresh(project_dir)
    if has_pending:
        print(
            "\n❌ ABORTADO: existe una pending_execution en este experimento -- "
            "no se puede garantizar que ambos motores solo reutilicen resultados "
            "existentes sin ejecutar nada nuevo. Resuélvela primero (corriendo "
            "el motor propio normalmente sobre este project_dir hasta que quede "
            "resuelta) antes de volver a intentar esta comparación."
        )
        return False
    if not all_fresh:
        print(
            "\n❌ ABORTADO (fail-closed): estas etapas NO están garantizadas como "
            "frescas:"
        )
        for item in problematic:
            print(f"    - {item}")
        print(
            "\nEste modo NUNCA ejecuta agentes ni realiza nuevas llamadas de "
            "generación al LLM -- no existe un flag para forzarlo. Si necesitas "
            "comparar los dos motores ejecutando de verdad, usa "
            "--mode=fresh-execution con dos project_dir independientes (nunca el "
            "mismo directorio para los dos motores)."
        )
        return False
    print("✅ Pre-flight OK: todas las etapas relevantes están COMPLETED y vigentes.\n")

    from src.orchestration.pipeline_orchestrator import run_pipeline
    from src.orchestration_langgraph.pipeline_graph import run_pipeline_via_langgraph

    # Seguro correr los dos motores sobre el MISMO project_dir aquí, y
    # solo aquí: el preflight ya confirmó que ninguno de los dos va a
    # escribir nada nuevo en pipeline_state.json (SKIPPED_FRESH no
    # comete ninguna transacción) -- el primero no modifica lo que el
    # segundo va a ver.
    print(f"Corriendo motor propio sobre {project_dir}...")
    outcomes_a = run_pipeline(project_dir)

    print(f"\nCorriendo LangGraph sobre el MISMO {project_dir}...")
    outcomes_b = run_pipeline_via_langgraph(project_dir)

    matches = _print_comparison("motor propio", outcomes_a, "LangGraph", outcomes_b)

    non_fresh = [o for o in outcomes_b if o.status not in ("SKIPPED_FRESH", "ALREADY_TERMINAL")]
    if non_fresh:
        print(
            "\n⚠️  AVISO: LangGraph ejecutó algo que NO fue SKIPPED_FRESH/ALREADY_TERMINAL "
            f"({[o.key for o in non_fresh]}) -- ocurrió una ejecución no prevista pese al "
            "pre-flight, lo que puede implicar nueva ejecución (posiblemente llamadas al "
            "LLM). Repórtalo: es una discrepancia real entre el chequeo de este script y "
            "run_stage() que hay que investigar antes de confiar en este modo de nuevo."
        )
    return matches


def run_fresh_execution_mode(project_dir_a: str, project_dir_b: str) -> bool:
    from src.orchestration.pipeline_orchestrator import run_pipeline
    from src.orchestration_langgraph.pipeline_graph import run_pipeline_via_langgraph

    print(f"Corriendo motor propio DESDE CERO sobre {project_dir_a}...")
    outcomes_a = run_pipeline(project_dir_a, force_rerun=True)

    print(f"\nCorriendo LangGraph DESDE CERO sobre {project_dir_b}...")
    outcomes_b = run_pipeline_via_langgraph(project_dir_b, force_rerun=True)

    return _print_comparison("motor propio", outcomes_a, "LangGraph", outcomes_b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["skipped-fresh", "fresh-execution"], default="skipped-fresh")
    parser.add_argument("--project-dir", help="Requerido para --mode=skipped-fresh.")
    parser.add_argument("--project-dir-a", help="Requerido para --mode=fresh-execution (motor propio).")
    parser.add_argument("--project-dir-b", help="Requerido para --mode=fresh-execution (LangGraph).")
    args = parser.parse_args()

    if args.mode == "skipped-fresh":
        if not args.project_dir:
            parser.error("--mode=skipped-fresh requiere --project-dir")
        matches = run_skipped_fresh_mode(args.project_dir)
    else:
        if not args.project_dir_a or not args.project_dir_b:
            parser.error("--mode=fresh-execution requiere --project-dir-a y --project-dir-b")
        matches = run_fresh_execution_mode(args.project_dir_a, args.project_dir_b)

    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
