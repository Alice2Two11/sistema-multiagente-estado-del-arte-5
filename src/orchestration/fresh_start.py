# ============================================================
# FRESH START -- reinicio explícito y auditable del estado de un
# experimento, para usar cuando se aplicó un cambio de código real
# (un parche, un fix) y hace falta garantizar que NINGÚN resultado
# calculado con la versión anterior del código se reutilice por error.
#
# Por qué existe: build_current_extraction_signature() (y las firmas
# equivalentes de las demás etapas) hashean los DATOS de entrada y
# strings de versión mantenidos a mano -- nunca el código fuente en
# sí. Si se corrige un bug real sin subir esos strings de versión (algo
# fácil de olvidar), la huella no cambia, y SKIPPED_FRESH puede reusar
# un resultado calculado con el bug todavía presente. fresh_start()
# no resuelve esa causa de fondo (para eso habría que fechar cada
# firma contra un hash real del código, un cambio mayor); resuelve el
# síntoma de forma explícita y controlada: te obliga a decidir "quiero
# que TODO se recalcule ahora", en vez de confiar en que la huella lo
# detecte sola.
#
# Regla de uso: correlo cada vez que apliques un parche de código
# (tuyo o de un fix) ANTES de la siguiguiente corrida de ese
# experimento, y también cada vez que una corrida se haya interrumpido
# de forma no limpia (proceso matado, Colab desconectado a medias).
# ============================================================

from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from src.orchestration.stage_execution import resolve_state_path
from src.state.pipeline_state import CycleState
from src.state.state_store import StateStore

WRITER_VERIFIER_CYCLE_NAME = "writer_verifier"
CYCLE_ROUNDS_DIRECTORY_NAME = "writer_verifier_cycle"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FreshStartReport(NamedTuple):
    """Resumen de lo que fresh_start() encontró y reinició, para poder
    imprimirlo o loguearlo -- nunca se aplica en silencio."""

    stages_reset: tuple[str, ...]
    cycle_was_active: bool
    cycle_rounds_used_before: int
    pending_execution_cleared: bool
    cycle_rounds_dir_backed_up_to: str | None


def perform_fresh_start(project_dir: str | Path) -> FreshStartReport:
    """Reinicia, de forma explícita y auditable, todo lo que podría
    hacer que una corrida nueva reutilice en silencio trabajo calculado
    con una versión de código anterior:

    1. attempts_used = 0 en TODAS las etapas (CANONICAL_STAGE_ORDER).
    2. El ciclo writer_verifier (06<->07) vuelve a NOT_STARTED, con
       rounds_used = 0.
    3. pending_execution se limpia (nunca se asume una ejecución
       colgada como válida).
    4. Si existen carpetas de rondas ya persistidas en disco
       (05_outputs/writer_verifier_cycle/round_NN/), se MUEVEN a un
       backup con timestamp -- nunca se borran -- para que
       cycle_round_persistence.py no rechace escribir una ronda nueva
       por encontrar la ronda anterior todavía en su lugar.

    Devuelve un FreshStartReport con el detalle de lo que se hizo, para
    que el caller lo imprima explícitamente (nunca queda un reset
    silencioso sin registro).
    """

    state_path, experiment_id, run_id = resolve_state_path(project_dir)
    store = StateStore(state_path)
    state = store.load()

    # 1) attempts_used = 0 en todas las etapas ya registradas en el estado.
    stages_reset = []
    new_stages = dict(state.stages)
    for stage_key, stage in new_stages.items():
        if stage.attempts_used != 0:
            stages_reset.append(stage_key)
        new_stages[stage_key] = replace(stage, attempts_used=0)

    # 2) el ciclo writer_verifier vuelve a su estado inicial.
    new_cycles = dict(state.cycles)
    existing_cycle = new_cycles.get(WRITER_VERIFIER_CYCLE_NAME)
    cycle_was_active = bool(existing_cycle and existing_cycle.status == "ACTIVE")
    cycle_rounds_used_before = existing_cycle.rounds_used if existing_cycle else 0
    new_cycles[WRITER_VERIFIER_CYCLE_NAME] = CycleState(
        rounds_used=0,
        max_rounds=existing_cycle.max_rounds if existing_cycle else 3,
        status="NOT_STARTED",
        last_return_reason=None,
        unresolved_claims_count=0,
    )

    # 3) ninguna ejecución pendiente colgada.
    pending_execution_cleared = state.pending_execution is not None

    new_state = replace(
        state,
        stages=new_stages,
        cycles=new_cycles,
        pending_execution=None,
    )
    store.save(new_state)

    # 4) mueve a un lado (nunca borra) las rondas físicas del ciclo, si existen.
    experiment_dir = state_path.parents[2]  # .../<experiment_id>/05_outputs/00_orchestrator_planner/pipeline_state.json
    cycle_rounds_dir = experiment_dir / "05_outputs" / CYCLE_ROUNDS_DIRECTORY_NAME
    backed_up_to = None
    if cycle_rounds_dir.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = cycle_rounds_dir.parent / f"{CYCLE_ROUNDS_DIRECTORY_NAME}_BACKUP_{ts}"
        shutil.move(str(cycle_rounds_dir), str(backup_dir))
        backed_up_to = str(backup_dir)

    return FreshStartReport(
        stages_reset=tuple(stages_reset),
        cycle_was_active=cycle_was_active,
        cycle_rounds_used_before=cycle_rounds_used_before,
        pending_execution_cleared=pending_execution_cleared,
        cycle_rounds_dir_backed_up_to=backed_up_to,
    )


def print_fresh_start_report(report: FreshStartReport) -> None:
    print("=" * 70)
    print("FRESH START")
    print("=" * 70)
    if report.stages_reset:
        print(f"attempts_used reiniciado en {len(report.stages_reset)} etapa(s): {', '.join(report.stages_reset)}")
    else:
        print("attempts_used ya estaba en 0 en todas las etapas -- nada que reiniciar ahí.")
    if report.cycle_was_active:
        print(f"Ciclo writer_verifier: estaba ACTIVE con {report.cycle_rounds_used_before} ronda(s) usada(s) -- reiniciado a NOT_STARTED.")
    else:
        print("Ciclo writer_verifier: no estaba ACTIVE -- reiniciado igual, por seguridad.")
    if report.pending_execution_cleared:
        print("pending_execution: había una ejecución pendiente registrada -- se limpió.")
    else:
        print("pending_execution: no había ninguna pendiente.")
    if report.cycle_rounds_dir_backed_up_to:
        print(f"writer_verifier_cycle/ (rondas físicas en disco): movida a {report.cycle_rounds_dir_backed_up_to}")
    else:
        print("writer_verifier_cycle/: no existía en disco -- nada que mover.")
    print("=" * 70)
