"""Reconciliación de ``pending_execution`` cuando apunta a una etapa
DISTINTA de la que el motor de orquestación está a punto de intentar.

Extraído de ``stage_execution.py`` (Bloque 2 de la migración a LangGraph,
MAIN 5) sin cambiar una sola línea de lógica -- solo la ubicación, en su
propio módulo pequeño y verificable, tal como preveía el plan de
migración. En el Bloque 1 esta función ya había salido de
``pipeline_orchestrator.py`` (quedó dentro de ``stage_execution.py``); este
paso solo la separa en un archivo propio, sin volver a tocar
``pipeline_orchestrator.py`` más que para actualizar de dónde la importa.

Responsabilidad, sin cambios: ``StateStore.prepare_execution`` mantiene un
único slot GLOBAL de ``pending_execution`` (no uno por etapa). Si una
ejecución anterior de OTRA etapa quedó interrumpida antes de comprometer,
ese slot bloquea la preparación de CUALQUIER etapa -- incluida la que el
motor de orquestación está intentando ahora. Esta función resuelve esa
pending vía el protocolo OFICIAL de la etapa a la que realmente pertenece
(``run_stage()`` sobre su propio ``StageSpec``), sin leer ni escribir
``pending_execution`` directamente.

Invocable independientemente del bucle ``for`` de ninguna máquina de
estados concreta -- recibe todo lo que necesita como parámetros
(``registry``, ``current_stage``, ``attempt_numbers``), nunca variables de
un bucle externo. Es exactamente la pieza que un nodo de arranque de
LangGraph (``reconcile_pending``, ver el diseño de MAIN 5) invocará antes
de decidir por dónde entrar al grafo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.orchestration.stage_execution import StageOutcome, StageSpec, run_stage


def _reconcile_pending_execution_for_other_stage(
    *,
    store,
    project_dir: Path,
    registry: Mapping[str, "StageSpec"],
    current_stage: str,
    attempt_numbers: Mapping[str, int],
    observations: Mapping[str, Any] | None,
) -> tuple[list["StageOutcome"], bool]:
    """Si ``state.pending_execution`` existe y apunta a una etapa DISTINTA
    de ``current_stage``, la reconcilia vía el protocolo OFICIAL de esa
    otra etapa (``run_stage`` sobre su propio ``StageSpec``) antes de que
    se intente preparar ``current_stage``.

    Motivo: ``StateStore.prepare_execution`` mantiene un único slot GLOBAL
    de ``pending_execution`` (no uno por etapa) -- una ejecución
    interrumpida de OTRA etapa (ej. 07 crasheando antes de comprometer)
    deja ese slot ocupado y bloquea la preparación de CUALQUIER otra
    etapa, incluida la que el pipeline está intentando ahora, con
    ``RuntimeError("a pending execution already exists")``. Esta función
    nunca lee ni escribe ``pending_execution`` directamente -- delega
    por completo en ``run_stage()`` para la etapa a la que realmente
    pertenece, que ya sabe resolverla oficialmente (COMMIT del resultado
    persistido, liberar para reejecutar, o lo que corresponda según su
    propio protocolo -- para 07 esto enruta a
    ``_run_verification_stage``/``resume_agent07_execution``).

    Devuelve ``(outcomes_a_agregar, debe_detenerse)``: si
    ``debe_detenerse`` es ``True``, el llamador no debe intentar preparar
    ``current_stage`` en esta vuelta (o bien la pending sigue sin
    resolverse tras el intento oficial, o bien apunta a una etapa sin
    ``StageSpec`` registrado -- inconsistencia real que se reporta
    explícitamente, nunca se oculta ni se fuerza)."""

    state = store.load()
    pending = state.pending_execution
    if pending is None or pending.target_stage == current_stage:
        return [], False

    pending_stage_key = pending.target_stage
    if pending_stage_key not in registry:
        return (
            [
                StageOutcome(
                    key=current_stage,
                    label=registry[current_stage].label if current_stage in registry else current_stage,
                    status="FAILED",
                    execution_status=None,
                    quality_status=None,
                    warnings=(),
                    error={
                        "type": "PendingExecutionUnknownTargetStage",
                        "message": (
                            f"pending_execution.target_stage={pending_stage_key!r} "
                            "no está en el registro de etapas -- no se puede "
                            "reconciliar automáticamente."
                        ),
                    },
                    attempt_number=0,
                    next_action="HALT_STAGE",
                    target_stage=None,
                    reason_code="PENDING_EXECUTION_UNKNOWN_TARGET_STAGE",
                )
            ],
            True,
        )

    pending_spec = registry[pending_stage_key]
    reconcile_outcome = run_stage(
        store=store,
        project_dir=project_dir,
        spec=pending_spec,
        attempt_number=attempt_numbers.get(pending_stage_key, 1),
        observations=observations,
        force_rerun=False,
    )

    state = store.load()
    if state.pending_execution is not None:
        # El protocolo oficial de esa etapa no logró liberar la pending
        # (ej. sigue EXECUTED_NOT_COMMITTED esperando otra vuelta) -- no
        # se fuerza nada más. El llamador se detiene aquí en vez de
        # intentar preparar current_stage, que el store rechazaría de
        # nuevo con el mismo error.
        return [reconcile_outcome], True

    return [reconcile_outcome], False
