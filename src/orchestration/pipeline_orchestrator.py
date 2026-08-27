"""Máquina de estados propia que orquesta las etapas 03-08 usando
exclusivamente los componentes de ``src/``.

Bloque 1 de la migración a LangGraph (MAIN 5): la infraestructura de
ejecución de UNA etapa (registro de ``StageSpec``, construcción de agente+
``AgentInput`` por etapa, resolución de ``pending_execution``, chequeo de
fingerprints/``SKIPPED_FRESH``, construcción de ``StageOutcome``) ya NO vive
en este archivo -- se extrajo a ``stage_execution.py``/
``stage_constructors.py`` (neutrales respecto al motor de orquestación) sin
cambiar una sola línea de su lógica. Este módulo importa esas piezas y
conserva ÚNICAMENTE lo que es específico de esta máquina de estados: el
bucle ``run_pipeline`` con su variable ``current_stage``, la función
``_apply_stage_transition`` que pega ``decision_engine`` con ese bucle, el
logging de consola, y la CLI.

Diseño (sin cambios respecto a antes de la extracción)
------
Cada etapa migrada ya expone dos piezas reutilizables:

1. Un constructor ``build_real_<etapa>_execution(project_dir, attempt_number)``
   (en ``src/adapters/*_runtime.py``) que arma el agente/capability real y su
   ``AgentInput`` a partir de ``active_experiment.json`` y los artefactos ya
   generados en el proyecto.
2. Un protocolo transaccional (en ``src/runtime/*_protocol.py``) que envuelve
   PREPARE → EXECUTE → persist → COMMIT sobre ``StateStore``, convirtiendo
   cualquier fallo de preparación (dependencia faltante, credencial ausente,
   etc.) en un ``AgentResult`` ``FAILED`` comprometido igualmente al estado,
   en vez de dejar una excepción sin registrar.

``run_pipeline`` interpreta ``RequestedTransition`` en vez de limitarse a
recorrer ``STAGE_ORDER`` en orden fijo. La semántica de decisión
(ADVANCE/RETRY/RETURN/HALT_STAGE/STOP_PIPELINE, vigencia por fingerprints,
invalidación en cascada) vive en ``decision_engine.py``; este módulo solo la
conecta con la ejecución real de cada etapa:

- reutiliza el ``pipeline_state.json`` canónico del experimento activo;
- si una etapa ya quedó ``COMPLETED`` y sus fingerprints siguen vigentes, la
  salta (``SKIPPED_FRESH``); si los datos de entrada cambiaron, la reejecuta
  aunque no se haya pedido ``force_rerun``;
- si existe una ejecución PENDING de un run anterior, la resuelve (COMMITTED
  o REEXECUTE) antes de continuar;
- sigue la transición que cada etapa solicitó (validada primero: ver
  ``decision_engine.validate_transition``), no un orden fijo.

No orquesta notebooks ni depende de ellos: sólo importa símbolos de ``src/``.

Todas las 7 etapas (03 a 08) tienen ``StageSpec`` ejecutable hoy -- ver
``src.orchestration.stage_execution._stage_registry``.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

from src.orchestration import decision_engine as de
from src.orchestration.decision_engine import (
    CANONICAL_STAGE_ORDER,
    apply_return_with_cycle,
    resolve_cycle_if_active,
)
from src.orchestration.stage_execution import (
    DRAFT_STAGE_NAME,
    StageOutcome,
    StageSpec,
    _check_already_terminal_state,
    _outcome_from_committed_stage,
    _outcome_from_result,
    _stage_registry,
    ensure_pipeline_state,
    load_active_experiment,
    resolve_state_path,
    run_stage,
)
from src.orchestration.pending_reconciliation import (
    _reconcile_pending_execution_for_other_stage,
)

# Re-exportados para compatibilidad temporal con cualquier import directo
# preexistente de `pipeline_orchestrator` (tests, notebooks) -- su
# definición real ya vive en stage_constructors.py. Se retiran junto con
# el resto de este archivo al final de la migración (Bloque 6).
from src.orchestration.stage_constructors import (  # noqa: F401
    _draft_runtime_transaction,
    _experimental_evaluation_execution,
    _experimental_verification_execution,
    _quantitative_runtime_transaction,
    _real_draft_execution,
    _real_extraction_execution,
    _real_outline_execution,
    _real_quantitative_execution,
    _real_thematic_execution,
    _resolve_draft_execution_mode,
    _run_evaluation_stage,
    _run_verification_stage,
)

# Etapas con StageSpec ejecutable hoy.
STAGE_ORDER: tuple[str, ...] = (
    "03_agente_extraccion_kb",
    "03B_extraccion_cuantitativa_kb",
    "04_agente_analisis_tematico",
    "05_generador_esquema",
    DRAFT_STAGE_NAME,
    "07_agente_verificador",
    "08_evaluacion_experimental",
)


# ---------------------------------------------------------------------------
# Interpretación de RequestedTransition dentro de esta máquina de estados
# ---------------------------------------------------------------------------


def _apply_stage_transition(
    outcome: "StageOutcome",
    *,
    store,
    stage_key: str,
    attempt_number: int,
    attempt_numbers: dict[str, int],
    until: str | None,
    outcomes: list["StageOutcome"],
) -> tuple[str | None, bool]:
    """Interpreta ``outcome.next_action`` con la MISMA semántica que
    gobierna el bucle principal de ``run_pipeline`` (ADVANCE con
    resolución de ciclo, RETRY, RETURN con ``apply_return_with_cycle`` y
    posible agotamiento del ciclo, o HALT_STAGE/STOP_PIPELINE) --
    factorizada para poder aplicarse tanto a la etapa que el bucle está
    procesando en su vuelta normal como a una etapa reconciliada fuera de
    orden (ver ``_reconcile_pending_execution_for_other_stage``): antes
    de esta función, la transición de una etapa reconciliada (ej. 07,
    resuelta porque tenía una ``pending_execution`` vieja) se ignoraba
    por completo -- el bucle seguía su recorrido normal desde
    ``current_stage`` sin importar si la reconciliación había producido
    HALT_STAGE, RETURN o ADVANCE, lo que permitía llegar a intentar una
    etapa posterior (06) que ya no correspondía tocar.

    Devuelve ``(nuevo_current_stage_o_None, debe_detenerse)`` -- si
    ``debe_detenerse`` es ``True``, el llamador debe terminar el bucle
    (pipeline completo, HALT_STAGE/STOP_PIPELINE, o ciclo agotado -- en
    este último caso ya se agregó el ``StageOutcome`` de
    ``CYCLE_EXHAUSTED`` a ``outcomes`` antes de devolver)."""

    if until is not None and stage_key == until:
        return None, True

    if outcome.next_action == "ADVANCE":
        if outcome.target_stage is None:
            return None, True  # pipeline completo
        if stage_key == de.WRITER_VERIFIER_TRIGGER_STAGE:
            resolve_cycle_if_active(store)
        return outcome.target_stage, False

    if outcome.next_action == "RETRY":
        attempt_numbers[stage_key] = attempt_number + 1
        return stage_key, False

    if outcome.next_action == "RETURN":
        cycle_result = apply_return_with_cycle(
            store,
            from_stage=stage_key,
            target_stage=outcome.target_stage,
            reason=f"INVALIDATED_BY_RETURN_FROM_{stage_key}",
        )
        if cycle_result.cycle_exhausted:
            outcomes.append(
                StageOutcome(
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
            )
            return None, True
        for stage_key_to_clear in CANONICAL_STAGE_ORDER[
            CANONICAL_STAGE_ORDER.index(outcome.target_stage) :
        ]:
            attempt_numbers.pop(stage_key_to_clear, None)
        return outcome.target_stage, False

    # HALT_STAGE o STOP_PIPELINE: se detiene el bucle.
    return None, True


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------


def run_pipeline(
    project_dir: str | Any,
    *,
    start_stage: str | None = None,
    until: str | None = None,
    attempt_numbers: Mapping[str, int] | None = None,
    force_rerun: bool = False,
    max_iterations: int = 50,
    observations: Mapping[str, Any] | None = None,
) -> list[StageOutcome]:
    """Corre el pipeline interpretando las transiciones solicitadas por cada etapa.

    Bucle guiado por ``RequestedTransition`` validado con
    ``decision_engine.validate_transition``:

    - ``ADVANCE`` → sigue a la etapa objetivo (por defecto la siguiente).
    - ``RETRY`` → reintenta la misma etapa (respetando el límite de intentos).
    - ``RETURN`` → invalida la etapa objetivo y todas las posteriores
      (``decision_engine.invalidate_from``) y continúa desde ahí.
    - ``HALT_STAGE`` / ``STOP_PIPELINE`` → detiene el bucle.

    ``until``: si se da, se detiene apenas la etapa con esa clave produce un
    resultado (antes de avanzar a la siguiente), incluso si el resultado
    pedía ADVANCE.

    ``force_rerun`` sólo se aplica a ``start_stage`` (o a la primera etapa si
    no se indica); las etapas alcanzadas después por ADVANCE/RETRY/RETURN se
    evalúan normalmente (con su propio chequeo de fingerprints).
    """

    attempt_numbers = dict(attempt_numbers or {})
    store = ensure_pipeline_state(project_dir)
    registry = {spec.key: spec for spec in _stage_registry()}

    if until is not None and until not in STAGE_ORDER:
        raise ValueError(f"Etapa desconocida en 'until': {until}")

    current_stage = start_stage or STAGE_ORDER[0]
    outcomes: list[StageOutcome] = []
    force_rerun_current = force_rerun

    terminal_outcome = _check_already_terminal_state(
        store=store, registry=registry, start_stage=start_stage, force_rerun=force_rerun
    )
    if terminal_outcome is not None:
        _print_outcome(terminal_outcome)
        return [terminal_outcome]

    for _ in range(max_iterations):
        if current_stage not in registry:
            outcomes.append(
                StageOutcome(
                    key=current_stage,
                    label=f"(sin StageSpec ejecutable todavía: {current_stage})",
                    status="REACHED_UNREGISTERED_STAGE",
                    execution_status=None,
                    quality_status=None,
                    warnings=(),
                    error=None,
                    attempt_number=0,
                    next_action="STOP_PIPELINE",
                    target_stage=None,
                    reason_code="STAGE_NOT_REGISTERED",
                )
            )
            break

        spec = registry[current_stage]

        reconcile_outcomes, must_stop = _reconcile_pending_execution_for_other_stage(
            store=store, project_dir=project_dir, registry=registry, current_stage=current_stage,
            attempt_numbers=attempt_numbers, observations=observations,
        )
        if reconcile_outcomes:
            reconcile_outcome = reconcile_outcomes[0]
            outcomes.append(reconcile_outcome)
            _print_outcome(reconcile_outcome)
            if must_stop:
                # Inconsistencia real (etapa de la pending sin StageSpec
                # registrado) o la pending sigue sin resolverse tras el
                # intento oficial -- no hay transición válida que
                # despachar; se detiene aquí, igual que antes.
                break
            # La transición REAL de la etapa reconciliada (HALT_STAGE,
            # RETURN o ADVANCE) gobierna el flujo principal a partir de
            # aquí -- nunca se ignora para seguir el recorrido normal
            # desde current_stage.
            reconciled_stage_key = reconcile_outcome.key
            reconciled_attempt_number = attempt_numbers.get(reconciled_stage_key, 1)
            new_stage, should_stop = _apply_stage_transition(
                reconcile_outcome, store=store, stage_key=reconciled_stage_key,
                attempt_number=reconciled_attempt_number, attempt_numbers=attempt_numbers,
                until=until, outcomes=outcomes,
            )
            if should_stop:
                break
            current_stage = new_stage
            continue

        attempt_number = attempt_numbers.get(current_stage, 1)
        outcome = run_stage(
            store=store,
            project_dir=project_dir,
            spec=spec,
            attempt_number=attempt_number,
            observations=observations,
            force_rerun=force_rerun_current,
        )
        force_rerun_current = False
        outcomes.append(outcome)
        _print_outcome(outcome)

        new_stage, should_stop = _apply_stage_transition(
            outcome, store=store, stage_key=current_stage, attempt_number=attempt_number,
            attempt_numbers=attempt_numbers, until=until, outcomes=outcomes,
        )
        if should_stop:
            break
        current_stage = new_stage
    else:
        raise RuntimeError(
            "run_pipeline alcanzó max_iterations sin converger a un estado "
            "terminal; posible ciclo ADVANCE/RETURN entre etapas."
        )

    return outcomes


def _print_outcome(outcome: StageOutcome) -> None:
    print(
        f"[{outcome.status:24s}] {outcome.label:45s} "
        f"execution={outcome.execution_status} quality={outcome.quality_status} "
        f"next={outcome.next_action}->{outcome.target_stage}"
    )
    for warning in outcome.warnings:
        print(f"    warning: {warning}")
    if outcome.error:
        print(f"    error: {outcome.error}")


# ---------------------------------------------------------------------------
# CLI para uso directo en Colab: `python -m src.orchestration.pipeline_orchestrator`
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", required=True, help="Ruta a PROJECT_DIR (contiene active_experiment.json)."
    )
    parser.add_argument(
        "--until",
        default=None,
        choices=STAGE_ORDER,
        help="Detenerse tras completar esta etapa (por defecto corre hasta 06).",
    )
    parser.add_argument(
        "--start-stage",
        default=None,
        choices=STAGE_ORDER,
        help=(
            "Empezar el recorrido directamente en esta etapa, en vez de "
            "STAGE_ORDER[0] -- ejecuta ÚNICAMENTE esta etapa y las que "
            "resulten de sus transiciones reales (nunca las anteriores). "
            "Con start-stage explícito, el chequeo de estado ya-terminal "
            "se omite deliberadamente (se respeta la petición explícita "
            "del llamador, igual que --force-rerun) -- si la etapa ya "
            "está COMPLETED y vigente (fingerprints sin cambios), sigue "
            "reconociéndose SKIPPED_FRESH con normalidad; si su último "
            "commit fue FAILED (ej. HALT_STAGE), esto la reintenta con "
            "un decision_id nuevo, SIN --force-rerun y sin tocar ninguna "
            "etapa previa."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Reejecuta la etapa inicial aunque ya esté COMPLETED y vigente en pipeline_state.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    outcomes = run_pipeline(
        args.project_dir,
        start_stage=args.start_stage,
        until=args.until,
        force_rerun=args.force_rerun,
    )
    return 0 if all(o.status not in {"FAILED", "REACHED_UNREGISTERED_STAGE"} for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
