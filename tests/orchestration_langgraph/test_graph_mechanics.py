"""Tests unitarios de la variante LangGraph (MAIN 5, Bloque 3).

Cubren exclusivamente la MECÁNICA del grafo (estructura, ruteo) con dobles
de ``StageOutcome`` -- no requieren Chroma/OpenAI ni un experimento real
(eso queda para la batería de equivalencia del Bloque 4, contra
``pipeline_orchestrator.py``, sobre un experimento real conocido).
"""

from __future__ import annotations

import ast
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.orchestration.decision_engine import CANONICAL_STAGE_ORDER, MAX_ATTEMPTS_DEFAULT
from src.orchestration.stage_execution import StageOutcome
from src.orchestration_langgraph.routing import END, resolve_transition
from src.state.pipeline_state import CycleState, PipelineIdentity, PipelineState
from src.state.state_store import StateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_store(tmp_path: Path) -> StateStore:
    state_path = tmp_path / "pipeline_state.json"
    store = StateStore(state_path)
    now = datetime.now(timezone.utc).isoformat()
    store.initialize(
        PipelineState(
            identity=PipelineIdentity(
                experiment_id="test_experiment",
                run_id="test_experiment",
                created_at=now,
                updated_at=now,
                schema_version="1.0",
            )
        )
    )
    return store


def _outcome(
    *,
    key: str,
    next_action: str,
    target_stage: str | None = None,
    status: str = "COMMITTED",
) -> StageOutcome:
    return StageOutcome(
        key=key,
        label=f"(dummy {key})",
        status=status,
        execution_status="COMPLETED",
        quality_status="APPROVED",
        warnings=(),
        error=None,
        attempt_number=1,
        next_action=next_action,
        target_stage=target_stage,
        reason_code="DUMMY",
    )


# ---------------------------------------------------------------------------
# resolve_transition() -- ADVANCE / RETRY / HALT_STAGE / STOP_PIPELINE / until
# (no requieren tocar StateStore más allá de leerlo, salvo el caso RETURN)
# ---------------------------------------------------------------------------


def test_advance_routes_to_target_stage(tmp_path):
    store = _fresh_store(tmp_path)
    outcome = _outcome(key="03_agente_extraccion_kb", next_action="ADVANCE", target_stage="03B_extraccion_cuantitativa_kb")
    target, attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="03_agente_extraccion_kb", attempt_number=1,
        attempt_numbers={}, until=None,
    )
    assert target == "03B_extraccion_cuantitativa_kb"
    assert synthetic is None
    assert attempt_numbers == {}


def test_advance_with_target_none_is_pipeline_complete(tmp_path):
    store = _fresh_store(tmp_path)
    outcome = _outcome(key="08_evaluacion_experimental", next_action="ADVANCE", target_stage=None)
    target, _attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="08_evaluacion_experimental", attempt_number=1,
        attempt_numbers={}, until=None,
    )
    assert target == END
    assert synthetic is None


def test_retry_self_loops_and_increments_attempt_number(tmp_path):
    store = _fresh_store(tmp_path)
    outcome = _outcome(key="03_agente_extraccion_kb", next_action="RETRY")
    target, attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="03_agente_extraccion_kb", attempt_number=1,
        attempt_numbers={}, until=None,
    )
    assert target == "03_agente_extraccion_kb"
    assert attempt_numbers["03_agente_extraccion_kb"] == 2
    assert synthetic is None


def test_halt_stage_routes_to_end(tmp_path):
    store = _fresh_store(tmp_path)
    outcome = _outcome(key="07_agente_verificador", next_action="HALT_STAGE")
    target, _attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="07_agente_verificador", attempt_number=1,
        attempt_numbers={}, until=None,
    )
    assert target == END
    assert synthetic is None


def test_stop_pipeline_routes_to_end(tmp_path):
    store = _fresh_store(tmp_path)
    outcome = _outcome(key="05_generador_esquema", next_action="STOP_PIPELINE")
    target, _attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="05_generador_esquema", attempt_number=1,
        attempt_numbers={}, until=None,
    )
    assert target == END
    assert synthetic is None


def test_until_reached_routes_to_end_even_on_advance(tmp_path):
    store = _fresh_store(tmp_path)
    outcome = _outcome(key="06_agente_redactor", next_action="ADVANCE", target_stage="07_agente_verificador")
    target, _attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="06_agente_redactor", attempt_number=1,
        attempt_numbers={}, until="06_agente_redactor",
    )
    assert target == END
    assert synthetic is None


# ---------------------------------------------------------------------------
# resolve_transition() -- RETURN / ciclo 06<->07 (sí toca StateStore)
# ---------------------------------------------------------------------------


def test_return_non_cycle_pair_invalidates_and_routes_to_target(tmp_path):
    # Cualquier par RETURN que no sea exactamente 07->06 no activa
    # CycleState -- solo invalidate_from.
    store = _fresh_store(tmp_path)
    outcome = _outcome(key="05_generador_esquema", next_action="RETURN", target_stage="04_agente_analisis_tematico")
    target, attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="05_generador_esquema", attempt_number=1,
        attempt_numbers={"04_agente_analisis_tematico": 1, "05_generador_esquema": 1}, until=None,
    )
    assert target == "04_agente_analisis_tematico"
    assert synthetic is None
    # attempt_numbers de la etapa objetivo en adelante se limpian.
    assert "04_agente_analisis_tematico" not in attempt_numbers
    assert "05_generador_esquema" not in attempt_numbers


def test_return_writer_verifier_cycle_increments_rounds_used(tmp_path):
    store = _fresh_store(tmp_path)
    outcome = _outcome(key="07_agente_verificador", next_action="RETURN", target_stage="06_agente_redactor")
    target, _attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="07_agente_verificador", attempt_number=1,
        attempt_numbers={}, until=None,
    )
    assert target == "06_agente_redactor"
    assert synthetic is None
    state = store.load()
    cycle = state.cycles["writer_verifier"]
    assert cycle.rounds_used == 1
    assert cycle.status == "ACTIVE"


def test_return_writer_verifier_cycle_exhausted_routes_to_end(tmp_path):
    store = _fresh_store(tmp_path)
    # Agota el ciclo de antemano (max_rounds=3 por defecto -- 3 rondas ya usadas).
    state = store.load()
    from dataclasses import replace

    exhausted_cycles = dict(state.cycles)
    exhausted_cycles["writer_verifier"] = CycleState(rounds_used=3, max_rounds=3, status="ACTIVE")
    store.save(replace(state, cycles=exhausted_cycles))

    outcome = _outcome(key="07_agente_verificador", next_action="RETURN", target_stage="06_agente_redactor")
    target, _attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="07_agente_verificador", attempt_number=1,
        attempt_numbers={}, until=None,
    )
    assert target == END
    assert synthetic is not None
    assert synthetic.status == "CYCLE_EXHAUSTED"
    assert synthetic.reason_code == "WRITER_VERIFIER_CYCLE_EXHAUSTED"

    state = store.load()
    assert state.cycles["writer_verifier"].status == "EXHAUSTED"


def test_advance_from_07_resolves_active_cycle(tmp_path):
    # Si 07 hace ADVANCE (no RETURN) con un ciclo ACTIVE, el ciclo debe
    # quedar RESOLVED -- mismo comportamiento que
    # pipeline_orchestrator._apply_stage_transition.
    store = _fresh_store(tmp_path)
    state = store.load()
    from dataclasses import replace

    active_cycles = dict(state.cycles)
    active_cycles["writer_verifier"] = CycleState(rounds_used=1, max_rounds=3, status="ACTIVE")
    store.save(replace(state, cycles=active_cycles))

    outcome = _outcome(key="07_agente_verificador", next_action="ADVANCE", target_stage="08_evaluacion_experimental")
    target, _attempt_numbers, synthetic = resolve_transition(
        outcome, store=store, stage_key="07_agente_verificador", attempt_number=1,
        attempt_numbers={}, until=None,
    )
    assert target == "08_evaluacion_experimental"
    assert synthetic is None
    state = store.load()
    assert state.cycles["writer_verifier"].status == "RESOLVED"


# ---------------------------------------------------------------------------
# route_after_stage() -- callback trivial del edge condicional
# ---------------------------------------------------------------------------


def test_route_after_stage_reads_route_target():
    from src.orchestration_langgraph.routing import route_after_stage

    assert route_after_stage({"route_target": "04_agente_analisis_tematico"}) == "04_agente_analisis_tematico"
    assert route_after_stage({"route_target": END}) == END


def test_route_after_stage_raises_if_route_target_missing():
    from src.orchestration_langgraph.routing import route_after_stage

    with pytest.raises(RuntimeError, match="route_target"):
        route_after_stage({})


# ---------------------------------------------------------------------------
# Estructura del grafo compilado
# ---------------------------------------------------------------------------


def test_graph_compiles_with_expected_nodes():
    from src.orchestration_langgraph.pipeline_graph import compile_pipeline_graph

    app = compile_pipeline_graph()
    nodes = set(app.get_graph().nodes.keys())
    for stage_key in CANONICAL_STAGE_ORDER:
        assert stage_key in nodes
    assert "reconcile_pending" in nodes


def test_stage_node_factory_produces_one_callable_per_stage():
    from src.orchestration.stage_execution import _stage_registry
    from src.orchestration_langgraph.nodes import make_stage_node

    registry = {spec.key: spec for spec in _stage_registry()}
    for stage_key in CANONICAL_STAGE_ORDER:
        node = make_stage_node(registry[stage_key])
        assert callable(node)
        assert node.__name__ == f"node_{stage_key}"


# ---------------------------------------------------------------------------
# LangGraph nunca importa pipeline_orchestrator.py (requisito explícito)
# ---------------------------------------------------------------------------


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


def test_orchestration_langgraph_never_imports_pipeline_orchestrator():
    package_dir = Path(__file__).resolve().parents[2] / "src" / "orchestration_langgraph"
    for py_file in package_dir.glob("*.py"):
        imported = _imported_module_names(py_file)
        offending = {name for name in imported if "pipeline_orchestrator" in name}
        assert not offending, f"{py_file.name} importa pipeline_orchestrator: {offending}"
