"""Equivalencia estructural entre la máquina de estados propia
(``pipeline_orchestrator.run_pipeline``) y la variante LangGraph
(``orchestration_langgraph.pipeline_graph.run_pipeline_via_langgraph``).

Límite honesto de este archivo: NO corre contra Chroma/OpenAI reales ni
contra un experimento real -- ese entorno no existe en el sandbox donde se
escribió esto. Lo que SÍ hace, con rigor: monta un ``StageSpec`` registry
falso (7 dobles deterministas, uno por etapa) cuyo ``build_execution``/
``runtime_transaction`` devuelven un guion FIJO de ``AgentResult`` -- pero
todo lo demás (``StateStore`` real, ``decision_engine`` real,
``run_stage()`` real, ``_apply_stage_transition``/``resolve_transition``
reales) es la maquinaria de producción sin mockear. Corre AMBOS motores
contra el MISMO guion, cada uno sobre su propio ``StateStore`` en un
directorio temporal separado, y compara sus secuencias de
``StageOutcome`` estructuralmente (nunca byte a byte -- excluye
timestamps/IDs no deterministas, per el criterio ya acordado).

El guion ejercita explícitamente los 4 casos mínimos pedidos:
ADVANCE normal (03->03B->04->05->06), un RETRY real (03B), un RETURN de
ciclo 06<->07 real, y un HALT_STAGE real (07, segunda vez, imitando
AGENT07_NON_CORRECTABLE_ISSUE).

Para la equivalencia REAL contra un experimento tuyo (Chroma/OpenAI
reales), ver ``scripts/run_langgraph_equivalence_check.py`` -- mismo
patrón de comparación estructural, pero corriendo los DOS motores contra
tu ``active_experiment.json`` real, en Colab.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

import pytest

from src.contracts.agent_result import (
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.orchestration.decision_engine import CANONICAL_STAGE_ORDER
from src.orchestration.stage_execution import StageSpec
from src.state.state_store import StateStore


# ---------------------------------------------------------------------------
# Guion determinista: por cada etapa, la secuencia de AgentResult que su
# runtime_transaction falso va a "producir" en visitas sucesivas.
# ---------------------------------------------------------------------------


def _agent_result(
    *,
    action: str,
    target_stage: str | None,
    execution_status: ExecutionStatus = ExecutionStatus.COMPLETED,
    quality_status: QualityStatus = QualityStatus.APPROVED,
    attempt_number: int = 1,
    reason_code: str = "SCRIPTED",
    error: dict | None = None,
) -> AgentResult:
    now = datetime.now(timezone.utc).isoformat()
    if execution_status == ExecutionStatus.FAILED and error is None:
        error = {"type": "ScriptedFailure", "message": "scripted for equivalence test"}
    return AgentResult(
        execution_status=execution_status,
        quality_status=quality_status,
        decision=DecisionInfo(code=reason_code, rationale="scripted for equivalence test"),
        quality_metrics={"technical": {}, "scientific": {}},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction(action),
            target_stage=target_stage,
            reason_code=reason_code,
            requires_human_confirmation=False,
        ),
        output_artifacts={},
        tool_usage=ToolUsage(),
        attempt_number=attempt_number,
        started_at=now,
        completed_at=now,
        error=error,
    )


def _build_script() -> dict[str, list[AgentResult]]:
    """Una lista de AgentResult por etapa, consumida en orden en visitas
    sucesivas. Cubre: ADVANCE normal, un RETRY real (03B), un RETURN de
    ciclo 06<->07 real, y un HALT_STAGE real (07, segunda visita)."""

    return {
        "03_agente_extraccion_kb": [
            _agent_result(action="ADVANCE", target_stage="03B_extraccion_cuantitativa_kb"),
        ],
        "03B_extraccion_cuantitativa_kb": [
            # 1a visita: RETRY real.
            _agent_result(
                action="RETRY", target_stage=None,
                execution_status=ExecutionStatus.FAILED, quality_status=QualityStatus.REJECTED,
            ),
            # 2a visita (tras el self-loop): ADVANCE.
            _agent_result(action="ADVANCE", target_stage="04_agente_analisis_tematico", attempt_number=2),
        ],
        "04_agente_analisis_tematico": [
            _agent_result(action="ADVANCE", target_stage="05_generador_esquema"),
        ],
        "05_generador_esquema": [
            _agent_result(action="ADVANCE", target_stage="06_agente_redactor"),
        ],
        "06_agente_redactor": [
            # 1a visita: ADVANCE hacia 07.
            _agent_result(action="ADVANCE", target_stage="07_agente_verificador"),
            # 2a visita (tras el RETURN de 07): ADVANCE de nuevo hacia 07.
            _agent_result(action="ADVANCE", target_stage="07_agente_verificador"),
        ],
        "07_agente_verificador": [
            # 1a visita: RETURN real hacia 06 (ciclo writer_verifier, ronda 1).
            _agent_result(action="RETURN", target_stage="06_agente_redactor"),
            # 2a visita: HALT_STAGE real (imita AGENT07_NON_CORRECTABLE_ISSUE).
            _agent_result(
                action="HALT_STAGE", target_stage=None,
                quality_status=QualityStatus.NEEDS_REVISION, reason_code="AGENT07_NON_CORRECTABLE_ISSUE",
            ),
        ],
        # 08 nunca se alcanza en este guion (07 se detiene con HALT_STAGE
        # en su 2a visita) -- deliberado, replica exactamente el patrón
        # real de tus 10 corridas (PARTIAL_HALT en 07).
        "08_evaluacion_experimental": [],
    }


@dataclass
class _FakeTransactionResult:
    agent_result: AgentResult


class _ScriptedStageBehavior:
    """Cierra sobre la lista de AgentResult de UNA etapa y los va
    devolviendo en orden, uno por llamada real a runtime_transaction --
    nunca más llamadas que resultados en el guion (falla explícito si el
    motor bajo prueba visita la etapa más veces de las previstas)."""

    def __init__(self, stage_key: str, results: list[AgentResult]) -> None:
        self.stage_key = stage_key
        self._results = list(results)
        self._index = 0

    def build_execution(self, project_dir, attempt_number: int):
        return object(), object()  # agente/AgentInput dummy -- nunca se inspecciona

    def runtime_transaction(self, *, store: StateStore, build_execution, attempt_number: int, observations=None):
        if self._index >= len(self._results):
            raise AssertionError(
                f"{self.stage_key}: el motor bajo prueba la visitó más veces "
                f"de las {len(self._results)} previstas en el guion."
            )
        result = self._results[self._index]
        self._index += 1
        prepared = store.prepare_execution(
            target_stage=self.stage_key,
            intended_action="SCRIPTED_EXECUTION",
            attempt_number=attempt_number,
        )
        persisted_path = store.persist_agent_result(prepared.decision_id, result)
        committed_state = store.commit_execution(
            decision_id=prepared.decision_id,
            result=result,
            stage_name=self.stage_key,
            fingerprints={
                "input": "scripted",
                "config": "scripted",
                "dependencies": "scripted",
                "composite": "scripted",
            },
            observations=dict(observations or {}),
        )
        return _FakeTransactionResult(agent_result=result)

    def resolve_resume(self, *, store, agent_input, observations=None):  # pragma: no cover
        raise AssertionError(
            f"{self.stage_key}: resolve_resume no debía invocarse en este guion "
            "(ninguna corrida se interrumpe a mitad de una etapa)."
        )


def _build_fake_registry() -> list[StageSpec]:
    script = _build_script()
    return [
        StageSpec(
            key=stage_key,
            label=f"(guion determinista: {stage_key})",
            build_execution=behavior.build_execution,
            runtime_transaction=behavior.runtime_transaction,
            resolve_resume=behavior.resolve_resume,
            build_fingerprints=None,
        )
        for stage_key in CANONICAL_STAGE_ORDER
        for behavior in [_ScriptedStageBehavior(stage_key, script[stage_key])]
    ]


def _write_active_experiment(project_dir) -> None:
    import json

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "active_experiment.json").write_text(
        json.dumps({"active_experiment_id": "equivalence_test"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _normalize_outcomes(outcomes) -> list[tuple[Any, ...]]:
    """Vista estructural comparable: excluye deliberadamente timestamps y
    cualquier ID no determinista (ninguno vive hoy en StageOutcome, pero
    se deja explícito el criterio -- ver punto 1 del diseño aprobado)."""

    return [
        (o.key, o.status, o.next_action, o.target_stage, o.reason_code, o.execution_status, o.quality_status)
        for o in outcomes
    ]


# ---------------------------------------------------------------------------
# La comparación real
# ---------------------------------------------------------------------------


def _patch_stage_registry_for_pipeline_orchestrator(monkeypatch, registry) -> None:
    """``pipeline_orchestrator.py`` importa ``_stage_registry`` con
    ``from ... import _stage_registry`` -- un nombre propio en su
    namespace, resuelto al momento del import. Parchear el atributo en
    ``stage_execution`` NO lo alcanza una vez que ``pipeline_orchestrator``
    ya está en ``sys.modules`` (cacheado desde un test anterior de la
    misma sesión de pytest) -- hay que parchear el nombre exacto donde
    vive, en cada módulo consumidor por separado."""

    monkeypatch.setattr("src.orchestration.pipeline_orchestrator._stage_registry", lambda: registry)


def _patch_stage_registry_for_langgraph(monkeypatch, registry) -> None:
    """Mismo motivo que arriba -- ``pipeline_graph.py`` Y ``nodes.py``
    importan ``_stage_registry`` cada uno por su cuenta; hay que
    parchear las dos ubicaciones, no solo una."""

    monkeypatch.setattr("src.orchestration_langgraph.pipeline_graph._stage_registry", lambda: registry)
    monkeypatch.setattr("src.orchestration_langgraph.nodes._stage_registry", lambda: registry)


def test_langgraph_matches_pipeline_orchestrator_structurally(tmp_path, monkeypatch):

    # Cada motor recibe su PROPIO registro de dobles (mismo guion, objetos
    # StageSpec/behaviors distintos) -- así cada uno consume su propio
    # StateStore sin compartir contadores de visita entre sí.
    registry_a = _build_fake_registry()
    registry_b = _build_fake_registry()

    _patch_stage_registry_for_pipeline_orchestrator(monkeypatch, registry_a)
    from src.orchestration.pipeline_orchestrator import run_pipeline

    store_a_dir = tmp_path / "motor_propio"
    _write_active_experiment(store_a_dir)
    outcomes_pipeline_orchestrator = run_pipeline(str(store_a_dir), max_iterations=20)

    _patch_stage_registry_for_langgraph(monkeypatch, registry_b)
    from src.orchestration_langgraph.pipeline_graph import run_pipeline_via_langgraph

    store_b_dir = tmp_path / "langgraph"
    _write_active_experiment(store_b_dir)
    outcomes_langgraph = run_pipeline_via_langgraph(str(store_b_dir), recursion_limit=40)

    normalized_a = _normalize_outcomes(outcomes_pipeline_orchestrator)
    normalized_b = _normalize_outcomes(outcomes_langgraph)

    assert normalized_a == normalized_b, (
        "Las secuencias de StageOutcome difieren estructuralmente entre "
        "pipeline_orchestrator.run_pipeline y "
        "orchestration_langgraph.run_pipeline_via_langgraph:\n"
        f"motor propio: {normalized_a}\n"
        f"langgraph:    {normalized_b}"
    )

    # Confirmaciones explícitas de que el guion realmente ejercitó los 4
    # casos mínimos pedidos, no solo que las dos listas coincidan entre sí.
    keys_and_actions = [(o.key, o.next_action) for o in outcomes_pipeline_orchestrator]
    assert ("03B_extraccion_cuantitativa_kb", "RETRY") in keys_and_actions, "falta el caso RETRY real"
    assert ("07_agente_verificador", "RETURN") in keys_and_actions, "falta el caso RETURN real"
    assert ("07_agente_verificador", "HALT_STAGE") in keys_and_actions, "falta el caso HALT_STAGE real"
    assert outcomes_pipeline_orchestrator[-1].next_action == "HALT_STAGE"
    # execution_status=COMPLETED con next_action=HALT_STAGE es el patrón
    # real (AGENT07_NON_CORRECTABLE_ISSUE, no un fallo técnico) -- status
    # se deriva de execution_status, no de next_action (ver run_stage()).
    assert outcomes_pipeline_orchestrator[-1].status == "COMMITTED"
    assert outcomes_pipeline_orchestrator[-1].execution_status == "COMPLETED"

    # El ciclo writer_verifier debe haber quedado con la MISMA ronda usada
    # en los dos StateStore -- confirma que apply_return_with_cycle corrió
    # igual en ambos motores, no solo que los StageOutcome coincidan.
    from src.orchestration.stage_execution import resolve_state_path

    state_path_a, _eid_a, _rid_a = resolve_state_path(store_a_dir)
    state_path_b, _eid_b, _rid_b = resolve_state_path(store_b_dir)
    store_a = StateStore(state_path_a)
    store_b = StateStore(state_path_b)
    cycle_a = store_a.load().cycles.get("writer_verifier")
    cycle_b = store_b.load().cycles.get("writer_verifier")
    assert cycle_a is not None and cycle_b is not None
    assert cycle_a.rounds_used == cycle_b.rounds_used == 1
    assert cycle_a.status == cycle_b.status == "ACTIVE"


# ---------------------------------------------------------------------------
# Camino normal completo: 03->03B->04->05->06->07(ADVANCE)->08->END
# ---------------------------------------------------------------------------


def _build_happy_path_script() -> dict[str, list[AgentResult]]:
    """Guion donde las 7 etapas hacen ADVANCE en cadena, terminando en 08
    con target_stage=None (pipeline completo) -- sin RETRY, sin RETURN.
    Complementa el guion de la 1a comparación (que deliberadamente
    terminaba en HALT_STAGE dentro de 07 y nunca llegaba a validar la
    terminación normal vía 08)."""

    order = list(CANONICAL_STAGE_ORDER)
    script: dict[str, list[AgentResult]] = {}
    for i, stage_key in enumerate(order):
        next_target = order[i + 1] if i + 1 < len(order) else None
        script[stage_key] = [_agent_result(action="ADVANCE", target_stage=next_target)]
    return script


def _build_fake_registry_from_script(script: dict[str, list[AgentResult]]) -> list[StageSpec]:
    return [
        StageSpec(
            key=stage_key,
            label=f"(guion determinista: {stage_key})",
            build_execution=behavior.build_execution,
            runtime_transaction=behavior.runtime_transaction,
            resolve_resume=behavior.resolve_resume,
            build_fingerprints=None,
        )
        for stage_key in CANONICAL_STAGE_ORDER
        for behavior in [_ScriptedStageBehavior(stage_key, script[stage_key])]
    ]


def test_happy_path_full_pipeline_matches_structurally(tmp_path, monkeypatch):

    script_a = _build_happy_path_script()
    script_b = _build_happy_path_script()
    registry_a = _build_fake_registry_from_script(script_a)
    registry_b = _build_fake_registry_from_script(script_b)

    _patch_stage_registry_for_pipeline_orchestrator(monkeypatch, registry_a)
    from src.orchestration.pipeline_orchestrator import run_pipeline

    store_a_dir = tmp_path / "happy_motor_propio"
    _write_active_experiment(store_a_dir)
    outcomes_a = run_pipeline(str(store_a_dir), max_iterations=20)

    _patch_stage_registry_for_langgraph(monkeypatch, registry_b)
    from src.orchestration_langgraph.pipeline_graph import run_pipeline_via_langgraph

    store_b_dir = tmp_path / "happy_langgraph"
    _write_active_experiment(store_b_dir)
    outcomes_b = run_pipeline_via_langgraph(str(store_b_dir), recursion_limit=20)

    assert _normalize_outcomes(outcomes_a) == _normalize_outcomes(outcomes_b)

    # Confirma explícitamente que SÍ llegó hasta el final vía 08, con
    # terminación normal -- la terminación normal que el primer test
    # (deliberadamente) nunca ejercitaba. decision_engine.validate_transition
    # normaliza ADVANCE con target_stage=None a STOP_PIPELINE/
    # PIPELINE_COMPLETE (confirmado en el propio código, no un supuesto) --
    # ese es el next_action real que ambos motores deben reportar para 08.
    assert [o.key for o in outcomes_a] == list(CANONICAL_STAGE_ORDER)
    assert outcomes_a[-1].key == "08_evaluacion_experimental"
    assert outcomes_a[-1].next_action == "STOP_PIPELINE"
    assert outcomes_a[-1].target_stage is None
    assert outcomes_a[-1].reason_code == "PIPELINE_COMPLETE"
    assert all(o.next_action == "ADVANCE" for o in outcomes_a[:-1])


# ---------------------------------------------------------------------------
# STOP_PIPELINE explícito
# ---------------------------------------------------------------------------


def test_stop_pipeline_matches_structurally(tmp_path, monkeypatch):

    # Escenario mínimo: 03 hace ADVANCE hacia 03B, pero 03B pide
    # STOP_PIPELINE directamente (no HALT_STAGE) -- ambos motores deben
    # detenerse ahí, con el mismo reason_code, sin tocar 04 en adelante.
    def _script() -> dict[str, list[AgentResult]]:
        script = {stage_key: [] for stage_key in CANONICAL_STAGE_ORDER}
        script["03_agente_extraccion_kb"] = [
            _agent_result(action="ADVANCE", target_stage="03B_extraccion_cuantitativa_kb")
        ]
        script["03B_extraccion_cuantitativa_kb"] = [
            _agent_result(
                action="STOP_PIPELINE", target_stage=None, reason_code="SCRIPTED_STOP_PIPELINE"
            )
        ]
        return script

    registry_a = _build_fake_registry_from_script(_script())
    registry_b = _build_fake_registry_from_script(_script())

    _patch_stage_registry_for_pipeline_orchestrator(monkeypatch, registry_a)
    from src.orchestration.pipeline_orchestrator import run_pipeline

    store_a_dir = tmp_path / "stop_motor_propio"
    _write_active_experiment(store_a_dir)
    outcomes_a = run_pipeline(str(store_a_dir), max_iterations=20)

    _patch_stage_registry_for_langgraph(monkeypatch, registry_b)
    from src.orchestration_langgraph.pipeline_graph import run_pipeline_via_langgraph

    store_b_dir = tmp_path / "stop_langgraph"
    _write_active_experiment(store_b_dir)
    outcomes_b = run_pipeline_via_langgraph(str(store_b_dir), recursion_limit=20)

    assert _normalize_outcomes(outcomes_a) == _normalize_outcomes(outcomes_b)

    assert [o.key for o in outcomes_a] == ["03_agente_extraccion_kb", "03B_extraccion_cuantitativa_kb"]
    assert outcomes_a[-1].next_action == "STOP_PIPELINE"
    assert outcomes_a[-1].reason_code == "SCRIPTED_STOP_PIPELINE"
    # Nunca debió alcanzar 04 en adelante en NINGUNO de los dos motores.
    visited_keys_a = {o.key for o in outcomes_a}
    visited_keys_b = {o.key for o in outcomes_b}
    for later_stage in ("04_agente_analisis_tematico", "05_generador_esquema", "06_agente_redactor"):
        assert later_stage not in visited_keys_a
        assert later_stage not in visited_keys_b


# ---------------------------------------------------------------------------
# Equivalencia de pending_execution / reconcile_pending
# ---------------------------------------------------------------------------


class _PendingReconciliationBehavior:
    """Doble para la etapa que tiene una pending_execution "vieja":
    resolve_resume delega en el protocolo REAL de StateStore
    (``store.resolve_resume``), que encuentra el AgentResult ya
    persistido (pero no comprometido) y lo comete -- exactamente el
    mismo camino que toma un runtime real (``verification_notebook.py``,
    etc.) al resolver una interrupción."""

    def __init__(self, stage_key: str) -> None:
        self.stage_key = stage_key

    def build_execution(self, project_dir, attempt_number: int):
        return object(), object()

    def resolve_resume(self, *, store: StateStore, agent_input, observations=None):
        return store.resolve_resume(
            stage_name=self.stage_key,
            fingerprints={
                "input": "scripted", "config": "scripted",
                "dependencies": "scripted", "composite": "scripted",
            },
            observations=observations,
        )

    def runtime_transaction(self, *, store, build_execution, attempt_number, observations=None):  # pragma: no cover
        raise AssertionError(
            f"{self.stage_key}: runtime_transaction no debía invocarse -- "
            "la pending_execution debía resolverse vía resolve_resume."
        )


def _prepare_stale_pending(project_dir, *, stage_key: str, scripted_result: AgentResult) -> None:
    """Simula una ejecución interrumpida: PREPARE + persistir el
    AgentResult, pero SIN comprometer -- deja pending_execution
    apuntando a ``stage_key`` en el StateStore real del experimento."""

    from src.orchestration.stage_execution import ensure_pipeline_state

    store = ensure_pipeline_state(project_dir)
    prepared = store.prepare_execution(
        target_stage=stage_key, intended_action="SCRIPTED_INTERRUPTED_EXECUTION", attempt_number=1,
    )
    store.persist_agent_result(prepared.decision_id, scripted_result)


def _build_pending_reconciliation_registry() -> list[StageSpec]:
    """03B tiene una pending_execution "vieja" que se reconcilia hacia
    04 (ADVANCE); 04 recibe un ScriptedStageBehavior normal que hace
    STOP_PIPELINE para terminar rápido y determinista. El resto nunca
    se visita."""

    pending_behavior = _PendingReconciliationBehavior("03B_extraccion_cuantitativa_kb")
    stage_04_behavior = _ScriptedStageBehavior(
        "04_agente_analisis_tematico",
        [_agent_result(action="STOP_PIPELINE", target_stage=None, reason_code="SCRIPTED_STOP_AFTER_RECONCILE")],
    )
    empty_behavior = {
        stage_key: _ScriptedStageBehavior(stage_key, [])
        for stage_key in CANONICAL_STAGE_ORDER
        if stage_key not in ("03B_extraccion_cuantitativa_kb", "04_agente_analisis_tematico")
    }

    specs = []
    for stage_key in CANONICAL_STAGE_ORDER:
        if stage_key == "03B_extraccion_cuantitativa_kb":
            behavior = pending_behavior
        elif stage_key == "04_agente_analisis_tematico":
            behavior = stage_04_behavior
        else:
            behavior = empty_behavior[stage_key]
        specs.append(
            StageSpec(
                key=stage_key,
                label=f"(guion determinista: {stage_key})",
                build_execution=behavior.build_execution,
                runtime_transaction=behavior.runtime_transaction,
                resolve_resume=behavior.resolve_resume,
                build_fingerprints=None,
            )
        )
    return specs


def test_pending_execution_reconciliation_matches_structurally(tmp_path, monkeypatch):

    pending_result = _agent_result(action="ADVANCE", target_stage="04_agente_analisis_tematico")

    registry_a = _build_pending_reconciliation_registry()
    registry_b = _build_pending_reconciliation_registry()

    # --- Motor propio ---
    _patch_stage_registry_for_pipeline_orchestrator(monkeypatch, registry_a)
    from src.orchestration.pipeline_orchestrator import run_pipeline

    store_a_dir = tmp_path / "pending_motor_propio"
    _write_active_experiment(store_a_dir)
    _prepare_stale_pending(store_a_dir, stage_key="03B_extraccion_cuantitativa_kb", scripted_result=pending_result)
    # entry_stage = 04 (lo que el motor "intenta" ahora); la pending
    # apunta a 03B, una etapa DISTINTA -- exactamente el escenario que
    # _reconcile_pending_execution_for_other_stage/reconcile_pending
    # deben resolver antes de tocar 04.
    outcomes_a = run_pipeline(str(store_a_dir), start_stage="04_agente_analisis_tematico", max_iterations=20)

    # --- LangGraph ---
    _patch_stage_registry_for_langgraph(monkeypatch, registry_b)
    from src.orchestration_langgraph.pipeline_graph import run_pipeline_via_langgraph

    store_b_dir = tmp_path / "pending_langgraph"
    _write_active_experiment(store_b_dir)
    _prepare_stale_pending(store_b_dir, stage_key="03B_extraccion_cuantitativa_kb", scripted_result=pending_result)
    outcomes_b = run_pipeline_via_langgraph(
        str(store_b_dir), start_stage="04_agente_analisis_tematico", recursion_limit=20
    )

    assert _normalize_outcomes(outcomes_a) == _normalize_outcomes(outcomes_b)

    # La pending de 03B debe reconciliarse PRIMERO (antes de tocar 04) en
    # los dos motores, y el pipeline debe CONTINUAR hacia 04 después
    # (porque el AgentResult reconciliado pedía ADVANCE->04), no
    # detenerse solo por haber encontrado una pending.
    assert [o.key for o in outcomes_a] == ["03B_extraccion_cuantitativa_kb", "04_agente_analisis_tematico"]
    assert outcomes_a[0].next_action == "ADVANCE"
    assert outcomes_a[0].target_stage == "04_agente_analisis_tematico"
    assert outcomes_a[1].next_action == "STOP_PIPELINE"
    assert outcomes_a[1].reason_code == "SCRIPTED_STOP_AFTER_RECONCILE"

    # Tras la reconciliación, pending_execution debe quedar en None en
    # AMBOS StateStore -- confirma que el protocolo oficial (store.
    # resolve_resume) realmente la liberó, no solo que los outcomes
    # coincidan superficialmente.
    from src.orchestration.stage_execution import ensure_pipeline_state

    state_a = ensure_pipeline_state(store_a_dir).load()
    state_b = ensure_pipeline_state(store_b_dir).load()
    assert state_a.pending_execution is None
    assert state_b.pending_execution is None
