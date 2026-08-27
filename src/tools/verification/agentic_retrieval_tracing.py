"""Agentic Retrieval -- Bloque 6 (corregido): instrumentación y
trazabilidad experimental. NO cambia decisiones, retrieval, planner,
grader ni VerificationAgent -- hace observable el ciclo ya funcional
(Bloques 1-5).

CORRECCIÓN (B6-TRACE-CONSISTENCY-FIX): el trace observacional
distingue explícitamente dos historias, en vez de mezclarlas:

- ``decision_steps``: copia DIRECTA de ``AgenticRetrievalResult.steps``
  (Bloque 2, contrato cerrado) -- incluye TODAS las decisiones del
  planner/Python, incluida ``ACCEPT_EVIDENCE`` (que no ejecuta
  retrieval, no genera una nueva Observation -- no tiene sentido
  inventarle query_before/after).

- ``retrieval_transitions``: derivada del wrapper before/after sobre
  ``execute_action_fn`` (solo REWRITE_QUERY/ADJUST_TOP_K, las únicas
  acciones que ejecutan retrieval real), RECONCILIADA contra
  ``result.outcome``: confirmado que ``run_agentic_retrieval_cycle``
  (Bloque 2) hace ``steps.append(...)`` ANTES de
  ``_validate_improvement_transition(...)`` -- por tanto
  ``result.steps`` (y por extensión ``decision_steps``) SÍ incluye el
  step que produjo una transición rechazada cuando
  ``outcome == AGENTIC_TRANSITION_INVALID``, pero esa transición NUNCA
  fue aceptada por el controller. La reconciliación elimina esa última
  entrada de ``retrieval_transitions`` -- nunca se presenta un intento
  rechazado como transición exitosa. No se modifica el controller para
  esto: la reconciliación ocurre aquí, en runtime, después de que
  ``run_agentic_retrieval_cycle`` retorna.

Si ``FINISH_UNRESOLVED`` ocurre como outcome forzado (sin
``selected_action`` real), no se inventa ningún decision step para
él -- permanece únicamente como ``outcome``.

Fuente estructurada exclusivamente: ``decision_steps`` = copia directa
de ``result.steps``; ``retrieval_transitions`` = las
``AgenticRetrievalObservation`` reales capturadas antes/después de
cada llamada exitosa al executor. Nunca se parsean logs/prompts/strings.

Todos los campos de nivel superior pedidos (claim_id,
initial_grade_result, initial_candidate_count,
initial_max_relevance_score, initial_top_k, planner_invoked, outcome,
final_grade_result, final_candidate_count, final_max_relevance_score,
query_rewrite_count, agentic_additional_retrievals_used,
effective_budget_for_verify_claim) ya existen, producidos por
``_run_agentic_retrieval_for_claim``/``rag_record["agentic_retrieval"]``
(Bloque 5) -- reutilizados tal cual, sin recomputar.

El resultado vive exclusivamente en ``record`` (segundo valor de
``_independent_retrieve_claim``, separado de ``updated``/``ctx`` -- el
contexto científico que consume el LLM) -- nunca lo contamina."""

from __future__ import annotations

from typing import Any, Callable


def build_traced_execute_action_fn(
    execute_action_fn: Callable[[str, str, Any], Any],
) -> tuple[Callable[[str, str, Any], Any], list[dict[str, Any]]]:
    """Envuelve ``execute_action_fn`` (executor de Bloque 4) SIN
    modificar su comportamiento -- captura el ``AgenticRetrievalObservation``
    real recibido (antes) y el real devuelto (después) de cada llamada
    EXITOSA a la tool (REWRITE_QUERY/ADJUST_TOP_K -- las únicas que
    pasan por ``execute_action_fn``, nunca ACCEPT_EVIDENCE).

    Retorna ``(traced_fn, raw_transition_attempts)`` -- ``raw_transition_
    attempts`` se llena en vivo; PUEDE incluir un intento final que el
    controller termine rechazando (``AGENTIC_TRANSITION_INVALID``) --
    la reconciliación contra ``result.outcome`` ocurre en
    ``build_agentic_retrieval_trace``, no aquí."""
    raw_transition_attempts: list[dict[str, Any]] = []

    def traced_execute_action_fn(selected_action: str, decision_basis: str, observation: Any) -> Any:
        after = execute_action_fn(selected_action, decision_basis, observation)
        # Se registra tras retorno exitoso de la tool -- pero el
        # controller aún puede rechazar esta transición
        # (_validate_improvement_transition, después de este punto). Si
        # execute_action_fn en sí lanza una excepción (ej. el retriever
        # falla), esta línea nunca se alcanza -- ese intento tampoco
        # queda registrado (misma semántica atómica de Bloque 4).
        raw_transition_attempts.append({
            "retrieval_round": after.retrieval_round,
            "selected_action": selected_action,
            "decision_basis": decision_basis,
            "query_before": observation.current_query,
            "query_after": after.current_query,
            "top_k_before": observation.current_top_k,
            "top_k_after": after.current_top_k,
            "candidate_count_before": observation.candidate_count,
            "candidate_count_after": after.candidate_count,
            "max_relevance_before": observation.max_relevance_score,
            "max_relevance_after": after.max_relevance_score,
            "reason_codes_before": observation.reason_codes,
            "reason_codes_after": after.reason_codes,
            "remaining_budget_before": observation.remaining_retrieval_budget,
            "remaining_budget_after": after.remaining_retrieval_budget,
        })
        return after

    return traced_execute_action_fn, raw_transition_attempts


def _reconcile_retrieval_transitions(
    *, raw_transition_attempts: list[dict[str, Any]], outcome: str | None,
) -> tuple[dict[str, Any], ...]:
    """Fail-closed contra el resultado REAL validado del controller: si
    ``outcome == AGENTIC_TRANSITION_INVALID``, el último intento
    capturado por el wrapper corresponde exactamente al que
    ``_validate_improvement_transition`` rechazó (confirmado: en Bloque
    2, ``steps.append`` ocurre antes de esa validación, así que el
    wrapper -- que se ejecuta dentro de esa misma llamada a
    ``execute_action_fn`` -- ya lo registró como intento, pero el
    controller nunca lo aceptó como transición real). Se elimina esa
    última entrada -- nunca se presenta como step exitoso."""
    transitions = list(raw_transition_attempts)
    if outcome == "AGENTIC_TRANSITION_INVALID" and transitions:
        transitions.pop()
    return tuple(transitions)


def build_agentic_retrieval_trace(
    *,
    claim_id: str,
    initial_candidate_count: int,
    initial_max_relevance_score: float,
    initial_top_k: int,
    raw_transition_attempts: list[dict[str, Any]],
    agentic_result: dict[str, Any],
) -> dict[str, Any]:
    """Ensambla la estructura de trazabilidad completa, reutilizando
    exclusivamente datos ya producidos por Bloques 1-5 -- sin
    recalcular ni reconstruir ninguna métrica desde logs o strings.

    ``decision_steps`` = copia directa de ``agentic_result["steps"]``
    (a su vez, copia directa de ``AgenticRetrievalResult.steps``,
    Bloque 2) -- incluye ACCEPT_EVIDENCE cuando ocurre.
    ``retrieval_transitions`` = ``raw_transition_attempts`` reconciliado
    contra ``agentic_result["outcome"]`` -- solo transiciones realmente
    aceptadas por el controller."""
    final_observation = agentic_result.get("final_observation")
    outcome = agentic_result["outcome"]

    decision_steps = tuple(dict(step) for step in agentic_result.get("steps", ()))
    retrieval_transitions = _reconcile_retrieval_transitions(
        raw_transition_attempts=raw_transition_attempts, outcome=outcome,
    )

    trace: dict[str, Any] = {
        "claim_id": claim_id,
        "initial_grade_result": agentic_result["initial_grade_result"],
        "initial_candidate_count": initial_candidate_count,
        "initial_max_relevance_score": initial_max_relevance_score,
        "initial_top_k": initial_top_k,
        "planner_invoked": agentic_result["planner_invoked"],
        "decision_steps": decision_steps,
        "retrieval_transitions": retrieval_transitions,
        "outcome": outcome,
        "agentic_additional_retrievals_used": agentic_result["agentic_additional_retrievals_used"],
        "effective_budget_for_verify_claim": agentic_result["effective_budget_for_verify_claim"],
    }

    if final_observation is not None:
        trace["final_grade_result"] = final_observation["grade_result"]
        trace["final_candidate_count"] = final_observation["candidate_count"]
        trace["final_max_relevance_score"] = final_observation["max_relevance_score"]
        trace["query_rewrite_count"] = final_observation["query_rewrite_count"]
    else:
        # SUFFICIENT inicial: el "final" coincide con el inicial -- no
        # se ejecutó el ciclo, no hay final_observation distinto.
        trace["final_grade_result"] = agentic_result["initial_grade_result"]
        trace["final_candidate_count"] = initial_candidate_count
        trace["final_max_relevance_score"] = initial_max_relevance_score
        trace["query_rewrite_count"] = 0

    return trace
