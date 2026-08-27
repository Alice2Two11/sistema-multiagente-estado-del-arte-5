"""Agentic Retrieval (Stage 07, pre-verificación) -- Bloque 2:
controller (Observation, gate de acciones, ciclo).

    claim
      -> RETRIEVE (determinista, siempre, query=claim_text, top_k=top_k_initial)
      -> GRADE_EVIDENCE (determinista, automático -- src.tools.verification.
         agentic_retrieval_grader.grade_evidence, Bloque 1)
      -> SUFFICIENT? -> ACCEPT_EVIDENCE (determinista, forzado por el
         controller, el planner NUNCA es consultado en este caso)
      -> INSUFFICIENT, budget agotado?
           sí -> minimum_viable_evidence? true -> ACCEPT_EVIDENCE (forzado)
                                            false -> FINISH_UNRESOLVED (forzado)
           no -> planner elige entre {REWRITE_QUERY, ADJUST_TOP_K} (primera
                 insuficiencia) o {REWRITE_QUERY, ADJUST_TOP_K, ACCEPT_EVIDENCE
                 si minimum_viable_evidence} (insuficiencias posteriores)
      -> [tool ejecuta la acción elegida] -> RETRIEVE -> GRADE_EVIDENCE -> ...

Punto de decisión ReAct genuino, y el ÚNICO: cuando existe evidencia
insuficiente con presupuesto disponible, el planner elige realmente
entre REWRITE_QUERY y ADJUST_TOP_K (dos acciones con efectos
observables distintos, ninguna reducible a la otra) -- y, tras al
menos un intento de mejora, también entre aceptar lo que hay
(ACCEPT_EVIDENCE) si es mínimamente viable. RETRIEVE/GRADE_EVIDENCE
NUNCA son decisiones del planner -- son consecuencias obligatorias.
ACCEPT_EVIDENCE NUNCA está disponible en la primera insuficiencia
(``retrieval_round == 0``) -- el sistema intenta mejorar la
recuperación primero, tal como se acordó explícitamente.

Autocontenido: NO reutiliza ``react_prompting.py`` (auditado en el
diseño: importa y valida directamente contra ``REACT_ACTIONS``/
``DECISION_BASIS_VALUES`` del dominio post-verificación descartado --
no es genérico tal como está escrito). Este módulo define su propio
prompt/parseo mínimo, mismo patrón conceptual (JSON de 2 claves, sin
rationale libre, fail-closed ante malformado).

Presupuesto: reutiliza directamente ``remaining_retrieval_budget``
(derivado del mismo ``max_additional_retrieval_requests`` compartido
con ``verify_claim``, ver Bloque 1
``compute_effective_budget_for_verify_claim``) como único límite del
ciclo -- sin crear ``MAX_AGENTIC_RAG_STEPS_PER_CLAIM`` como contador
independiente (decisión ya tomada: evitar contadores redundantes).

Este bloque NO toca verification_runtime.py, verification_agent.py, ni
el retriever -- solo el controller y sus tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from src.config.agentic_retrieval_policy_config import GRADE_REASON_CODES

# ---------------------------------------------------------------------------
# Acciones planner-seleccionables -- SOLO estas 3. RETRIEVE/GRADE_EVIDENCE
# son consecuencias obligatorias (nunca elegidas por el planner).
# FINISH_UNRESOLVED es un outcome forzado por el controller, nunca una
# acción que el planner elija directamente.
# ---------------------------------------------------------------------------

AGENTIC_RETRIEVAL_ACTIONS = ("REWRITE_QUERY", "ADJUST_TOP_K", "ACCEPT_EVIDENCE")

FINISH_UNRESOLVED = "FINISH_UNRESOLVED"
AGENTIC_PLANNER_FAILED = "AGENTIC_PLANNER_FAILED"

# decision_basis -- enum cerrado, refleja 1:1 los reason codes del
# grader (Bloque 1) más un valor para la aceptación deliberada pese a
# huecos. Sin rationale libre ni chain-of-thought.
AGENTIC_DECISION_BASIS_VALUES = (
    "EVIDENCE_INSUFFICIENT_LOW_CANDIDATE_COUNT",
    "EVIDENCE_INSUFFICIENT_LOW_SOURCE_DIVERSITY",
    "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE",
    "EVIDENCE_INSUFFICIENT_LOW_COVERAGE",
    "EVIDENCE_ACCEPTABLE_DESPITE_GAPS",
)


def validate_selected_action(value: str) -> str:
    if value not in AGENTIC_RETRIEVAL_ACTIONS:
        raise ValueError(
            f"selected_action debe ser una de {AGENTIC_RETRIEVAL_ACTIONS}, recibido {value!r}."
        )
    return value


def validate_decision_basis(value: str) -> str:
    if value not in AGENTIC_DECISION_BASIS_VALUES:
        raise ValueError(
            f"decision_basis debe ser uno de {AGENTIC_DECISION_BASIS_VALUES}, recibido {value!r}."
        )
    return value


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgenticRetrievalObservation:
    """Estado real observable del ciclo de recuperación para UN claim,
    en un momento dado. Todos los campos derivan de resultados reales
    de RETRIEVE/GRADE_EVIDENCE (Bloque 1) -- no hay estados
    inventados."""

    claim_id: str
    claim_text: str
    current_query: str
    retrieval_round: int  # 0 = resultado de la recuperación inicial
    current_top_k: int
    effective_top_k_max: int
    remaining_retrieval_budget: int
    candidate_count: int
    # evidence_ids representa TODOS los candidatos materializados en esta
    # ronda -- el mismo conjunto que candidate_count cuenta en
    # grade_evidence (Bloque 1), no un subconjunto. Invariante obligatoria:
    # candidate_count == len(evidence_ids) (ver __post_init__).
    evidence_ids: tuple[str, ...]
    max_relevance_score: float
    grade_result: str  # "SUFFICIENT" | "INSUFFICIENT"
    reason_codes: tuple[str, ...]
    minimum_viable_evidence: bool
    query_rewrite_count: int

    def __post_init__(self) -> None:
        # --- Tipos estrictos: bool es subclase de int en Python ---
        for int_field_name, int_field_value in (
            ("retrieval_round", self.retrieval_round),
            ("current_top_k", self.current_top_k),
            ("effective_top_k_max", self.effective_top_k_max),
            ("remaining_retrieval_budget", self.remaining_retrieval_budget),
            ("candidate_count", self.candidate_count),
            ("query_rewrite_count", self.query_rewrite_count),
        ):
            if isinstance(int_field_value, bool) or not isinstance(int_field_value, int):
                raise TypeError(
                    f"{int_field_name} debe ser int, recibido "
                    f"{type(int_field_value).__name__} ({int_field_value!r})."
                )

        if self.retrieval_round < 0:
            raise ValueError(f"retrieval_round debe ser >= 0, recibido {self.retrieval_round!r}")
        if self.remaining_retrieval_budget < 0:
            raise ValueError(
                f"remaining_retrieval_budget debe ser >= 0, recibido {self.remaining_retrieval_budget!r}"
            )
        if self.candidate_count < 0:
            raise ValueError(f"candidate_count debe ser >= 0, recibido {self.candidate_count!r}")
        if self.current_top_k <= 0:
            raise ValueError(f"current_top_k debe ser > 0, recibido {self.current_top_k!r}")
        if self.effective_top_k_max <= 0:
            raise ValueError(f"effective_top_k_max debe ser > 0, recibido {self.effective_top_k_max!r}")
        if self.query_rewrite_count < 0:
            raise ValueError(f"query_rewrite_count debe ser >= 0, recibido {self.query_rewrite_count!r}")
        if self.current_top_k > self.effective_top_k_max:
            raise ValueError(
                f"current_top_k ({self.current_top_k}) no puede exceder "
                f"effective_top_k_max ({self.effective_top_k_max})."
            )
        if self.query_rewrite_count > self.retrieval_round:
            raise ValueError(
                f"query_rewrite_count ({self.query_rewrite_count}) no puede exceder "
                f"retrieval_round ({self.retrieval_round}) -- cada REWRITE_QUERY incrementa "
                "ambos en 1, cada ADJUST_TOP_K solo incrementa retrieval_round; "
                "query_rewrite_count > retrieval_round es un estado imposible según las "
                "transiciones del controller."
            )
        if self.candidate_count > self.current_top_k:
            raise ValueError(
                f"candidate_count ({self.candidate_count}) no puede exceder current_top_k "
                f"({self.current_top_k}) -- confirmado contra Agent07ChromaRetriever."
                "retrieve_more (verification_incremental_retriever.py): "
                "'if len(selected) >= self.top_k: break' trunca siempre el conjunto que "
                "alimenta grade_evidence a top_k. No confundir con effective_top_k_max/"
                "fetch_k, que acotan la petición bruta a Chroma, no el pool materializado."
            )

        if (
            not isinstance(self.max_relevance_score, (int, float))
            or isinstance(self.max_relevance_score, bool)
            or not (0.0 <= float(self.max_relevance_score) <= 1.0)
        ):
            raise ValueError(
                f"max_relevance_score debe estar en [0.0, 1.0], recibido {self.max_relevance_score!r}."
            )

        # Tipado estricto: claim_id/claim_text/current_query deben ser str
        # reales, no cualquier valor coaccionable con str(...) -- bool/int/
        # float/None/listas se rechazan explícitamente. Especialmente
        # importante para current_query, que Bloque 3 modificará de verdad
        # (REWRITE_QUERY).
        for str_field_name, str_field_value in (
            ("claim_id", self.claim_id),
            ("claim_text", self.claim_text),
            ("current_query", self.current_query),
        ):
            if not isinstance(str_field_value, str):
                raise TypeError(
                    f"{str_field_name} debe ser str real, recibido "
                    f"{type(str_field_value).__name__} ({str_field_value!r})."
                )
            if not str_field_value.strip():
                raise ValueError(f"{str_field_name} no puede estar vacío.")

        # Si nunca hubo REWRITE_QUERY, la query debe seguir siendo el
        # claim original -- ninguna transición válida (initial retrieval,
        # ADJUST_TOP_K) modifica current_query sin pasar por REWRITE_QUERY,
        # que es lo único que incrementa query_rewrite_count. No se impone
        # la inversa (query_rewrite_count > 0 no implica current_query !=
        # claim_text -- una reescritura podría, en teoría, converger de
        # nuevo al texto original).
        if self.query_rewrite_count == 0 and self.current_query != self.claim_text:
            raise ValueError(
                "query_rewrite_count=0 implica current_query == claim_text -- "
                f"recibido current_query={self.current_query!r} distinto de "
                f"claim_text={self.claim_text!r}. Ninguna transición válida puede "
                "producir este estado sin haber pasado por REWRITE_QUERY."
            )

        if self.grade_result not in ("SUFFICIENT", "INSUFFICIENT"):
            raise ValueError(f"grade_result inválido: {self.grade_result!r}")

        # reason_codes endurecido con el mismo rigor que evidence_ids:
        # tuple real (no list -- una lista mutable dentro de una dataclass
        # frozen=True rompe la expectativa de inmutabilidad del estado),
        # cada elemento str real, perteneciente a GRADE_REASON_CODES
        # (Bloque 1, sin duplicar vocabulario), sin duplicados.
        if not isinstance(self.reason_codes, tuple):
            raise TypeError(
                f"reason_codes debe ser tuple real, recibido {type(self.reason_codes).__name__} "
                f"({self.reason_codes!r})."
            )
        for code in self.reason_codes:
            if not isinstance(code, str):
                raise TypeError(
                    f"reason_codes contiene un elemento no-str: {code!r} ({type(code).__name__})."
                )
            if code not in GRADE_REASON_CODES:
                raise ValueError(
                    f"reason_codes contiene {code!r}, fuera de GRADE_REASON_CODES {GRADE_REASON_CODES} "
                    "(Bloque 1) -- se reutiliza el vocabulario del grader, no se duplica."
                )
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError(
                f"reason_codes contiene duplicados: {self.reason_codes!r} -- cada reason_code "
                "debe aparecer como máximo una vez."
            )

        # Coherencia con el contrato real del grader de Bloque 1
        # (confirmado: grade_result = "INSUFFICIENT" if reason_codes else
        # "SUFFICIENT" -- no existe caso legítimo de excepción):
        if self.grade_result == "SUFFICIENT" and self.reason_codes:
            raise ValueError(
                "grade_result=SUFFICIENT debe tener reason_codes vacío -- "
                f"recibido {self.reason_codes!r}."
            )
        if self.grade_result == "INSUFFICIENT" and not self.reason_codes:
            raise ValueError(
                "grade_result=INSUFFICIENT debe tener al menos un reason_code -- "
                "recibido reason_codes vacío."
            )

        if not isinstance(self.minimum_viable_evidence, bool):
            raise TypeError(
                f"minimum_viable_evidence debe ser bool real, recibido "
                f"{type(self.minimum_viable_evidence).__name__} ({self.minimum_viable_evidence!r}) -- "
                "un str/int truthy podría producir ACCEPT_EVIDENCE indebidamente al agotarse "
                "el presupuesto."
            )

        # SUFFICIENT es un criterio más estricto que minimum_viable_evidence
        # por diseño (Bloque 1: min_relevance_score del grader > el de
        # minimum_viable) -- SUFFICIENT + minimum_viable_evidence=False es
        # un estado imposible: si la evidencia ya pasó el criterio más
        # estricto, necesariamente pasa el más laxo.
        if self.grade_result == "SUFFICIENT" and self.minimum_viable_evidence is not True:
            raise ValueError(
                "grade_result=SUFFICIENT implica minimum_viable_evidence=True -- "
                f"recibido minimum_viable_evidence={self.minimum_viable_evidence!r}. "
                "SUFFICIENT es un criterio más estricto que minimum_viable_evidence por "
                "diseño; esta combinación es un estado imposible."
            )

        # Endurecimiento del contrato de evidence_ids -- será esencial para
        # trazabilidad en Bloque 3: tuple real (no list), cada elemento str
        # real no vacío, todos únicos.
        if not isinstance(self.evidence_ids, tuple):
            raise TypeError(
                f"evidence_ids debe ser tuple real, recibido {type(self.evidence_ids).__name__} "
                f"({self.evidence_ids!r})."
            )
        for evidence_id in self.evidence_ids:
            if not isinstance(evidence_id, str):
                raise TypeError(
                    f"evidence_ids contiene un elemento no-str: {evidence_id!r} "
                    f"({type(evidence_id).__name__})."
                )
            if not evidence_id.strip():
                raise ValueError("evidence_ids no puede contener IDs vacíos.")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError(
                f"evidence_ids contiene IDs duplicados: {self.evidence_ids!r} -- "
                "todos los IDs deben ser únicos."
            )

        # evidence_ids representa TODOS los candidatos materializados en
        # esta ronda (mismo conjunto que candidate_count cuenta en
        # grade_evidence, Bloque 1) -- no un subconjunto. Ambos deben
        # coincidir en cardinalidad.
        if self.candidate_count != len(self.evidence_ids):
            raise ValueError(
                f"candidate_count ({self.candidate_count}) debe ser igual a "
                f"len(evidence_ids) ({len(self.evidence_ids)}) -- evidence_ids representa "
                "todos los candidatos materializados en esta ronda, el mismo conjunto que "
                "candidate_count cuenta (Bloque 1, grade_evidence)."
            )

        # minimum_viable_evidence=True requiere evidencia real -- no puede
        # afirmarse viabilidad sobre un conjunto vacío. Por consecuencia
        # (SUFFICIENT ya implica minimum_viable_evidence=True, arriba),
        # SUFFICIENT también implica candidate_count >= 1 -- ambos chequeos
        # explícitos para fail-closed directo y mensajes claros.
        if self.minimum_viable_evidence and self.candidate_count < 1:
            raise ValueError(
                "minimum_viable_evidence=True requiere candidate_count >= 1 -- "
                f"recibido candidate_count={self.candidate_count!r}. No puede haber "
                "evidencia mínimamente viable sobre un conjunto vacío."
            )
        if self.grade_result == "SUFFICIENT" and self.candidate_count < 1:
            raise ValueError(
                "grade_result=SUFFICIENT requiere candidate_count >= 1 -- "
                f"recibido candidate_count={self.candidate_count!r}."
            )

        # candidate_count/evidence_ids/max_relevance_score describen el
        # mismo resultado materializado del retrieval -- confirmado contra
        # el grader real de Bloque 1 (grade_evidence: max_relevance_score
        # = max(scores) if scores else 0.0): si no hay candidatos, el
        # score máximo es 0.0, nunca positivo. No se impone la inversa
        # (max_relevance_score=0.0 no implica candidate_count=0 --
        # candidatos con relevancia 0 son posibles).
        if self.candidate_count == 0 and self.max_relevance_score != 0.0:
            raise ValueError(
                f"candidate_count=0 requiere max_relevance_score=0.0 -- recibido "
                f"{self.max_relevance_score!r}. Sin candidatos no puede existir un score "
                "máximo positivo procedente de ellos (Bloque 1, grade_evidence)."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "current_query": self.current_query,
            "retrieval_round": self.retrieval_round,
            "current_top_k": self.current_top_k,
            "effective_top_k_max": self.effective_top_k_max,
            "remaining_retrieval_budget": self.remaining_retrieval_budget,
            "candidate_count": self.candidate_count,
            "evidence_ids": list(self.evidence_ids),
            "max_relevance_score": self.max_relevance_score,
            "grade_result": self.grade_result,
            "reason_codes": list(self.reason_codes),
            "minimum_viable_evidence": self.minimum_viable_evidence,
            "query_rewrite_count": self.query_rewrite_count,
        }


# ---------------------------------------------------------------------------
# Gate de decisión -- único punto ReAct genuino del sistema auditado.
# ---------------------------------------------------------------------------


def determine_forced_outcome(observation: AgenticRetrievalObservation) -> str | None:
    """Casos que el controller resuelve SIN consultar al planner:
    - SUFFICIENT: ACCEPT_EVIDENCE automático (no hay decisión que tomar).
    - Presupuesto agotado (INSUFFICIENT): ACCEPT_EVIDENCE si
      minimum_viable_evidence, FINISH_UNRESOLVED si no -- NUNCA
      convertido en una elección del planner."""
    if observation.grade_result == "SUFFICIENT":
        return "ACCEPT_EVIDENCE"
    if observation.remaining_retrieval_budget <= 0:
        return "ACCEPT_EVIDENCE" if observation.minimum_viable_evidence else FINISH_UNRESOLVED
    return None


def compute_allowed_actions(observation: AgenticRetrievalObservation) -> tuple[str, ...]:
    """Gate planner-seleccionable. Llamar SOLO después de confirmar que
    ``determine_forced_outcome`` devolvió ``None`` (evidencia
    insuficiente, presupuesto disponible).

    - ``ADJUST_TOP_K`` solo si ``current_top_k < effective_top_k_max``
      (restricción estructural real, no artificial).
    - ``REWRITE_QUERY`` siempre disponible mientras haya presupuesto
      (sin cap independiente -- comparte el mismo presupuesto).
    - ``ACCEPT_EVIDENCE`` NUNCA en la primera insuficiencia
      (``retrieval_round == 0``) -- solo tras al menos un intento de
      mejora (``retrieval_round >= 1``), y solo si
      ``minimum_viable_evidence``.
    """
    if observation.grade_result == "SUFFICIENT" or observation.remaining_retrieval_budget <= 0:
        return ()

    actions: list[str] = []
    if observation.current_top_k < observation.effective_top_k_max:
        actions.append("ADJUST_TOP_K")
    actions.append("REWRITE_QUERY")

    if observation.retrieval_round >= 1 and observation.minimum_viable_evidence:
        actions.append("ACCEPT_EVIDENCE")

    return tuple(actions)


# ---------------------------------------------------------------------------
# Planner -- prompt mínimo + parseo fail-closed, autocontenido.
# ---------------------------------------------------------------------------


class AgenticPlannerResponseError(ValueError):
    """La respuesta del planner no pudo interpretarse como una decisión
    válida tras agotar los reintentos de parseo."""


MAX_PLANNER_PARSE_RETRIES = 2


def build_agentic_planner_prompt(
    *, observation: AgenticRetrievalObservation, allowed_actions: tuple[str, ...]
) -> str:
    if not allowed_actions:
        raise ValueError(
            "build_agentic_planner_prompt requiere allowed_actions no vacío -- "
            "si no hay ninguna acción autorizada, el controller no debe invocar "
            "al planner (cierre determinista vía determine_forced_outcome)."
        )
    for action in allowed_actions:
        validate_selected_action(action)

    return (
        "Eres el planner de recuperación agentic para UN claim científico, "
        "Stage 07 (pre-verificación).\n"
        "Debes elegir EXACTAMENTE UNA acción de la lista permitida. No "
        "expliques tu razonamiento en texto libre.\n\n"
        f"OBSERVATION:\n{json.dumps(observation.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        f"ACCIONES PERMITIDAS (elige solo una, ninguna otra es válida):\n"
        f"{json.dumps(list(allowed_actions), ensure_ascii=False)}\n\n"
        f"VALORES VÁLIDOS DE decision_basis:\n"
        f"{json.dumps(list(AGENTIC_DECISION_BASIS_VALUES), ensure_ascii=False)}\n\n"
        "Responde ÚNICAMENTE con un objeto JSON de exactamente estas dos claves, "
        "sin texto adicional, sin markdown:\n"
        '{"selected_action": "<una de las acciones permitidas>", '
        '"decision_basis": "<uno de los valores válidos>"}'
    )


def _validate_action_decision_basis_coherence(
    *, selected_action: str, decision_basis: str, observation_reason_codes: tuple[str, ...]
) -> None:
    """Coherencia obligatoria entre la acción elegida y su justificación
    -- ambos enums ya se validaron por separado, esto valida la
    COMBINACIÓN:
    - ACCEPT_EVIDENCE únicamente con EVIDENCE_ACCEPTABLE_DESPITE_GAPS.
    - REWRITE_QUERY/ADJUST_TOP_K (acciones de mejora) únicamente con un
      decision_basis cuyo reason code correspondiente esté REALMENTE
      presente en observation_reason_codes -- no basta con que el
      decision_basis pertenezca al enum general."""
    if selected_action == "ACCEPT_EVIDENCE":
        if decision_basis != "EVIDENCE_ACCEPTABLE_DESPITE_GAPS":
            raise AgenticPlannerResponseError(
                f"PLANNER_RESPONSE_INCOHERENT_DECISION_BASIS: ACCEPT_EVIDENCE solo admite "
                f"decision_basis='EVIDENCE_ACCEPTABLE_DESPITE_GAPS', recibido {decision_basis!r}."
            )
        return

    # REWRITE_QUERY / ADJUST_TOP_K (acciones de mejora)
    if decision_basis == "EVIDENCE_ACCEPTABLE_DESPITE_GAPS":
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_INCOHERENT_DECISION_BASIS: {selected_action!r} no puede "
            "justificarse con 'EVIDENCE_ACCEPTABLE_DESPITE_GAPS' -- esa justificación es "
            "exclusiva de ACCEPT_EVIDENCE."
        )
    corresponding_reason_code = decision_basis.removeprefix("EVIDENCE_INSUFFICIENT_")
    if corresponding_reason_code not in observation_reason_codes:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_INCOHERENT_DECISION_BASIS: decision_basis={decision_basis!r} "
            f"implica reason_code={corresponding_reason_code!r}, que no está presente en "
            f"la Observation actual (reason_codes={observation_reason_codes!r})."
        )


def parse_agentic_planner_response(
    raw_text: str, *, allowed_actions: tuple[str, ...], observation_reason_codes: tuple[str, ...] = (),
) -> dict[str, str]:
    text = (raw_text or "").strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise AgenticPlannerResponseError(
            "PLANNER_RESPONSE_NOT_PURE_JSON_OBJECT: debe ser exactamente un objeto JSON."
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgenticPlannerResponseError(f"PLANNER_RESPONSE_INVALID_JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise AgenticPlannerResponseError("PLANNER_RESPONSE_ROOT_NOT_OBJECT")

    expected_keys = {"selected_action", "decision_basis"}
    if set(payload.keys()) != expected_keys:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_UNEXPECTED_KEYS: se esperaban exactamente "
            f"{sorted(expected_keys)}, se recibió {sorted(payload.keys())} -- "
            "no se acepta 'rationale' ni ningún otro campo."
        )

    selected_action = payload["selected_action"]
    decision_basis = payload["decision_basis"]

    if not isinstance(selected_action, str) or selected_action not in AGENTIC_RETRIEVAL_ACTIONS:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_INVALID_SELECTED_ACTION: {selected_action!r} "
            "no pertenece a AGENTIC_RETRIEVAL_ACTIONS."
        )
    if selected_action not in allowed_actions:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_ACTION_NOT_ALLOWED: {selected_action!r} no está "
            f"en el conjunto autorizado {allowed_actions!r}."
        )
    if not isinstance(decision_basis, str) or decision_basis not in AGENTIC_DECISION_BASIS_VALUES:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_INVALID_DECISION_BASIS: {decision_basis!r} "
            "no pertenece a AGENTIC_DECISION_BASIS_VALUES."
        )

    _validate_action_decision_basis_coherence(
        selected_action=selected_action, decision_basis=decision_basis,
        observation_reason_codes=observation_reason_codes,
    )

    return {"selected_action": selected_action, "decision_basis": decision_basis}


def invoke_agentic_planner_with_retry(
    *,
    invoke_fn: Callable[[str], str],
    prompt: str,
    allowed_actions: tuple[str, ...],
    observation_reason_codes: tuple[str, ...] = (),
    max_retries: int = MAX_PLANNER_PARSE_RETRIES,
) -> dict[str, str]:
    last_error: Exception | None = None
    attempts = max(1, max_retries + 1)
    for _ in range(attempts):
        raw = invoke_fn(prompt)
        try:
            return parse_agentic_planner_response(
                raw, allowed_actions=allowed_actions, observation_reason_codes=observation_reason_codes,
            )
        except AgenticPlannerResponseError as exc:
            last_error = exc
            continue
    raise AgenticPlannerResponseError(
        f"PLANNER_RESPONSE_RETRIES_EXHAUSTED tras {attempts} intentos: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Validación de la transición producida por execute_action_fn -- el
# presupuesto compartido es la garantía de terminación del ciclo, así
# que no basta con confiar en que la tool devuelva algo razonable.
# ---------------------------------------------------------------------------

AGENTIC_TRANSITION_INVALID = "AGENTIC_TRANSITION_INVALID"


class AgenticRetrievalActionUnavailable(Exception):
    """E2E-BUG-01 (contract fix): excepción tipada de integración -- la
    acción seleccionada era legal según ``compute_allowed_actions`` para
    la Observation actual, pero no puede ejecutarse con los datos
    concretos disponibles (ej. ``generate_query_rewrite`` sin
    vocabulario nuevo genuino que incorporar). NO significa fallo
    técnico global, claim unsupported, presupuesto agotado, transición
    inválida ni fallo del planner -- el executor (Bloque 4/runtime) es
    responsable de traducir la condición legítima específica
    (``QUERY_REWRITE_UNAVAILABLE``) a esta excepción; cualquier otro
    error debe seguir propagándose sin conversión."""


EXECUTION_STATUS_EXECUTED = "EXECUTED"
EXECUTION_STATUS_ACTION_UNAVAILABLE = "ACTION_UNAVAILABLE"
EXECUTION_STATUS_TERMINAL = "TERMINAL"


def _validate_improvement_transition(
    *, action: str, before: AgenticRetrievalObservation, after: AgenticRetrievalObservation
) -> None:
    """Fail-closed: para REWRITE_QUERY/ADJUST_TOP_K (las únicas
    acciones que ejecuta una tool real vía execute_action_fn -- nunca
    ACCEPT_EVIDENCE, que termina el ciclo antes de llegar aquí),
    verifica que la Observation resultante cumpla exactamente el
    contrato de esa acción -- incluidos los campos que NINGUNA acción
    de mejora puede tocar (identidad del claim y el tope estructural
    de top_k), no solo los que cada una sí modifica."""
    # Invariantes compartidos: ninguna acción de mejora puede alterar
    # la identidad del claim ni el tope estructural de top_k.
    if after.claim_id != before.claim_id:
        raise ValueError(
            f"{action}: claim_id_after ({after.claim_id!r}) debe ser igual a "
            f"claim_id_before ({before.claim_id!r})."
        )
    if after.claim_text != before.claim_text:
        raise ValueError(
            f"{action}: claim_text_after debe ser igual a claim_text_before -- "
            "ninguna acción de mejora reescribe el claim."
        )
    if after.effective_top_k_max != before.effective_top_k_max:
        raise ValueError(
            f"{action}: effective_top_k_max_after ({after.effective_top_k_max}) debe ser "
            f"igual a effective_top_k_max_before ({before.effective_top_k_max}) -- es un "
            "tope estructural fijo del ciclo, no algo que la tool pueda modificar."
        )

    if after.retrieval_round != before.retrieval_round + 1:
        raise ValueError(
            f"{action}: retrieval_round_after ({after.retrieval_round}) debe ser "
            f"retrieval_round_before + 1 ({before.retrieval_round + 1})."
        )
    if after.remaining_retrieval_budget != before.remaining_retrieval_budget - 1:
        raise ValueError(
            f"{action}: remaining_retrieval_budget_after ({after.remaining_retrieval_budget}) "
            f"debe ser remaining_retrieval_budget_before - 1 ({before.remaining_retrieval_budget - 1}) "
            "-- el presupuesto compartido es la garantía de terminación del ciclo."
        )

    if action == "REWRITE_QUERY":
        if after.current_query == before.current_query:
            raise ValueError("REWRITE_QUERY: current_query_after debe ser distinto de current_query_before.")
        if after.query_rewrite_count != before.query_rewrite_count + 1:
            raise ValueError(
                f"REWRITE_QUERY: query_rewrite_count_after ({after.query_rewrite_count}) debe ser "
                f"query_rewrite_count_before + 1 ({before.query_rewrite_count + 1})."
            )
        if after.current_top_k != before.current_top_k:
            raise ValueError(
                f"REWRITE_QUERY: current_top_k_after ({after.current_top_k}) debe ser igual a "
                f"current_top_k_before ({before.current_top_k}) -- REWRITE_QUERY no toca top_k."
            )
    elif action == "ADJUST_TOP_K":
        if after.current_query != before.current_query:
            raise ValueError(
                "ADJUST_TOP_K: current_query_after debe ser EXACTAMENTE igual a "
                "current_query_before -- ADJUST_TOP_K no toca la query."
            )
        if after.current_top_k <= before.current_top_k:
            raise ValueError(
                f"ADJUST_TOP_K: current_top_k_after ({after.current_top_k}) debe ser mayor que "
                f"current_top_k_before ({before.current_top_k})."
            )
        if after.current_top_k > before.effective_top_k_max:
            raise ValueError(
                f"ADJUST_TOP_K: current_top_k_after ({after.current_top_k}) no puede exceder "
                f"effective_top_k_max ({before.effective_top_k_max})."
            )
        if after.query_rewrite_count != before.query_rewrite_count:
            raise ValueError(
                f"ADJUST_TOP_K: query_rewrite_count_after ({after.query_rewrite_count}) debe ser "
                f"igual a query_rewrite_count_before ({before.query_rewrite_count}) -- "
                "ADJUST_TOP_K no toca la query."
            )
    else:
        raise ValueError(f"_validate_improvement_transition: acción inesperada {action!r}.")


def _infer_deterministic_decision_basis(observation: AgenticRetrievalObservation) -> str:
    """Cuando solo hay 1 acción disponible, Python ejecuta directamente
    sin consultar al planner -- pero el step sigue registrando un
    decision_basis, derivado determinísticamente del primer reason_code
    presente en la Observation (mismo vocabulario, prefijo
    "EVIDENCE_INSUFFICIENT_", reutilizado de Bloque 1 sin duplicar)."""
    if not observation.reason_codes:
        raise ValueError(
            "No se puede inferir decision_basis determinista sin reason_codes -- "
            "esto no debería ocurrir para grade_result=INSUFFICIENT (invariante ya "
            "validada en __post_init__)."
        )
    return "EVIDENCE_INSUFFICIENT_" + observation.reason_codes[0]


# ---------------------------------------------------------------------------
# Ciclo
# ---------------------------------------------------------------------------


@dataclass
class AgenticRetrievalResult:
    claim_id: str
    outcome: str  # "ACCEPT_EVIDENCE" | FINISH_UNRESOLVED | AGENTIC_PLANNER_FAILED | AGENTIC_TRANSITION_INVALID
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_observation: AgenticRetrievalObservation | None = None


def run_agentic_retrieval_cycle(
    *,
    initial_observation: AgenticRetrievalObservation,
    invoke_planner_fn: Callable[[str], str],
    execute_action_fn: Callable[[str, str, AgenticRetrievalObservation], AgenticRetrievalObservation],
) -> AgenticRetrievalResult:
    """Ejecuta el ciclo desde ``initial_observation`` (resultado ya
    materializado de RETRIEVE inicial + GRADE_EVIDENCE) hasta
    ACCEPT_EVIDENCE/FINISH_UNRESOLVED/AGENTIC_PLANNER_FAILED/
    AGENTIC_TRANSITION_INVALID.

    El planner LLM SOLO se invoca cuando ``compute_allowed_actions``
    devuelve 2 o más acciones -- con exactamente 1 acción disponible,
    Python la ejecuta directamente (``decision_basis`` derivado
    determinísticamente); con 0, ``determine_forced_outcome`` ya cerró
    el ciclo antes de llegar aquí.

    ``execute_action_fn(selected_action, decision_basis, observation)
    -> nueva Observation`` es la tool real (inyectada) que ejecuta
    REWRITE_QUERY/ADJUST_TOP_K (incluye el RETRIEVE + GRADE_EVIDENCE
    subsiguientes, ambos deterministas). ``decision_basis`` se pasa
    EXACTAMENTE tal como se resolvió para este step (respuesta
    validada real del planner cuando lo hubo, o el valor determinista
    ya calculado por Python cuando el planner no fue consultado -- ver
    ``_infer_deterministic_decision_basis``) -- CONTRATO AMPLIADO en
    Bloque 4: antes esta firma solo recibía ``(selected_action,
    observation)``, perdiendo qué reason_code específico motivó la
    decisión cuando ``observation.reason_codes`` tenía más de un
    elemento. Sin este dato, un executor no puede derivar
    ``rewrite_reason`` (Bloque 3) con fidelidad al planner real -- solo
    podría aproximarlo (ej. tomar el primer reason_code), perdiendo
    trazabilidad semántica. Su resultado se valida contra el contrato
    exacto de la acción (``_validate_improvement_transition``) antes de
    continuar -- fail-closed si la tool no respeta el presupuesto
    compartido o el contrato de la acción."""
    observation = initial_observation
    steps: list[dict[str, Any]] = []
    # E2E-BUG-01: exclusiones LOCALES a la Observation actual -- una
    # acción legal según compute_allowed_actions puede resultar no
    # ejecutable con los datos concretos disponibles (ver
    # AgenticRetrievalActionUnavailable). Se reinicia en cuanto se
    # obtiene una nueva Observation real (evidencia distinta puede
    # volver viable una acción que antes no lo era).
    unavailable_actions_for_current_observation: set[str] = set()

    while True:
        forced_outcome = determine_forced_outcome(observation)
        if forced_outcome is not None:
            return AgenticRetrievalResult(
                claim_id=observation.claim_id, outcome=forced_outcome, steps=steps, final_observation=observation,
            )

        allowed_actions = tuple(
            a for a in compute_allowed_actions(observation)
            if a not in unavailable_actions_for_current_observation
        )
        if not allowed_actions:
            # 0 acciones efectivas -- ya sea porque compute_allowed_actions
            # no ofrecía ninguna (salvaguarda, no debería alcanzarse dado
            # determine_forced_outcome) o porque todas las legales
            # resultaron ACTION_UNAVAILABLE para esta Observation.
            return AgenticRetrievalResult(
                claim_id=observation.claim_id, outcome=FINISH_UNRESOLVED, steps=steps, final_observation=observation,
            )

        if len(allowed_actions) == 1:
            # Una sola acción efectiva -- Python ya sabe qué hacer, el
            # planner NUNCA se invoca (ni siquiera una segunda vez tras
            # descartar una acción no ejecutable).
            selected_action = allowed_actions[0]
            decision_basis = _infer_deterministic_decision_basis(observation)
            planner_invoked = False
        else:
            # El prompt se construye con allowed_actions_effective -- el
            # planner nunca puede volver a elegir una acción ya marcada
            # no ejecutable para esta misma Observation.
            prompt = build_agentic_planner_prompt(observation=observation, allowed_actions=allowed_actions)
            try:
                decision = invoke_agentic_planner_with_retry(
                    invoke_fn=invoke_planner_fn, prompt=prompt, allowed_actions=allowed_actions,
                    observation_reason_codes=observation.reason_codes,
                )
            except AgenticPlannerResponseError:
                return AgenticRetrievalResult(
                    claim_id=observation.claim_id, outcome=AGENTIC_PLANNER_FAILED, steps=steps, final_observation=observation,
                )
            selected_action = decision["selected_action"]
            decision_basis = decision["decision_basis"]
            planner_invoked = True

        step_number = len(steps) + 1

        if selected_action == "ACCEPT_EVIDENCE":
            steps.append({
                "step_number": step_number,
                "selected_action": selected_action,
                "decision_basis": decision_basis,
                "planner_invoked": planner_invoked,
                "execution_status": EXECUTION_STATUS_TERMINAL,
            })
            return AgenticRetrievalResult(
                claim_id=observation.claim_id, outcome="ACCEPT_EVIDENCE", steps=steps, final_observation=observation,
            )

        observation_before = observation
        try:
            observation_after = execute_action_fn(selected_action, decision_basis, observation_before)
        except AgenticRetrievalActionUnavailable as exc:
            # E2E-DIAG-02: instrumentación temporal, solo diagnóstico --
            # no cambia lógica ni contratos, no captura nada nuevo.
            print(
                "AGENTIC_ACTION_UNAVAILABLE_CAUGHT",
                type(exc).__module__,
                type(exc).__qualname__,
                repr(exc),
            )
            # La acción era legal para esta Observation, pero no pudo
            # ejecutarse con los datos concretos disponibles -- NO
            # consume budget/round (no hubo retrieval, no hay nueva
            # Observation), NO aparece como retrieval_transition
            # (Bloque 6 -- nunca se llega a construir una), pero SÍ
            # queda auditada en decision_steps. Se excluye SOLO para
            # esta Observation -- el bucle vuelve a evaluar
            # allowed_actions_effective sin ella, sin avanzar
            # observation, sin volver a intentarla indefinidamente
            # (queda en unavailable_actions_for_current_observation
            # hasta la próxima Observation real).
            steps.append({
                "step_number": step_number,
                "selected_action": selected_action,
                "decision_basis": decision_basis,
                "planner_invoked": planner_invoked,
                "execution_status": EXECUTION_STATUS_ACTION_UNAVAILABLE,
            })
            unavailable_actions_for_current_observation.add(selected_action)
            continue

        steps.append({
            "step_number": step_number,
            "selected_action": selected_action,
            "decision_basis": decision_basis,
            "planner_invoked": planner_invoked,
            "execution_status": EXECUTION_STATUS_EXECUTED,
        })

        try:
            _validate_improvement_transition(action=selected_action, before=observation_before, after=observation_after)
        except ValueError:
            return AgenticRetrievalResult(
                claim_id=observation_before.claim_id, outcome=AGENTIC_TRANSITION_INVALID,
                steps=steps, final_observation=observation_before,
            )
        observation = observation_after
        unavailable_actions_for_current_observation = set()
