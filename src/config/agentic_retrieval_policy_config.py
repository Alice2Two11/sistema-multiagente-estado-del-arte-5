"""Agentic Retrieval (Stage 07, pre-verificación) -- Bloque 1:
configuración propia.

Contratos independientes del ReAct post-verificación descartado
(``react_policy_config.py``) -- no se arrastra ningún enum de ese
dominio (``VERIFY_CURRENT_EVIDENCE``, ``PROPOSE_CORRECTION``,
``FINISH_RESOLVED``, etc.). Auditoría confirmada: ``react_prompting.py``
importa y valida directamente contra ``REACT_ACTIONS``/
``DECISION_BASIS_VALUES`` de ese módulo -- no es genérico tal como está
escrito, así que este dominio nuevo no lo reutiliza sin cambios.

Ruta real de Stage 07 reconstruida y confirmada (evidencia directa del
código, no inferencia):

    Chroma.query(n_results=fetch_k)
        -> scoring (1-distance, fused_rrf_score) + dedupe + filtro de
           fuente autorizada
        -> corte real del retriever: len(selected) >= top_k -> break
        -> select_evidence_for_scientific_judgment
           (trunca de verdad aquí: max_llm_evidence_chunks_per_claim,
           max_llm_evidence_per_source, max_contrast_evidence_chunks,
           max_total_evidence_chars)
        -> máximo max_llm_evidence_chunks_per_claim (hoy 8) al LLM de
           verificación

``effective_top_k_max = fetch_k`` (hoy 35) -- confirmado como el único
límite real que acota cuántos candidatos entran al pool antes del
corte por ``top_k``. Los campos ``max_total_candidates_per_claim``/
``max_final_evidence_per_claim``/``max_candidates_per_source`` de
``verification_policy_config.py`` tienen CERO consumidores en todo
``src/`` (confirmado por grep exhaustivo) -- son configuración
vestigial sin efecto en el flujo real, no caps activos. No se cuentan
como restricción de ``ADJUST_TOP_K``.

IMPORTANTE: los thresholds de este módulo son PARAMETRIZABLES, no
valores científicamente definitivos -- quedan pendientes de
calibración empírica en un conjunto de desarrollo separado del
conjunto de evaluación final, antes de congelarse para cualquier
corrida experimental comparativa (BASELINE vs AGENTIC_RAG).

Este bloque NO toca verification_runtime.py, verification_agent.py, ni
el retriever -- solo configuración y (en el módulo hermano
``agentic_retrieval_grader.py``) el grader determinista puro."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GRADE_EVIDENCE: solo SUFFICIENT/INSUFFICIENT -- CONTRADICTORY queda
# exclusivamente en VerificationAgent.verify_claim(), que sí tiene el
# contrato/campos para determinarlo (contradiction_type,
# contradiction_evidence_ids). El grader pre-verificación no tiene
# fundamento en sus señales (candidate_count, diversidad, similitud,
# cobertura léxica) para juzgar contradicción científica.
# ---------------------------------------------------------------------------

GRADE_RESULT_VALUES = ("SUFFICIENT", "INSUFFICIENT")

# Reason codes del grader -- auditados contra los campos REALMENTE
# disponibles en la salida de retrieve_more (Agent07ChromaRetriever):
# source_filename, chunk_id, text, score (1-distance). Ninguno requiere
# metadata nueva ni cambios en Chroma/embeddings.
GRADE_REASON_CODES = (
    "LOW_CANDIDATE_COUNT",      # candidate_count < umbral_mínimo_conteo
    "LOW_SOURCE_DIVERSITY",     # distinct(source_filename) < umbral_mínimo_diversidad
    "LOW_RELEVANCE",            # max(score) < umbral_mínimo_relevancia
    "LOW_COVERAGE",             # overlap léxico(claim_text, candidatos) < umbral_mínimo_cobertura
)


def validate_grade_result(value: str) -> str:
    if value not in GRADE_RESULT_VALUES:
        raise ValueError(
            f"grade_result debe ser uno de {GRADE_RESULT_VALUES}, recibido {value!r}."
        )
    return value


def validate_grade_reason_code(value: str) -> str:
    if value not in GRADE_REASON_CODES:
        raise ValueError(
            f"reason code de grade debe ser uno de {GRADE_REASON_CODES}, recibido {value!r}."
        )
    return value


# ---------------------------------------------------------------------------
# Thresholds del grader -- PARAMETRIZABLES, pendientes de calibración
# empírica (ver docstring del módulo). Los valores por defecto aquí son
# de arranque razonable, no resultado de ningún experimento todavía.
# ---------------------------------------------------------------------------

DEFAULT_GRADER_THRESHOLDS = {
    "min_candidate_count": 2,
    "min_source_diversity": 2,       # solo exigible si candidate_count >= min_candidate_count_for_diversity_check
    "min_candidate_count_for_diversity_check": 4,
    "min_relevance_score": 0.3,      # sobre 1-distance (similitud coseno), rango [0, 1]
    "min_lexical_overlap_ratio": 0.15,  # fracción de términos del claim presentes en algún candidato
}


def validate_grader_thresholds(thresholds: dict) -> dict:
    """Fail-closed: exige exactamente las 5 claves definidas arriba, con
    tipos y rangos razonables. No acepta thresholds parciales ni
    valores fuera de un rango sensato -- evita configuraciones
    silenciosamente inválidas (ej. min_relevance_score=5, fuera de
    [0,1])."""
    expected_keys = set(DEFAULT_GRADER_THRESHOLDS.keys())
    if set(thresholds.keys()) != expected_keys:
        raise ValueError(
            f"grader thresholds debe tener exactamente las claves {sorted(expected_keys)}, "
            f"recibido {sorted(thresholds.keys())}."
        )
    for int_key in ("min_candidate_count", "min_source_diversity", "min_candidate_count_for_diversity_check"):
        value = thresholds[int_key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"{int_key} debe ser un entero >= 1, recibido {value!r} -- "
                "un umbral en 0 permitiría que evidencia vacía se marque SUFFICIENT."
            )
    if thresholds["min_candidate_count_for_diversity_check"] < thresholds["min_candidate_count"]:
        raise ValueError(
            "min_candidate_count_for_diversity_check "
            f"({thresholds['min_candidate_count_for_diversity_check']}) no puede ser menor que "
            f"min_candidate_count ({thresholds['min_candidate_count']}) -- el chequeo de "
            "diversidad no puede activarse con menos candidatos de los que ya se exigen "
            "como mínimo absoluto."
        )
    for ratio_key in ("min_relevance_score", "min_lexical_overlap_ratio"):
        value = thresholds[ratio_key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{ratio_key} debe estar en [0.0, 1.0], recibido {value!r}.")
    return dict(thresholds)


# ---------------------------------------------------------------------------
# minimum_viable_evidence -- definición objetiva, más laxa que SUFFICIENT
# del grader: "hay al menos un candidato mínimamente plausible", no
# "hay evidencia buena para juzgar con confianza".
# ---------------------------------------------------------------------------

DEFAULT_MINIMUM_VIABLE_THRESHOLDS = {
    "min_candidate_count": 1,
    "min_relevance_score": 0.15,  # deliberadamente más laxo que min_relevance_score del grader (0.3)
}


def validate_minimum_viable_thresholds(thresholds: dict) -> dict:
    expected_keys = set(DEFAULT_MINIMUM_VIABLE_THRESHOLDS.keys())
    if set(thresholds.keys()) != expected_keys:
        raise ValueError(
            f"minimum_viable thresholds debe tener exactamente las claves {sorted(expected_keys)}, "
            f"recibido {sorted(thresholds.keys())}."
        )
    value = thresholds["min_candidate_count"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"min_candidate_count debe ser un entero >= 1, recibido {value!r}.")
    value = thresholds["min_relevance_score"]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"min_relevance_score debe estar en [0.0, 1.0], recibido {value!r}.")
    return dict(thresholds)


def validate_threshold_coherence(*, grader_thresholds: dict, minimum_viable_thresholds: dict) -> None:
    """Garantiza que ``minimum_viable_evidence`` sea semánticamente MÁS
    LAXO o igual que ``SUFFICIENT`` del grader, nunca más estricto --
    si no se cumpliera, un claim podría marcarse "viable para aceptar"
    con un criterio más exigente que el que el propio grader usa para
    decir "esto es bueno", lo cual invertiría el sentido del diseño.

    Se llama independientemente de ``validate_grader_thresholds``/
    ``validate_minimum_viable_thresholds`` (que ya validan cada dict
    por separado) -- esta es la validación CRUZADA entre ambos."""
    grader_thresholds = validate_grader_thresholds(grader_thresholds)
    minimum_viable_thresholds = validate_minimum_viable_thresholds(minimum_viable_thresholds)

    if minimum_viable_thresholds["min_candidate_count"] > grader_thresholds["min_candidate_count"]:
        raise ValueError(
            "minimum_viable.min_candidate_count "
            f"({minimum_viable_thresholds['min_candidate_count']}) no puede ser mayor que "
            f"grader.min_candidate_count ({grader_thresholds['min_candidate_count']}) -- "
            "minimum_viable_evidence debe ser más laxo o igual que SUFFICIENT, nunca más estricto."
        )
    if minimum_viable_thresholds["min_relevance_score"] > grader_thresholds["min_relevance_score"]:
        raise ValueError(
            "minimum_viable.min_relevance_score "
            f"({minimum_viable_thresholds['min_relevance_score']}) no puede ser mayor que "
            f"grader.min_relevance_score ({grader_thresholds['min_relevance_score']}) -- "
            "minimum_viable_evidence debe ser más laxo o igual que SUFFICIENT, nunca más estricto."
        )


# ---------------------------------------------------------------------------
# effective_top_k_max -- confirmado con evidencia directa (ver docstring
# del módulo): es fetch_k, el único cap real que acota candidatos antes
# del corte por top_k. NO es max_llm_evidence_chunks_per_claim (que es
# un límite posterior y distinto, sobre cuánta evidencia sobrevive a la
# selección final, no sobre el tamaño del pool de candidatos).
# ---------------------------------------------------------------------------


def validate_effective_top_k_max(fetch_k: int) -> int:
    if not isinstance(fetch_k, int) or isinstance(fetch_k, bool) or fetch_k < 1:
        raise ValueError(f"fetch_k debe ser un entero >= 1, recibido {fetch_k!r}.")
    return fetch_k


def next_top_k(*, current_top_k: int, effective_top_k_max: int, step_multiplier: float = 1.5) -> int:
    """Determina el siguiente valor de top_k para ADJUST_TOP_K --
    SIEMPRE decidido por Python, nunca por el planner directamente
    (el planner solo elige la ACCIÓN, no el valor). Garantiza
    next_top_k > current_top_k y next_top_k <= effective_top_k_max.

    Fail-closed y estrictamente tipado: ``bool`` es subclase de ``int``
    en Python -- se rechaza explícitamente, junto con cualquier tipo
    no numérico o valor fuera de rango, ANTES de la comparación de
    negocio (``current_top_k >= effective_top_k_max``)."""
    if isinstance(current_top_k, bool) or not isinstance(current_top_k, int):
        raise TypeError(f"current_top_k debe ser int, recibido {type(current_top_k).__name__} ({current_top_k!r}).")
    if current_top_k <= 0:
        raise ValueError(f"current_top_k debe ser > 0, recibido {current_top_k!r}.")
    if isinstance(effective_top_k_max, bool) or not isinstance(effective_top_k_max, int):
        raise TypeError(f"effective_top_k_max debe ser int, recibido {type(effective_top_k_max).__name__} ({effective_top_k_max!r}).")
    if effective_top_k_max <= 0:
        raise ValueError(f"effective_top_k_max debe ser > 0, recibido {effective_top_k_max!r}.")
    if isinstance(step_multiplier, bool) or not isinstance(step_multiplier, (int, float)):
        raise TypeError(f"step_multiplier debe ser numérico, recibido {type(step_multiplier).__name__} ({step_multiplier!r}).")
    if step_multiplier <= 1.0:
        raise ValueError(f"step_multiplier debe ser > 1.0, recibido {step_multiplier!r}.")

    if current_top_k >= effective_top_k_max:
        raise ValueError(
            f"current_top_k={current_top_k} ya alcanzó effective_top_k_max="
            f"{effective_top_k_max}; no existe un siguiente valor válido."
        )
    candidate = int(round(current_top_k * step_multiplier))
    candidate = max(candidate, current_top_k + 1)  # garantiza estrictamente > current
    return min(candidate, effective_top_k_max)


# ---------------------------------------------------------------------------
# Budget compartido Agentic RAG <-> verify_claim -- ver diseño acordado:
# el retrieval inicial NO consume este presupuesto (reemplaza
# _independent_retrieve_claim); solo los retrievals derivados de
# REWRITE_QUERY/ADJUST_TOP_K lo consumen.
# ---------------------------------------------------------------------------


def compute_effective_budget_for_verify_claim(
    *, original_max_additional_retrieval_requests: int, agentic_additional_retrievals_used: int
) -> int:
    """effective_budget_for_verify_claim = original - usado_por_agentic.

    ``used == original`` -> remaining = 0 (agotamiento legítimo).
    ``used > original`` -> ``ValueError`` fail-closed: esto NUNCA debe
    ocurrir si el controller respeta su propio presupuesto -- saturar
    silenciosamente a 0 ocultaría una violación real del controller
    (gastó más de lo permitido), en vez de exponerla.

    Estrictamente tipado: rechaza ``bool``/``float``/``str``/``None``
    para ambos parámetros -- solo ``int`` real."""
    for name, value in (
        ("original_max_additional_retrieval_requests", original_max_additional_retrieval_requests),
        ("agentic_additional_retrievals_used", agentic_additional_retrievals_used),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} debe ser int, recibido {type(value).__name__} ({value!r}).")
        if value < 0:
            raise ValueError(f"{name} debe ser >= 0, recibido {value!r}.")

    if agentic_additional_retrievals_used > original_max_additional_retrieval_requests:
        raise ValueError(
            f"agentic_additional_retrievals_used={agentic_additional_retrievals_used} "
            f"excede original_max_additional_retrieval_requests="
            f"{original_max_additional_retrieval_requests}; esto indica una violación "
            "real del presupuesto por parte del controller, no debe saturarse a 0 "
            "silenciosamente."
        )
    return original_max_additional_retrieval_requests - agentic_additional_retrievals_used
