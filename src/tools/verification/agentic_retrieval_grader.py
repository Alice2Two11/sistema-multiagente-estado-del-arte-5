"""Agentic Retrieval (Stage 07, pre-verificación) -- Bloque 1: grader
determinista puro.

``GRADE_EVIDENCE`` -- decisión automática, NUNCA del planner (ver
diseño acordado: solo RETRIEVE/GRADE_EVIDENCE son consecuencias
obligatorias; REWRITE_QUERY/ADJUST_TOP_K/ACCEPT_EVIDENCE son las
únicas acciones planner-seleccionables, en un bloque posterior).

Determinista y auditable, sin LLM -- recomendación ya aprobada frente
a un grader LLM (evita una cadena de juicios de alucinación
compuestos). ``CONTRADICTORY`` excluido: esa evaluación exige
contratos que solo ``VerificationAgent.verify_claim`` tiene
(``contradiction_type``/``contradiction_evidence_ids``).

Los 4 reason codes usan exclusivamente campos que ya existen en la
salida real de ``Agent07ChromaRetriever.retrieve_more``
(``source_filename``, ``chunk_id``, ``text``,
``native_scores_by_retriever["chroma"]``) -- ninguno requiere
metadata nueva.

CONTRACT-FIX: ``extract_candidate_relevance_score`` es el extractor
CANÓNICO y único de la señal de relevancia -- corrige un bug real
donde tanto este módulo como Bloque 3 asumían una clave ``"score"``
directa que el retriever real nunca produce (el campo real es
``native_scores_by_retriever["chroma"]``). Bloque 3 importa esta misma
función, no reimplementa la extracción.

Este módulo es puro: no importa nada de ``verification_runtime.py``,
``verification_agent.py``, ni del retriever -- recibe candidatos y
texto de claim ya materializados, sin efectos secundarios."""

from __future__ import annotations

import math
import re
from typing import Any

from src.config.agentic_retrieval_policy_config import (
    DEFAULT_GRADER_THRESHOLDS,
    GRADE_REASON_CODES,
    GRADE_RESULT_VALUES,
    validate_grader_thresholds,
    validate_minimum_viable_thresholds,
)

# Stopwords mínimas ES/EN para el cálculo de cobertura léxica -- no es
# NLP complejo, solo excluye conectores triviales para no contarlos
# como "términos del claim" al medir overlap.
_STOPWORDS = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "y", "o",
    "que", "con", "por", "para", "es", "son", "se", "su", "sus", "al", "the",
    "an", "of", "in", "and", "or", "that", "with", "for", "is", "are", "to", "on", "as",
})


def _extract_terms(text: str) -> set[str]:
    """Tokeniza texto a un set de términos en minúsculas, excluyendo
    stopwords y tokens de 1-2 caracteres (ruido). Sin NLP semántico --
    solo comparación léxica literal, determinista y reproducible."""
    tokens = re.findall(r"[a-záéíóúñ0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _require_valid_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Validador mínimo, reutilizado por ``grade_evidence``/
    ``is_minimum_viable_evidence``/``_lexical_overlap_ratio`` -- el
    mismo principio fail-closed ya aplicado a
    ``extract_candidate_relevance_score`` para el score.

    Bloque 1 consume un SUBCONJUNTO del schema canónico completo del
    candidate (no usa ``chunk_id`` en ningún cálculo de sus métricas):
    exige ``candidate`` dict real, ``source_filename`` str real no
    vacío, ``text`` str real no vacío. La relevancia se valida por
    separado vía ``extract_candidate_relevance_score`` en cada punto
    donde se usa. Sin ``str(...)`` para corregir entradas inválidas --
    un ``source_filename=123`` o ``text=["foo","bar"]`` se rechaza, no
    se convierte silenciosamente."""
    if not isinstance(candidate, dict):
        raise TypeError(f"candidate debe ser dict real, recibido {type(candidate).__name__}.")
    source_filename = candidate.get("source_filename")
    if not isinstance(source_filename, str) or not source_filename.strip():
        raise ValueError(
            f"candidate.source_filename debe ser str real no vacío, recibido "
            f"{source_filename!r} ({type(source_filename).__name__})."
        )
    text = candidate.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"candidate.text debe ser str real no vacío, recibido {text!r} ({type(text).__name__})."
        )
    return candidate


def _lexical_overlap_ratio(claim_text: str, candidates: list[dict[str, Any]]) -> float:
    """Fracción de términos del claim que aparecen literalmente en AL
    MENOS UN candidato (unión de todos los textos de candidatos)."""
    claim_terms = _extract_terms(claim_text)
    if not claim_terms:
        return 0.0
    candidate_terms: set[str] = set()
    for candidate in candidates:
        candidate_terms |= _extract_terms(candidate["text"])
    overlap = claim_terms & candidate_terms
    return len(overlap) / len(claim_terms)


def extract_candidate_relevance_score(candidate: dict[str, Any]) -> float:
    """Extractor CANÓNICO y ÚNICO de la señal de relevancia de un
    candidato -- usado tanto por Bloque 1 como por Bloque 3 (Bloque 3
    lo importa directamente, no reimplementa la extracción).

    CONTRACT-FIX (esta ronda): corrige un bug contractual real --
    ``candidate.get("score", 0.0)`` asumía una clave ``"score"`` que
    el retriever real (``Agent07ChromaRetriever.retrieve_more``,
    ``verification_incremental_retriever.py``) NUNCA produce. El
    campo real, confirmado por lectura directa del retriever, es:

        candidate["native_scores_by_retriever"]["chroma"] == 1.0 - distance

    Se preserva exactamente la misma métrica que ya se pretendía usar
    (``1.0 - distance``) -- solo se corrige el acceso al campo, nunca
    se sustituye por ``fused_rrf_score`` (sin evidencia contractual de
    que esa fuera la señal aprobada).

    Contrato fail-closed, sin ``.get(..., 0.0)`` ni ningún fallback:
    - ``candidate`` debe ser dict.
    - ``native_scores_by_retriever`` debe existir y ser un mapping
      (``dict``).
    - la clave ``"chroma"`` debe existir dentro de él.
    - el valor debe ser numérico real (``int``/``float``), ``bool``
      rechazado, y finito (rechaza NaN/+inf/-inf -- de otro modo
      evadirían silenciosamente cualquier comparación de threshold,
      ``NaN < x``/``NaN >= x`` son ambas ``False`` en IEEE 754).
    - NO se impone rango ``[0,1]`` -- no está contractualmente
      confirmado.

    Si falta el score nativo de Chroma o el schema es inválido, eso es
    una violación del contrato del candidato -- NUNCA se interpreta
    como relevancia igual a cero."""
    if not isinstance(candidate, dict):
        raise TypeError(f"candidate debe ser dict, recibido {type(candidate).__name__}.")
    if "native_scores_by_retriever" not in candidate:
        raise ValueError(
            "candidate no contiene 'native_scores_by_retriever' -- violación del contrato "
            "del candidato real (Agent07ChromaRetriever.retrieve_more), no se asume score=0.0."
        )
    native_scores = candidate["native_scores_by_retriever"]
    if not isinstance(native_scores, dict):
        raise TypeError(
            f"candidate['native_scores_by_retriever'] debe ser mapping (dict), "
            f"recibido {type(native_scores).__name__}."
        )
    if "chroma" not in native_scores:
        raise ValueError(
            "candidate['native_scores_by_retriever'] no contiene la clave 'chroma' -- "
            "violación del contrato del candidato real."
        )
    raw = native_scores["chroma"]
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise TypeError(
            f"candidate['native_scores_by_retriever']['chroma'] debe ser numérico, "
            f"recibido {type(raw).__name__} ({raw!r})."
        )
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(
            f"candidate['native_scores_by_retriever']['chroma'] debe ser finito "
            f"(no NaN/+inf/-inf), recibido {value!r}."
        )
    return value


def grade_evidence(
    *,
    claim_text: str,
    candidates: list[dict[str, Any]],
    thresholds: dict | None = None,
) -> dict[str, Any]:
    """Evalúa si ``candidates`` (salida ya materializada de
    ``retrieve_more``, lista de dicts con al menos ``source_filename``/
    ``text``/``native_scores_by_retriever``) es suficiente para pasar
    a verificación.

    Retorna:
        {
            "grade_result": "SUFFICIENT" | "INSUFFICIENT",
            "reason_codes": tuple[str, ...],  # vacío si SUFFICIENT
            "candidate_count": int,
            "source_diversity": int,
            "max_relevance_score": float,
            "lexical_overlap_ratio": float,
        }

    Determinista: mismos inputs -> mismo output, siempre. No invoca
    ningún LLM ni tiene estado mutable entre llamadas.
    """
    if thresholds is None:
        thresholds = DEFAULT_GRADER_THRESHOLDS
    thresholds = validate_grader_thresholds(thresholds)

    candidates = [_require_valid_candidate(c) for c in candidates]

    candidate_count = len(candidates)
    source_diversity = len({c["source_filename"] for c in candidates})
    scores = [extract_candidate_relevance_score(c) for c in candidates]
    max_relevance_score = max(scores) if scores else 0.0
    lexical_overlap_ratio = _lexical_overlap_ratio(claim_text, candidates)

    reason_codes: list[str] = []

    if candidate_count < thresholds["min_candidate_count"]:
        reason_codes.append("LOW_CANDIDATE_COUNT")

    if (
        candidate_count >= thresholds["min_candidate_count_for_diversity_check"]
        and source_diversity < thresholds["min_source_diversity"]
    ):
        reason_codes.append("LOW_SOURCE_DIVERSITY")

    if max_relevance_score < thresholds["min_relevance_score"]:
        reason_codes.append("LOW_RELEVANCE")

    if lexical_overlap_ratio < thresholds["min_lexical_overlap_ratio"]:
        reason_codes.append("LOW_COVERAGE")

    grade_result = "INSUFFICIENT" if reason_codes else "SUFFICIENT"

    for code in reason_codes:
        assert code in GRADE_REASON_CODES, code  # invariante interna
    assert grade_result in GRADE_RESULT_VALUES, grade_result  # invariante interna

    return {
        "grade_result": grade_result,
        "reason_codes": tuple(reason_codes),
        "candidate_count": candidate_count,
        "source_diversity": source_diversity,
        "max_relevance_score": max_relevance_score,
        "lexical_overlap_ratio": lexical_overlap_ratio,
    }


def is_minimum_viable_evidence(
    *,
    candidates: list[dict[str, Any]],
    thresholds: dict,
    authorized_sources: frozenset[str] | set[str],
) -> bool:
    """Definición objetiva de ``minimum_viable_evidence`` -- MÁS LAXA
    que ``grade_evidence`` (SUFFICIENT): solo exige que exista al menos
    un candidato mínimamente plausible Y de una fuente REALMENTE
    autorizada (verificado por pertenencia explícita a
    ``authorized_sources``, no solo por tener un ``source_filename``
    no vacío -- nunca se asume silenciosamente que el caller ya filtró
    por autorización). Usada exclusivamente por el controller (bloque
    posterior) cuando el presupuesto de retrieval se agota, para
    decidir determinísticamente entre ACCEPT_EVIDENCE/FINISH_UNRESOLVED
    -- nunca consultada al planner.

    El candidato que satisface "viable" debe cumplir AMBAS condiciones
    a la vez (relevancia mínima Y fuente autorizada) -- no basta con
    que existan candidatos relevantes por un lado y candidatos
    autorizados por otro, sin relación entre sí."""
    thresholds = validate_minimum_viable_thresholds(thresholds)
    if not isinstance(authorized_sources, (frozenset, set)):
        raise TypeError(
            f"authorized_sources debe ser frozenset/set, recibido "
            f"{type(authorized_sources).__name__}."
        )
    candidates = [_require_valid_candidate(c) for c in candidates]

    candidate_count = len(candidates)
    if candidate_count < thresholds["min_candidate_count"]:
        return False

    scores = [extract_candidate_relevance_score(c) for c in candidates]
    max_relevance_score = max(scores) if scores else 0.0
    if max_relevance_score < thresholds["min_relevance_score"]:
        return False

    has_viable_authorized_candidate = any(
        extract_candidate_relevance_score(c) >= thresholds["min_relevance_score"]
        and c["source_filename"] in authorized_sources
        for c in candidates
    )
    return has_viable_authorized_candidate
