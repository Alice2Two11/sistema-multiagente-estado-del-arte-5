"""Agentic Retrieval de la Etapa 07 -- Bloque 1: evaluación automática de la evidencia recuperada.

Este módulo implementa GRADE_EVIDENCE, que determina de forma automática si la
evidencia recuperada para un claim es SUFFICIENT o INSUFFICIENT.

La decisión es determinística y no interviene el planner ni ningún LLM. El grader
evalúa la evidencia mediante reglas programadas y genera reason_codes cuando detecta
problemas como pocos candidatos, baja diversidad de fuentes, baja relevancia o
cobertura insuficiente.

GRADE_EVIDENCE no decide qué hacer después. Su función es únicamente evaluar la
calidad y suficiencia de la evidencia. Las decisiones posteriores, como
REWRITE_QUERY, ADJUST_TOP_K o ACCEPT_EVIDENCE, se gestionan en el controller
de Agentic Retrieval.

La contradicción no se evalúa en este bloque porque requiere información adicional
que se procesa posteriormente en VerificationAgent.verify_claim, como
contradiction_type y contradiction_evidence_ids.

El grader utiliza únicamente información que ya entrega el retriever, como:
source_filename, chunk_id, text y native_scores_by_retriever["chroma"].

La relevancia de cada candidato se obtiene mediante
extract_candidate_relevance_score, que actúa como función única para extraer
correctamente el score de Chroma y evitar que otros módulos interpreten el campo
de relevancia de forma diferente.

Este módulo es independiente del runtime y del agente verificador: recibe el claim
y los candidatos ya recuperados, los evalúa y devuelve el resultado sin modificar
otros componentes del sistema."""

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

    """Valida la estructura mínima que debe tener cada candidato de evidencia antes
    de usarlo en las funciones de evaluación del grader.
    
    Este validador es reutilizado por grade_evidence,
    is_minimum_viable_evidence y _lexical_overlap_ratio para comprobar que
    los campos básicos del candidato sean válidos antes de calcular métricas.
    
    La puntuación de relevancia no se valida aquí, sino mediante
    extract_candidate_relevance_score en los puntos donde realmente se necesita.
    
    La validación es estricta: no convierte automáticamente valores incorrectos
    con str(...). Por ejemplo, source_filename=123 o text=["foo", "bar"] se
    rechazan en lugar de transformarse silenciosamente en texto."""
    
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
    """Calcula qué proporción de los términos del claim aparece literalmente
    en al menos uno de los textos recuperados como evidencia."""
    claim_terms = _extract_terms(claim_text)
    if not claim_terms:
        return 0.0
    candidate_terms: set[str] = set()
    for candidate in candidates:
        candidate_terms |= _extract_terms(candidate["text"])
    overlap = claim_terms & candidate_terms
    return len(overlap) / len(claim_terms)


def extract_candidate_relevance_score(candidate: dict[str, Any]) -> float:
    """Extrae de forma única y consistente la puntuación de relevancia de un
    candidato, reutilizando la misma lógica en los distintos bloques que la necesitan.
    
    La relevancia se obtiene del score nativo de Chroma almacenado en
    native_scores_by_retriever["chroma"], que corresponde a la señal calculada por
    el retriever. No se utiliza fused_rrf_score como sustituto.
    
    La validación es estricta:
    - candidate debe ser un diccionario;
    - native_scores_by_retriever debe existir y ser un mapping válido;
    - debe existir la clave "chroma";
    - el valor debe ser numérico, no booleano y además finito.
    
    No se aplican valores por defecto ni conversiones silenciosas. Si el score nativo
    de Chroma falta o tiene una estructura inválida, se considera una violación del
    contrato del candidato y se genera un error en lugar de asumir relevancia cero."""
    
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
    """Evalúa si la evidencia recuperada para un claim es suficiente antes de pasar
    a la etapa de verificación.
    
    A partir de los candidatos recuperados, calcula de forma automática:
    - cantidad total de candidatos;
    - diversidad de fuentes;
    - mayor puntuación de relevancia encontrada;
    - proporción de términos del claim presentes en la evidencia.
    
    Con estos valores determina si la evidencia es SUFFICIENT o INSUFFICIENT.
    Cuando es insuficiente, también devuelve los reason_codes que indican qué
    criterios no se cumplieron.
    
    La evaluación es determinística: con los mismos candidatos y el mismo claim
    siempre produce el mismo resultado, sin utilizar LLM ni mantener estado entre llamadas.
    """
    if thresholds is None:
        thresholds = DEFAULT_GRADER_THRESHOLDS
    thresholds = validate_grader_thresholds(thresholds)

    candidates = [_require_valid_candidate(c) for c in candidates]

    candidate_count = len(candidates) #candidatos fueron recuperados
    source_diversity = len({c["source_filename"] for c in candidates}) #cuántas fuentes distintas hay
    scores = [extract_candidate_relevance_score(c) for c in candidates] #toma el score de relevancia más alto entre todas las evidencias recuperadas (verification_incremental_retriever.py) similitud de coseno
    max_relevance_score = max(scores) if scores else 0.0
    lexical_overlap_ratio = _lexical_overlap_ratio(claim_text, candidates) #qué proporción de los términos del claim aparece literalmente en los textos de la evidencia recuperada (_lexical_overlap_ratio)

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
    """Define cuándo una evidencia puede considerarse mínimamente viable.
    Este criterio es más flexible que el usado por grade_evidence para declarar
    la evidencia como SUFFICIENT. Aquí no se exige cumplir todos los criterios de
    suficiencia, sino encontrar al menos un candidato que pueda seguir utilizándose
    como evidencia básica.
    
    Para que un candidato sea considerado viable debe cumplir al mismo tiempo:
    - alcanzar la relevancia mínima establecida;
    - pertenecer a una fuente explícitamente autorizada en authorized_sources.
    
    No basta con que exista un candidato relevante y, por separado, otro candidato
    autorizado: ambas condiciones deben cumplirse sobre la misma evidencia.
    
    Este criterio lo utiliza únicamente el controller cuando se agota el presupuesto
    de recuperación. En ese momento decide automáticamente si acepta la evidencia
    disponible con ACCEPT_EVIDENCE o si finaliza el claim como FINISH_UNRESOLVED,
    sin consultar al planner."""
    # Se exige al menos 1 candidato, con un score de relevancia >= 0.15,
    # y además ese mismo candidato debe pertenecer a una fuente autorizada.
    # Si cualquiera de estas condiciones no se cumple, la evidencia no se considera
    # mínimamente viable y el claim puede finalizar como FINISH_UNRESOLVED.
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
