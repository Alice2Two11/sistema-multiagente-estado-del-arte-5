"""Agentic Retrieval de la Etapa 07 -- Bloque 3: reformulación determinística
de consultas con control de desviación semántica.

Este módulo se encarga únicamente de generar y validar una nueva versión de
current_query cuando el controller selecciona REWRITE_QUERY.

Este bloque modifica exclusivamente la consulta de búsqueda para
intentar recuperar evidencia más útil, manteniendo controles determinísticos
que evitan perder información del claim o introducir contenido numérico no
respaldado."""

from __future__ import annotations

import re
from typing import Any

from src.config.agentic_retrieval_policy_config import GRADE_REASON_CODES
from src.tools.verification.agentic_retrieval_grader import (
    _extract_terms,
    _STOPWORDS,
    extract_candidate_relevance_score,
)

REWRITE_REASON_VALUES = GRADE_REASON_CODES

NOT_DETERMINISTICALLY_ENFORCEABLE = (
    "polarity_or_negation_preservation",
    "causal_or_comparative_relation_preservation",
    "deep_semantic_central_content_preservation",
)

DEFAULT_MAX_REWRITTEN_QUERY_LENGTH = 500
DEFAULT_MAX_NEW_TERMS_PER_REWRITE = 8

_INSTRUCTION_LEAKAGE_PATTERNS = (
    re.compile(r"\bignore\s+(the\s+)?previous\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\banswer\s+as\b", re.IGNORECASE),
    re.compile(r"\bsummari[sz]e\b", re.IGNORECASE),
    re.compile(r"\brespond\s+with\b", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bdisregard\b", re.IGNORECASE),
)

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


class QueryRewriteError(ValueError):
    """El rewrite generado o propuesto viola un guardrail de query
    drift, o no es posible producir uno real -- fail-closed."""

def _normalize_for_equivalence(text: str) -> str:
    collapsed = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", collapsed).strip()

def _extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_PATTERN.findall(text))

def _require_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise QueryRewriteError(f"{name} debe ser str real, recibido {type(value).__name__} ({value!r}).")
    if not value.strip():
        raise QueryRewriteError(f"{name} no puede estar vacío.")
    return value

def _require_reason_codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise QueryRewriteError(f"reason_codes debe ser tuple real, recibido {type(value).__name__}.")
    if not value:
        raise QueryRewriteError("reason_codes no puede estar vacío.")
    for code in value:
        if not isinstance(code, str):
            raise QueryRewriteError(f"reason_codes contiene un elemento no-str: {code!r}.")
        if code not in GRADE_REASON_CODES:
            raise QueryRewriteError(
                f"reason_codes contiene {code!r}, fuera de GRADE_REASON_CODES {GRADE_REASON_CODES}."
            )
    if len(set(value)) != len(value):
        raise QueryRewriteError(f"reason_codes contiene duplicados: {value!r}.")
    return value

def _require_rewrite_reason(value: Any, reason_codes: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise QueryRewriteError(f"rewrite_reason debe ser str real, recibido {type(value).__name__}.")
    if value not in REWRITE_REASON_VALUES:
        raise QueryRewriteError(
            f"rewrite_reason={value!r} fuera de REWRITE_REASON_VALUES {REWRITE_REASON_VALUES}."
        )
    if value not in reason_codes:
        raise QueryRewriteError(
            f"rewrite_reason={value!r} no pertenece a reason_codes={reason_codes!r} -- "
            "el rewrite_reason debe corresponder a una causa real declarada en la Observation "
            "(decision_basis real del planner de Bloque 2), no elegirse por posición."
        )
    return value

def _require_authorized_sources(value: Any) -> frozenset[str] | set[str]:
    if not isinstance(value, (frozenset, set)):
        raise QueryRewriteError(
            f"authorized_sources debe ser frozenset/set, recibido {type(value).__name__} -- "
            "posible violación contractual upstream."
        )
    if not value:
        raise QueryRewriteError(
            "authorized_sources vacío -- no existe ninguna fuente autorizada desde la cual "
            "realizar expansión; REWRITE_QUERY no es viable bajo este algoritmo."
        )
    for source in value:
        if not isinstance(source, str) or not source.strip():
            raise QueryRewriteError(f"authorized_sources contiene un elemento inválido: {source!r}.")
    return value
  

def _require_candidates(value: Any) -> list[dict[str, Any]]:
    """Valida estrictamente que cada candidato contenga los campos obligatorios
    source_filename, chunk_id, text y una puntuación de relevancia válida.
    No convierte ni completa automáticamente valores incorrectos: si falta un campo,
    tiene un tipo inválido o viene vacío, el candidato se rechaza.
    La relevancia se obtiene mediante extract_candidate_relevance_score, reutilizando
    el extractor canónico del Bloque 1 para leer correctamente
    native_scores_by_retriever["chroma"], sin asumir una clave "score" directa."""
    
    if not isinstance(value, list):
        raise QueryRewriteError(f"candidates debe ser list, recibido {type(value).__name__}.")
    for candidate in value:
        if not isinstance(candidate, dict):
            raise QueryRewriteError(f"candidates contiene un elemento no-dict: {candidate!r}.")
        source_filename = candidate.get("source_filename")
        if not isinstance(source_filename, str) or not source_filename.strip():
            raise QueryRewriteError(
                f"candidate.source_filename debe ser str real no vacío, recibido "
                f"{source_filename!r} ({type(source_filename).__name__})."
            )
        text = candidate.get("text")
        if not isinstance(text, str) or not text.strip():
            raise QueryRewriteError(
                f"candidate.text debe ser str real no vacío, recibido {text!r} ({type(text).__name__})."
            )
        try:
            extract_candidate_relevance_score(candidate)
        except (TypeError, ValueError) as exc:
            raise QueryRewriteError(f"candidate.score inválido: {exc}") from exc
        # chunk_id OBLIGATORIO -- es el identificador estable real
        # confirmado en el retriever, usado para el tie-break.
        chunk_id = candidate.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise QueryRewriteError(
                f"candidate.chunk_id debe ser str real no vacío, recibido "
                f"{chunk_id!r} ({type(chunk_id).__name__})."
            )
    return value


def _extract_terms_in_order(text: str, stopwords: frozenset[str]) -> list[str]:
    """Como ``_extract_terms`` pero preservando el ORDEN de aparición en
    el texto (no un set) -- necesario para conservar "primer orden de
    aparición útil" dentro de cada candidato al seleccionar términos por
    relevancia, sin reordenar alfabéticamente."""
    tokens = re.findall(r"[a-záéíóúñ0-9]+", text.lower())
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if len(token) > 2 and token not in stopwords and token not in seen:
            ordered.append(token)
            seen.add(token)
    return ordered


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise QueryRewriteError(f"{name} debe ser int real > 0, recibido {value!r}.")
    return value


def _rank_authorized_candidates(
    candidates: list[dict[str, Any]], authorized_sources: frozenset[str] | set[str]
) -> list[dict[str, Any]]:
    """Filtra los candidatos para conservar únicamente evidencias provenientes de
    fuentes autorizadas y las ordena de mayor a menor según su relevancia en Chroma.
    La puntuación se obtiene mediante extract_candidate_relevance_score, reutilizando
    el extractor canónico del Bloque 1.
    Si dos candidatos tienen el mismo score, se aplica un desempate determinístico:
    primero por source_filename y luego por chunk_id en orden ascendente.
    """
    authorized = [c for c in candidates if c["source_filename"] in authorized_sources]
    return sorted(
        authorized,
        key=lambda c: (
            -extract_candidate_relevance_score(c),
            c["source_filename"],
            c["chunk_id"],
        ),
    )


def _collect_authorized_candidate_content(
    candidates: list[dict[str, Any]], authorized_sources: frozenset[str] | set[str]
) -> tuple[set[str], set[str]]:
    """Retorna (términos, números) presentes en candidatos AUTORIZADOS
    """
    terms: set[str] = set()
    numbers: set[str] = set()
    for candidate in candidates:
        source_filename = str(candidate.get("source_filename", "")).strip()
        if source_filename not in authorized_sources:
            continue
        text = str(candidate.get("text", ""))
        terms |= _extract_terms(text)
        numbers |= _extract_numbers(text)
    return terms, numbers








def generate_query_rewrite(
    *,
    claim_text: str,
    current_query: str,
    reason_codes: tuple[str, ...],
    rewrite_reason: str,
    candidates: list[dict[str, Any]],
    authorized_sources: frozenset[str] | set[str],
    max_new_terms: int = DEFAULT_MAX_NEW_TERMS_PER_REWRITE,
    max_length: int = DEFAULT_MAX_REWRITTEN_QUERY_LENGTH,
) -> dict[str, Any]:

    # Valida todos los datos de entrada antes de intentar construir una nueva query.
    # Comprueba que los textos no estén vacíos, que los reason_codes sean válidos,
    # que rewrite_reason corresponda realmente a una causa detectada por el grader,
    # que los candidatos tengan la estructura esperada, que existan fuentes autorizadas
    claim_text = _require_nonempty_str(claim_text, "claim_text")
    current_query = _require_nonempty_str(current_query, "current_query")
    reason_codes = _require_reason_codes(reason_codes)
    rewrite_reason = _require_rewrite_reason(rewrite_reason, reason_codes)
    candidates = _require_candidates(candidates)
    authorized_sources = _require_authorized_sources(authorized_sources)
    max_new_terms = _require_positive_int(max_new_terms, "max_new_terms")
    max_length = _require_positive_int(max_length, "max_length")

    # Extrae el vocabulario y los números que ya aparecen en el claim o en la query actual.
    # Esto permite evitar duplicados y distinguir posteriormente qué contenido fue realmente
    # incorporado durante la reformulación.
    existing_vocabulary = _extract_terms(claim_text) | _extract_terms(current_query)
    existing_numbers = _extract_numbers(claim_text) | _extract_numbers(current_query)

    # Conserva únicamente candidatos provenientes de fuentes autorizadas y los ordena
    # de mayor a menor según su relevancia en Chroma.
    ranked_candidates = _rank_authorized_candidates(candidates, authorized_sources)

    # Recorre los candidatos en orden de relevancia y selecciona términos nuevos.
    # Solo incorpora términos que no estuvieran ya presentes en el claim o en la query,
    # evita duplicarlos y respeta el límite máximo definido por max_new_terms max_new_terms = 8
    selected_terms: list[str] = []
    seen: set[str] = set()

    for candidate in ranked_candidates:
        if len(selected_terms) >= max_new_terms:
            break

        candidate_terms = _extract_terms_in_order(
            str(candidate.get("text", "")),
            _STOPWORDS,
        )

        for term in candidate_terms:
            if len(selected_terms) >= max_new_terms:
                break

            if term in existing_vocabulary or term in seen:
                continue

            selected_terms.append(term)
            seen.add(term)

    # Si ningún candidato autorizado aporta términos realmente nuevos,
    # no es posible producir una reformulación distinta de la query actual.
    # En ese caso REWRITE_QUERY se considera una acción no ejecutable.
    if not selected_terms:
        raise QueryRewriteError(
            "QUERY_REWRITE_UNAVAILABLE: no hay términos nuevos disponibles en candidatos "
            "autorizados que no estén ya presentes en claim_text/current_query -- no es "
            "posible producir un rewrite real bajo este algoritmo."
        )

    # Construye la nueva query de forma aditiva:
    # conserva completa la consulta anterior y añade al final los términos nuevos.
    rewritten_query = f"{current_query} {' '.join(selected_terms)}".strip()

    # Reúne todos los valores numéricos presentes en los candidatos autorizados
    # y priorizados, para poder identificar cuáles números nuevos provienen realmente
    # de la evidencia recuperada.
    candidate_numbers_ranked_first = set()

    for candidate in ranked_candidates:
        candidate_numbers_ranked_first |= _extract_numbers(
            str(candidate.get("text", ""))
        )

    # Registra los términos nuevos utilizados en la reformulación que no son
    # exclusivamente valores numéricos.
    source_terms_used = tuple(
        t for t in selected_terms if not t.isdigit()
    )

    # Identifica los números introducidos por los nuevos términos y conserva únicamente
    # aquellos que aparecen realmente en la evidencia y que no estaban ya presentes
    # en el claim o en la query anterior.
    introduced_text = " ".join(selected_terms)
    numbers_in_selected_terms = _extract_numbers(introduced_text)

    source_numbers_used = tuple(
        sorted(
            numbers_in_selected_terms
            & (candidate_numbers_ranked_first - existing_numbers)
        )
    )

    # Construye el resultado de la reformulación, conservando la query anterior,
    # la nueva query generada, el motivo del rewrite y la procedencia del contenido
    # adicional incorporado.
    result = {
        "previous_query": current_query,
        "rewritten_query": rewritten_query,
        "rewrite_reason": rewrite_reason,
        "source_terms_used": source_terms_used,
        "source_numbers_used": source_numbers_used,
    }

    # Antes de devolver la nueva query, valida que la reformulación cumpla las reglas
    # del sistema: preservar la consulta previa, respetar el motivo de reformulación,
    # no introducir números no autorizados, usar contenido proveniente de fuentes
    # permitidas y mantenerse dentro del límite máximo de longitud.
    validate_query_rewrite(
        previous_query=result["previous_query"],
        rewritten_query=result["rewritten_query"],
        claim_text=claim_text,
        reason_codes=reason_codes,
        rewrite_reason=result["rewrite_reason"],
        source_terms_used=result["source_terms_used"],
        source_numbers_used=result["source_numbers_used"],
        candidates=candidates,
        authorized_sources=authorized_sources,
        max_length=max_length,
    )

    # Devuelve únicamente una reformulación que ya pasó todas las validaciones.
    return result



# Valida que una query reformulada sea realmente nueva, conserve íntegra la
# consulta anterior y añada únicamente términos o números provenientes de
# evidencia autorizada.
#
# También comprueba que todo el contenido nuevo quede correctamente trazado en
# source_terms_used y source_numbers_used, que no se exceda la longitud máxima DEFAULT_MAX_REWRITTEN_QUERY_LENGTH = 500
# y que la expansión no contenga estructuras de instrucciones o caracteres inválidos.
#
# Si alguna de estas reglas falla, rechaza la reformulación mediante QueryRewriteError.
def validate_query_rewrite(
    *,
    previous_query: str,
    rewritten_query: str,
    claim_text: str,
    reason_codes: tuple[str, ...],
    rewrite_reason: str,
    source_terms_used: tuple[str, ...],
    source_numbers_used: tuple[str, ...] = (),
    candidates: list[dict[str, Any]],
    authorized_sources: frozenset[str] | set[str],
    max_length: int = DEFAULT_MAX_REWRITTEN_QUERY_LENGTH,
) -> None:
    previous_query = _require_nonempty_str(previous_query, "previous_query")
    claim_text = _require_nonempty_str(claim_text, "claim_text")
    max_length = _require_positive_int(max_length, "max_length")
    authorized_sources = _require_authorized_sources(authorized_sources)
    candidates = _require_candidates(candidates)
    reason_codes = _require_reason_codes(reason_codes)
    rewrite_reason = _require_rewrite_reason(rewrite_reason, reason_codes)

    if not isinstance(rewritten_query, str):
        raise QueryRewriteError(
            f"rewritten_query debe ser str real, recibido {type(rewritten_query).__name__}."
        )
    if not rewritten_query.strip():
        raise QueryRewriteError("rewritten_query no puede estar vacía.")
    if rewritten_query == previous_query:
        raise QueryRewriteError("rewritten_query idéntica a previous_query -- no es un rewrite real.")
    if _normalize_for_equivalence(rewritten_query) == _normalize_for_equivalence(previous_query):
        raise QueryRewriteError(
            "rewritten_query equivalente a previous_query tras normalización trivial "
            "(case/espacios/puntuación) -- no constituye un rewrite real."
        )
    if len(rewritten_query) > max_length:
        raise QueryRewriteError(
            f"rewritten_query excede max_length={max_length} caracteres (recibida {len(rewritten_query)})."
        )
      
    if not rewritten_query.startswith(previous_query):
        raise QueryRewriteError(
            "rewritten_query no conserva previous_query íntegra como bloque inicial -- "
            "la estrategia aditiva exige previous_query + expansión, nunca alteración "
            "del contenido previo."
        )
    remainder = rewritten_query[len(previous_query):]
    if remainder and not remainder[0].isspace():
        raise QueryRewriteError(
            f"rewritten_query concatena contenido pegado directamente al final de "
            f"previous_query sin separador (remainder={remainder!r}) -- no es una "
            "expansión real, podría alterar el último token de previous_query."
        )

    previous_terms = _extract_terms(previous_query)
    claim_terms = _extract_terms(claim_text)
    rewritten_terms = _extract_terms(rewritten_query)
    previous_numbers = _extract_numbers(previous_query)
    claim_numbers = _extract_numbers(claim_text)
    rewritten_numbers = _extract_numbers(rewritten_query)


    authorized_candidate_terms, authorized_candidate_numbers = _collect_authorized_candidate_content(
        candidates, authorized_sources
    )

    allowed_vocabulary = claim_terms | previous_terms | authorized_candidate_terms
    introduced_terms = rewritten_terms - allowed_vocabulary
    if introduced_terms:
        raise QueryRewriteError(
            f"rewritten_query introduce términos no presentes en claim_text/previous_query/"
            f"candidate texts autorizados: {sorted(introduced_terms)!r}."
        )

    allowed_numbers = claim_numbers | previous_numbers | authorized_candidate_numbers
    introduced_numbers = rewritten_numbers - allowed_numbers
    if introduced_numbers:
        raise QueryRewriteError(
            f"rewritten_query introduce números no presentes en inputs autorizados: "
            f"{sorted(introduced_numbers)!r}."
        )

    if not isinstance(source_terms_used, tuple):
        raise QueryRewriteError(f"source_terms_used debe ser tuple, recibido {type(source_terms_used).__name__}.")
    for term in source_terms_used:
        if not isinstance(term, str) or not term.strip():
            raise QueryRewriteError(f"source_terms_used contiene un término inválido: {term!r}.")
    if len(set(source_terms_used)) != len(source_terms_used):
        raise QueryRewriteError(f"source_terms_used contiene duplicados: {source_terms_used!r}.")

    remainder_terms_in_order = _extract_terms_in_order(remainder, _STOPWORDS)
    real_introduced_terms_in_order = tuple(
        t for t in remainder_terms_in_order
        if not t.isdigit() and t not in claim_terms and t not in previous_terms
    )
    if source_terms_used != real_introduced_terms_in_order:
        raise QueryRewriteError(
            f"source_terms_used ({source_terms_used!r}) no coincide EN ORDEN con los "
            f"términos realmente introducidos por la expansión ({real_introduced_terms_in_order!r}) "
            "-- la traza debe representar el orden real en que el generador los incorporó, "
            "no solo el mismo conjunto; tampoco puede introducir el mismo término más de una vez."
        )
    for term in source_terms_used:
        if term not in authorized_candidate_terms:
            raise QueryRewriteError(
                f"source_terms_used contiene {term!r}, que no proviene de ningún candidate text "
                "autorizado -- violación de trazabilidad."
            )

    if not isinstance(source_numbers_used, tuple):
        raise QueryRewriteError(
            f"source_numbers_used debe ser tuple, recibido {type(source_numbers_used).__name__}."
        )
    for number in source_numbers_used:
        if not isinstance(number, str) or not number.strip():
            raise QueryRewriteError(f"source_numbers_used contiene un valor inválido: {number!r}.")
    if len(set(source_numbers_used)) != len(source_numbers_used):
        raise QueryRewriteError(f"source_numbers_used contiene duplicados: {source_numbers_used!r}.")

    real_introduced_numbers = rewritten_numbers - (claim_numbers | previous_numbers)
    if set(source_numbers_used) != real_introduced_numbers:
        raise QueryRewriteError(
            f"source_numbers_used ({sorted(source_numbers_used)!r}) no coincide exactamente con "
            f"los números realmente introducidos ({sorted(real_introduced_numbers)!r}) -- la "
            "query recibió información nueva de la evidencia sin trazarla, o la traza declara "
            "números que no fueron realmente incorporados."
        )
    for number in source_numbers_used:
        if number not in authorized_candidate_numbers:
            raise QueryRewriteError(
                f"source_numbers_used contiene {number!r}, que no proviene de ningún candidate "
                "text autorizado -- violación de trazabilidad."
            )
          
    for pattern in _INSTRUCTION_LEAKAGE_PATTERNS:
        if pattern.search(remainder):
            raise QueryRewriteError(
                f"la expansión introducida por el rewrite contiene una estructura de "
                f"instrucción/prompt ({pattern.pattern!r}) ajena a una consulta científica "
                "de retrieval."
            )

    if "\x00" in rewritten_query or any(ord(ch) < 9 for ch in rewritten_query):
        raise QueryRewriteError("rewritten_query contiene caracteres de control inválidos.")
