"""Agentic Retrieval (Stage 07, pre-verificación) -- Bloque 3
(corregido, ronda 2): REWRITE_QUERY real + guardrails de query drift.

Alcance exclusivo de este bloque: generar y validar una reformulación
de ``current_query``, de forma completamente aislada -- NO ejecuta
retrieval, NO toca ``verification_runtime.py``/``verification_agent.py``/
``verification_incremental_retriever.py``, NO modifica ningún contador
del controller (``query_rewrite_count``/``retrieval_round``/
``remaining_retrieval_budget``/``current_top_k``/``effective_top_k_max``
-- esos los produce el controller/action adapter de un bloque
posterior, consumiendo el resultado de este módulo).

Contrato real confirmado antes de implementar (solo lectura, sin
modificar nada):
- Query inicial de Stage 07: ``claim_text``/``original_claim_text``
  literal (``verification_incremental_retriever.py:81-85``), sin
  transformación.
- ``retrieve_more(self, request)`` NO acepta hoy ningún campo de query
  override -- siempre deriva ``claim_text`` internamente. Bloque 4
  deberá adaptar esto; no se toca aquí.
- El retriever real usa ``allowed_source_filenames``/``allowed_sources``
  para el mismo concepto que aquí llamamos ``authorized_sources``
  (Bloque 1/2) -- discrepancia de nomenclatura documentada, no
  resuelta en este bloque.
- Señal de ranking: CONTRACT-FIX (esta ronda) -- el campo ``score``
  usado hasta ahora NUNCA existió como clave directa en el candidato
  real. El campo real, confirmado por lectura directa del retriever,
  es ``candidate["native_scores_by_retriever"]["chroma"]`` (== ``1.0 -
  distance``). Se corrige el ACCESO al campo, preservando exactamente
  la misma métrica -- nunca sustituida por ``fused_rrf_score`` (sin
  evidencia contractual de que fuera la señal aprobada). Extraída
  mediante ``extract_candidate_relevance_score`` (extractor canónico
  de Bloque 1, importado aquí -- nunca reimplementado).

Estrategia de generación: DETERMINISTA POR REGLAS, sin LLM -- decisión
ya aprobada. No se introduce ningún modelo nuevo.

Selección de términos por RELEVANCIA, no alfabética (corrección de
esta ronda): candidatos autorizados ordenados por ``score`` DESCENDENTE
(tie-break determinista: ``source_filename`` ASC, luego ``chunk_id``
ASC), términos nuevos extraídos candidate por candidate en ese orden,
preservando primer orden de aparición útil, hasta ``max_new_terms``.
Nunca se une todo el vocabulario en un set antes de priorizar --
eso destruiría la procedencia/ranking.

Estrategia ADITIVA, impuesta por el validador: el rewrite nunca elimina
contenido de ``previous_query`` -- todos sus términos y números deben
seguir presentes en ``rewritten_query``.

Operación pública atómica: ``generate_query_rewrite`` SIEMPRE valida su
propio resultado antes de retornarlo. ``validate_query_rewrite`` es
TAMBIÉN fail-closed por sí misma (corrección de esta ronda) -- no
depende de haber sido invocada exclusivamente desde
``generate_query_rewrite``; endurece sus propios inputs
(``previous_query``/``claim_text``/``max_length``) al entrar.

Preservación semántica: de las propiedades pedidas originalmente, solo
son determinísticamente verificables con confianza: preservación
ADITIVA completa de todos los términos/números de ``previous_query``,
y no introducción de números nuevos no autorizados. El resto se
declara explícitamente en ``NOT_DETERMINISTICALLY_ENFORCEABLE``.
"""

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
    """Endurecido (CONTRACT-FIX de esta ronda): source_filename/chunk_id/
    text/score son OBLIGATORIOS -- confirmado que el retriever real
    (``verification_incremental_retriever.py``) siempre los produce.
    Sin coacción vía ``str(...)`` ni fallbacks ``.get(..., default)``.

    El score se valida vía ``extract_candidate_relevance_score``
    (extractor canónico, importado de Bloque 1 -- no reimplementado
    aquí) -- valida el schema real ``native_scores_by_retriever
    ["chroma"]``, nunca una clave ``"score"`` inexistente."""
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
        # score OBLIGATORIO -- participa en el ranking determinista;
        # extract_candidate_relevance_score ya lanza (ValueError/
        # TypeError de Bloque 1) si native_scores_by_retriever/"chroma"
        # está ausente o mal formado -- se re-envuelve en
        # QueryRewriteError para mantener un único tipo de excepción
        # público de este módulo.
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
    """Filtra a solo autorizados y ordena por relevancia nativa de Chroma
    DESCENDENTE (``extract_candidate_relevance_score``, extractor
    canónico -- nunca reimplementado aquí), con tie-break determinista
    (source_filename ASC, chunk_id ASC) -- preserva procedencia/ranking,
    nunca une en un set antes de esto.

    ``source_filename``/``chunk_id`` ya fueron validados como str reales
    no vacíos por ``_require_candidates`` -- se acceden directamente,
    sin ``.get(..., "")``."""
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
    -- usado solo para validación de pertenencia (orden irrelevante
    aquí), no para la selección/ranking en generación."""
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
    """Genera y valida atómicamente una propuesta de reformulación
    determinista de ``current_query``, seleccionando términos por
    RELEVANCIA (candidatos ordenados por ``score`` descendente, no
    alfabéticamente) -- ver ``_rank_authorized_candidates``. Nunca
    retorna un resultado sin validar; nunca retorna un NO-OP.

    Retorna:
        {
            "previous_query": str,
            "rewritten_query": str,
            "rewrite_reason": str,
            "source_terms_used": tuple[str, ...],
            "source_numbers_used": tuple[str, ...],
        }
    """
    claim_text = _require_nonempty_str(claim_text, "claim_text")
    current_query = _require_nonempty_str(current_query, "current_query")
    reason_codes = _require_reason_codes(reason_codes)
    rewrite_reason = _require_rewrite_reason(rewrite_reason, reason_codes)
    candidates = _require_candidates(candidates)
    authorized_sources = _require_authorized_sources(authorized_sources)
    max_new_terms = _require_positive_int(max_new_terms, "max_new_terms")
    max_length = _require_positive_int(max_length, "max_length")

    existing_vocabulary = _extract_terms(claim_text) | _extract_terms(current_query)
    existing_numbers = _extract_numbers(claim_text) | _extract_numbers(current_query)

    ranked_candidates = _rank_authorized_candidates(candidates, authorized_sources)

    selected_terms: list[str] = []
    seen: set[str] = set()
    for candidate in ranked_candidates:
        if len(selected_terms) >= max_new_terms:
            break
        candidate_terms = _extract_terms_in_order(str(candidate.get("text", "")), _STOPWORDS)
        for term in candidate_terms:
            if len(selected_terms) >= max_new_terms:
                break
            if term in existing_vocabulary or term in seen:
                continue
            selected_terms.append(term)
            seen.add(term)

    if not selected_terms:
        raise QueryRewriteError(
            "QUERY_REWRITE_UNAVAILABLE: no hay términos nuevos disponibles en candidatos "
            "autorizados que no estén ya presentes en claim_text/current_query -- no es "
            "posible producir un rewrite real bajo este algoritmo."
        )

    rewritten_query = f"{current_query} {' '.join(selected_terms)}".strip()

    candidate_numbers_ranked_first = set()
    for candidate in ranked_candidates:
        candidate_numbers_ranked_first |= _extract_numbers(str(candidate.get("text", "")))

    source_terms_used = tuple(t for t in selected_terms if not t.isdigit())
    source_numbers_used = tuple(
        sorted({t for t in selected_terms if t.isdigit()} & (candidate_numbers_ranked_first - existing_numbers))
    )

    result = {
        "previous_query": current_query,
        "rewritten_query": rewritten_query,
        "rewrite_reason": rewrite_reason,
        "source_terms_used": source_terms_used,
        "source_numbers_used": source_numbers_used,
    }

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

    return result


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
    """Fail-closed POR SÍ MISMA (corrección de esta ronda) -- no depende
    de haber sido invocada exclusivamente desde ``generate_query_
    rewrite``: endurece sus propios inputs (``previous_query``/
    ``claim_text``/``max_length``) al entrar, igual que
    ``generate_query_rewrite`` hace con los suyos.

    Valida un rewrite propuesto contra todos los guardrails de query
    drift, incluida la estrategia ADITIVA y la trazabilidad completa y
    exacta de ``source_terms_used``/``source_numbers_used``. Lanza
    ``QueryRewriteError`` si cualquiera falla.

    No modifica ningún contador del controller. No ejecuta retrieval.
    """
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

    # Estrategia aditiva impuesta como CONTRATO ESTRUCTURAL EXACTO
    # (corrección de esta ronda): confirmado que el generador real
    # SIEMPRE produce previous_query + " " + expansión, sin reordenar
    # ni normalizar previous_query -- el validador exige exactamente
    # ese contrato, no solo preservación del set de palabras largas
    # (que dejaría pasar la eliminación de tokens cortos científicamente
    # relevantes como "AI"/"R"/"C++", stopwords, o puntuación técnica).
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

    # Traza ORDENADA (corrección de esta ronda): el generador extrae
    # términos en orden real de aparición dentro de la expansión
    # (remainder), usando el mismo criterio que _extract_terms_in_order
    # -- no basta con igualdad de conjuntos, source_terms_used debe
    # coincidir EXACTAMENTE en orden con los términos nuevos realmente
    # introducidos por el remainder (no por previous_query/claim_text).
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

    # source_numbers_used SÍ se mantiene como comparación de conjunto/
    # orden canónico (sorted) -- confirmado que el generador real
    # deliberadamente los ordena canónicamente (sorted(...)), no por
    # aparición; no se cambia por simetría con source_terms_used.
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

    # Instruction/prompt leakage: SOLO sobre la expansión introducida
    # (remainder), nunca sobre previous_query -- previous_query es
    # inmutable y ya fue validado/aceptado antes de este ciclo; el
    # validator de Bloque 3 solo es responsable de que REWRITE_QUERY no
    # introduzca leakage nuevo, no de re-juzgar contenido preexistente.
    for pattern in _INSTRUCTION_LEAKAGE_PATTERNS:
        if pattern.search(remainder):
            raise QueryRewriteError(
                f"la expansión introducida por el rewrite contiene una estructura de "
                f"instrucción/prompt ({pattern.pattern!r}) ajena a una consulta científica "
                "de retrieval."
            )

    if "\x00" in rewritten_query or any(ord(ch) < 9 for ch in rewritten_query):
        raise QueryRewriteError("rewritten_query contiene caracteres de control inválidos.")
