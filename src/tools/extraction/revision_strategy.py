"""Directed revision planning for Agent 03 attempt 2.

This module does not perform I/O, construct clients, or execute transactions.
It classifies preliminary scientific cards and normalizes only unambiguous
LLM payload shapes required by the approved attempt-2 policy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .card_validation import build_quality_row, required_fields_for_card
from .review_exclusion import is_unknown_document_type
from .corpus_eligibility import is_corpus_include


REVISION_PLAN_COLUMNS = [
    "source_filename",
    "invalid_title",
    "missing_fields",
    "evidence_missing",
    "retrieved_unique_chunks",
    "primary_reason_code",
    "underlying_reason_code",
    "recommended_strategy",
]

INVALID_TITLE_VALUES = {
    "",
    "error",
    "no especificado",
    "nan",
}


class InvalidCardSchemaError(ValueError):
    """Raised when a parsed LLM payload is not one unambiguous card object."""


def normalize_card_payload(payload: Any) -> dict[str, Any]:
    """Return one card object, accepting a singleton list of one mapping.

    Lists with zero, multiple, or non-mapping elements are ambiguous and are
    rejected explicitly instead of failing later with a TypeError.
    """

    if isinstance(payload, Mapping):
        return dict(payload)

    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], Mapping):
            return dict(payload[0])
        raise InvalidCardSchemaError(
            "INVALID_LLM_OUTPUT: se esperaba un objeto JSON o una lista "
            "con exactamente un objeto válido."
        )

    raise InvalidCardSchemaError(
        "INVALID_CARD_SCHEMA: la salida LLM no es un objeto JSON."
    )


def is_invalid_title(card: Mapping[str, Any]) -> bool:
    return str(card.get("title", "")).strip().casefold() in INVALID_TITLE_VALUES


def missing_critical_fields(card: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    invalid_strings = {
        "",
        "error",
        "no especificado",
        "none",
        "nan",
    }
    # Contrato SEMÁNTICAMENTE CONDICIONAL (ver card_validation.py,
    # required_fields_for_card): target_domain solo se exige para
    # papers que reclaman un dominio de aplicación concreto (estudios
    # empíricos/domain-specific) -- un paper metodológico/fundacional/
    # de propósito general nunca debe bloquearse por carecer de un
    # dominio que su contribución no reclama.
    for field in required_fields_for_card(dict(card)):
        value = card.get(field)
        if value is None:
            missing.append(field)
        elif isinstance(value, str) and value.strip().casefold() in invalid_strings:
            missing.append(field)
        elif isinstance(value, list) and not value:
            missing.append(field)
    return missing


def _invalid_output_sources(
    extraction_errors: Sequence[Mapping[str, Any]],
) -> set[str]:
    sources: set[str] = set()
    for row in extraction_errors:
        text = (
            str(row.get("error_type", ""))
            + " "
            + str(row.get("error_message", ""))
        ).casefold()
        if any(
            token in text
            for token in (
                "list indices must be integers",
                "invalid_llm_output",
                "invalid_card_schema",
                "json",
                "parse",
            )
        ):
            sources.add(str(row.get("source_filename", "")))
    return sources


def unique_chunks_by_source(
    retrieval_trace_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    chunks: dict[str, set[str]] = defaultdict(set)
    for row in retrieval_trace_rows:
        source = str(row.get("source_filename", ""))
        chunk_id = str(row.get("chunk_id", ""))
        if source and chunk_id:
            chunks[source].add(chunk_id)
    return {source: len(values) for source, values in chunks.items()}


def build_revision_plan(
    cards: Sequence[Mapping[str, Any]],
    extraction_errors: Sequence[Mapping[str, Any]],
    retrieval_trace_rows: Sequence[Mapping[str, Any]],
    *,
    low_evidence_chunk_threshold: int = 3,
) -> list[dict[str, Any]]:
    invalid_output_sources = _invalid_output_sources(extraction_errors)
    chunk_counts = unique_chunks_by_source(retrieval_trace_rows)
    rows: list[dict[str, Any]] = []

    for card in cards:
        source = str(card.get("source_filename", ""))

        # Corpus eligibility gate (ver corpus_eligibility.py): SOLO
        # las fichas INCLUDE entran al quality gate científico -- un
        # documento EXCLUDE (review/fuera de scope) o QUARANTINE
        # (título/metadata irrecuperable, contenido insuficiente,
        # relevancia indeterminable) nunca participa en el plan de
        # revisión, nunca se le exige methods_or_models/evaluation_
        # metrics/main_results, y nunca bloquea Stage03 por sí solo.
        # is_corpus_include trata la ausencia del campo canónico como
        # INCLUDE (retrocompatible con cualquier card que aún no pasó
        # por el eligibility gate). La ficha en sí no se toca aquí --
        # solo se salta del plan.
        if not is_corpus_include(card):
            continue

        invalid_title = is_invalid_title(card)
        missing_fields = missing_critical_fields(card)
        evidence_missing = not bool(card.get("evidence"))
        unique_chunks = int(chunk_counts.get(source, 0))
        invalid_output = source in invalid_output_sources

        if invalid_output:
            primary_reason = "INVALID_LLM_OUTPUT"
            if unique_chunks <= low_evidence_chunk_threshold:
                strategy = "REPAIR_SCHEMA_EXPANDED_EVIDENCE"
            else:
                strategy = "REPAIR_SCHEMA_REUSE_RETRIEVAL"
        elif invalid_title and set(missing_fields).issubset({"title"}):
            primary_reason = "MISSING_OR_INVALID_TITLE"
            strategy = "REPAIR_TITLE_ONLY"
        elif evidence_missing and unique_chunks <= low_evidence_chunk_threshold:
            primary_reason = "INSUFFICIENT_EVIDENCE"
            strategy = "EXPAND_EVIDENCE"
        elif missing_fields:
            primary_reason = "MISSING_CRITICAL_FIELDS"
            strategy = "REPAIR_MISSING_FIELDS"
        elif invalid_title:
            primary_reason = "MISSING_OR_INVALID_TITLE"
            strategy = "REPAIR_TITLE_ONLY"
        else:
            continue

        # Política general (multidominio, nunca depende de nombres de
        # archivo/título/tarea científica concretos): una ficha con
        # tipo documental UNKNOWN (ninguna señal -- ni document_type/
        # paper_type, ni marcador bibliográfico de título -- ver
        # review_exclusion.py) que ADEMÁS resulta inválida por
        # cualquier motivo (aquí es donde SIEMPRE llega el caso: nunca
        # se le "asume" review, así que el gate normal de campos
        # críticos siempre se evalúa primero) se distingue con un
        # reason_code explícito propio -- NUNCA se excluye
        # silenciosamente, y sigue bloqueando exactamente igual que
        # antes (retry/halt), pero la auditoría deja claro que la
        # causa de fondo es "no sabemos si esto es una review o un
        # paper primario", no simplemente "campos faltantes" -- la
        # causa raíz original (missing fields, título inválido, etc.)
        # se conserva en underlying_reason_code para no perder
        # información. Una ficha con tipo documental CONOCIDO (KEEP
        # confiable: paper_type/document_type explícito y distinto de
        # review) sigue el camino histórico sin cambios.
        underlying_reason = primary_reason
        if is_unknown_document_type(card):
            primary_reason = "DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID"

        rows.append({
            "source_filename": source,
            "invalid_title": bool(invalid_title),
            "missing_fields": ";".join(missing_fields),
            "evidence_missing": bool(evidence_missing),
            "retrieved_unique_chunks": unique_chunks,
            "primary_reason_code": primary_reason,
            "underlying_reason_code": underlying_reason,
            "recommended_strategy": strategy,
        })

    return rows


def plan_by_source(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("source_filename", "")): dict(row)
        for row in rows
        if str(row.get("source_filename", ""))
    }
