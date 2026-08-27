"""Initial card extraction and bad-card repair from notebook 03.

This module preserves the characterized behavior of cells 6 and 7. Retrieval,
context construction, prompt building, LLM calls, JSON parsing, message
construction, and raw-error writing are all injected by the caller.

It performs no CSV/JSONL I/O, opens no Chroma collection, builds no client,
creates no directories, and reads no global configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .card_validation import is_bad_card, normalize_card


class _DefaultMessage:
    """Minimal message used only when no external factory is supplied."""

    def __init__(self, content: Any):
        self.content = content


REPAIR_PROMPT_SUFFIX = """

IMPORTANTE FINAL:
Devuelve únicamente un objeto JSON válido.
No escribas explicación antes ni después.
No uses comillas sin escapar dentro de los valores.
Si no sabes un dato, usa "no especificado".
"""

INITIAL_ERROR_CARD_FIELDS = [
    "source_filename",
    "title",
    "paper_type",
    "research_problem",
    "objective",
    "task_type",
    "target_domain",
    "target_variable_or_object",
    "temporal_horizon_or_scope",
    "methods_or_models",
    "method_families",
    "datasets_or_case_study",
    "input_variables_or_data_sources",
    "evaluation_metrics",
    "main_results",
    "reported_best_method_or_model",
    "limitations_or_gaps",
    "contribution",
    "relevance_for_state_of_art",
    "domain_specific_notes",
    "evidence",
    "retrieved_chunk_ids",
]


def build_initial_error_card(
    source_filename: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "source_filename": source_filename,
        "title": "error",
        "paper_type": "error",
        "research_problem": "error",
        "objective": "error",
        "task_type": "error",
        "target_domain": "error",
        "target_variable_or_object": "error",
        "temporal_horizon_or_scope": "error",
        "methods_or_models": [],
        "method_families": [],
        "datasets_or_case_study": "error",
        "input_variables_or_data_sources": [],
        "evaluation_metrics": [],
        "main_results": "error",
        "reported_best_method_or_model": "error",
        "limitations_or_gaps": "error",
        "contribution": "error",
        "relevance_for_state_of_art": "error",
        "domain_specific_notes": str(error),
        "evidence": [],
        "retrieved_chunk_ids": [],
    }


def extract_card_for_source(
    source_filename: str,
    *,
    retrieve: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    build_context: Callable[..., str],
    prompt_builder: Callable[..., str],
    experiment_profile: Any,
    llm: Any,
    json_parser: Callable[[Any], Any],
    message_factory: Callable[..., Any] = _DefaultMessage,
    max_chunks_per_paper: int,
    max_context_chars: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    selected_chunks, trace_rows = retrieve(
        source_filename=source_filename,
        max_chunks=max_chunks_per_paper,
    )

    context = build_context(
        selected_chunks=selected_chunks,
        max_context_chars=max_context_chars,
    )

    prompt = prompt_builder(
        source_filename=source_filename,
        context=context,
        experiment_profile=experiment_profile,
    )

    response = llm.invoke([
        message_factory(
            content=prompt
        )
    ])

    card = json_parser(
        response.content
    )

    card["source_filename"] = (
        source_filename
    )

    card["retrieved_chunk_ids"] = [
        chunk["chunk_id"]
        for chunk in selected_chunks
    ]

    return (
        card,
        trace_rows,
    )


def run_initial_extraction(
    source_filenames: Sequence[str],
    *,
    retrieve: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    build_context: Callable[..., str],
    prompt_builder: Callable[..., str],
    experiment_profile: Any,
    llm: Any,
    json_parser: Callable[[Any], Any],
    message_factory: Callable[..., Any] = _DefaultMessage,
    max_chunks_per_paper: int,
    max_context_chars: int,
    created_at: str,
) -> dict[str, Any]:
    cards = []
    retrieval_trace_rows = []
    extraction_errors = []
    llm_calls = 0

    for source_filename in source_filenames:
        try:
            selected_chunks, trace_rows = retrieve(
                source_filename=source_filename,
                max_chunks=max_chunks_per_paper,
            )

            retrieval_trace_rows.extend(
                trace_rows
            )

            context = build_context(
                selected_chunks=selected_chunks,
                max_context_chars=max_context_chars,
            )

            prompt = prompt_builder(
                source_filename=source_filename,
                context=context,
                experiment_profile=experiment_profile,
            )

            llm_calls += 1

            response = llm.invoke([
                message_factory(
                    content=prompt
                )
            ])

            card = json_parser(
                response.content
            )

            card["source_filename"] = (
                source_filename
            )

            card["retrieved_chunk_ids"] = [
                chunk["chunk_id"]
                for chunk in selected_chunks
            ]

            cards.append(card)

        except Exception as error:
            extraction_errors.append({
                "source_filename": (
                    source_filename
                ),
                "stage": (
                    "initial_extraction"
                ),
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(
                    error
                ),
                "created_at": (
                    created_at
                ),
            })

            cards.append(
                build_initial_error_card(
                    source_filename,
                    error,
                )
            )

    return {
        "cards": cards,
        "retrieval_trace_rows": (
            retrieval_trace_rows
        ),
        "extraction_errors": (
            extraction_errors
        ),
        "llm_calls": llm_calls,
    }


def generate_repaired_card_for_source(
    source_filename: str,
    *,
    retrieve: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    build_context: Callable[..., str],
    prompt_builder: Callable[..., str],
    experiment_profile: Any,
    repair_llm: Any,
    json_parser: Callable[[Any], Any],
    message_factory: Callable[..., Any] = _DefaultMessage,
    repair_max_chunks_per_paper: int,
    repair_max_context_chars: int,
) -> tuple[
    dict[str, Any],
    str,
    list[dict[str, Any]],
]:
    selected_chunks, trace_rows = retrieve(
        source_filename=source_filename,
        max_chunks=(
            repair_max_chunks_per_paper
        ),
    )

    context = build_context(
        selected_chunks=selected_chunks,
        max_context_chars=(
            repair_max_context_chars
        ),
    )

    prompt = prompt_builder(
        source_filename=source_filename,
        context=context,
        experiment_profile=experiment_profile,
    )

    prompt += REPAIR_PROMPT_SUFFIX

    response = repair_llm.invoke([
        message_factory(
            content=prompt
        )
    ])

    raw_response = response.content

    card = json_parser(
        raw_response
    )

    card = normalize_card(
        card,
        source_filename,
    )

    card["retrieved_chunk_ids"] = [
        chunk["chunk_id"]
        for chunk in selected_chunks
    ]

    return (
        card,
        raw_response,
        trace_rows,
    )


def raw_error_filename(
    source_filename: str,
) -> str:
    safe_name = (
        source_filename
        .replace("/", "_")
        .replace("\\", "_")[:120]
    )

    return f"{safe_name}.txt"


def run_bad_card_repair(
    cards: list[dict[str, Any]],
    *,
    should_rebuild_extraction: Any,
    generate_card: Callable[
        [str],
        tuple[
            dict[str, Any],
            str,
            list[dict[str, Any]],
        ],
    ],
    created_at: str,
    raw_errors_directory: str | Path,
    raw_writer: Any,
    retrieval_trace_rows: list[dict[str, Any]] | None = None,
    extraction_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retrieval_trace_rows = (
        retrieval_trace_rows
        if retrieval_trace_rows is not None
        else []
    )

    extraction_errors = (
        extraction_errors
        if extraction_errors is not None
        else []
    )

    run_bad_card_repair = (
        should_rebuild_extraction
    )

    bad_sources = [
        card["source_filename"]
        for card in cards
        if is_bad_card(card)
    ]

    if not run_bad_card_repair:
        return {
            "cards": cards,
            "bad_sources": bad_sources,
            "retrieval_trace_rows": (
                retrieval_trace_rows
            ),
            "extraction_errors": (
                extraction_errors
            ),
            "new_cards_by_source": {},
            "bad_after_repair": (
                bad_sources
            ),
            "repair_calls": 0,
        }

    if not bad_sources:
        return {
            "cards": cards,
            "bad_sources": [],
            "retrieval_trace_rows": (
                retrieval_trace_rows
            ),
            "extraction_errors": (
                extraction_errors
            ),
            "new_cards_by_source": {},
            "bad_after_repair": [],
            "repair_calls": 0,
        }

    new_cards_by_source = {}
    repair_calls = 0

    for source_filename in bad_sources:
        raw_response = ""

        try:
            repair_calls += 1

            (
                repaired_card,
                raw_response,
                trace_rows,
            ) = generate_card(
                source_filename
            )

            retrieval_trace_rows.extend(
                trace_rows
            )

            if is_bad_card(
                repaired_card
            ):
                raise ValueError(
                    "La ficha reparada conserva "
                    "un título inválido."
                )

            new_cards_by_source[
                source_filename
            ] = repaired_card

        except Exception as error:
            extraction_errors.append({
                "source_filename": (
                    source_filename
                ),
                "stage": "repair",
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(
                    error
                ),
                "created_at": (
                    created_at
                ),
            })

            raw_path = (
                Path(raw_errors_directory)
                / raw_error_filename(
                    source_filename
                )
            )

            raw_writer.write(
                raw_path,
                str(raw_response),
                encoding="utf-8",
            )

    updated_cards = []

    for card in cards:
        source_filename = card[
            "source_filename"
        ]

        updated_cards.append(
            new_cards_by_source.get(
                source_filename,
                card,
            )
        )

    bad_after_repair = [
        card["source_filename"]
        for card in updated_cards
        if is_bad_card(card)
    ]

    return {
        "cards": updated_cards,
        "bad_sources": bad_sources,
        "retrieval_trace_rows": (
            retrieval_trace_rows
        ),
        "extraction_errors": (
            extraction_errors
        ),
        "new_cards_by_source": (
            new_cards_by_source
        ),
        "bad_after_repair": (
            bad_after_repair
        ),
        "repair_calls": (
            repair_calls
        ),
    }
