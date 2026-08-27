"""Title-repair helpers extracted from notebook 03 cell 10.

The module preserves the characterized behavior of the notebook. It receives
all runtime dependencies by parameter, performs no file I/O, reads no global
configuration, and never constructs an LLM client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Sequence

import pandas as pd

from .retrieval import get_first_chunks_context


TITLE_REPAIR_INVALID_VALUES = [
    "",
    "no especificado",
]


class _DefaultMessage:
    """Minimal message object used when no external factory is supplied."""

    def __init__(self, content: Any):
        self.content = content


def select_missing_title_cards(
    cards: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        card
        for card in cards
        if str(
            card.get("title", "")
        ).strip().lower()
        in [
            "",
            "no especificado",
        ]
    ]


def build_title_repair_prompt(
    source_filename: Any,
    context: Any,
) -> str:
    return f"""
Extrae el título exacto del paper usando SOLO el contexto.

Reglas:
- No inventes.
- Si no aparece el título, responde "no especificado".
- Devuelve SOLO JSON válido.

Archivo:
{source_filename}

Contexto:
{context}

Formato:
{{
  "title": ""
}}
"""


def repair_title_with_llm(
    source_filename: Any,
    *,
    df_chunks_clean: pd.DataFrame,
    title_repair_first_chunks: int,
    repair_llm: Any,
    json_parser: Callable[[Any], Any],
    message_factory: Callable[..., Any] = _DefaultMessage,
    on_before_invoke: Callable[[], None] | None = None,
) -> str:
    context = get_first_chunks_context(
        source_filename,
        title_repair_first_chunks,
        df_chunks_clean=df_chunks_clean,
    )

    prompt = build_title_repair_prompt(
        source_filename,
        context,
    )

    if on_before_invoke is not None:
        on_before_invoke()

    response = repair_llm.invoke([
        message_factory(
            content=prompt
        )
    ])

    data = json_parser(
        response.content
    )

    title = str(
        data.get("title", "")
    ).strip()

    return (
        title
        if title
        else "no especificado"
    )


def run_title_repair(
    cards: list[dict[str, Any]],
    *,
    df_chunks_clean: pd.DataFrame,
    title_repair_first_chunks: int,
    repair_llm: Any,
    json_parser: Callable[[Any], Any],
    should_rebuild_extraction: Any,
    message_factory: Callable[..., Any] = _DefaultMessage,
    created_at: str | None = None,
    timestamp_factory: Callable[[], str] | None = None,
    extraction_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    extraction_errors = (
        extraction_errors
        if extraction_errors is not None
        else []
    )

    missing_title_cards = (
        select_missing_title_cards(cards)
    )

    repair_titles = (
        should_rebuild_extraction
        and bool(missing_title_cards)
    )

    llm_calls = 0

    def count_llm_call() -> None:
        nonlocal llm_calls
        llm_calls += 1

    if repair_titles:
        for card in missing_title_cards:
            source_filename = card.get(
                "source_filename"
            )

            try:
                card["title"] = (
                    repair_title_with_llm(
                        source_filename,
                        df_chunks_clean=(
                            df_chunks_clean
                        ),
                        title_repair_first_chunks=(
                            title_repair_first_chunks
                        ),
                        repair_llm=repair_llm,
                        json_parser=json_parser,
                        message_factory=(
                            message_factory
                        ),
                        on_before_invoke=(
                            count_llm_call
                        ),
                    )
                )

            except Exception as error:
                if timestamp_factory is not None:
                    error_created_at = (
                        timestamp_factory()
                    )
                elif created_at is not None:
                    error_created_at = created_at
                else:
                    error_created_at = (
                        datetime.now().isoformat()
                    )

                extraction_errors.append({
                    "source_filename": (
                        source_filename
                    ),
                    "stage": "title_repair",
                    "error_type": (
                        type(error).__name__
                    ),
                    "error_message": str(error),
                    "created_at": (
                        error_created_at
                    ),
                })

    return {
        "cards": cards,
        "missing_title_cards": (
            missing_title_cards
        ),
        "repair_titles": repair_titles,
        "extraction_errors": (
            extraction_errors
        ),
        "llm_calls": llm_calls,
    }
