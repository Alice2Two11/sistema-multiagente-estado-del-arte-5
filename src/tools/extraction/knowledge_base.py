"""Deterministic Scientific Knowledge Base construction from notebook 03.

This module preserves the characterized behavior of cell 12. It performs no
file I/O. Existence checks and any previously loaded CSV DataFrame are supplied
by the caller.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .card_validation import first_available, list_to_str


KNOWLEDGE_BASE_COLUMNS = [
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
    "relevance_level",
    "include_in_state_of_art",
    "relevance_reason",
    "retrieved_chunk_ids",
    "num_evidence_items",
]


def build_knowledge_base_rows(
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    kb_rows = []

    for card in cards:
        kb_rows.append({
            "source_filename": card.get(
                "source_filename"
            ),
            "title": card.get("title"),
            "paper_type": card.get(
                "paper_type"
            ),
            "research_problem": card.get(
                "research_problem"
            ),
            "objective": card.get(
                "objective"
            ),
            "task_type": card.get(
                "task_type"
            ),
            "target_domain": card.get(
                "target_domain"
            ),
            "target_variable_or_object": (
                card.get(
                    "target_variable_or_object"
                )
            ),
            "temporal_horizon_or_scope": (
                card.get(
                    "temporal_horizon_or_scope"
                )
            ),
            "methods_or_models": list_to_str(
                card.get(
                    "methods_or_models"
                )
            ),
            "method_families": list_to_str(
                first_available(
                    card,
                    "method_families",
                    "method_family",
                    default=[],
                )
            ),
            "datasets_or_case_study": (
                first_available(
                    card,
                    "datasets_or_case_study",
                    "dataset_or_case_study",
                )
            ),
            "input_variables_or_data_sources": (
                list_to_str(
                    first_available(
                        card,
                        "input_variables_or_data_sources",
                        "input_variables",
                        default=[],
                    )
                )
            ),
            "evaluation_metrics": list_to_str(
                card.get(
                    "evaluation_metrics"
                )
            ),
            "main_results": card.get(
                "main_results"
            ),
            "reported_best_method_or_model": (
                first_available(
                    card,
                    "reported_best_method_or_model",
                    "reported_best_model",
                )
            ),
            "limitations_or_gaps": card.get(
                "limitations_or_gaps"
            ),
            "contribution": card.get(
                "contribution"
            ),
            "relevance_for_state_of_art": (
                card.get(
                    "relevance_for_state_of_art"
                )
            ),
            "domain_specific_notes": (
                card.get(
                    "domain_specific_notes"
                )
            ),
            "relevance_level": card.get(
                "relevance_level"
            ),
            "include_in_state_of_art": (
                card.get(
                    "include_in_state_of_art"
                )
            ),
            "relevance_reason": card.get(
                "relevance_reason",
                card.get("reason", ""),
            ),
            "retrieved_chunk_ids": (
                list_to_str(
                    card.get(
                        "retrieved_chunk_ids",
                        [],
                    )
                )
            ),
            "num_evidence_items": (
                len(
                    card.get(
                        "evidence",
                        [],
                    )
                )
                if isinstance(
                    card.get("evidence"),
                    list,
                )
                else 0
            ),
        })

    return kb_rows


def execute_knowledge_base_branch(
    cards: list[dict[str, Any]],
    *,
    kb_csv_exists: bool,
    kb_jsonl_exists: bool,
    kb_should_recreate: bool,
    existing_csv_dataframe: pd.DataFrame | None = None,
) -> tuple[
    str,
    pd.DataFrame,
    list[dict[str, Any]] | None,
]:
    if (
        kb_csv_exists
        and kb_jsonl_exists
        and not kb_should_recreate
    ):
        if existing_csv_dataframe is None:
            raise ValueError(
                "existing_csv_dataframe is required to characterize pd.read_csv"
            )

        return (
            "reused",
            existing_csv_dataframe.copy(),
            None,
        )

    kb_rows = build_knowledge_base_rows(
        cards
    )
    df_kb = pd.DataFrame(kb_rows)

    return (
        "created",
        df_kb,
        kb_rows,
    )
