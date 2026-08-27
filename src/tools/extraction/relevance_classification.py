"""Relevance classification orchestration extracted from notebook 03.

The module preserves the characterized behavior of cell 11. Prompt building,
message construction, LLM invocation, and JSON parsing are injected by the
caller. The module performs no file I/O and reads no global configuration.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence


RELEVANCE_COMPACT_CARD_FIELDS = [
    "source_filename", "title", "paper_type", "research_problem",
    "objective", "task_type", "target_domain", "target_variable_or_object",
    "temporal_horizon_or_scope", "methods_or_models", "method_families",
    "datasets_or_case_study", "input_variables_or_data_sources",
    "evaluation_metrics", "main_results", "reported_best_method_or_model",
    "limitations_or_gaps", "contribution", "relevance_for_state_of_art",
]

RELEVANCE_CLASSIFICATION_RESPONSE_FIELDS = [
    "task_type", "target_domain", "method_families", "relevance_level",
    "include_in_state_of_art", "relevance_reason",
]


def build_relevance_compact_card(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_filename": card.get("source_filename"),
        "title": card.get("title"),
        "paper_type": card.get("paper_type"),
        "research_problem": card.get("research_problem"),
        "objective": card.get("objective"),
        "task_type": card.get("task_type"),
        "target_domain": card.get("target_domain"),
        "target_variable_or_object": card.get("target_variable_or_object"),
        "temporal_horizon_or_scope": card.get("temporal_horizon_or_scope"),
        "methods_or_models": card.get("methods_or_models"),
        "method_families": card.get("method_families"),
        "datasets_or_case_study": card.get("datasets_or_case_study"),
        "input_variables_or_data_sources": card.get("input_variables_or_data_sources"),
        "evaluation_metrics": card.get("evaluation_metrics"),
        "main_results": card.get("main_results"),
        "reported_best_method_or_model": card.get("reported_best_method_or_model"),
        "limitations_or_gaps": card.get("limitations_or_gaps"),
        "contribution": card.get("contribution"),
        "relevance_for_state_of_art": card.get("relevance_for_state_of_art"),
    }


def classify_card_relevance(
    card: Mapping[str, Any],
    *,
    experiment_profile: Any,
    prompt_builder: Callable[..., str],
    llm: Any,
    json_parser: Callable[[Any], Any],
    message_factory: Callable[..., Any] | None = None,
) -> Any:
    compact_card = build_relevance_compact_card(card)
    prompt = prompt_builder(card=compact_card, experiment_profile=experiment_profile)

    if message_factory is None:
        class _InjectedMessageFallback:
            def __init__(self, content: Any) -> None:
                self.content = content
        message_factory = _InjectedMessageFallback

    response = llm.invoke([message_factory(content=prompt)])
    return json_parser(response.content)


def has_valid_classification(card: Mapping[str, Any]) -> bool:
    relevance_level = str(card.get("relevance_level", "")).strip().lower()
    include_value = card.get("include_in_state_of_art")
    reason = str(card.get("relevance_reason", card.get("reason", ""))).strip()

    has_level = relevance_level not in [
        "", "nan", "none", "no especificado", "error",
    ]
    has_include = (
        include_value is not None
        and str(include_value).strip().lower() not in ["", "nan", "none"]
    )
    has_reason = reason.lower() not in ["", "nan", "none", "no especificado"]
    return has_level and has_include and has_reason


def determine_relevance_reclassification(
    cards: Sequence[Mapping[str, Any]],
    *,
    should_rebuild_extraction: Any,
) -> tuple[bool, bool]:
    cards_need_classification = any(
        not has_valid_classification(card) for card in cards
    )
    reclassify_relevance = bool(
        should_rebuild_extraction or cards_need_classification
    )
    return cards_need_classification, reclassify_relevance


def apply_relevance_classification_success(
    card: dict[str, Any],
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    card.update({
        "task_type": classification.get("task_type", card.get("task_type")),
        "target_domain": classification.get("target_domain", card.get("target_domain")),
        "method_families": classification.get(
            "method_families", card.get("method_families", [])
        ),
        "relevance_level": classification.get("relevance_level"),
        "include_in_state_of_art": classification.get("include_in_state_of_art"),
        "relevance_reason": classification.get(
            "relevance_reason", classification.get("reason", "")
        ),
    })
    return card


def apply_relevance_classification_error(
    card: dict[str, Any],
    error: Exception,
    *,
    created_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    card.update({
        "relevance_level": "error",
        "include_in_state_of_art": False,
        "relevance_reason": str(error),
    })
    error_row = {
        "source_filename": card.get("source_filename"),
        "stage": "relevance_classification",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "created_at": created_at,
    }
    return card, error_row


def run_relevance_classification(
    cards: list[dict[str, Any]],
    *,
    should_rebuild_extraction: Any,
    classify: Callable[[dict[str, Any]], Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    cards_need_classification, reclassify_relevance = (
        determine_relevance_reclassification(
            cards,
            should_rebuild_extraction=should_rebuild_extraction,
        )
    )

    if not reclassify_relevance:
        return {
            "cards_need_classification": cards_need_classification,
            "reclassify_relevance": False,
            "cards": cards,
            "errors": [],
            "kb_should_recreate": False,
            "classification_calls": 0,
        }

    classified_cards = []
    extraction_errors = []
    classification_calls = 0

    for card in cards:
        try:
            classification_calls += 1
            classification = classify(card)
            apply_relevance_classification_success(card, classification)
        except Exception as error:
            _, error_row = apply_relevance_classification_error(
                card, error, created_at=created_at
            )
            extraction_errors.append(error_row)
        classified_cards.append(card)

    return {
        "cards_need_classification": cards_need_classification,
        "reclassify_relevance": True,
        "cards": classified_cards,
        "errors": extraction_errors,
        "kb_should_recreate": True,
        "classification_calls": classification_calls,
    }
