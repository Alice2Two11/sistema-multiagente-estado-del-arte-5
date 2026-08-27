"""Card validation and deterministic row-building helpers from notebook 03.

This module preserves the characterized behavior of cells 7, 8, and 9.
It performs no LLM calls, no repair, no relevance classification, and no I/O.
"""

from __future__ import annotations

from typing import Any


CARD_LIST_FIELDS = [
    "methods_or_models",
    "method_families",
    "input_variables_or_data_sources",
    "evaluation_metrics",
    "evidence",
]

CARD_REQUIRED_FIELDS = [
    "source_filename",
    "title",
    "research_problem",
    "objective",
    "task_type",
    "target_domain",
    "methods_or_models",
    "main_results",
    "evidence",
]

# Contrato de campos críticos SEMÁNTICAMENTE CONDICIONAL: no todo
# campo en CARD_REQUIRED_FIELDS aplica a todo tipo de paper por igual.
# target_domain describe el DOMINIO DE APLICACIÓN de un estudio
# empírico/domain-specific -- para un paper metodológico, fundacional
# o de propósito general, el dominio de aplicación simplemente no es
# un atributo que el paper reclame (su contribución es el método en
# sí, no un dominio concreto). Exigirlo ahí fuerza al LLM a inventar
# un dominio que el paper nunca declara -- exactamente lo que este
# contrato evita: NUNCA se rellena, se reconoce como no aplicable.
#
# UNIVERSAL_REQUIRED_FIELDS: exigidos para CUALQUIER tipo de paper,
# independientemente de su rol científico -- identidad documental y
# científica básica.
UNIVERSAL_REQUIRED_FIELDS = [
    field for field in CARD_REQUIRED_FIELDS if field != "target_domain"
]

# CONDITIONALLY_REQUIRED_FIELDS: solo se exigen cuando el paper SÍ es
# de un tipo que reclama un dominio de aplicación (estudio empírico/
# domain-specific) -- ver is_domain_agnostic_paper().
CONDITIONALLY_REQUIRED_FIELDS = ["target_domain"]

# Marcadores de ROL científico del paper (paper_type/task_type) que
# indican que es metodológico/fundacional/de propósito general --
# términos que describen el TIPO de contribución (nunca un dominio de
# aplicación concreto como ECG, NLP, energía solar, ciberseguridad,
# etc.), así que esta lista es multidominio por construcción: nunca
# necesita ampliarse para un dominio nuevo.
_DOMAIN_AGNOSTIC_ROLE_MARKERS = frozenset({
    "methodological_proposal", "methodological", "method_proposal",
    "methodology_proposal", "foundational_method", "foundational",
    "general_method", "general_purpose_method", "general purpose method",
    "algorithm_proposal", "framework_proposal", "framework",
    "theoretical", "theoretical_contribution", "architecture_proposal",
    "model_proposal", "domain_agnostic", "domain-agnostic",
})

# Valores de target_domain que YA declaran explícitamente "sin dominio
# de aplicación concreto" -- una señal directa, independiente de
# paper_type/task_type. Nunca deben tratarse como "campo faltante".
_DOMAIN_AGNOSTIC_TARGET_DOMAIN_VALUES = frozenset({
    "general", "general purpose", "general-purpose", "domain-agnostic",
    "domain agnostic", "n/a", "not applicable", "no aplica",
})


def _normalize_role(value: Any) -> str:
    return str(value or "").strip().casefold()


def is_domain_agnostic_paper(card: dict[str, Any]) -> bool:
    """True si el ROL científico del paper (``paper_type``/
    ``task_type``, o un ``target_domain`` ya declarado explícitamente
    como "general"/"domain-agnostic") indica que es metodológico,
    fundacional o de propósito general -- nunca tiene un dominio de
    aplicación concreto que reportar. Puramente determinista, sin LLM,
    sin vocabulario de ningún dominio específico -- multidominio por
    construcción."""

    if _normalize_role(card.get("paper_type")) in _DOMAIN_AGNOSTIC_ROLE_MARKERS:
        return True
    if _normalize_role(card.get("task_type")) in _DOMAIN_AGNOSTIC_ROLE_MARKERS:
        return True
    if _normalize_role(card.get("target_domain")) in _DOMAIN_AGNOSTIC_TARGET_DOMAIN_VALUES:
        return True
    return False


def required_fields_for_card(card: dict[str, Any]) -> list[str]:
    """Contrato de campos críticos para ESTA ficha concreta,
    semánticamente condicional: los campos universales siempre se
    exigen; ``target_domain`` solo se exige si el paper NO es
    domain-agnostic (``is_domain_agnostic_paper``). Nunca inventa ni
    rellena valores -- solo decide qué campos participan en la
    validación de completitud."""

    fields = list(UNIVERSAL_REQUIRED_FIELDS)
    if not is_domain_agnostic_paper(card):
        fields.extend(CONDITIONALLY_REQUIRED_FIELDS)
    return fields

SUMMARY_COLUMNS = [
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
    "retrieved_chunk_ids",
    "num_evidence_items",
]

QUALITY_COLUMNS = [
    "source_filename",
    "title",
    "missing_fields",
    "num_missing_fields",
    "num_evidence_items",
    "num_retrieved_chunks",
    "methods_or_models",
    "method_families",
    "evaluation_metrics",
    "reported_best_method_or_model",
]


def is_bad_card(card: dict[str, Any]) -> bool:
    title = str(card.get("title", "")).strip().lower()
    return title in ["", "error", "no especificado", "nan"]


def normalize_card(
    card: dict[str, Any],
    source_filename: str,
    *,
    card_list_fields: list[str] = CARD_LIST_FIELDS,
) -> dict[str, Any]:
    card["source_filename"] = source_filename

    for field in card_list_fields:
        value = card.get(field)

        if value is None:
            card[field] = []

        elif isinstance(value, str):
            if value.strip().lower() in [
                "",
                "no especificado",
                "none",
                "nan",
            ]:
                card[field] = []
            else:
                card[field] = [value]

        elif not isinstance(value, list):
            card[field] = [value]

    return card


def list_to_str(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(
            str(item)
            for item in value
        )

    return str(value or "")


def first_available(
    card: dict[str, Any],
    *keys: str,
    default: Any = "",
) -> Any:
    for key in keys:
        value = card.get(key)

        if value not in [
            None,
            "",
            [],
            {},
        ]:
            return value

    return default


def build_summary_row(
    card: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_filename": card.get(
            "source_filename"
        ),
        "title": card.get("title"),
        "paper_type": card.get("paper_type"),
        "research_problem": card.get(
            "research_problem"
        ),
        "objective": card.get("objective"),
        "task_type": card.get("task_type"),
        "target_domain": card.get(
            "target_domain"
        ),
        "target_variable_or_object": card.get(
            "target_variable_or_object"
        ),
        "temporal_horizon_or_scope": card.get(
            "temporal_horizon_or_scope"
        ),
        "methods_or_models": list_to_str(
            card.get("methods_or_models")
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
            card.get("evaluation_metrics")
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
        "relevance_for_state_of_art": card.get(
            "relevance_for_state_of_art"
        ),
        "domain_specific_notes": card.get(
            "domain_specific_notes"
        ),
        "retrieved_chunk_ids": list_to_str(
            card.get(
                "retrieved_chunk_ids",
                [],
            )
        ),
        "num_evidence_items": (
            len(card.get("evidence", []))
            if isinstance(
                card.get("evidence"),
                list,
            )
            else 0
        ),
    }


def build_quality_row(
    card: dict[str, Any],
    *,
    required_fields: list[str] = CARD_REQUIRED_FIELDS,
) -> dict[str, Any]:
    missing_fields = []

    for field in required_fields:
        value = card.get(field)

        if value is None:
            missing_fields.append(field)

        elif (
            isinstance(value, str)
            and value.strip().lower()
            in ["", "no especificado"]
        ):
            missing_fields.append(field)

        elif (
            isinstance(value, list)
            and len(value) == 0
        ):
            missing_fields.append(field)

    evidence_count = (
        len(card.get("evidence", []))
        if isinstance(
            card.get("evidence"),
            list,
        )
        else 0
    )

    return {
        "source_filename": card.get(
            "source_filename"
        ),
        "title": card.get("title"),
        "missing_fields": "; ".join(
            missing_fields
        ),
        "num_missing_fields": len(
            missing_fields
        ),
        "num_evidence_items": (
            evidence_count
        ),
        "num_retrieved_chunks": len(
            card.get(
                "retrieved_chunk_ids",
                [],
            )
        ),
        "methods_or_models": list_to_str(
            card.get("methods_or_models")
        ),
        "method_families": list_to_str(
            card.get("method_families")
        ),
        "evaluation_metrics": list_to_str(
            card.get("evaluation_metrics")
        ),
        "reported_best_method_or_model": (
            card.get(
                "reported_best_method_or_model"
            )
        ),
    }
