from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.config.draft_writing_policy_config import get_draft_writing_policy


_ORGANIZATIONAL_SOURCE_REQUIREMENTS = {
    "none",
    "no_sources",
    "source_free",
    "organizational",
    "not_required",
}
_SUBSTANTIVE_SOURCE_REQUIREMENTS = {
    "required",
    "requires_sources",
    "sources_required",
    "evidence_required",
    "documentary_evidence_required",
}
_ORGANIZATIONAL_SECTION_TYPES = {
    "organizational",
    "introduction",
    "introductory",
    "conclusion",
    "conclusions",
    "closing",
}
_SUBSTANTIVE_SECTION_TYPES = {
    "substantive",
    "body",
    "analysis",
    "thematic_analysis",
    "discussion",
    "methods",
    "results",
    "synthesis",
}
_ORGANIZATIONAL_TITLES = {
    "introduccion",
    "introducción",
    "introduction",
    "conclusion",
    "conclusión",
    "conclusiones",
    "conclusions",
    "cierre",
    "closing",
}


@dataclass(frozen=True)
class SectionClassification:
    section_id: str
    budget_type: str
    reason: str

    @property
    def is_organizational(self) -> bool:
        return self.budget_type == "source_free_organizational"


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalized_label(value: Any) -> str:
    return _safe_text(value).casefold().replace("-", "_").replace(" ", "_")


def _require_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"SOURCE_AWARE_BUDGET_INVALID_TYPE:{field_name}:expected_integer"
        )
    return value


def _explicit_structural_decisions(section: Mapping[str, Any]) -> list[tuple[bool, str]]:
    decisions: list[tuple[bool, str]] = []

    if "requires_sources" in section and section.get("requires_sources") is not None:
        value = section.get("requires_sources")
        if not isinstance(value, bool):
            raise ValueError(
                "SOURCE_AWARE_BUDGET_INVALID_TYPE:requires_sources:expected_boolean"
            )
        decisions.append((not value, "requires_sources"))

    source_requirement = _normalized_label(section.get("source_requirement"))
    if source_requirement:
        if source_requirement in _ORGANIZATIONAL_SOURCE_REQUIREMENTS:
            decisions.append((True, "source_requirement"))
        elif source_requirement in _SUBSTANTIVE_SOURCE_REQUIREMENTS:
            decisions.append((False, "source_requirement"))

    section_type = _normalized_label(section.get("section_type"))
    if section_type:
        if section_type in _ORGANIZATIONAL_SECTION_TYPES:
            decisions.append((True, "section_type"))
        elif section_type in _SUBSTANTIVE_SECTION_TYPES:
            decisions.append((False, "section_type"))

    return decisions


def classify_section_for_budget(section: Mapping[str, Any]) -> SectionClassification:
    if not isinstance(section, Mapping):
        raise ValueError("SOURCE_AWARE_BUDGET_INVALID_TYPE:section:expected_mapping")

    section_id = _safe_text(section.get("section_id"))
    if not section_id:
        raise ValueError("SOURCE_AWARE_BUDGET_INVALID:section_id:must_be_nonempty")

    structural = _explicit_structural_decisions(section)
    if structural:
        decisions = {decision for decision, _ in structural}
        if len(decisions) > 1:
            raise ValueError(
                f"SOURCE_AWARE_BUDGET_CONFLICTING_STRUCTURAL_SIGNALS:{section_id}"
            )
        is_organizational = next(iter(decisions))
        reasons = "+".join(sorted({reason for _, reason in structural}))
        return SectionClassification(
            section_id=section_id,
            budget_type=(
                "source_free_organizational"
                if is_organizational
                else "source_aware_substantive"
            ),
            reason=f"structured:{reasons}",
        )

    if "papers_to_use" in section:
        papers = section.get("papers_to_use")
        if papers is None:
            papers = []
        if isinstance(papers, (str, bytes)) or not isinstance(papers, Sequence):
            raise ValueError(
                f"SOURCE_AWARE_BUDGET_INVALID_TYPE:papers_to_use:{section_id}:expected_sequence"
            )
        is_organizational = len(papers) == 0
        return SectionClassification(
            section_id=section_id,
            budget_type=(
                "source_free_organizational"
                if is_organizational
                else "source_aware_substantive"
            ),
            reason=(
                "papers_to_use:empty"
                if is_organizational
                else "papers_to_use:assigned"
            ),
        )

    title = _safe_text(
        section.get("section_title", section.get("title", ""))
    ).casefold()
    if title in _ORGANIZATIONAL_TITLES:
        return SectionClassification(
            section_id=section_id,
            budget_type="source_free_organizational",
            reason="title_fallback:organizational",
        )

    return SectionClassification(
        section_id=section_id,
        budget_type="source_aware_substantive",
        reason="title_fallback:substantive_default",
    )


def _distribute_exact(total: int, count: int) -> list[int]:
    if count <= 0:
        if total == 0:
            return []
        raise ValueError(
            "SOURCE_AWARE_BUDGET_IMPOSSIBLE:positive_total_without_sections"
        )
    base, residue = divmod(total, count)
    return [base + (1 if index < residue else 0) for index in range(count)]


def _build_organizational_budget(target: int, policy: Mapping[str, Any]) -> dict[str, Any]:
    minimum = int(policy["organizational_minimum_words"])
    maximum = int(policy["organizational_maximum_words"])
    if not minimum <= target <= maximum:
        raise ValueError(
            "SOURCE_AWARE_BUDGET_IMPOSSIBLE:organizational_target_outside_configured_bounds"
        )
    return {
        "target_words": target,
        "minimum_words": minimum,
        "maximum_words": maximum,
        "budget_type": "source_free_organizational",
    }


def _build_substantive_budget(target: int, policy: Mapping[str, Any]) -> dict[str, Any]:
    if target < 1:
        raise ValueError(
            "SOURCE_AWARE_BUDGET_IMPOSSIBLE:substantive_section_requires_at_least_one_target_word"
        )
    minimum = max(1, int(target * float(policy["substantive_minimum_ratio"])))
    maximum = max(target, int(target * float(policy["substantive_maximum_ratio"])))
    return {
        "target_words": target,
        "minimum_words": minimum,
        "maximum_words": maximum,
        "budget_type": "source_aware_substantive",
    }


def assign_source_aware_section_budgets(
    outline_sections: Sequence[Mapping[str, Any]] | None,
    target_total_words: int,
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Assign deterministic, exact-total budgets to every outline section.

    Structural section signals take priority over paper assignments. Paper
    assignments take priority over title vocabulary. Title vocabulary is only a
    compatibility fallback when no structured or paper-assignment signal is
    available.
    """
    total = _require_integer(target_total_words, "target_total_words")
    if total < 0:
        raise ValueError(
            "SOURCE_AWARE_BUDGET_INVALID:target_total_words:must_be_greater_than_or_equal_to_0"
        )

    if outline_sections is None:
        sections: list[Mapping[str, Any]] = []
    elif isinstance(outline_sections, (str, bytes)) or not isinstance(
        outline_sections, Sequence
    ):
        raise ValueError(
            "SOURCE_AWARE_BUDGET_INVALID_TYPE:outline_sections:expected_sequence"
        )
    else:
        sections = list(outline_sections)

    if not sections:
        if total == 0:
            return {}
        raise ValueError(
            "SOURCE_AWARE_BUDGET_IMPOSSIBLE:target_words_without_sections"
        )

    resolved_policy = get_draft_writing_policy(policy)
    classifications = [classify_section_for_budget(section) for section in sections]
    section_ids = [classification.section_id for classification in classifications]
    if len(section_ids) != len(set(section_ids)):
        raise ValueError("SOURCE_AWARE_BUDGET_INVALID:section_id:must_be_unique")

    organizational = [item for item in classifications if item.is_organizational]
    substantive = [item for item in classifications if not item.is_organizational]

    if total < len(sections):
        raise ValueError(
            "SOURCE_AWARE_BUDGET_IMPOSSIBLE:target_total_words_smaller_than_section_count"
        )

    organizational_minimum = int(resolved_policy["organizational_minimum_words"])
    organizational_maximum = int(resolved_policy["organizational_maximum_words"])
    organizational_preferred = int(resolved_policy["organizational_target_words"])

    if organizational and not substantive:
        aggregate_minimum = len(organizational) * organizational_minimum
        aggregate_maximum = len(organizational) * organizational_maximum
        if total < aggregate_minimum or total > aggregate_maximum:
            raise ValueError(
                "SOURCE_AWARE_BUDGET_IMPOSSIBLE:organizational_only_total_outside_configured_bounds"
            )
        organizational_targets = _distribute_exact(total, len(organizational))
        substantive_targets: list[int] = []
    elif organizational:
        minimum_organizational_total = len(organizational) * organizational_minimum
        maximum_organizational_total = min(
            len(organizational) * organizational_maximum,
            total - len(substantive),
        )
        if maximum_organizational_total < minimum_organizational_total:
            raise ValueError(
                "SOURCE_AWARE_BUDGET_IMPOSSIBLE:insufficient_words_for_configured_organizational_minimums"
            )
        preferred_total = len(organizational) * organizational_preferred
        organizational_total = min(
            max(preferred_total, minimum_organizational_total),
            maximum_organizational_total,
        )
        organizational_targets = _distribute_exact(
            organizational_total,
            len(organizational),
        )
        substantive_targets = _distribute_exact(
            total - organizational_total,
            len(substantive),
        )
    else:
        organizational_targets = []
        substantive_targets = _distribute_exact(total, len(substantive))

    organizational_by_id = {
        item.section_id: target
        for item, target in zip(organizational, organizational_targets, strict=True)
    }
    substantive_by_id = {
        item.section_id: target
        for item, target in zip(substantive, substantive_targets, strict=True)
    }

    budgets: dict[str, dict[str, Any]] = {}
    for classification in classifications:
        if classification.is_organizational:
            budget = _build_organizational_budget(
                organizational_by_id[classification.section_id],
                resolved_policy,
            )
        else:
            budget = _build_substantive_budget(
                substantive_by_id[classification.section_id],
                resolved_policy,
            )
        budget["classification_reason"] = classification.reason
        budgets[classification.section_id] = budget

    if sum(item["target_words"] for item in budgets.values()) != total:
        raise RuntimeError("SOURCE_AWARE_BUDGET_INTERNAL_ERROR:target_sum_mismatch")
    return budgets
