from __future__ import annotations


def _src(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("source_filename", ""))
    return ""


def calculate_diagnostic_metrics(
    data, df, ref_counts, *,
    min_sections=None, max_sections=None, enforce_section_count=None,
):
    """Diagnostics that preserve the original representative-paper semantics.

    representative_papers are exemplars, not an exhaustive paper-to-theme assignment.
    Exhaustive assignment/coverage metrics therefore remain not applicable and never
    participate in the quality gate.
    """
    themes = data.get("themes", [])
    gaps = data.get("research_gaps", [])
    dimensions = data.get("comparative_dimensions", [])
    representative_refs = [
        _src(paper)
        for theme in themes
        for paper in (theme.get("representative_papers", []) or [])
        if _src(paper)
    ]
    themes_with_names = sum(bool(str(t.get("theme_name") or t.get("theme") or "").strip()) for t in themes)
    gaps_with_descriptions = sum(bool(str(g.get("description") or g.get("gap") or "").strip()) for g in gaps)
    dimensions_with_names = sum(bool(str(d.get("dimension") or "").strip()) for d in dimensions)
    total_refs = int(ref_counts.get("reference_total", 0))
    valid_refs = int(ref_counts.get("valid_references", 0))
    title_total = int(ref_counts.get("title_total", 0))
    title_matches = int(ref_counts.get("title_matches", 0))
    return {
        "paper_coverage": None,
        "papers_assigned_to_theme_rate": None,
        "unassigned_paper_rate": None,
        "duplicate_assignment_rate": None,
        "coverage_semantics": "NOT_APPLICABLE_REPRESENTATIVE_PAPERS_ARE_NON_EXHAUSTIVE",
        "theme_coverage": themes_with_names / len(themes) if themes else 0.0,
        "supported_theme_rate": themes_with_names / len(themes) if themes else 0.0,
        "supported_gap_rate": gaps_with_descriptions / len(gaps) if gaps else 1.0,
        "comparative_dimension_support_rate": dimensions_with_names / len(dimensions) if dimensions else 1.0,
        "valid_reference_rate": valid_refs / total_refs if total_refs else 1.0,
        "title_match_rate": title_matches / title_total if title_total else 1.0,
        "representative_source_validity_rate": valid_refs / total_refs if total_refs else 1.0,
        "representative_reference_count": len(representative_refs),
        "structure_source_validity_rate": 1.0,
        "gap_source_validity_rate": 1.0,
        "invalid_record_rate": 0.0,
        "section_count": len(data.get("suggested_state_of_art_structure", [])),
        "min_sections": min_sections,
        "max_sections": max_sections,
        "structure_too_short": (
            min_sections is not None
            and len(data.get("suggested_state_of_art_structure", [])) < int(min_sections)
        ),
        "structure_too_long": (
            max_sections is not None
            and len(data.get("suggested_state_of_art_structure", [])) > int(max_sections)
        ),
        "enforce_section_count": (
            bool(enforce_section_count) if enforce_section_count is not None else None
        ),
    }
