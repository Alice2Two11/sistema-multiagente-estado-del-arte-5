from __future__ import annotations

from copy import deepcopy
from typing import Any


THEMATIC_ALIAS_VERSION = "v16_deterministic_alias_mapping_1"


def _list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise ValueError("INVALID_THEMATIC_SCHEMA")


def _first_text(record: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_value(record: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _normalize_sources(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)):
        return [value]
    return []


def _record_repair(repairs: list[dict], *, block: str, index: int, action: str, source_key: str | None = None, target_key: str | None = None, value: Any = None) -> None:
    item = {"type": action, "block": block, "index": index}
    if source_key is not None:
        item["source_key"] = source_key
    if target_key is not None:
        item["target_key"] = target_key
    if value is not None:
        item["value"] = value
    repairs.append(item)


def normalize_theme_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    normalized: list[dict] = []
    repairs: list[dict] = []
    issues: list[dict] = []
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            issues.append({"code": "INVALID_THEME_RECORD", "index": index, "value": repr(raw)})
            continue
        record = deepcopy(raw)
        theme_id = _first_text(record, ("theme_id", "id"))
        if not theme_id:
            theme_id = f"T{index}"
            _record_repair(repairs, block="themes", index=index, action="DETERMINISTIC_ID_GENERATED", target_key="theme_id", value=theme_id)
        theme_name = _first_text(record, ("theme_name", "theme", "name", "title"))
        if not record.get("theme_name") and theme_name:
            source_key = next((k for k in ("theme", "name", "title") if record.get(k)), None)
            _record_repair(repairs, block="themes", index=index, action="ALIAS_MAPPED", source_key=source_key, target_key="theme_name")
        description = _first_text(record, ("description", "summary", "content", "evidence"))
        if not description and theme_name:
            description = theme_name
            _record_repair(repairs, block="themes", index=index, action="THEME_LABEL_USED_AS_DESCRIPTION", source_key="theme_name", target_key="description")
        representative = _normalize_sources(_first_value(record, ("representative_papers", "papers", "sources", "representative_sources"), []))
        normalized.append({
            **record,
            "theme_id": theme_id,
            "theme_name": theme_name,
            "description": description,
            "representative_papers": representative,
        })
    return normalized, repairs, issues


def normalize_gap_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    normalized: list[dict] = []
    repairs: list[dict] = []
    issues: list[dict] = []
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            issues.append({"code": "INVALID_GAP_RECORD", "index": index, "value": repr(raw)})
            continue
        record = deepcopy(raw)
        gap_id = _first_text(record, ("gap_id", "id"))
        if not gap_id:
            gap_id = f"G{index}"
            _record_repair(repairs, block="research_gaps", index=index, action="DETERMINISTIC_ID_GENERATED", target_key="gap_id", value=gap_id)
        description = _first_text(record, ("description", "gap", "name", "title"))
        if not record.get("description") and description:
            source_key = next((k for k in ("gap", "name", "title") if record.get(k)), None)
            _record_repair(repairs, block="research_gaps", index=index, action="ALIAS_MAPPED", source_key=source_key, target_key="description")
        basis = _first_text(record, ("basis", "evidence", "rationale", "justification", "description", "gap"))
        sources = _normalize_sources(_first_value(record, ("supporting_sources", "sources", "papers", "references"), []))
        if not record.get("supporting_sources") and sources:
            source_key = next((k for k in ("sources", "papers", "references") if record.get(k)), None)
            _record_repair(repairs, block="research_gaps", index=index, action="ALIAS_MAPPED", source_key=source_key, target_key="supporting_sources")
        normalized.append({
            **record,
            "gap_id": gap_id,
            "description": description,
            "basis": basis,
            "supporting_sources": sources,
        })
    return normalized, repairs, issues


def normalize_structure_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    normalized: list[dict] = []
    repairs: list[dict] = []
    issues: list[dict] = []
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            issues.append({"code": "INVALID_STRUCTURE_RECORD", "index": index, "value": repr(raw)})
            continue
        record = deepcopy(raw)
        section_id = _first_text(record, ("section_id", "id"))
        if not section_id:
            section_id = f"S{index}"
            _record_repair(repairs, block="suggested_state_of_art_structure", index=index, action="DETERMINISTIC_ID_GENERATED", target_key="section_id", value=section_id)
        section_title = _first_text(record, ("section_title", "section", "title", "name"))
        if not record.get("section_title") and section_title:
            source_key = next((k for k in ("section", "title", "name") if record.get(k)), None)
            _record_repair(repairs, block="suggested_state_of_art_structure", index=index, action="ALIAS_MAPPED", source_key=source_key, target_key="section_title")
        description = _first_text(record, ("description", "content", "summary", "purpose"))
        if not record.get("description") and description:
            source_key = next((k for k in ("content", "summary", "purpose") if record.get(k)), None)
            _record_repair(repairs, block="suggested_state_of_art_structure", index=index, action="ALIAS_MAPPED", source_key=source_key, target_key="description")
        sources = _normalize_sources(_first_value(record, ("recommended_sources", "sources", "papers", "references"), []))
        normalized.append({
            **record,
            "section_id": section_id,
            "section_title": section_title,
            "description": description,
            "recommended_sources": sources,
        })
    return normalized, repairs, issues


def normalize_dimension_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    normalized: list[dict] = []
    repairs: list[dict] = []
    issues: list[dict] = []
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            issues.append({"code": "INVALID_COMPARATIVE_DIMENSION_RECORD", "index": index, "value": repr(raw)})
            continue
        record = deepcopy(raw)
        dimension = _first_text(record, ("dimension", "name", "title"))
        description = _first_text(record, ("description", "content", "summary"))
        sources = _normalize_sources(_first_value(record, ("relevant_sources", "sources", "papers", "references"), []))
        if not record.get("relevant_sources") and sources:
            source_key = next((k for k in ("sources", "papers", "references") if record.get(k)), None)
            _record_repair(repairs, block="comparative_dimensions", index=index, action="ALIAS_MAPPED", source_key=source_key, target_key="relevant_sources")
        normalized.append({
            **record,
            "dimension": dimension,
            "description": description,
            "relevant_sources": sources,
        })
    return normalized, repairs, issues


def normalize_thematic_output(payload: Any, *, return_repairs: bool = False):
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("INVALID_LLM_OUTPUT")

    out = deepcopy(payload)
    top_aliases = {
        "gaps": "research_gaps",
        "structure": "suggested_state_of_art_structure",
        "dimensions": "comparative_dimensions",
    }
    top_repairs: list[dict] = []
    for source_key, target_key in top_aliases.items():
        if target_key not in out and source_key in out:
            out[target_key] = out[source_key]
            top_repairs.append({"type": "ALIAS_MAPPED", "block": "root", "source_key": source_key, "target_key": target_key})

    for key in ("themes", "research_gaps", "suggested_state_of_art_structure", "comparative_dimensions"):
        out[key] = _list(out.get(key, []))
    if not any(out[key] for key in ("themes", "research_gaps", "suggested_state_of_art_structure", "comparative_dimensions")):
        raise ValueError("EMPTY_THEMATIC_OUTPUT")

    themes, theme_repairs, theme_issues = normalize_theme_records(out["themes"])
    gaps, gap_repairs, gap_issues = normalize_gap_records(out["research_gaps"])
    structure, structure_repairs, structure_issues = normalize_structure_records(out["suggested_state_of_art_structure"])
    dimensions, dimension_repairs, dimension_issues = normalize_dimension_records(out["comparative_dimensions"])

    out["themes"] = themes
    out["research_gaps"] = gaps
    out["suggested_state_of_art_structure"] = structure
    out["comparative_dimensions"] = dimensions

    issues = theme_issues + gap_issues + structure_issues + dimension_issues
    repairs = top_repairs + theme_repairs + gap_repairs + structure_repairs + dimension_repairs
    if return_repairs:
        return out, issues, repairs
    return out, issues


def inspect_thematic_payload(payload: dict) -> dict:
    """Count recoverable raw records before flattening."""
    return {
        "raw_theme_records": len(_list(payload.get("themes", []))),
        "raw_gap_records": len(_list(payload.get("research_gaps", payload.get("gaps", [])))),
        "raw_structure_records": len(_list(payload.get("suggested_state_of_art_structure", payload.get("structure", [])))),
        "raw_comparative_dimension_records": len(_list(payload.get("comparative_dimensions", payload.get("dimensions", [])))),
    }


def validate_json_to_tables(raw_counts: dict, table_counts: dict) -> tuple[list[str], dict]:
    codes: list[str] = []
    mapping = (
        ("raw_theme_records", "flattened_theme_semantic_rows", "THEME_FLATTENING_FAILED"),
        ("raw_gap_records", "flattened_gap_semantic_rows", "GAP_FLATTENING_FAILED"),
        ("raw_structure_records", "flattened_structure_semantic_rows", "STRUCTURE_FLATTENING_FAILED"),
        ("raw_comparative_dimension_records", "flattened_comparative_dimension_semantic_rows", "COMPARATIVE_DIMENSION_FLATTENING_FAILED"),
    )
    for raw_key, flat_key, code in mapping:
        if int(raw_counts.get(raw_key, 0)) > 0 and int(table_counts.get(flat_key, 0)) == 0:
            codes.extend(("INVALID_THEMATIC_SCHEMA", code, "ALIAS_MAPPING_REQUIRED"))
    return list(dict.fromkeys(codes)), {**raw_counts, **table_counts}
