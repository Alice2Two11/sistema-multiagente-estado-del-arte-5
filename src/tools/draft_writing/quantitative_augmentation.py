from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import json
import math
from typing import Any

from .numeric_literals import numeric_literal_exists_in_text
from .retrieval import safe_str

ChunkKey = tuple[str, str]
EvidenceRow = dict[str, Any]

DEFAULT_CONFIRMED_STATUSES: tuple[str, ...] = (
    "confirmed_in_source_chunk",
    "confirmed_literal_in_source_chunk",
)

_VALUE_FIELDS: tuple[str, ...] = ("value", "numeric_value", "reported_value", "raw_value")
_CONTEXT_FIELDS: tuple[str, ...] = ("metric", "unit", "method", "dataset", "condition")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() == "nan" else text


def _normalize_context(value: Any) -> str:
    return " ".join(_clean_text(value).casefold().split())


def _validate_non_negative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"QUANTITATIVE_AUGMENTATION_INVALID_TYPE:{name}:expected_integer")
    if value < 0:
        raise ValueError(
            f"QUANTITATIVE_AUGMENTATION_INVALID:{name}:must_be_greater_than_or_equal_to_0"
        )
    return value


def _validate_positive_integer(name: str, value: Any) -> int:
    result = _validate_non_negative_integer(name, value)
    if result <= 0:
        raise ValueError(f"QUANTITATIVE_AUGMENTATION_INVALID:{name}:must_be_greater_than_0")
    return result


def normalize_chunk_ids(value: Any) -> list[str]:
    """Normalize historical chunk-id representations without inventing IDs."""
    if value is None:
        return []
    raw_items: list[Any]
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    elif isinstance(value, set):
        raw_items = sorted(value, key=lambda item: _clean_text(item))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed: Any = None
        if text[:1] in "[{" and text[-1:] in "]}":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, (list, tuple, set)):
            raw_items = list(parsed)
        else:
            normalized = text.replace(";", ",").replace("|", ",").replace("\n", ",")
            raw_items = normalized.split(",")
    else:
        raw_items = [value]

    output: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        chunk_id = _clean_text(item)
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        output.append(chunk_id)
    return output


def _row_value(row: Mapping[str, Any]) -> str:
    for field in _VALUE_FIELDS:
        value = _clean_text(row.get(field))
        if value:
            return value
    return ""


def _row_chunk_ids(row: Mapping[str, Any]) -> list[str]:
    direct = normalize_chunk_ids(row.get("chunk_id"))
    checked = normalize_chunk_ids(row.get("source_chunk_ids_checked"))
    output: list[str] = []
    seen: set[str] = set()
    for chunk_id in direct + checked:
        if chunk_id not in seen:
            seen.add(chunk_id)
            output.append(chunk_id)
    return output


def _coverage_key(row: Mapping[str, Any], value: str) -> str:
    parts = [_normalize_context(row.get(field)) for field in _CONTEXT_FIELDS]
    parts.insert(1, _normalize_context(value))
    return "|".join(parts)


def _row_identifier(row: Mapping[str, Any], fallback_index: int) -> str:
    for field in ("row_id", "result_id", "quantitative_row_id", "id"):
        value = _clean_text(row.get(field))
        if value:
            return value
    return f"row_{fallback_index:06d}"


def _extract_rows(quantitative_context: Any) -> list[Mapping[str, Any]]:
    if isinstance(quantitative_context, list):
        rows = quantitative_context
    elif isinstance(quantitative_context, Mapping):
        rows = quantitative_context.get("quantitative_results", [])
    else:
        rows = []
    return [row for row in rows if isinstance(row, Mapping)]


def normalize_confirmed_quantitative_rows(
    quantitative_context: Any,
    *,
    confirmed_statuses: Iterable[str] = DEFAULT_CONFIRMED_STATUSES,
) -> list[EvidenceRow]:
    """Normalize only rows whose verification status is explicitly allowed."""
    allowed_statuses = {
        _normalize_context(status) for status in confirmed_statuses if _normalize_context(status)
    }
    normalized: list[EvidenceRow] = []
    for index, row in enumerate(_extract_rows(quantitative_context), start=1):
        status = _normalize_context(row.get("verification_status"))
        source = _clean_text(row.get("source_filename"))
        value = _row_value(row)
        chunk_ids = _row_chunk_ids(row)
        if status not in allowed_statuses or not source or not value or not chunk_ids:
            continue
        item = dict(row)
        item["_row_id"] = _row_identifier(row, index)
        item["_verification_status"] = status
        item["_source_filename"] = source
        item["_chunk_ids"] = chunk_ids
        item["_value"] = value
        item["_coverage_key"] = _coverage_key(row, value)
        item["_metric"] = _clean_text(row.get("metric"))
        normalized.append(item)
    return normalized


def _chunk_lookup(chunks_df) -> dict[ChunkKey, str]:
    if chunks_df is None or getattr(chunks_df, "empty", True):
        return {}
    lookup: dict[ChunkKey, str] = {}
    for _, row in chunks_df.iterrows():
        source = _clean_text(row.get("source_filename"))
        chunk_id = _clean_text(row.get("chunk_id"))
        text = _clean_text(row.get("text") or row.get("chunk_text"))
        if source and chunk_id and text:
            lookup[(source, chunk_id)] = text
    return lookup


def build_quantitative_chunk_candidates(
    chunks_df,
    quantitative_context: Any,
    *,
    allowed_papers: Sequence[str],
    valid_source_chunk_pairs: Iterable[ChunkKey] | None = None,
    confirmed_statuses: Iterable[str] = DEFAULT_CONFIRMED_STATUSES,
    max_quantitative_rows_per_section: int | None = None,
    base_evidence: Sequence[EvidenceRow] = (),
) -> list[EvidenceRow]:
    """Build one citable candidate per real source/chunk pair."""
    allowed = {_clean_text(source) for source in allowed_papers if _clean_text(source)}
    valid_pairs = set(valid_source_chunk_pairs or ())
    lookup = _chunk_lookup(chunks_df)
    rows = normalize_confirmed_quantitative_rows(
        quantitative_context, confirmed_statuses=confirmed_statuses
    )
    if max_quantitative_rows_per_section is not None:
        limit = _validate_non_negative_integer(
            "max_quantitative_rows_per_section", max_quantitative_rows_per_section
        )
        rows = rows[:limit]

    base_by_key = {
        (_clean_text(row.get("source_filename")), _clean_text(row.get("chunk_id"))): row
        for row in base_evidence
    }
    grouped: dict[ChunkKey, EvidenceRow] = {}
    for row in rows:
        source = row["_source_filename"]
        if source not in allowed:
            continue
        for chunk_id in row["_chunk_ids"]:
            key = (source, chunk_id)
            if key not in lookup:
                continue
            if valid_pairs and key not in valid_pairs:
                continue
            text = lookup[key]
            value = row["_value"]
            if not numeric_literal_exists_in_text(value, text):
                continue
            candidate = grouped.setdefault(
                key,
                {
                    "source_filename": source,
                    "chunk_id": chunk_id,
                    "text": text,
                    "selection_bucket": "quantitative_candidate",
                    "quantitative_values": [],
                    "quantitative_coverage_keys": [],
                    "quantitative_row_ids": [],
                    "quantitative_metrics": [],
                    "verification_statuses": [],
                    "_coverage_key_to_row": {},
                },
            )
            mapping = candidate["_coverage_key_to_row"]
            coverage_key = row["_coverage_key"]
            if coverage_key not in mapping:
                mapping[coverage_key] = row
                candidate["quantitative_coverage_keys"].append(coverage_key)
            for field, value_to_add in (
                ("quantitative_values", value),
                ("quantitative_row_ids", row["_row_id"]),
                ("quantitative_metrics", row["_metric"]),
                ("verification_statuses", row["_verification_status"]),
            ):
                if value_to_add and value_to_add not in candidate[field]:
                    candidate[field].append(value_to_add)

            base_row = base_by_key.get(key)
            if base_row is not None:
                for field in (
                    "retrieval_source",
                    "retrieval_sources",
                    "chroma_rank",
                    "csv_rank",
                    "rrf_score",
                    "score",
                ):
                    if field in base_row:
                        candidate[field] = deepcopy(base_row[field])

    candidates = list(grouped.values())
    for candidate in candidates:
        for field in (
            "quantitative_values",
            "quantitative_coverage_keys",
            "quantitative_row_ids",
            "quantitative_metrics",
            "verification_statuses",
        ):
            candidate[field] = sorted(candidate[field])
    candidates.sort(key=lambda row: (row["source_filename"], row["chunk_id"]))
    return candidates


def _context_compatible(row: Mapping[str, Any], text: str) -> bool:
    """Require identifiable contextual fields to be observable in the same chunk."""
    normalized_text = _normalize_context(text)
    for field in ("dataset", "method", "condition"):
        expected = _normalize_context(row.get(field))
        if expected and expected not in normalized_text:
            return False
    return True


def _covered_keys_from_base(
    base_rows: Sequence[EvidenceRow], candidates: Sequence[EvidenceRow]
) -> set[str]:
    by_key = {
        (_clean_text(row.get("source_filename")), _clean_text(row.get("chunk_id"))): row
        for row in base_rows
    }
    covered: set[str] = set()
    for candidate in candidates:
        key = (candidate["source_filename"], candidate["chunk_id"])
        base = by_key.get(key)
        if base is None:
            continue
        text = _clean_text(base.get("text"))
        for coverage_key, row in candidate["_coverage_key_to_row"].items():
            if numeric_literal_exists_in_text(row["_value"], text) and _context_compatible(row, text):
                covered.add(coverage_key)
    return covered


def augment_evidence_with_quantitative_chunks_greedy(
    base_evidence: Sequence[EvidenceRow],
    chunks_df,
    quantitative_context: Any,
    *,
    allowed_papers: Sequence[str],
    top_k_evidence_per_section: int,
    quantitative_evidence_quota: int,
    max_evidence_chars: int,
    max_candidates_per_source: int,
    valid_source_chunk_pairs: Iterable[ChunkKey] | None = None,
    confirmed_statuses: Iterable[str] = DEFAULT_CONFIRMED_STATUSES,
    max_quantitative_rows_per_section: int | None = None,
) -> list[EvidenceRow]:
    """Reserve optional slots for confirmed, literal, citable quantitative evidence."""
    top_k = _validate_non_negative_integer(
        "top_k_evidence_per_section", top_k_evidence_per_section
    )
    quota = _validate_non_negative_integer(
        "quantitative_evidence_quota", quantitative_evidence_quota
    )
    char_limit = _validate_positive_integer("max_evidence_chars", max_evidence_chars)
    per_source_limit = _validate_positive_integer(
        "max_candidates_per_source", max_candidates_per_source
    )
    if quota > top_k:
        raise ValueError(
            "QUANTITATIVE_AUGMENTATION_INVALID:quantitative_evidence_quota:"
            "must_be_less_than_or_equal_to_top_k_evidence_per_section"
        )
    if top_k == 0:
        return []

    allowed = {_clean_text(source) for source in allowed_papers if _clean_text(source)}
    valid_pairs = set(valid_source_chunk_pairs or ())
    base_valid: list[EvidenceRow] = []
    seen_base: set[ChunkKey] = set()
    for original in base_evidence:
        row = deepcopy(dict(original))
        source = _clean_text(row.get("source_filename"))
        chunk_id = _clean_text(row.get("chunk_id"))
        text = _clean_text(row.get("text"))
        key = (source, chunk_id)
        if not source or not chunk_id or not text or key in seen_base:
            continue
        if source not in allowed or (valid_pairs and key not in valid_pairs):
            continue
        seen_base.add(key)
        row["source_filename"] = source
        row["chunk_id"] = chunk_id
        row["text"] = text
        base_valid.append(row)

    candidates = build_quantitative_chunk_candidates(
        chunks_df,
        quantitative_context,
        allowed_papers=sorted(allowed),
        valid_source_chunk_pairs=valid_pairs,
        confirmed_statuses=confirmed_statuses,
        max_quantitative_rows_per_section=max_quantitative_rows_per_section,
        base_evidence=base_valid,
    )

    base_slots = top_k - quota
    selected: list[EvidenceRow] = []
    selected_keys: set[ChunkKey] = set()
    source_counts: Counter[str] = Counter()
    used_chars = 0

    def can_add(row: Mapping[str, Any]) -> bool:
        source = _clean_text(row.get("source_filename"))
        chunk_id = _clean_text(row.get("chunk_id"))
        text = _clean_text(row.get("text"))
        key = (source, chunk_id)
        return bool(
            source
            and chunk_id
            and text
            and source in allowed
            and key not in selected_keys
            and (not valid_pairs or key in valid_pairs)
            and source_counts[source] < per_source_limit
            and used_chars + len(text) <= char_limit
        )

    def add_row(original: Mapping[str, Any], bucket: str, marginal_gain: int | None = None) -> bool:
        nonlocal used_chars
        if not can_add(original):
            return False
        row = deepcopy(dict(original))
        row.pop("_coverage_key_to_row", None)
        source = _clean_text(row.get("source_filename"))
        chunk_id = _clean_text(row.get("chunk_id"))
        text = _clean_text(row.get("text"))
        row["source_filename"] = source
        row["chunk_id"] = chunk_id
        row["text"] = text
        row["selection_bucket"] = bucket
        row["selection_order"] = len(selected) + 1
        if marginal_gain is not None:
            row["quantitative_marginal_gain"] = marginal_gain
        selected.append(row)
        selected_keys.add((source, chunk_id))
        source_counts[source] += 1
        used_chars += len(text)
        return True

    for row in base_valid:
        if len(selected) >= base_slots:
            break
        add_row(row, _clean_text(row.get("selection_bucket")) or "hybrid_base_reserved")

    covered_keys = _covered_keys_from_base(selected, candidates)
    selected_sources = {row["source_filename"] for row in selected}
    quantitative_added = 0
    remaining_candidates = list(candidates)

    while quantitative_added < quota:
        scored: list[tuple[Any, ...]] = []
        for candidate in remaining_candidates:
            key = (candidate["source_filename"], candidate["chunk_id"])
            if key in selected_keys or not can_add(candidate):
                continue
            marginal = set(candidate["quantitative_coverage_keys"]) - covered_keys
            if not marginal:
                continue
            metrics = {
                candidate["_coverage_key_to_row"][coverage_key].get("_metric", "")
                for coverage_key in marginal
                if coverage_key in candidate["_coverage_key_to_row"]
            }
            metrics.discard("")
            source_diversity = 1 if candidate["source_filename"] not in selected_sources else 0
            hybrid_score = float(
                candidate.get("rrf_score", candidate.get("score", 0.0)) or 0.0
            )
            scored.append(
                (
                    -len(marginal),
                    -len(metrics),
                    -source_diversity,
                    -hybrid_score,
                    candidate["source_filename"],
                    candidate["chunk_id"],
                    candidate,
                    marginal,
                )
            )
        if not scored:
            break
        scored.sort(key=lambda item: item[:-2])
        *_, chosen, marginal = scored[0]
        if add_row(chosen, "quantitative_greedy", len(marginal)):
            quantitative_added += 1
            covered_keys.update(marginal)
            selected_sources.add(chosen["source_filename"])
        remaining_candidates = [
            row
            for row in remaining_candidates
            if (row["source_filename"], row["chunk_id"])
            != (chosen["source_filename"], chosen["chunk_id"])
        ]

    for row in base_valid:
        if len(selected) >= top_k:
            break
        add_row(row, "hybrid_returned_slot")

    return selected[:top_k]
