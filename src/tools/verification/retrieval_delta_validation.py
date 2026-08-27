"""Validación del delta incremental de retrieval adicional (Fase 4T/4U, Agente 07).

Extraído mecánicamente de src.tools.verification.validation (Bloque C,
C4) -- sin cambios de comportamiento, lógica, firmas ni mensajes de
error respecto al código original. validation.py reexporta estos
símbolos para preservar el contrato de importación existente.
"""
from __future__ import annotations

from typing import Any, Mapping

from src.config.verification_policy_config import (
    CONTRADICTION_SIGNAL_CODES,
    RETRIEVAL_TECHNICAL_STATUSES,
    TECHNICAL_ISSUE_CODES,
)


ADDITIONAL_RETRIEVAL_DELTA_ACCUMULATIVE_FIELDS = (
    "rounds_executed", "total_candidates_seen", "total_unique_candidates_seen",
    "queries_executed_total", "new_unique_pairs_seen",
)
ADDITIONAL_RETRIEVAL_DELTA_UNION_FIELDS = (
    "queries", "discarded_candidates", "retrieval_trace", "contradiction_signals",
    "technical_issue_codes",
)
ADDITIONAL_RETRIEVAL_DELTA_SNAPSHOT_FIELDS = (
    "coverage_after", "stop_reason", "technical_status", "queries_remaining",
)
ADDITIONAL_RETRIEVAL_DELTA_DERIVED_FIELDS = (
    "total_unique_candidates_retained", "new_unique_pairs_selected",
    "structural_coverage_improved", "structural_coverage_improved_this_delta",
)
ADDITIONAL_RETRIEVAL_DELTA_ALLOWED_FIELDS = frozenset(
    ("selected_candidates", "deterministic_validation")
    + ADDITIONAL_RETRIEVAL_DELTA_ACCUMULATIVE_FIELDS
    + ADDITIONAL_RETRIEVAL_DELTA_UNION_FIELDS
    + ADDITIONAL_RETRIEVAL_DELTA_SNAPSHOT_FIELDS
    + ADDITIONAL_RETRIEVAL_DELTA_DERIVED_FIELDS
)
ADDITIONAL_RETRIEVAL_MUTABLE_DETERMINISTIC_FIELDS = frozenset({
    "numeric_pairs_valid", "comparative_coverage_ok", "attribution_coverage_ok",
    "missing_structural_elements",
})
ADDITIONAL_RETRIEVAL_PROTECTED_CANDIDATE_FIELDS = frozenset({
    "authorized_for_section", "outside_section_sources", "usage_allowed",
    "is_inherited", "retrieval_scope", "canonical_text", "contractual_text",
})
ADDITIONAL_RETRIEVAL_CANDIDATE_ALLOWED_FIELDS = frozenset({
    "source_filename", "chunk_id", "text", "retrieval_sources", "query_ids",
    "all_native_ranks", "native_ranks_by_retriever", "native_scores_by_retriever",
    "native_score_types_by_retriever", "first_seen_round", "last_seen_round",
    "fused_rrf_score", "text_variants", "contradiction_signals",
})
ADDITIONAL_RETRIEVAL_STOP_REASONS = frozenset({
    "NOT_ATTEMPTED", "STRUCTURAL_COVERAGE_SATISFIED", "NO_NEW_EVIDENCE",
    "BUDGET_EXHAUSTED",
})
ADDITIONAL_RETRIEVAL_COVERAGE_FIELDS = frozenset({
    "structural_coverage_ok", "resolved_evidence_count", "authorized_evidence_count",
    "quantitative_coverage_ok", "comparative_coverage_ok", "attribution_coverage_ok",
    "missing_structural_elements",
})


def _delta_string_sequence(value: Any, field: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:{field}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or (nonempty and not item.strip()):
            raise ValueError(f"ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:{field}")
        out.append(item.strip())
    return tuple(out)


def _validate_incremental_candidate(row: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:selected_candidate")
    candidate = dict(row)
    source = candidate.get("source_filename")
    chunk = candidate.get("chunk_id")
    if not isinstance(source, str) or not source.strip() or not isinstance(chunk, str) or not chunk.strip():
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_CANDIDATE_IDENTITY_INVALID")
    protected = sorted(set(candidate) & ADDITIONAL_RETRIEVAL_PROTECTED_CANDIDATE_FIELDS)
    if protected:
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_PROTECTED_CANDIDATE_FIELD:" + ",".join(protected))
    unknown = sorted(set(candidate) - ADDITIONAL_RETRIEVAL_CANDIDATE_ALLOWED_FIELDS)
    if strict and unknown:
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_UNKNOWN_FIELD:selected_candidate:" + ",".join(unknown))
    if "text" in candidate and (not isinstance(candidate["text"], str) or not candidate["text"].strip()):
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:text")
    for field in ("retrieval_sources", "query_ids"):
        if field in candidate:
            candidate[field] = _delta_string_sequence(candidate[field], field)
    if "all_native_ranks" in candidate:
        if type(candidate["all_native_ranks"]) not in (list, tuple) or any(type(x) is not int or x < 1 for x in candidate["all_native_ranks"]):
            raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:all_native_ranks")
        candidate["all_native_ranks"] = tuple(candidate["all_native_ranks"])
    maps = {
        "native_ranks_by_retriever": lambda v: type(v) is int and v >= 1,
        "native_scores_by_retriever": lambda v: type(v) in (int, float),
        "native_score_types_by_retriever": lambda v: isinstance(v, str) and bool(v.strip()),
    }
    names_by_map: dict[str, set[str]] = {}
    for field, validator in maps.items():
        if field not in candidate:
            continue
        value = candidate[field]
        if not isinstance(value, Mapping) or any(not isinstance(k, str) or not k.strip() or not validator(v) for k, v in value.items()):
            raise ValueError(f"ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:{field}")
        names_by_map[field] = set(value)
        candidate[field] = {k: value[k] for k in sorted(value)}
    if len(names_by_map) > 1:
        nonempty_sets = [v for v in names_by_map.values() if v]
        if nonempty_sets and any(v != nonempty_sets[0] for v in nonempty_sets[1:]):
            raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:retriever_name_maps_inconsistent")
    for field in ("first_seen_round", "last_seen_round"):
        if field in candidate and (type(candidate[field]) is not int or candidate[field] < 0):
            raise ValueError(f"ADDITIONAL_RETRIEVAL_DELTA_COUNTER_INVALID:{field}")
    if "first_seen_round" in candidate and "last_seen_round" in candidate and candidate["first_seen_round"] > candidate["last_seen_round"]:
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:round_range")
    if "fused_rrf_score" in candidate and type(candidate["fused_rrf_score"]) not in (int, float):
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:fused_rrf_score")
    if "text_variants" in candidate:
        if type(candidate["text_variants"]) not in (list, tuple) or any(not isinstance(x, Mapping) for x in candidate["text_variants"]):
            raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:text_variants")
        candidate["text_variants"] = tuple(dict(x) for x in candidate["text_variants"])
    if "contradiction_signals" in candidate:
        signals = _delta_string_sequence(candidate["contradiction_signals"], "contradiction_signals")
        invalid = sorted(set(signals) - set(CONTRADICTION_SIGNAL_CODES))
        if invalid:
            raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:contradiction_signals:" + ",".join(invalid))
        candidate["contradiction_signals"] = signals
    return candidate


def _validate_coverage_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:coverage_after")
    unknown = sorted(set(value) - ADDITIONAL_RETRIEVAL_COVERAGE_FIELDS)
    if unknown:
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_UNKNOWN_FIELD:coverage_after:" + ",".join(unknown))
    out = dict(value)
    for field in ("structural_coverage_ok", "quantitative_coverage_ok", "comparative_coverage_ok", "attribution_coverage_ok"):
        if field in out and type(out[field]) is not bool:
            raise ValueError(f"ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:coverage_after:{field}")
    for field in ("resolved_evidence_count", "authorized_evidence_count"):
        if field in out and (type(out[field]) is not int or out[field] < 0):
            raise ValueError(f"ADDITIONAL_RETRIEVAL_DELTA_COUNTER_INVALID:coverage_after:{field}")
    if "missing_structural_elements" in out:
        out["missing_structural_elements"] = _delta_string_sequence(out["missing_structural_elements"], "coverage_after:missing_structural_elements")
    return out


def validate_additional_retrieval_delta(delta: Mapping[str, Any], *, strict: bool = True) -> dict[str, Any]:
    if not isinstance(delta, Mapping):
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:root")
    unknown = sorted(set(delta) - ADDITIONAL_RETRIEVAL_DELTA_ALLOWED_FIELDS)
    if strict and unknown:
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_UNKNOWN_FIELD:" + ",".join(unknown))
    value = dict(delta)
    for field in ADDITIONAL_RETRIEVAL_DELTA_ACCUMULATIVE_FIELDS + ("queries_remaining",):
        if field in value and (type(value[field]) is not int or value[field] < 0):
            raise ValueError(f"ADDITIONAL_RETRIEVAL_DELTA_COUNTER_INVALID:{field}")
    if "selected_candidates" in value:
        if type(value["selected_candidates"]) not in (list, tuple):
            raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:selected_candidates")
        value["selected_candidates"] = tuple(_validate_incremental_candidate(row, strict=strict) for row in value["selected_candidates"])
    for field in ADDITIONAL_RETRIEVAL_DELTA_UNION_FIELDS:
        if field in value and type(value[field]) not in (list, tuple):
            raise ValueError(f"ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:{field}")
    if "queries" in value and any(not isinstance(x, Mapping) for x in value["queries"]):
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:queries")
    if "retrieval_trace" in value and any(not isinstance(x, (Mapping, str)) for x in value["retrieval_trace"]):
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:retrieval_trace")
    if "technical_issue_codes" in value:
        invalid = sorted(set(value["technical_issue_codes"]) - set(TECHNICAL_ISSUE_CODES))
        if invalid:
            raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:technical_issue_codes:" + ",".join(invalid))
    if "contradiction_signals" in value:
        invalid = sorted(set(value["contradiction_signals"]) - set(CONTRADICTION_SIGNAL_CODES))
        if invalid:
            raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:contradiction_signals:" + ",".join(invalid))
    if "technical_status" in value and value["technical_status"] not in RETRIEVAL_TECHNICAL_STATUSES:
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:technical_status")
    if "stop_reason" in value and value["stop_reason"] not in ADDITIONAL_RETRIEVAL_STOP_REASONS:
        raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:stop_reason")
    if "coverage_after" in value:
        value["coverage_after"] = _validate_coverage_snapshot(value["coverage_after"])
    if "deterministic_validation" in value:
        dv = value["deterministic_validation"]
        if not isinstance(dv, Mapping):
            raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:deterministic_validation")
        forbidden = sorted(set(dv) - ADDITIONAL_RETRIEVAL_MUTABLE_DETERMINISTIC_FIELDS)
        if forbidden:
            raise ValueError("ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:deterministic_validation:" + ",".join(forbidden))
        dv = dict(dv)
        for field in ("numeric_pairs_valid", "comparative_coverage_ok", "attribution_coverage_ok"):
            if field in dv and type(dv[field]) is not bool:
                raise ValueError(f"ADDITIONAL_RETRIEVAL_DELTA_FIELD_INVALID:deterministic_validation:{field}")
        if "missing_structural_elements" in dv:
            dv["missing_structural_elements"] = _delta_string_sequence(dv["missing_structural_elements"], "deterministic_validation:missing_structural_elements")
        value["deterministic_validation"] = dv
    return value
