"""Phase 6.6: deterministic multi-proposal resolution for Agent 07.

This module consumes only a validated ProvisionalVerificationTraceabilityBundle.
It performs no LLM calls, retrieval, draft writes, or physical correction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.tools.verification.validation import (
    validate_provisional_verification_traceability_bundle_contract,
    validate_sha256_hex,
)
from src.tools.verification.corrections import fingerprint_text

PAIR_RELATIONS = (
    "INDEPENDENT",
    "REDUNDANT",
    "OVERLAPPING_COMPATIBLE",
    "OVERLAPPING_CONFLICTING",
    "SEMANTICALLY_CONFLICTING",
    "ORDER_DEPENDENT",
    "NOT_COMPARABLE",
)
PLAN_TYPES = (
    "NO_ACCEPTED_CORRECTIONS",
    "SINGLE_ACCEPTED_CORRECTION",
    "MULTIPLE_COMPATIBLE_CORRECTIONS",
    "MULTIPLE_REDUNDANT_CORRECTIONS",
    "MULTIPLE_CONFLICTING_CORRECTIONS",
    "MANUAL_REVIEW_REQUIRED",
    "BLOCKED_INVALID_BUNDLE",
)
RESOLUTION_STATUSES = ("COMPLETED", "PARTIAL", "BLOCKED")
RESOLUTION_FP_VERSION = "AGENT07_MULTI_PROPOSAL_RESOLUTION_V1"
RESOLUTION_AUDIT_FP_VERSION = "AGENT07_MULTI_PROPOSAL_AUDIT_V1"
PAIR_REASON_CODES = (
    "PAIR_CONTEXT_NOT_COMPARABLE", "PAIR_ORIGINAL_FINGERPRINT_MISMATCH", "PAIR_SPAN_INVALID",
    "PAIR_SAME_SPAN_AND_REPLACEMENT", "PAIR_OVERLAP_SAME_VIRTUAL_RESULT",
    "PAIR_OVERLAPPING_INCOMPATIBLE_REPLACEMENTS", "PAIR_NEW_ISSUE_CONFLICTS_WITH_RESOLUTION",
    "PAIR_REPLACEMENT_REFERENCES_OTHER_TARGET", "PAIR_POSITIONALLY_DISJOINT",
    "PAIR_SEMANTIC_COMPATIBILITY_NOT_ASSERTED",
)
RESOLUTION_ISSUE_CODES = ("MULTI_PROPOSAL_BLOCKED_INVALID_BUNDLE", "MULTI_PROPOSAL_CONFLICT")
RESOLUTION_WARNING_CODES = ("MULTI_PROPOSAL_MANUAL_REVIEW_REQUIRED",)
INDIVIDUAL_DECISIONS = ("ACCEPT_FOR_07C", "REJECT_PROPOSAL", "DEFER_TO_MANUAL_REVIEW")
BLOCKING_RELATIONS = ("OVERLAPPING_CONFLICTING", "SEMANTICALLY_CONFLICTING", "ORDER_DEPENDENT", "NOT_COMPARABLE")

PAIR_REASON_MATRIX = {
    "INDEPENDENT": frozenset({"PAIR_POSITIONALLY_DISJOINT", "PAIR_SEMANTIC_COMPATIBILITY_NOT_ASSERTED"}),
    "REDUNDANT": frozenset({"PAIR_SAME_SPAN_AND_REPLACEMENT"}),
    "OVERLAPPING_COMPATIBLE": frozenset({"PAIR_OVERLAP_SAME_VIRTUAL_RESULT"}),
    "OVERLAPPING_CONFLICTING": frozenset({"PAIR_OVERLAPPING_INCOMPATIBLE_REPLACEMENTS"}),
    "SEMANTICALLY_CONFLICTING": frozenset({"PAIR_NEW_ISSUE_CONFLICTS_WITH_RESOLUTION"}),
    "ORDER_DEPENDENT": frozenset({"PAIR_REPLACEMENT_REFERENCES_OTHER_TARGET"}),
    "NOT_COMPARABLE": frozenset({"PAIR_CONTEXT_NOT_COMPARABLE", "PAIR_ORIGINAL_FINGERPRINT_MISMATCH", "PAIR_SPAN_INVALID"}),
}


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _ordered(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _span(row: Mapping[str, Any]) -> tuple[int, int, str] | None:
    span = row.get("target_span_in_claim")
    if not isinstance(span, Mapping):
        return None
    start, end, text = span.get("start"), span.get("end"), span.get("text")
    if type(start) is not int or type(end) is not int or not isinstance(text, str):
        return None
    if start < 0 or end < start or end - start != len(text):
        return None
    return start, end, text


@dataclass(frozen=True, slots=True)
class SelectedCorrectionPatch:
    correction_id: str
    claim_id: str
    section_id: str
    original_claim_fingerprint: str
    target_span_in_claim: Mapping[str, Any]
    replacement_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorrectionPairRelation:
    claim_id: str
    section_id: str
    left_correction_id: str
    right_correction_id: str
    relation_type: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClaimCorrectionResolutionPlan:
    claim_id: str
    section_id: str
    plan_type: str
    individual_decisions: Mapping[str, str]
    accepted_correction_ids: tuple[str, ...]
    rejected_correction_ids: tuple[str, ...]
    deferred_correction_ids: tuple[str, ...]
    selected_correction_ids: tuple[str, ...]
    redundant_correction_ids: tuple[str, ...]
    application_order: tuple[str, ...]
    selected_patch_records: tuple[Mapping[str, Any], ...]
    original_claim_text: str
    virtual_result_text: str | None
    candidate_resolved_issue_codes: tuple[str, ...]
    accepted_resolved_issue_codes: tuple[str, ...]
    provisional_remaining_issue_codes: tuple[str, ...]
    new_issue_codes: tuple[str, ...]
    manual_review_required: bool
    eligible_for_07c: bool
    requires_07c: bool
    blocks_07c: bool
    claim_resolution_plan_fingerprint: str
    correction_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProvisionalMultiProposalResolutionResult:
    claim_resolution_plans: tuple[Mapping[str, Any], ...]
    pair_relations: tuple[Mapping[str, Any], ...]
    resolution_issue_codes: tuple[str, ...]
    resolution_warnings: tuple[str, ...]
    resolution_status: str
    aggregation_status: str
    eligible_for_07c: bool
    multi_proposal_resolution_fingerprint: str | None
    multi_proposal_audit_fingerprint: str
    source_bundle_audit_fingerprint: str
    correction_applied: bool = False
    official_artifacts_created: bool = False
    additional_llm_calls: int = 0
    additional_retrieval_rounds: int = 0
    result_contract_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_correction_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> CorrectionPairRelation:
    left, right = dict(left), dict(right)
    if left["correction_id"] > right["correction_id"]:
        left, right = right, left
    relation = "NOT_COMPARABLE"
    reasons: tuple[str, ...] = ("PAIR_CONTEXT_NOT_COMPARABLE",)
    ls, rs = _span(left), _span(right)
    if left.get("original_claim_fingerprint") != right.get("original_claim_fingerprint"):
        relation, reasons = "NOT_COMPARABLE", ("PAIR_ORIGINAL_FINGERPRINT_MISMATCH",)
    elif ls is None or rs is None:
        relation, reasons = "NOT_COMPARABLE", ("PAIR_SPAN_INVALID",)
    else:
        l0, l1, ltext = ls
        r0, r1, rtext = rs
        overlap = max(l0, r0) < min(l1, r1)
        same_span = (l0, l1) == (r0, r1)
        if same_span and left.get("replacement_text") == right.get("replacement_text"):
            relation, reasons = "REDUNDANT", ("PAIR_SAME_SPAN_AND_REPLACEMENT",)
        elif overlap and left.get("proposed_claim_text") == right.get("proposed_claim_text"):
            relation, reasons = "OVERLAPPING_COMPATIBLE", ("PAIR_OVERLAP_SAME_VIRTUAL_RESULT",)
        elif overlap:
            relation, reasons = "OVERLAPPING_CONFLICTING", ("PAIR_OVERLAPPING_INCOMPATIBLE_REPLACEMENTS",)
        elif set(left.get("new_issue_codes", ())) & set(right.get("resolved_issue_codes", ())):
            relation, reasons = "SEMANTICALLY_CONFLICTING", ("PAIR_NEW_ISSUE_CONFLICTS_WITH_RESOLUTION",)
        elif set(right.get("new_issue_codes", ())) & set(left.get("resolved_issue_codes", ())):
            relation, reasons = "SEMANTICALLY_CONFLICTING", ("PAIR_NEW_ISSUE_CONFLICTS_WITH_RESOLUTION",)
        elif (rtext and rtext in str(left.get("replacement_text", ""))) or (ltext and ltext in str(right.get("replacement_text", ""))):
            relation, reasons = "ORDER_DEPENDENT", ("PAIR_REPLACEMENT_REFERENCES_OTHER_TARGET",)
        else:
            relation, reasons = "INDEPENDENT", ("PAIR_POSITIONALLY_DISJOINT", "PAIR_SEMANTIC_COMPATIBILITY_NOT_ASSERTED")
    return CorrectionPairRelation(
        claim_id=str(left["claim_id"]), section_id=str(left["section_id"]),
        left_correction_id=str(left["correction_id"]), right_correction_id=str(right["correction_id"]),
        relation_type=relation, reason_codes=reasons,
    )


def _apply_virtual(original: str, rows: Sequence[Mapping[str, Any]]) -> tuple[str, tuple[str, ...]]:
    ordered_rows = sorted(rows, key=lambda r: (_span(r)[0], r["correction_id"]), reverse=True)
    text = original
    applied: list[str] = []
    for row in ordered_rows:
        span = _span(row)
        if span is None:
            raise ValueError("RESOLUTION_SPAN_INVALID")
        start, end, target = span
        if original[start:end] != target:
            raise ValueError("RESOLUTION_TARGET_TEXT_MISMATCH")
        # Right-to-left application preserves every remaining original offset.
        text = text[:start] + str(row["replacement_text"]) + text[end:]
        applied.append(str(row["correction_id"]))
    return text, tuple(applied)


def _validate_pair(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"claim_id","section_id","left_correction_id","right_correction_id","relation_type","reason_codes"}
    if set(value) != required:
        raise ValueError("PAIR_RELATION_SCHEMA_INVALID")
    v = dict(value)
    if v["relation_type"] not in PAIR_RELATIONS:
        raise ValueError("PAIR_RELATION_TYPE_INVALID")
    for field in ("claim_id","section_id","left_correction_id","right_correction_id"):
        if not isinstance(v[field], str) or not v[field]:
            raise ValueError("PAIR_RELATION_ID_INVALID")
    if v["left_correction_id"] >= v["right_correction_id"]:
        raise ValueError("PAIR_RELATION_ORDER_INVALID")
    v["reason_codes"] = tuple(v["reason_codes"])
    if not v["reason_codes"] or any(code not in PAIR_REASON_CODES for code in v["reason_codes"]):
        raise ValueError("PAIR_RELATION_REASON_CODE_INVALID")
    if frozenset(v["reason_codes"]) != PAIR_REASON_MATRIX[v["relation_type"]]:
        raise ValueError("PAIR_RELATION_REASON_TYPE_MISMATCH")
    return v


def _validate_selected_patch(value: Mapping[str, Any], *, plan: Mapping[str, Any]) -> dict[str, Any]:
    required = set(SelectedCorrectionPatch.__dataclass_fields__)
    if set(value) != required:
        raise ValueError("SELECTED_CORRECTION_PATCH_SCHEMA_INVALID")
    patch = dict(value)
    for field in ("correction_id", "claim_id", "section_id", "original_claim_fingerprint", "replacement_text"):
        if not isinstance(patch[field], str) or (field != "replacement_text" and not patch[field]):
            raise ValueError("SELECTED_CORRECTION_PATCH_FIELD_INVALID")
    if patch["claim_id"] != plan["claim_id"] or patch["section_id"] != plan["section_id"]:
        raise ValueError("SELECTED_CORRECTION_PATCH_CONTEXT_MISMATCH")
    if patch["original_claim_fingerprint"] != fingerprint_text(plan["original_claim_text"]):
        raise ValueError("SELECTED_CORRECTION_PATCH_ORIGINAL_FINGERPRINT_MISMATCH")
    span = _span({"target_span_in_claim": patch["target_span_in_claim"]})
    if span is None:
        raise ValueError("SELECTED_CORRECTION_PATCH_SPAN_INVALID")
    start, end, target = span
    if end > len(plan["original_claim_text"]) or plan["original_claim_text"][start:end] != target:
        raise ValueError("SELECTED_CORRECTION_PATCH_TARGET_MISMATCH")
    return patch


def _expected_patch_order(patches: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        p["correction_id"]
        for p in sorted(
            patches,
            key=lambda p: (
                int(p["target_span_in_claim"]["start"]),
                int(p["target_span_in_claim"]["end"]),
                str(p["correction_id"]),
            ),
            reverse=True,
        )
    )


def _reconstruct_from_patches(original: str, patches: Sequence[Mapping[str, Any]], order: Sequence[str]) -> str:
    by_id = {p["correction_id"]: p for p in patches}
    text = original
    for correction_id in order:
        patch = by_id[correction_id]
        span = patch["target_span_in_claim"]
        start, end = int(span["start"]), int(span["end"] )
        text = text[:start] + patch["replacement_text"] + text[end:]
    return text


def _plan_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": RESOLUTION_FP_VERSION,
        **{k: _plain(v) for k, v in plan.items() if k not in {"claim_resolution_plan_fingerprint", "correction_applied"}},
        "correction_applied": False,
    }


def validate_claim_correction_resolution_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    required = set(ClaimCorrectionResolutionPlan.__dataclass_fields__)
    if set(value) != required:
        raise ValueError("CLAIM_RESOLUTION_PLAN_SCHEMA_INVALID")
    v = dict(value)
    if v["plan_type"] not in PLAN_TYPES:
        raise ValueError("CLAIM_RESOLUTION_PLAN_TYPE_INVALID")
    if type(v["correction_applied"]) is not bool or v["correction_applied"]:
        raise ValueError("CLAIM_RESOLUTION_PLAN_APPLICATION_INVALID")
    for field in ("claim_id", "section_id", "original_claim_text"):
        if not isinstance(v[field], str) or (field != "original_claim_text" and not v[field]):
            raise ValueError("CLAIM_RESOLUTION_PLAN_ID_INVALID")
    bool_fields=("manual_review_required","eligible_for_07c","requires_07c","blocks_07c")
    if any(type(v[f]) is not bool for f in bool_fields):
        raise ValueError("CLAIM_RESOLUTION_PLAN_BOOLEAN_INVALID")
    seq_fields=("accepted_correction_ids","rejected_correction_ids","deferred_correction_ids","selected_correction_ids","redundant_correction_ids","application_order","candidate_resolved_issue_codes","accepted_resolved_issue_codes","provisional_remaining_issue_codes","new_issue_codes")
    for field in seq_fields:
        v[field] = tuple(v[field])
        if len(v[field]) != len(set(v[field])):
            raise ValueError(f"CLAIM_RESOLUTION_PLAN_DUPLICATE:{field}")
    accepted=set(v["accepted_correction_ids"]); rejected=set(v["rejected_correction_ids"]); deferred=set(v["deferred_correction_ids"])
    if accepted & rejected or accepted & deferred or rejected & deferred:
        raise ValueError("CLAIM_RESOLUTION_PLAN_DECISION_SETS_OVERLAP")
    decisions=dict(v["individual_decisions"])
    if set(decisions) != accepted | rejected | deferred:
        raise ValueError("CLAIM_RESOLUTION_PLAN_DECISION_KEYS_MISMATCH")
    expected_decisions={**{x:"ACCEPT_FOR_07C" for x in accepted}, **{x:"REJECT_PROPOSAL" for x in rejected}, **{x:"DEFER_TO_MANUAL_REVIEW" for x in deferred}}
    if decisions != expected_decisions or any(x not in INDIVIDUAL_DECISIONS for x in decisions.values()):
        raise ValueError("CLAIM_RESOLUTION_PLAN_INDIVIDUAL_DECISION_MISMATCH")
    selected=set(v["selected_correction_ids"]); redundant=set(v["redundant_correction_ids"])
    if not selected.issubset(accepted) or not redundant.issubset(accepted):
        raise ValueError("CLAIM_RESOLUTION_PLAN_ACCEPTED_MEMBERSHIP_INVALID")
    if selected & redundant:
        raise ValueError("CLAIM_RESOLUTION_PLAN_REDUNDANCY_INVALID")
    patches = tuple(_validate_selected_patch(p, plan=v) for p in v["selected_patch_records"])
    patch_ids = tuple(p["correction_id"] for p in patches)
    if len(patch_ids) != len(set(patch_ids)) or set(patch_ids) != selected:
        raise ValueError("CLAIM_RESOLUTION_PLAN_PATCH_MEMBERSHIP_MISMATCH")
    expected_order = _expected_patch_order(patches)
    if tuple(v["application_order"]) != expected_order:
        raise ValueError("CLAIM_RESOLUTION_PLAN_APPLICATION_ORDER_MISMATCH")
    sorted_spans = sorted((_span({"target_span_in_claim": p["target_span_in_claim"]})[:2] for p in patches))
    if any(left[1] > right[0] for left, right in zip(sorted_spans, sorted_spans[1:])):
        raise ValueError("CLAIM_RESOLUTION_PLAN_PATCH_OVERLAP")
    if not set(v["accepted_resolved_issue_codes"]).issubset(set(v["candidate_resolved_issue_codes"])):
        raise ValueError("CLAIM_RESOLUTION_PLAN_ACCEPTED_ISSUES_INVALID")
    pt=v["plan_type"]
    if pt=="NO_ACCEPTED_CORRECTIONS":
        if accepted or selected or v["application_order"] or v["virtual_result_text"] is not None or v["requires_07c"] or v["blocks_07c"] or v["eligible_for_07c"]:
            raise ValueError("CLAIM_RESOLUTION_PLAN_NO_ACCEPTED_INVALID")
    elif pt=="SINGLE_ACCEPTED_CORRECTION":
        if len(accepted)!=1 or len(selected)!=1 or not v["requires_07c"] or v["blocks_07c"]:
            raise ValueError("CLAIM_RESOLUTION_PLAN_SINGLE_INVALID")
    elif pt=="MULTIPLE_COMPATIBLE_CORRECTIONS":
        if len(selected)<2 or v["blocks_07c"]:
            raise ValueError("CLAIM_RESOLUTION_PLAN_COMPATIBLE_INVALID")
    elif pt=="MULTIPLE_REDUNDANT_CORRECTIONS":
        if len(accepted)<2 or not redundant or not selected or v["blocks_07c"]:
            raise ValueError("CLAIM_RESOLUTION_PLAN_REDUNDANT_INVALID")
    elif pt=="MULTIPLE_CONFLICTING_CORRECTIONS":
        if not v["manual_review_required"] or selected or v["application_order"] or not v["blocks_07c"]:
            raise ValueError("CLAIM_RESOLUTION_PLAN_CONFLICT_INVALID")
    elif pt=="MANUAL_REVIEW_REQUIRED":
        if not v["manual_review_required"] or selected or v["application_order"] or not v["blocks_07c"]:
            raise ValueError("CLAIM_RESOLUTION_PLAN_MANUAL_INVALID")
    if v["eligible_for_07c"] != (v["requires_07c"] and not v["blocks_07c"] and not v["manual_review_required"]):
        raise ValueError("CLAIM_RESOLUTION_PLAN_ELIGIBILITY_INVALID")
    if selected and v["virtual_result_text"] is None:
        raise ValueError("CLAIM_RESOLUTION_PLAN_VIRTUAL_TEXT_REQUIRED")
    if selected:
        reconstructed = _reconstruct_from_patches(v["original_claim_text"], patches, v["application_order"])
        if v["virtual_result_text"] != reconstructed:
            raise ValueError("CLAIM_RESOLUTION_PLAN_VIRTUAL_RESULT_MISMATCH")
    elif v["virtual_result_text"] is not None:
        raise ValueError("CLAIM_RESOLUTION_PLAN_VIRTUAL_TEXT_FORBIDDEN")
    expected = _sha256(_plan_payload(v))
    if v["claim_resolution_plan_fingerprint"] != expected:
        raise ValueError("CLAIM_RESOLUTION_PLAN_FINGERPRINT_MISMATCH")
    v["individual_decisions"] = dict(sorted(decisions.items()))
    v["selected_patch_records"] = patches
    return v

def _build_plan(claim: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], pair_relations: list[CorrectionPairRelation]) -> ClaimCorrectionResolutionPlan:
    rows = sorted((dict(r) for r in rows), key=lambda r: r["correction_id"])
    decisions = {r["correction_id"]: r.get("acceptance_decision") for r in rows if r.get("acceptance_decision")}
    accepted = tuple(r["correction_id"] for r in rows if r.get("acceptance_decision") == "ACCEPT_FOR_07C")
    rejected = tuple(r["correction_id"] for r in rows if r.get("acceptance_decision") == "REJECT_PROPOSAL")
    deferred = tuple(r["correction_id"] for r in rows if r.get("acceptance_decision") == "DEFER_TO_MANUAL_REVIEW")
    accepted_rows = [r for r in rows if r["correction_id"] in accepted]
    selected = list(accepted)
    redundant: list[str] = []
    claim_pairs = [p for p in pair_relations if p.claim_id == claim["claim_id"]]
    # Canonical redundant connected components.
    parent = {cid: cid for cid in accepted}
    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a: str,b: str) -> None:
        ra,rb=find(a),find(b)
        if ra!=rb: parent[max(ra,rb)] = min(ra,rb)
    for p in claim_pairs:
        if p.relation_type == "REDUNDANT": union(p.left_correction_id,p.right_correction_id)
    groups: dict[str,list[str]] = {}
    for cid in accepted: groups.setdefault(find(cid),[]).append(cid)
    for members in groups.values():
        members.sort(); redundant.extend(members[1:])
    selected = [cid for cid in selected if cid not in redundant]
    blocking_relations = {"OVERLAPPING_CONFLICTING","SEMANTICALLY_CONFLICTING","ORDER_DEPENDENT","NOT_COMPARABLE"}
    has_conflict = any(p.relation_type in blocking_relations for p in claim_pairs)
    manual = bool(deferred) or has_conflict
    virtual: str | None = None
    order: tuple[str, ...] = ()
    if not accepted:
        plan_type = "MANUAL_REVIEW_REQUIRED" if deferred else "NO_ACCEPTED_CORRECTIONS"
    elif manual:
        plan_type = "MULTIPLE_CONFLICTING_CORRECTIONS" if len(accepted) > 1 and has_conflict else "MANUAL_REVIEW_REQUIRED"
    elif len(accepted) == 1:
        plan_type = "SINGLE_ACCEPTED_CORRECTION"
    elif redundant and len(selected) == 1:
        plan_type = "MULTIPLE_REDUNDANT_CORRECTIONS"
    else:
        plan_type = "MULTIPLE_COMPATIBLE_CORRECTIONS"
    eligible = plan_type in {"SINGLE_ACCEPTED_CORRECTION","MULTIPLE_COMPATIBLE_CORRECTIONS","MULTIPLE_REDUNDANT_CORRECTIONS"} and not manual
    if not eligible:
        selected = []
    selected_rows: list[dict[str, Any]] = []
    selected_patches: tuple[dict[str, Any], ...] = ()
    if eligible:
        selected_rows = [r for r in accepted_rows if r["correction_id"] in selected]
        selected_patches = tuple({
            "correction_id": r["correction_id"], "claim_id": r["claim_id"], "section_id": r["section_id"],
            "original_claim_fingerprint": r["original_claim_fingerprint"],
            "target_span_in_claim": dict(r["target_span_in_claim"]), "replacement_text": r["replacement_text"],
        } for r in selected_rows)
        order = _expected_patch_order(selected_patches)
        virtual = _reconstruct_from_patches(str(claim["original_claim_text"]), selected_patches, order)
    candidate = _ordered(code for r in rows for code in r.get("resolved_issue_codes", ()))
    accepted_resolved = _ordered(code for r in accepted_rows for code in r.get("resolved_issue_codes", ()))
    remaining = tuple(claim.get("provisional_remaining_issue_codes", ()))
    new = _ordered(code for r in accepted_rows for code in r.get("new_issue_codes", ()))
    provisional = ClaimCorrectionResolutionPlan(
        claim_id=claim["claim_id"], section_id=claim["section_id"], plan_type=plan_type,
        individual_decisions=dict(sorted(decisions.items())), accepted_correction_ids=accepted,
        rejected_correction_ids=rejected, deferred_correction_ids=deferred,
        selected_correction_ids=tuple(sorted(selected)), redundant_correction_ids=tuple(sorted(redundant)),
        application_order=order, selected_patch_records=selected_patches, original_claim_text=claim["original_claim_text"], virtual_result_text=virtual,
        candidate_resolved_issue_codes=candidate, accepted_resolved_issue_codes=accepted_resolved,
        provisional_remaining_issue_codes=remaining, new_issue_codes=new,
        manual_review_required=manual, eligible_for_07c=eligible,
        requires_07c=bool(accepted), blocks_07c=manual,
        claim_resolution_plan_fingerprint="", correction_applied=False,
    )
    fp = _sha256(_plan_payload(provisional.to_dict()))
    return replace(provisional, claim_resolution_plan_fingerprint=fp)


def _normalized_result_payload(plans: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]], status: str, aggregation_status: str, eligible: bool) -> dict[str, Any]:
    return {
        "version": RESOLUTION_FP_VERSION,
        "claim_resolution_plans": tuple(plans), "pair_relations": tuple(pairs),
        "resolution_status": status, "aggregation_status": aggregation_status,
        "eligible_for_07c": eligible, "correction_applied": False,
        "official_artifacts_created": False, "additional_llm_calls": 0,
        "additional_retrieval_rounds": 0,
    }


def _audit_result_payload(normalized_fp: str | None, pairs: Sequence[Mapping[str, Any]], issues: Sequence[str], warnings: Sequence[str], blocked_claims: Sequence[str], source_audit_fp: str) -> dict[str, Any]:
    return {
        "version": RESOLUTION_AUDIT_FP_VERSION,
        "normalized_resolution_fingerprint": normalized_fp,
        "source_bundle_audit_fingerprint": source_audit_fp,
        "pair_relations": tuple(pairs), "resolution_issue_codes": tuple(issues),
        "resolution_warnings": tuple(warnings), "blocked_claims": tuple(blocked_claims),
    }


def _validate_pair_coverage(plans: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> None:
    plan_by_key={(p["claim_id"],p["section_id"]):p for p in plans}
    observed=set()
    blocking_by_plan: dict[tuple[str,str], bool]={}
    for pair in pairs:
        key=(pair["claim_id"],pair["section_id"])
        plan=plan_by_key.get(key)
        if plan is None:
            raise ValueError("PAIR_RELATION_PLAN_NOT_FOUND")
        accepted=set(plan["accepted_correction_ids"])
        ids={pair["left_correction_id"],pair["right_correction_id"]}
        if not ids.issubset(accepted):
            raise ValueError("PAIR_RELATION_CORRECTION_NOT_ACCEPTED")
        pk=(key,pair["left_correction_id"],pair["right_correction_id"])
        if pk in observed: raise ValueError("PAIR_RELATION_DUPLICATE")
        observed.add(pk)
        if pair["relation_type"] in BLOCKING_RELATIONS: blocking_by_plan[key]=True
    for key,plan in plan_by_key.items():
        accepted=sorted(plan["accepted_correction_ids"])
        expected={(key,accepted[i],accepted[j]) for i in range(len(accepted)) for j in range(i+1,len(accepted))}
        actual={x for x in observed if x[0]==key}
        if actual != expected: raise ValueError("PAIR_RELATION_COVERAGE_MISMATCH")
        has_blocking=blocking_by_plan.get(key,False)
        if plan["plan_type"]=="MULTIPLE_COMPATIBLE_CORRECTIONS" and has_blocking:
            raise ValueError("CLAIM_RESOLUTION_PLAN_COMPATIBLE_HAS_BLOCKING_RELATION")
        if plan["plan_type"]=="MULTIPLE_CONFLICTING_CORRECTIONS" and not has_blocking:
            raise ValueError("CLAIM_RESOLUTION_PLAN_CONFLICT_NOT_DEMONSTRATED")


def validate_provisional_multi_proposal_resolution_result(value: Mapping[str, Any], *, allow_unvalidated: bool = False) -> dict[str, Any]:
    required = set(ProvisionalMultiProposalResolutionResult.__dataclass_fields__)
    if set(value) != required: raise ValueError("MULTI_PROPOSAL_RESULT_SCHEMA_INVALID")
    v=dict(value)
    if v["resolution_status"] not in RESOLUTION_STATUSES or v["aggregation_status"] not in {"VALID","PARTIAL","INVALID"}:
        raise ValueError("MULTI_PROPOSAL_RESULT_STATUS_INVALID")
    if any(type(v[f]) is not bool or v[f] for f in ("correction_applied","official_artifacts_created")):
        raise ValueError("MULTI_PROPOSAL_RESULT_INVARIANT_INVALID")
    if any(type(v[f]) is not int or v[f]!=0 for f in ("additional_llm_calls","additional_retrieval_rounds")):
        raise ValueError("MULTI_PROPOSAL_RESULT_INVARIANT_INVALID")
    source=v["source_bundle_audit_fingerprint"]
    try:
        validate_sha256_hex(source, field="source_bundle_audit_fingerprint")
    except ValueError as exc:
        raise ValueError("MULTI_PROPOSAL_SOURCE_AUDIT_FINGERPRINT_INVALID") from exc
    issues=tuple(v["resolution_issue_codes"]); warnings=tuple(v["resolution_warnings"])
    if any(x not in RESOLUTION_ISSUE_CODES for x in issues): raise ValueError("MULTI_PROPOSAL_ISSUE_CODE_INVALID")
    if any(x not in RESOLUTION_WARNING_CODES for x in warnings): raise ValueError("MULTI_PROPOSAL_WARNING_CODE_INVALID")
    pairs=tuple(_validate_pair(p) for p in v["pair_relations"])
    plans=tuple(validate_claim_correction_resolution_plan(p) for p in v["claim_resolution_plans"])
    if list(plans)!=sorted(plans,key=lambda p:(p["section_id"],p["claim_id"])): raise ValueError("MULTI_PROPOSAL_PLAN_ORDER_INVALID")
    if list(pairs)!=sorted(pairs,key=lambda p:(p["section_id"],p["claim_id"],p["left_correction_id"],p["right_correction_id"])): raise ValueError("MULTI_PROPOSAL_PAIR_ORDER_INVALID")
    _validate_pair_coverage(plans,pairs)
    if v["aggregation_status"]=="INVALID":
        if v["resolution_status"]!="BLOCKED" or plans or pairs or v["eligible_for_07c"] or v["multi_proposal_resolution_fingerprint"] is not None:
            raise ValueError("MULTI_PROPOSAL_INVALID_BUNDLE_RULE")
    else:
        if v["resolution_status"]=="BLOCKED": raise ValueError("MULTI_PROPOSAL_STATUS_COHERENCE_INVALID")
        expected_eligible=any(p["requires_07c"] for p in plans) and not any(p["blocks_07c"] for p in plans)
        if v["eligible_for_07c"] != expected_eligible: raise ValueError("MULTI_PROPOSAL_GLOBAL_ELIGIBILITY_MISMATCH")
        has_noneligible=any(p["requires_07c"] and not p["eligible_for_07c"] for p in plans)
        expected_status="PARTIAL" if v["aggregation_status"]=="PARTIAL" or has_noneligible else "COMPLETED"
        if v["resolution_status"] != expected_status: raise ValueError("MULTI_PROPOSAL_RESOLUTION_STATUS_MISMATCH")
        normalized_expected=_sha256(_normalized_result_payload(plans,pairs,v["resolution_status"],v["aggregation_status"],v["eligible_for_07c"]))
        if v["multi_proposal_resolution_fingerprint"] != normalized_expected:
            raise ValueError("MULTI_PROPOSAL_RESOLUTION_FINGERPRINT_MISMATCH")
    blocked=tuple(p["claim_id"] for p in plans if p["blocks_07c"])
    audit_expected=_sha256(_audit_result_payload(v["multi_proposal_resolution_fingerprint"],pairs,issues,warnings,blocked,source))
    if v["multi_proposal_audit_fingerprint"] != audit_expected:
        raise ValueError("MULTI_PROPOSAL_AUDIT_FINGERPRINT_MISMATCH")
    if not allow_unvalidated and v["result_contract_valid"] is not True: raise ValueError("MULTI_PROPOSAL_RESULT_CONTRACT_NOT_VALID")
    v["pair_relations"],v["claim_resolution_plans"],v["resolution_issue_codes"],v["resolution_warnings"]=pairs,plans,issues,warnings
    return v

def resolve_multiple_correction_proposals(bundle: Any) -> ProvisionalMultiProposalResolutionResult:
    b = validate_provisional_verification_traceability_bundle_contract(_plain(bundle))
    if b["aggregation_status"] == "INVALID":
        issues = ("MULTI_PROPOSAL_BLOCKED_INVALID_BUNDLE",)
        audit = _sha256(_audit_result_payload(None, (), issues, (), (), b.get("aggregation_audit_fingerprint")))
        payload = ProvisionalMultiProposalResolutionResult(
            claim_resolution_plans=(), pair_relations=(), resolution_issue_codes=issues,
            resolution_warnings=(), resolution_status="BLOCKED", aggregation_status="INVALID",
            eligible_for_07c=False, multi_proposal_resolution_fingerprint=None,
            multi_proposal_audit_fingerprint=audit, source_bundle_audit_fingerprint=b.get("aggregation_audit_fingerprint"), result_contract_valid=False,
        ).to_dict()
    else:
        claims = sorted((dict(x) for x in b["claim_traceability_rows"]), key=lambda x:(x["section_id"],x["claim_id"]))
        corrections = sorted((dict(x) for x in b["correction_traceability_rows"]), key=lambda x:(x["section_id"],x["claim_id"],x["correction_id"]))
        by_claim: dict[tuple[str,str], list[dict[str,Any]]] = {}
        for row in corrections: by_claim.setdefault((row["claim_id"],row["section_id"]),[]).append(row)
        pairs: list[CorrectionPairRelation] = []
        for key, rows in sorted(by_claim.items()):
            accepted = [r for r in rows if r.get("acceptance_decision") == "ACCEPT_FOR_07C"]
            for i in range(len(accepted)):
                for j in range(i+1,len(accepted)):
                    pairs.append(classify_correction_pair(accepted[i],accepted[j]))
        pairs.sort(key=lambda p:(p.section_id,p.claim_id,p.left_correction_id,p.right_correction_id))
        plans = [_build_plan(c, by_claim.get((c["claim_id"],c["section_id"]),()), pairs) for c in claims]
        plans.sort(key=lambda p:(p.section_id,p.claim_id))
        blocked = tuple(p.claim_id for p in plans if p.blocks_07c)
        status = "PARTIAL" if blocked or b["aggregation_status"] == "PARTIAL" else "COMPLETED"
        eligible = any(p.requires_07c for p in plans) and not any(p.blocks_07c for p in plans)
        plan_dicts = tuple(p.to_dict() for p in plans); pair_dicts = tuple(p.to_dict() for p in pairs)
        normalized_fp = _sha256(_normalized_result_payload(plan_dicts,pair_dicts,status,b["aggregation_status"],eligible))
        issues = _ordered("MULTI_PROPOSAL_CONFLICT" for p in plans if p.plan_type=="MULTIPLE_CONFLICTING_CORRECTIONS")
        warnings = _ordered("MULTI_PROPOSAL_MANUAL_REVIEW_REQUIRED" for p in plans if p.manual_review_required)
        audit = _sha256(_audit_result_payload(normalized_fp,pair_dicts,issues,warnings,blocked,b.get("aggregation_audit_fingerprint")))
        payload = ProvisionalMultiProposalResolutionResult(
            claim_resolution_plans=plan_dicts, pair_relations=pair_dicts,
            resolution_issue_codes=issues, resolution_warnings=warnings,
            resolution_status=status, aggregation_status=b["aggregation_status"], eligible_for_07c=eligible,
            multi_proposal_resolution_fingerprint=normalized_fp, multi_proposal_audit_fingerprint=audit,
            source_bundle_audit_fingerprint=b.get("aggregation_audit_fingerprint"), result_contract_valid=False,
        ).to_dict()
    validate_provisional_multi_proposal_resolution_result(payload, allow_unvalidated=True)
    payload["result_contract_valid"] = True
    result = ProvisionalMultiProposalResolutionResult(**payload)
    validate_provisional_multi_proposal_resolution_result(result.to_dict())
    return result
