"""Propuestas de corrección localizadas por claim.

Nunca modifica el borrador. Toda propuesta queda pendiente de reverificación.
No contiene OpenAI, red, runtime ni PipelineState.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import re
import unicodedata
from typing import Any, Mapping, Protocol, Sequence

from src.config.verification_policy_config import (
    ATTRIBUTION_RELATIONS, CORRECTION_ACTION_TYPES, CORRECTION_CHANGE_SCOPES,
    CORRECTION_COORDINATE_BASES, CORRECTION_COORDINATE_SYSTEMS,
    CORRECTION_DECISIONS, CORRECTION_LOCALIZATION_METHODS,
    CORRECTION_PROPOSAL_STATUSES, CORRECTION_REASON_CODES,
    CORRECTION_SEMANTIC_CHANGE_LEVELS, get_verification_input_policy,
)
from src.tools.verification.validation import (
    quantitative_pair_supported, metric_context_supported, canonical_correction_evidence_text,
)

class CorrectionLLM(Protocol):
    def invoke(self, messages: Sequence[Mapping[str, str]]) -> str: ...

@dataclass(frozen=True, slots=True)
class TextSpan:
    coordinate_base: str
    coordinate_system: str
    base_text_fingerprint: str
    start: int
    end: int
    text: str

@dataclass(frozen=True, slots=True)
class CorrectionProposal:
    correction_id: str
    claim_id: str
    section_id: str
    correction_decision: str
    action_type: str | None
    proposal_status: str
    original_text: str
    original_claim_fingerprint: str
    original_section_fingerprint: str
    claim_span_in_section: dict[str, Any] | None
    target_span_in_claim: dict[str, Any] | None
    localization_method: str | None
    target_text: str
    target_text_fingerprint: str
    replacement_text: str
    proposed_claim_text: str
    evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    change_scope: str
    semantic_change_level: str
    old_citation_refs: tuple[dict[str, str], ...]
    new_citation_refs: tuple[dict[str, str], ...]
    citation_text_span: dict[str, Any] | None
    old_numeric_pairs: tuple[tuple[str, str], ...]
    new_numeric_pairs: tuple[tuple[str, str], ...]
    metric_context: str
    unit_context: str
    old_attribution_elements: tuple[str, ...]
    new_attribution_elements: tuple[str, ...]
    attribution_relation: str | None
    new_entities: tuple[str, ...]
    new_citations: tuple[dict[str, str], ...]
    new_attributions: tuple[str, ...]
    new_conditions: tuple[str, ...]
    new_technical_terms: tuple[str, ...]
    llm_correction_recommendation: bool
    requires_manual_review: bool
    accepted_for_reverification: bool
    correction_applied: bool
    final_proposal_status: str
    proposal_fingerprint: str
    prompt_version: str
    validation_issue_codes: tuple[str, ...]
    decision_path: tuple[str, ...]
    raw_attempts: tuple[dict[str, Any], ...]
    retry_metrics: dict[str, int]

    def to_dict(self) -> dict[str, Any]: return asdict(self)

def fingerprint_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def build_correction_proposal_fingerprint_payload(
    *, original_claim_fingerprint: str, original_section_fingerprint: str,
    target_text_fingerprint: str, claim_id: str, action_type: str,
    target_span: Mapping[str, Any], replacement_text: str,
    evidence_ids: Sequence[str], prompt_version: str,
) -> dict[str, Any]:
    """Payload contractual congelado en Fase 5T."""
    return {
        "original_claim_fingerprint": str(original_claim_fingerprint),
        "original_section_fingerprint": str(original_section_fingerprint),
        "target_text_fingerprint": str(target_text_fingerprint),
        "claim_id": str(claim_id),
        "action_type": str(action_type),
        "target_span": dict(target_span),
        "replacement_text": str(replacement_text),
        "evidence_ids": list(evidence_ids),
        "prompt_version": str(prompt_version),
    }


def compute_correction_proposal_fingerprint(**kwargs: Any) -> str:
    payload = build_correction_proposal_fingerprint_payload(**kwargs)
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_empty_correction_proposal_fingerprint_payload(
    *, claim_id: str, decision: str, status: str, prompt_version: str,
) -> dict[str, Any]:
    """Payload histórico real usado por _empty_proposal(...)."""
    return {
        "claim_id": str(claim_id),
        "decision": str(decision),
        "status": str(status),
        "prompt_version": str(prompt_version),
    }


def compute_empty_correction_proposal_fingerprint(
    *, claim_id: str, decision: str, status: str, prompt_version: str,
) -> str:
    payload = build_empty_correction_proposal_fingerprint_payload(
        claim_id=claim_id, decision=decision, status=status, prompt_version=prompt_version,
    )
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def _normalize_surface_with_map(text: str) -> tuple[str, tuple[int, ...]]:
    """Normaliza solo superficie y mantiene un mapa a offsets originales."""
    out: list[str] = []
    mapping: list[int] = []
    quote_map = {'“':'"','”':'"','‘':"'",'’':"'"}
    pending_space = False
    for idx, raw in enumerate(text):
        chars = unicodedata.normalize("NFKC", raw)
        for ch in chars:
            ch = quote_map.get(ch, ch)
            if ch.isspace():
                pending_space = bool(out)
                continue
            if ch in ",.;:!?" and out and out[-1] == " ":
                out.pop(); mapping.pop()
            elif pending_space and out and out[-1] != " ":
                out.append(" "); mapping.append(idx)
            pending_space = False
            out.append(ch); mapping.append(idx)
    while out and out[-1] == " ": out.pop(); mapping.pop()
    return "".join(out), tuple(mapping)

def _normalize_surface(text: str) -> str:
    return _normalize_surface_with_map(text)[0]

def _validate_span_shape(span: Mapping[str, Any], *, base: str, fingerprint: str, text: str) -> TextSpan:
    if span.get("coordinate_base") != base or span.get("coordinate_system") != "PYTHON_CODEPOINT_OFFSETS":
        raise ValueError("SPAN_COORDINATE_BASE_INVALID")
    if span.get("base_text_fingerprint") != fingerprint:
        raise ValueError("ORIGINAL_FINGERPRINT_MISMATCH")
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int or not (0 <= start < end <= len(text)):
        raise ValueError("TARGET_SPAN_NOT_FOUND")
    exact = text[start:end]
    if exact != span.get("text"):
        raise ValueError("TARGET_TEXT_MISMATCH")
    return TextSpan(base, "PYTHON_CODEPOINT_OFFSETS", fingerprint, start, end, exact)

def validate_claim_span_in_section(section_text: str, original_claim_text: str, span: Mapping[str, Any], *, section_fingerprint: str) -> TextSpan:
    result = _validate_span_shape(span, base="SECTION_TEXT", fingerprint=section_fingerprint, text=section_text)
    if result.text != original_claim_text:
        raise ValueError("CLAIM_SPAN_TEXT_MISMATCH")
    return result

def locate_target(original_claim_text: str, target_text: str, *, claim_fingerprint: str,
                  explicit_span: Mapping[str, Any] | None = None) -> tuple[TextSpan, str]:
    if fingerprint_text(original_claim_text) != claim_fingerprint:
        raise ValueError("ORIGINAL_FINGERPRINT_MISMATCH")
    if explicit_span is not None:
        result = _validate_span_shape(explicit_span, base="CLAIM_TEXT", fingerprint=claim_fingerprint, text=original_claim_text)
        if result.text != target_text:
            raise ValueError("TARGET_TEXT_MISMATCH")
        return result, "CONTRACTUAL_SPAN"
    starts=[]; pos=0
    while True:
        idx=original_claim_text.find(target_text,pos)
        if idx<0: break
        starts.append(idx); pos=idx+1
    if len(starts)==1:
        st=starts[0]
        return TextSpan("CLAIM_TEXT","PYTHON_CODEPOINT_OFFSETS",claim_fingerprint,st,st+len(target_text),target_text),"EXACT_UNIQUE_MATCH"
    if len(starts)>1: raise ValueError("AMBIGUOUS_TARGET_SPAN")
    normalized_original, mapping = _normalize_surface_with_map(original_claim_text)
    normalized_target = _normalize_surface(target_text)
    matches=[]; pos=0
    while normalized_target:
        idx=normalized_original.find(normalized_target,pos)
        if idx<0: break
        end_idx=idx+len(normalized_target)-1
        start_orig=mapping[idx]; end_orig=mapping[end_idx]+1
        exact=original_claim_text[start_orig:end_orig]
        if _normalize_surface(exact)==normalized_target:
            matches.append((start_orig,end_orig,exact))
        pos=idx+1
    uniq=[]
    for item in matches:
        if item[:2] not in [x[:2] for x in uniq]: uniq.append(item)
    if len(uniq)==1:
        st,en,exact=uniq[0]
        return TextSpan("CLAIM_TEXT","PYTHON_CODEPOINT_OFFSETS",claim_fingerprint,st,en,exact),"NORMALIZED_UNIQUE_MATCH"
    if len(uniq)>1: raise ValueError("AMBIGUOUS_TARGET_SPAN")
    raise ValueError("TARGET_SPAN_NOT_FOUND")

def apply_localized_change(original: str, span: Mapping[str, Any], replacement: str) -> str:
    if span.get("coordinate_base") != "CLAIM_TEXT" or span.get("coordinate_system") != "PYTHON_CODEPOINT_OFFSETS":
        raise ValueError("SPAN_COORDINATE_BASE_INVALID")
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int or not (0 <= start <= end <= len(original)):
        raise ValueError("TARGET_SPAN_NOT_FOUND")
    if original[start:end] != span.get("text"):
        raise ValueError("TARGET_TEXT_MISMATCH")
    return original[:start] + replacement + original[end:]

def _pairs(value: Any, field: str) -> tuple[tuple[str, str], ...]:
    if type(value) not in (list, tuple): raise ValueError(f"CORRECTION_FIELD_INVALID:{field}")
    out=[]
    for row in value:
        if type(row) not in (list, tuple) or len(row)!=2 or not all(isinstance(x,str) for x in row):
            raise ValueError(f"CORRECTION_FIELD_INVALID:{field}")
        out.append((row[0].strip(), row[1].strip()))
    return tuple(out)

def _refs(value: Any, field: str) -> tuple[dict[str,str], ...]:
    if type(value) not in (list, tuple): raise ValueError(f"CORRECTION_FIELD_INVALID:{field}")
    out=[]
    for row in value:
        if not isinstance(row, Mapping) or not isinstance(row.get("source_filename"),str) or not isinstance(row.get("chunk_id"),str):
            raise ValueError(f"CORRECTION_FIELD_INVALID:{field}")
        out.append({"source_filename":row["source_filename"].strip(),"chunk_id":row["chunk_id"].strip()})
    return tuple(out)

def _strings(value: Any, field: str) -> tuple[str,...]:
    if type(value) not in (list,tuple) or any(not isinstance(x,str) or not x.strip() for x in value):
        raise ValueError(f"CORRECTION_FIELD_INVALID:{field}")
    return tuple(x.strip() for x in value)

def validate_correction_response(
    value: Mapping[str, Any], *, allowed_evidence_ids: Sequence[str],
    expected_claim_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping): raise ValueError("CORRECTION_RESPONSE_NOT_OBJECT")
    required=("claim_id","correction_decision","action_type","target_text","replacement_text","evidence_ids",
              "reason_codes","change_scope","semantic_change_level","old_citation_refs","new_citation_refs",
              "old_numeric_pairs","new_numeric_pairs","metric_context","unit_context","old_attribution_elements",
              "new_attribution_elements","attribution_relation","new_entities","new_attributions","new_conditions",
              "new_technical_terms","citation_text_span","llm_correction_recommendation")
    unknown=set(value)-set(required); missing=set(required)-set(value)
    if unknown: raise ValueError("CORRECTION_RESPONSE_UNKNOWN_FIELDS:"+",".join(sorted(unknown)))
    if missing: raise ValueError("CORRECTION_RESPONSE_MISSING_FIELDS:"+",".join(sorted(missing)))
    out=dict(value)
    if expected_claim_id is not None and out.get("claim_id") != expected_claim_id:
        raise ValueError("CORRECTION_RESPONSE_CLAIM_ID_MISMATCH")
    if out["correction_decision"] not in CORRECTION_DECISIONS: raise ValueError("CORRECTION_DECISION_UNKNOWN")
    action=out["action_type"]
    if out["correction_decision"]=="PROPOSE_CHANGE":
        if action not in CORRECTION_ACTION_TYPES: raise ValueError("CORRECTION_ACTION_UNKNOWN")
        if out["llm_correction_recommendation"] is not True:
            raise ValueError("CORRECTION_RECOMMENDATION_CONTRADICTION")
    else:
        if action is not None: raise ValueError("CORRECTION_ACTION_NOT_ALLOWED_FOR_DECISION")
        if out["llm_correction_recommendation"] is not False:
            raise ValueError("CORRECTION_RECOMMENDATION_CONTRADICTION")
    if out["change_scope"] not in CORRECTION_CHANGE_SCOPES: raise ValueError("CORRECTION_CHANGE_SCOPE_UNKNOWN")
    if out["semantic_change_level"] not in CORRECTION_SEMANTIC_CHANGE_LEVELS: raise ValueError("CORRECTION_SEMANTIC_LEVEL_UNKNOWN")
    if type(out["llm_correction_recommendation"]) is not bool: raise ValueError("CORRECTION_FIELD_INVALID:llm_correction_recommendation")
    out["evidence_ids"]=_strings(out["evidence_ids"],"evidence_ids")
    if not set(out["evidence_ids"]).issubset(set(allowed_evidence_ids)): raise ValueError("UNKNOWN_EVIDENCE_ID")
    out["reason_codes"]=_strings(out["reason_codes"],"reason_codes")
    if not set(out["reason_codes"]).issubset(set(CORRECTION_REASON_CODES)): raise ValueError("CORRECTION_REASON_CODE_UNKNOWN")
    for field in ("new_entities","new_attributions","new_conditions","new_technical_terms","old_attribution_elements","new_attribution_elements"):
        out[field]=_strings(out[field],field)
    out["old_numeric_pairs"]=_pairs(out["old_numeric_pairs"],"old_numeric_pairs")
    out["new_numeric_pairs"]=_pairs(out["new_numeric_pairs"],"new_numeric_pairs")
    out["old_citation_refs"]=_refs(out["old_citation_refs"],"old_citation_refs")
    out["new_citation_refs"]=_refs(out["new_citation_refs"],"new_citation_refs")
    if out["citation_text_span"] is not None and not isinstance(out["citation_text_span"], Mapping):
        raise ValueError("CORRECTION_FIELD_INVALID:citation_text_span")
    if action=="CORRECT_ATTRIBUTION" and out["attribution_relation"] not in ATTRIBUTION_RELATIONS:
        raise ValueError("ATTRIBUTION_RELATION_INVALID")
    if action!="CORRECT_ATTRIBUTION" and out["attribution_relation"] is not None:
        raise ValueError("ATTRIBUTION_RELATION_NOT_APPLICABLE")
    for field in ("claim_id","target_text","replacement_text","metric_context","unit_context"):
        if not isinstance(out[field],str): raise ValueError(f"CORRECTION_FIELD_INVALID:{field}")
    if out["correction_decision"]=="PROPOSE_CHANGE":
        if out["semantic_change_level"]!="MINIMAL": raise ValueError("AUTOMATIC_PROPOSAL_REQUIRES_MINIMAL_CHANGE")
        if not out["evidence_ids"]: raise ValueError("AUTOMATIC_PROPOSAL_REQUIRES_EVIDENCE")
        if action!="REMOVE_UNSUPPORTED_FRAGMENT" and not out["replacement_text"]: raise ValueError("REPLACEMENT_TEXT_REQUIRED")
        if action=="REPLACE_NUMERIC_VALUE" and (not out["old_numeric_pairs"] or not out["new_numeric_pairs"]): raise ValueError("NUMERIC_PAIRS_REQUIRED")
        if action=="CORRECT_ATTRIBUTION" and (not out["old_attribution_elements"] or not out["new_attribution_elements"] or not out["attribution_relation"]): raise ValueError("ATTRIBUTION_FIELDS_REQUIRED")
        if action=="REPLACE_CITATION" and (not out["old_citation_refs"] or not out["new_citation_refs"]): raise ValueError("CITATION_REFS_REQUIRED")
        if action=="REPLACE_CITATION" and out["citation_text_span"] is None: raise ValueError("CITATION_TEXT_SPAN_REQUIRED")
        if action=="ADD_QUALIFICATION" and not out["new_conditions"]: raise ValueError("QUALIFICATION_CONDITIONS_REQUIRED")
        if action=="NARROW_SCOPE" and not out["new_conditions"]: raise ValueError("NARROW_SCOPE_CONDITIONS_REQUIRED")

        # Matriz cerrada: una acción no puede transportar una segunda clase de cambio.
        forbidden_nonempty = {
            "REPLACE_NUMERIC_VALUE": ("old_citation_refs","new_citation_refs","old_attribution_elements","new_attribution_elements","new_attributions","new_conditions","new_entities","new_technical_terms"),
            "CORRECT_ATTRIBUTION": ("old_citation_refs","new_citation_refs","old_numeric_pairs","new_numeric_pairs","new_conditions","new_entities","new_technical_terms"),
            "REPLACE_CITATION": ("old_numeric_pairs","new_numeric_pairs","old_attribution_elements","new_attribution_elements","new_attributions","new_conditions","new_entities","new_technical_terms"),
            "ADD_QUALIFICATION": ("old_citation_refs","new_citation_refs","old_numeric_pairs","new_numeric_pairs","old_attribution_elements","new_attribution_elements","new_attributions","new_entities","new_technical_terms"),
            "REMOVE_UNSUPPORTED_FRAGMENT": ("new_citation_refs","new_numeric_pairs","new_attribution_elements","new_attributions","new_conditions","new_entities","new_technical_terms"),
            "NARROW_SCOPE": ("old_citation_refs","new_citation_refs","old_numeric_pairs","new_numeric_pairs","old_attribution_elements","new_attribution_elements","new_attributions","new_entities","new_technical_terms"),
            "SPLIT_CLAIM": (),
        }
        if any(bool(out[field]) for field in forbidden_nonempty.get(action, ())):
            raise ValueError("CORRECTION_ACTION_FIELD_MATRIX_VIOLATION")
    return out

def _balanced(text: str) -> bool:
    pairs={')':'(',']':'[','}':'{'}; stack=[]
    for ch in text:
        if ch in '([{': stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop()!=pairs[ch]: return False
    return not stack

def validate_text_integrity(text: str) -> tuple[str,...]:
    issues=[]
    if re.search(r"\s{2,}", text) or re.search(r"\s+[,.!?;:]", text): issues.append("WHITESPACE_INTEGRITY_INVALID")
    if not _balanced(text): issues.append("BRACKET_BALANCE_INVALID")
    if text.count('[')!=text.count(']'): issues.append("CITATION_SYNTAX_INVALID")
    if re.search(r"[,;:]\s*[.!?]", text) or re.search(r"[.!?]{2,}", text): issues.append("PUNCTUATION_INTEGRITY_INVALID")
    return tuple(issues)

def _spans_overlap(a: Mapping[str,Any], b: Mapping[str,Any]) -> bool:
    return max(a["start"],b["start"]) < min(a["end"],b["end"])

def _span_contract_tuple(span: Mapping[str,Any]) -> tuple[Any,...]:
    return (span.get("coordinate_base"), span.get("coordinate_system"), span.get("base_text_fingerprint"))

def _to_section_span(proposal: Mapping[str,Any]) -> dict[str,Any] | None:
    target=proposal.get("target_span_in_claim") or proposal.get("target_span") or {}
    claim_span=proposal.get("claim_span_in_section") or {}
    if not target or not claim_span: return None
    if target.get("coordinate_base")!="CLAIM_TEXT" or claim_span.get("coordinate_base")!="SECTION_TEXT": return None
    if target.get("coordinate_system")!=claim_span.get("coordinate_system"): return None
    return {"start":claim_span["start"]+target["start"],"end":claim_span["start"]+target["end"],
            "coordinate_base":"SECTION_TEXT","coordinate_system":claim_span["coordinate_system"],
            "base_text_fingerprint":claim_span["base_text_fingerprint"]}

def _prior_proposal_status_active(prior: Mapping[str, Any]) -> bool:
    status = prior.get("proposal_status", prior.get("final_proposal_status"))
    if status is None:
        # Compatibilidad con propuestas previas creadas antes de introducir el estado explícito.
        return True
    return status in {"PROPOSED", "ACCEPTED_FOR_REVERIFICATION"}

def _assess_prior_proposal(prior: Any, context: Mapping[str, Any]) -> tuple[bool, str | None, bool]:
    """Valida una propuesta previa antes de contarla o usarla para conflictos.

    Retorna (válida, reason_code_si_se_ignora, activa).
    """
    if not isinstance(prior, Mapping):
        return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
    for field in ("claim_id", "section_id"):
        if not isinstance(prior.get(field), str) or not prior.get(field).strip():
            return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
    if prior.get("section_id") != context.get("section_id"):
        return False, "PRIOR_CORRECTION_SECTION_MISMATCH", False
    claim_span = prior.get("claim_span_in_section")
    target = prior.get("target_span_in_claim")
    if not isinstance(claim_span, Mapping) or not isinstance(target, Mapping):
        return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
    required_span = ("coordinate_base", "coordinate_system", "base_text_fingerprint", "start", "end", "text")
    if any(k not in claim_span for k in required_span) or any(k not in target for k in required_span):
        return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
    if claim_span.get("coordinate_base") != "SECTION_TEXT" or target.get("coordinate_base") != "CLAIM_TEXT":
        return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
    if claim_span.get("coordinate_system") != "PYTHON_CODEPOINT_OFFSETS" or target.get("coordinate_system") != "PYTHON_CODEPOINT_OFFSETS":
        return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
    for span in (claim_span, target):
        if not isinstance(span.get("base_text_fingerprint"), str) or not span.get("base_text_fingerprint"):
            return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
        if type(span.get("start")) is not int or type(span.get("end")) is not int:
            return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
        if not isinstance(span.get("text"), str) or not span.get("text"):
            return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
        if not (0 <= span["start"] < span["end"]):
            return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False

    section_text = context.get("section_text")
    section_fp = context.get("section_fingerprint")
    if not isinstance(section_text, str) or not isinstance(section_fp, str):
        return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False
    if claim_span.get("base_text_fingerprint") != section_fp:
        return False, "STALE_PRIOR_CORRECTION_PROPOSAL", False
    if claim_span["end"] > len(section_text) or section_text[claim_span["start"]:claim_span["end"]] != claim_span["text"]:
        return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False

    prior_claim_text = claim_span["text"]
    expected_claim_fp = fingerprint_text(prior_claim_text)
    if target.get("base_text_fingerprint") != expected_claim_fp:
        return False, "STALE_PRIOR_CORRECTION_PROPOSAL", False
    if target["end"] > len(prior_claim_text) or prior_claim_text[target["start"]:target["end"]] != target["text"]:
        return False, "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", False

    if prior.get("claim_id") == context.get("claim_id"):
        current_claim_fp = context.get("claim_fingerprint")
        if target.get("base_text_fingerprint") != current_claim_fp:
            return False, "STALE_PRIOR_CORRECTION_PROPOSAL", False
    return True, None, _prior_proposal_status_active(prior)

def _valid_active_prior_proposals(context: Mapping[str, Any]) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    valid = []
    audit = []
    for prior in context.get("existing_correction_proposals", ()):
        ok, reason, active = _assess_prior_proposal(prior, context)
        if reason:
            audit.append(reason)
        if ok and active:
            valid.append(prior)
    valid.sort(key=lambda x: (str(x.get("claim_id", "")), str(x.get("correction_id", "")), (x.get("target_span_in_claim") or {}).get("start", -1)))
    return tuple(valid), tuple(sorted(set(audit)))

def preproposal_conflict_check(context: Mapping[str,Any], target_span: Mapping[str,Any]) -> tuple[tuple[str,...], tuple[str,...]]:
    issues=[]
    valid_priors, warnings = _valid_active_prior_proposals(context)
    current_claim=str(context.get("claim_id","")); current_claim_span=context.get("claim_span_in_section")
    for prior in valid_priors:
        prior_target=prior["target_span_in_claim"]
        if prior.get("claim_id")==current_claim:
            if _span_contract_tuple(prior_target)!=_span_contract_tuple(target_span):
                issues.append("FINGERPRINT_CONFLICT"); continue
            if _spans_overlap(target_span,prior_target): issues.append("OVERLAPPING_CORRECTIONS")
        else:
            if not current_claim_span: continue
            current={"claim_id":current_claim,"target_span_in_claim":target_span,"claim_span_in_section":current_claim_span}
            a=_to_section_span(current); b=_to_section_span(prior)
            if not a or not b: continue
            if _span_contract_tuple(a)!=_span_contract_tuple(b):
                issues.append("FINGERPRINT_CONFLICT"); continue
            if _spans_overlap(a,b): issues.append("OVERLAPPING_CORRECTIONS")
    return tuple(sorted(set(issues))), warnings

def postproposal_batch_conflict_analysis(proposals: Sequence[Mapping[str,Any]]) -> tuple[dict[str,Any],...]:
    ordered=sorted(proposals,key=lambda x:(x.get("section_id",""),x.get("claim_id",""),((x.get("target_span_in_claim") or x.get("target_span") or {}).get("start",-1)),x.get("correction_id","")))
    conflicts=[]
    for i,a in enumerate(ordered):
        for b in ordered[i+1:]:
            if a.get("section_id")!=b.get("section_id"): continue
            if a.get("claim_id")==b.get("claim_id"):
                sa=a.get("target_span_in_claim") or a.get("target_span") or {}; sb=b.get("target_span_in_claim") or b.get("target_span") or {}
                if _span_contract_tuple(sa)!=_span_contract_tuple(sb): ctype="FINGERPRINT_CONFLICT"
                elif _spans_overlap(sa,sb): ctype="SPAN_OVERLAP"
                else: continue
            else:
                sa=_to_section_span(a); sb=_to_section_span(b)
                if not sa or not sb: continue
                if _span_contract_tuple(sa)!=_span_contract_tuple(sb): ctype="FINGERPRINT_CONFLICT"
                elif _spans_overlap(sa,sb): ctype="SPAN_OVERLAP"
                else: continue
            ids=tuple(sorted((a.get("correction_id",""),b.get("correction_id",""))))
            conflicts.append({"conflict_id":"CF_"+sha256('|'.join(ids).encode()).hexdigest()[:12],"correction_ids":ids,"conflict_type":ctype,"manual_review_required":True})
    return tuple(conflicts)

def _extract_tokens(text: str) -> set[str]:
    return {x.casefold() for x in re.findall(r"\b[\w.-]+\b", text, re.UNICODE) if len(x)>2}

def validate_structural_novelty(response: Mapping[str,Any], evidence: Sequence[Mapping[str,Any]]) -> tuple[str,...]:
    corpus=' '.join(canonical_correction_evidence_text(x) for x in evidence if x.get('authorized_for_section'))
    supported=_extract_tokens(corpus)
    issues=[]
    if any(x.casefold() not in supported for x in response.get("new_entities",())): issues.append("NEW_ENTITY_UNSUPPORTED")
    if any(x.casefold() not in supported for x in response.get("new_technical_terms",())): issues.append("NEW_TECHNICAL_TERM_UNSUPPORTED")
    corpus_cf = corpus.casefold()
    if any(x.casefold() not in corpus_cf for x in response.get("new_conditions",())): issues.append("UNSUPPORTED_NEW_INFORMATION")
    return tuple(sorted(set(issues)))

def _contains_delimited(text: str, element: str) -> bool:
    if not element.strip(): return False
    pattern=r"(?<!\w)"+re.escape(element.strip())+r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE|re.UNICODE) is not None

def _citation_reference_markers(ref: Mapping[str, str]) -> tuple[str, ...]:
    source = str(ref.get("source_filename", "")).strip()
    chunk = str(ref.get("chunk_id", "")).strip()
    markers = []
    if source:
        basename = source.replace("\\", "/").rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0]
        markers.extend((source, basename, stem))
        markers.extend(token for token in re.split(r"[^\w]+", stem, flags=re.UNICODE) if len(token) >= 3)
    if chunk:
        markers.append(chunk)
    return tuple(sorted({m for m in markers if m}, key=lambda x: (-len(x), x.casefold())))

def _validate_citation_span(original: str, span: Mapping[str,Any], old_refs: Sequence[Mapping[str,str]]) -> TextSpan:
    result=_validate_span_shape(span, base="CLAIM_TEXT", fingerprint=fingerprint_text(original), text=original)
    linked = any(
        _contains_delimited(result.text, marker)
        for ref in old_refs
        for marker in _citation_reference_markers(ref)
    )
    if not linked:
        raise ValueError("CITATION_TEXT_REFERENCE_MISMATCH")
    return result

def _removal_semantics_unsafe(original: str, target: str) -> bool:
    protected=(" no "," not "," nunca "," never "," supera "," outperforms "," causa "," causes "," porque "," because ")
    padded=' '+target.casefold()+' '
    return any(tok in padded for tok in protected) or target.strip()==original.strip()

def propose_correction(context: Mapping[str,Any], *, llm: CorrectionLLM | None) -> CorrectionProposal:
    policy=get_verification_input_policy(context.get("policy")) if context.get("policy") else get_verification_input_policy()
    claim_id=str(context["claim_id"]); section_id=str(context["section_id"]); original=str(context["original_claim_text"])
    if not isinstance(context.get("claim_fingerprint"),str) or not context["claim_fingerprint"].strip(): raise ValueError("CLAIM_FINGERPRINT_REQUIRED")
    claim_fp=str(context["claim_fingerprint"]); section_fp=str(context.get("section_fingerprint") or "")
    if fingerprint_text(original)!=claim_fp: raise ValueError("ORIGINAL_FINGERPRINT_MISMATCH")
    decision_path=["CORRECTION_RECOMMENDATION_REVIEWED"]
    section_value=context.get("section_text"); claim_span_in_section=context.get("claim_span_in_section")
    if not isinstance(section_value,str) or not section_value or not isinstance(claim_span_in_section,Mapping):
        return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,"DEFER_TO_MANUAL_REVIEW","DEFERRED",decision_path,policy,issues=("CLAIM_SPAN_IN_SECTION_REQUIRED",),claim_span_in_section=None)
    section=section_value
    if not section_fp: raise ValueError("SECTION_FINGERPRINT_REQUIRED")
    if fingerprint_text(section)!=section_fp: raise ValueError("SECTION_FINGERPRINT_MISMATCH")
    validate_claim_span_in_section(section,original,claim_span_in_section,section_fingerprint=section_fp)
    eligibility=str(context.get("final_correction_eligibility","NO_CORRECTION_NEEDED"))
    if eligibility=="NO_CORRECTION_NEEDED":
        return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,"NO_CORRECTION","NOT_PROPOSED",decision_path+["ELIGIBILITY_NO_CORRECTION_NEEDED"],policy,reason_codes=("NO_AUTOMATIC_CORRECTION_REQUIRED",))
    if eligibility in {"MANUAL_REVIEW_REQUIRED","NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE"}:
        decision="DEFER_TO_MANUAL_REVIEW" if eligibility=="MANUAL_REVIEW_REQUIRED" else "NOT_CORRECTABLE"
        status="DEFERRED" if decision=="DEFER_TO_MANUAL_REVIEW" else "NOT_PROPOSED"
        cause="VERIFIER_REQUIRES_MANUAL_REVIEW" if eligibility=="MANUAL_REVIEW_REQUIRED" else "CLAIM_NOT_AUTOCORRECTABLE"
        return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,decision,status,decision_path+[f"ELIGIBILITY_{eligibility}"],policy,reason_codes=(cause,))
    valid_active_priors, prior_audit = _valid_active_prior_proposals(context)
    active_for_claim = tuple(p for p in valid_active_priors if p.get("claim_id") == claim_id)
    if len(active_for_claim) >= policy["max_correction_proposals_per_claim"]:
        return _empty_proposal(
            claim_id, section_id, original, claim_fp, section_fp,
            "DEFER_TO_MANUAL_REVIEW", "DEFERRED", decision_path, policy,
            issues=tuple(sorted(set(prior_audit + ("CORRECTION_PROPOSAL_LIMIT_REACHED",)))),
            claim_span_in_section=claim_span_in_section,
        )
    eligible=[dict(x) for x in context.get("eligible_evidence",()) if x.get("authorized_for_section") and x.get("usage_role") in {"SUPPORT","NUMERIC","ATTRIBUTION"}]
    if not eligible:
        return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,"DEFER_TO_MANUAL_REVIEW","DEFERRED",decision_path,policy,issues=("AUTHORIZED_CORRECTION_EVIDENCE_UNAVAILABLE",),claim_span_in_section=claim_span_in_section)
    if llm is None:
        return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,"DEFER_TO_MANUAL_REVIEW","DEFERRED",decision_path,policy,issues=("CORRECTION_LLM_UNAVAILABLE",),claim_span_in_section=claim_span_in_section)
    from src.tools.verification.prompting import build_correction_messages, parse_correction_response
    raw_attempts=[]; previous_errors=[]; parsed=None
    metrics={"llm_calls":0,"format_attempts":0,"format_retries":0,"schema_validation_attempts":0,"schema_retries":0,"total_response_retries":0}
    for attempt in range(1,policy["max_correction_llm_attempts"]+1):
        messages=build_correction_messages(context,eligible_evidence=eligible,previous_errors=previous_errors)
        try:
            metrics["llm_calls"]+=1
            raw=llm.invoke(messages)
        except Exception as exc:
            raw_attempts.append({"attempt_number":attempt,"tool_name":"CORRECTION_LLM","parse_status":"INVOCATION_FAILED","exception_type":type(exc).__name__,"exception_message_hash":sha256(str(exc).encode()).hexdigest(),"validation_errors":("CORRECTION_LLM_INVOCATION_FAILED",)})
            return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,"DEFER_TO_MANUAL_REVIEW","DEFERRED",decision_path,policy,issues=("CORRECTION_LLM_INVOCATION_FAILED",),raw_attempts=raw_attempts,retry_metrics=metrics,claim_span_in_section=claim_span_in_section)
        metrics["format_attempts"]+=1
        try:
            decoded=parse_correction_response(raw)
        except ValueError as exc:
            metrics["format_retries"]+=1; metrics["total_response_retries"]+=1
            previous_errors=[str(exc)]; raw_attempts.append({"attempt_number":attempt,"raw_text":raw,"parse_status":"FORMAT_INVALID","validation_errors":tuple(previous_errors)})
            if metrics["format_retries"] >= policy["max_correction_format_repair_attempts"] + 1: break
            continue
        metrics["schema_validation_attempts"]+=1
        try:
            parsed=validate_correction_response(decoded,allowed_evidence_ids=[x["evidence_id"] for x in eligible], expected_claim_id=claim_id)
            raw_attempts.append({"attempt_number":attempt,"raw_text":raw,"parse_status":"VALID","validation_errors":()})
            break
        except ValueError as exc:
            metrics["schema_retries"]+=1; metrics["total_response_retries"]+=1
            previous_errors=[str(exc)]; raw_attempts.append({"attempt_number":attempt,"raw_text":raw,"parse_status":"SCHEMA_INVALID","validation_errors":tuple(previous_errors)})
    if parsed is None:
        # Fallo real de validación (formato/esquema agotó sus reintentos) --
        # SÍ es un fallo técnico: validation_issue_codes se llena con los
        # errores reales acumulados (nunca inventados) además de la causa
        # semántica en reason_codes.
        real_validation_errors=tuple(sorted(set(
            err for attempt in raw_attempts for err in attempt.get("validation_errors",())
        )))
        return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,"DEFER_TO_MANUAL_REVIEW","DEFERRED",decision_path+["CORRECTION_RESPONSE_VALIDATION_EXHAUSTED"],policy,issues=real_validation_errors,reason_codes=("CORRECTION_RESPONSE_VALIDATION_EXHAUSTED",),raw_attempts=raw_attempts,retry_metrics=metrics,claim_span_in_section=claim_span_in_section)
    if parsed["correction_decision"]!="PROPOSE_CHANGE":
        status="DEFERRED" if parsed["correction_decision"]=="DEFER_TO_MANUAL_REVIEW" else "NOT_PROPOSED"
        # El propio LLM ya declaró SU causa real (reason_codes, validado
        # contra CORRECTION_REASON_CODES en validate_correction_response) --
        # se reutiliza tal cual, nunca se inventa una nueva. Solo si el LLM
        # legítimamente no reportó ninguna (lista vacía) se aplica el
        # respaldo semántico determinista de la decisión, para nunca dejar
        # esta decisión terminal sin causa.
        llm_reason_codes=tuple(parsed.get("reason_codes") or ())
        fallback_cause={
            "NO_CORRECTION": "NO_AUTOMATIC_CORRECTION_REQUIRED",
            "DEFER_TO_MANUAL_REVIEW": "VERIFIER_REQUIRES_MANUAL_REVIEW",
            "NOT_CORRECTABLE": "CLAIM_NOT_AUTOCORRECTABLE",
        }[parsed["correction_decision"]]
        return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,parsed["correction_decision"],status,decision_path+[f"LLM_DECISION_{parsed['correction_decision']}"],policy,reason_codes=llm_reason_codes or (fallback_cause,),raw_attempts=raw_attempts,retry_metrics=metrics,claim_span_in_section=claim_span_in_section)
    action=parsed["action_type"]
    if action=="SPLIT_CLAIM":
        return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,"DEFER_TO_MANUAL_REVIEW","DEFERRED",decision_path,policy,issues=("SPLIT_CLAIM_MANUAL_REVIEW_ONLY",),raw_attempts=raw_attempts,retry_metrics=metrics,claim_span_in_section=claim_span_in_section)
    try:
        span,method=locate_target(original,parsed["target_text"],claim_fingerprint=claim_fp,explicit_span=context.get("target_span_in_claim"))
    except ValueError as exc:
        return _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,"DEFER_TO_MANUAL_REVIEW","REJECTED",decision_path,policy,issues=(str(exc),),raw_attempts=raw_attempts,retry_metrics=metrics,claim_span_in_section=claim_span_in_section)
    span_dict=asdict(span)
    conflict_issues, prior_warnings=preproposal_conflict_check(context,span_dict)
    issues=list(conflict_issues); warnings=list(prior_warnings)
    replacement=parsed["replacement_text"]
    if len(replacement)>policy["max_replacement_chars"] or len(span.text)>policy["max_target_span_chars"]: issues.append("REPLACEMENT_SCOPE_EXCEEDED")
    if action=="REMOVE_UNSUPPORTED_FRAGMENT" and _removal_semantics_unsafe(original,span.text): issues.append("REMOVAL_ALTERS_SUPPORTED_MEANING")
    proposed=apply_localized_change(original,span_dict,replacement)
    issues.extend(validate_text_integrity(proposed))
    issues.extend(validate_structural_novelty(parsed,eligible))
    allowed_pairs={(x.get("source_filename"),x.get("chunk_id")) for x in eligible}
    if any((x["source_filename"],x["chunk_id"]) not in allowed_pairs for x in parsed["new_citation_refs"]): issues.append("UNAUTHORIZED_NEW_CITATION")
    evidence_texts=[canonical_correction_evidence_text(x) for x in eligible]
    for val,unit in parsed["new_numeric_pairs"]:
        if not any(quantitative_pair_supported(text,(val,unit)) for text in evidence_texts): issues.append("UNSUPPORTED_NEW_NUMERIC_VALUE")
    if parsed["metric_context"] and not any(metric_context_supported(text,parsed["metric_context"]) for text in evidence_texts): issues.append("NUMERIC_CONTEXT_MISMATCH")
    if action in {"ADD_QUALIFICATION", "NARROW_SCOPE"} and parsed["new_conditions"]:
        if "UNSUPPORTED_NEW_INFORMATION" not in issues: parsed["reason_codes"]=tuple(sorted(set(parsed["reason_codes"]+("SUPPORTED_NEW_QUALIFICATION",))))
    if action=="CORRECT_ATTRIBUTION":
        if parsed["attribution_relation"] not in ATTRIBUTION_RELATIONS: issues.append("ATTRIBUTION_RELATION_INVALID")
        corpus_text=" ".join(evidence_texts)
        required_attr=tuple(parsed["new_attributions"])+tuple(parsed["new_attribution_elements"])
        if any(not _contains_delimited(corpus_text,x) for x in required_attr): issues.append("UNSUPPORTED_NEW_ATTRIBUTION")
    citation_span_dict=None
    if action=="REPLACE_CITATION":
        try:
            citation_span_dict=asdict(_validate_citation_span(original,parsed["citation_text_span"],parsed["old_citation_refs"]))
        except ValueError as exc:
            issues.append(str(exc))
    proposal_fp=compute_correction_proposal_fingerprint(original_claim_fingerprint=claim_fp, original_section_fingerprint=section_fp, target_text_fingerprint=fingerprint_text(span.text), claim_id=claim_id, action_type=action, target_span=span_dict, replacement_text=replacement, evidence_ids=parsed["evidence_ids"], prompt_version=policy["correction_user_prompt_version"])
    cid=f"C07_{claim_id}_{proposal_fp[:12]}"
    accepted=not issues
    status="ACCEPTED_FOR_REVERIFICATION" if accepted else "REJECTED"
    decision_path += ["LOCALIZATION_CONFIRMED","AUTHORIZED_EVIDENCE_SELECTED","CORRECTION_LLM_INVOKED","DETERMINISTIC_VALIDATION_"+("PASSED" if accepted else "FAILED")]
    return CorrectionProposal(cid,claim_id,section_id,"PROPOSE_CHANGE",action,status,original,claim_fp,section_fp,
        dict(claim_span_in_section) if claim_span_in_section is not None else None,span_dict,method,span.text,fingerprint_text(span.text),replacement,proposed,parsed["evidence_ids"],parsed["reason_codes"],
        parsed["change_scope"],parsed["semantic_change_level"],parsed["old_citation_refs"],parsed["new_citation_refs"],
        citation_span_dict,parsed["old_numeric_pairs"],parsed["new_numeric_pairs"],
        parsed["metric_context"],parsed["unit_context"],parsed["old_attribution_elements"],parsed["new_attribution_elements"],
        parsed["attribution_relation"],parsed["new_entities"],parsed["new_citation_refs"],parsed["new_attributions"],
        parsed["new_conditions"],parsed["new_technical_terms"],parsed["llm_correction_recommendation"],not accepted,
        accepted,False,status,proposal_fp,policy["correction_user_prompt_version"],tuple(sorted(set(issues+warnings))),tuple(decision_path),tuple(raw_attempts),metrics)

# Decisiones terminales que SIEMPRE deben tener al menos un reason_code
# determinista y semántico -- nunca reason_codes=() vacío, sin importar
# la rama que las produzca. Verificado como invariante estructural en
# _empty_proposal, no confiado a que cada llamador lo recuerde.
_TERMINAL_DECISIONS_REQUIRING_REASON_CODE = frozenset({"NO_CORRECTION", "DEFER_TO_MANUAL_REVIEW", "NOT_CORRECTABLE"})


def _empty_proposal(claim_id,section_id,original,claim_fp,section_fp,decision,status,path,policy,issues=(),reason_codes=None,raw_attempts=(),retry_metrics=None,claim_span_in_section=None):
    # reason_codes es la CAUSA (siempre poblada para las 3 decisiones
    # terminales) -- distinta de issues/validation_issue_codes (solo se
    # llena cuando hubo un fallo de validación real). Si el llamador no
    # pasa reason_codes explícitamente, se reusa issues (compatibilidad
    # con las ramas que sí representan un fallo técnico real, donde
    # ambos campos deben coincidir: evidencia no disponible, LLM no
    # disponible, límite de propuestas alcanzado, etc.).
    if reason_codes is None:
        reason_codes = issues
    reason_codes = tuple(reason_codes)
    if decision in _TERMINAL_DECISIONS_REQUIRING_REASON_CODE and not reason_codes:
        # Nunca se llega aquí en la práctica -- cada rama que produce
        # una de estas 3 decisiones pasa su propio reason_codes real,
        # nunca inventado a posteriori. Esto es un guardrail estructural
        # para que un futuro llamador que lo olvide falle cerrado en
        # vez de persistir una causa vacía en silencio.
        raise ValueError(f"CORRECTION_PROPOSAL_TERMINAL_REASON_CODE_REQUIRED:{decision}")
    fp=compute_empty_correction_proposal_fingerprint(
        claim_id=claim_id, decision=decision, status=status,
        prompt_version=policy["correction_user_prompt_version"],
    )
    metrics=retry_metrics or {"llm_calls":0,"format_attempts":0,"format_retries":0,"schema_validation_attempts":0,"schema_retries":0,"total_response_retries":0}
    return CorrectionProposal("C07_"+claim_id+"_"+fp[:12],claim_id,section_id,decision,None,status,original,claim_fp,section_fp,
        dict(claim_span_in_section) if claim_span_in_section is not None else None,None,None,"",fingerprint_text(""),"",original,(),reason_codes,"NONE","NONE",(),(),None,(),(),"","",(),(),None,(),(),(),(),(),False,status in {"DEFERRED","REJECTED"},False,False,status,fp,policy["correction_user_prompt_version"],tuple(issues),tuple(path),tuple(raw_attempts),metrics)


# Phase 6.2: construcción virtual estrictamente en memoria.
def build_virtual_corrected_claim(
    original_claim_text: str,
    target_span_in_claim: Mapping[str, Any],
    replacement_text: str,
) -> str:
    """Construye el claim propuesto sin modificar archivos ni estado."""
    if not isinstance(original_claim_text, str) or not isinstance(replacement_text, str):
        raise ValueError("VIRTUAL_CLAIM_INPUT_INVALID")
    if not isinstance(target_span_in_claim, Mapping):
        raise ValueError("TARGET_SPAN_NOT_FOUND")
    if target_span_in_claim.get("coordinate_base") != "CLAIM_TEXT":
        raise ValueError("SPAN_COORDINATE_BASE_INVALID")
    if target_span_in_claim.get("coordinate_system") != "PYTHON_CODEPOINT_OFFSETS":
        raise ValueError("SPAN_COORDINATE_SYSTEM_INVALID")
    start, end = target_span_in_claim.get("start"), target_span_in_claim.get("end")
    if type(start) is not int or type(end) is not int or not (0 <= start < end <= len(original_claim_text)):
        raise ValueError("TARGET_SPAN_NOT_FOUND")
    if original_claim_text[start:end] != target_span_in_claim.get("text"):
        raise ValueError("TARGET_TEXT_MISMATCH")
    return original_claim_text[:start] + replacement_text + original_claim_text[end:]
