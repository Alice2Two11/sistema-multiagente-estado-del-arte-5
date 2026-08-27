"""Precondiciones contractuales y validaciones deterministas del Agente 07.

No modifica PipelineState. La evaluación de evidencia heredada se delega en
`evidence.assess_inherited_evidence`; este módulo conserva el contrato estricto
de entrada y expone validadores de schema/resultados provisionales.
"""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from src.config.verification_policy_config import (
    CLAIM_TYPES,
    RESOLUTION_ISSUE_CODES,
    RESOLUTION_STATUSES,
    RETRIEVAL_REASON_CODES,
    TEXT_MATCH_STATUSES,
    VERIFICATION_INTENSITIES,
    RETRIEVAL_TECHNICAL_STATUSES,
    CONTRADICTION_SIGNAL_CODES,
    get_verification_input_policy,
)
try:
    from src.contracts.agent_input import ArtifactReference
    from src.contracts.agent_result import ExecutionStatus, TransitionAction
    from src.state.pipeline_state import DecisionLogEntry, PipelineState
    from .evidence import provisional_evidence_schema, resolve_committed_artifact_reference
except ModuleNotFoundError:  # paquete mínimo auditable sin runtime contractual
    ArtifactReference = Any  # type: ignore
    DecisionLogEntry = Any  # type: ignore
    PipelineState = Any  # type: ignore
    class ExecutionStatus:
        COMPLETED = "COMPLETED"
    class TransitionAction:
        ADVANCE = "ADVANCE"
    def provisional_evidence_schema(): return ()
    def resolve_committed_artifact_reference(*args, **kwargs):
        raise RuntimeError("CONTRACT_RUNTIME_NOT_INCLUDED_IN_MINIMAL_PACKAGE")




from .commit_handoff_validation import (
    CommittedAgent06Input,
    _is_within,
    _latest_stage_decision,
    _require_result_field,
    _validate_committed_result,
    _validate_manifest,
    sha256_file,
    validate_committed_agent06_input,
    validate_provisional_evidence_output,
    validate_sha256_hex,
)  # Bloque C, C1 -- extraído mecánicamente a commit_handoff_validation.py; reexportado aquí para preservar el contrato de import existente.


# ---------------- Phase 4: juicio científico por claim ----------------
from dataclasses import dataclass as _dataclass
from typing import Sequence as _Sequence

from src.config.verification_policy_config import (
    ATTRIBUTION_ASSESSMENTS,
    CLAIM_EXECUTION_STATUSES,
    CLAIM_TECHNICAL_STATUSES,
    CONTRADICTION_TYPES,
    CORRECTION_ELIGIBILITIES,
    EXTRAPOLATION_ASSESSMENTS,
    HALLUCINATION_RISKS,
    NUMERIC_ASSESSMENTS,
    SCIENTIFIC_JUDGMENT_STATUSES,
    SCIENTIFIC_VERDICTS,
    SUPPORT_LEVELS,
    SUPPORT_LEVEL_BY_VERDICT,
    CLAIM_TYPES,
    VERIFICATION_INTENSITIES,
    SEMANTIC_REASON_CODES,
    ADDITIONAL_RETRIEVAL_REASON_CODES,
    TECHNICAL_ISSUE_CODES,
    get_verification_input_policy as _get_phase4_policy,
)


from .evidence_selection import (
    ClaimRetrievalTool,
    EvidenceSelection,
    _REQUIRED_LLM_FIELDS,
    _canonical_evidence_rows,
    allowed_verdicts_for_claim,
    canonical_correction_evidence_text,
    deterministic_precheck,
    select_evidence_for_scientific_judgment,
    validate_claim_verification_context,
)  # Bloque C, C2 -- extraído mecánicamente a evidence_selection.py; reexportado aquí para preservar el contrato de import existente.



from .llm_response_validation import (
    _require_exact_type,
    _require_string_list,
    compute_hallucination_risk,
    derive_semantic_issue_codes,
    determine_final_correction_eligibility,
    validate_llm_verification_response,
)  # Bloque C, C3 -- extraído mecánicamente a llm_response_validation.py; reexportado aquí para preservar el contrato de import existente.



# Fase 4T/4U: contrato estricto del delta incremental de retrieval.
from .retrieval_delta_validation import (
    ADDITIONAL_RETRIEVAL_CANDIDATE_ALLOWED_FIELDS,
    ADDITIONAL_RETRIEVAL_COVERAGE_FIELDS,
    ADDITIONAL_RETRIEVAL_DELTA_ACCUMULATIVE_FIELDS,
    ADDITIONAL_RETRIEVAL_DELTA_ALLOWED_FIELDS,
    ADDITIONAL_RETRIEVAL_DELTA_DERIVED_FIELDS,
    ADDITIONAL_RETRIEVAL_DELTA_SNAPSHOT_FIELDS,
    ADDITIONAL_RETRIEVAL_DELTA_UNION_FIELDS,
    ADDITIONAL_RETRIEVAL_MUTABLE_DETERMINISTIC_FIELDS,
    ADDITIONAL_RETRIEVAL_PROTECTED_CANDIDATE_FIELDS,
    ADDITIONAL_RETRIEVAL_STOP_REASONS,
    _delta_string_sequence,
    _validate_coverage_snapshot,
    _validate_incremental_candidate,
    validate_additional_retrieval_delta,
)  # Bloque C, C4 -- extraído mecánicamente a retrieval_delta_validation.py; reexportado aquí para preservar el contrato de import existente.



# Phase 5 correction validation facade.
from .numeric_validation import (
    _normalize_decimal_literal,
    extract_quantitative_pairs_strict,
    metric_context_supported,
    quantitative_pair_supported,
    validate_correction_proposal_response,
    validate_correction_text_integrity,
)  # Bloque C, C5 -- extraído mecánicamente a numeric_validation.py; reexportado aquí para preservar el contrato de import existente.


# Phase 6.1: contratos de reverificación virtual independiente previa a aplicación.
from src.config.verification_policy_config import (
    REVERIFICATION_ACCEPTANCE_DECISIONS,
    REVERIFICATION_EXECUTION_STATUSES,
    REVERIFICATION_PROCESS_NAME,
    REVERIFICATION_REASON_CODES,
    REVERIFICATION_RISK_DELTAS,
    REVERIFICATION_RISK_POLICY_VERSION,
    REVERIFICATION_SCIENTIFIC_OUTCOMES,
    REVERIFICATION_TECHNICAL_ISSUE_CODES,
)


from .reverification_contracts import (
    PRECHECK_STATUSES,
    _phase62_brackets_balanced,
    _reverification_nonempty_string,
    _reverification_string_tuple,
    _sha256_text,
    validate_correction_reverification_input_contract,
    validate_correction_reverification_result_contract,
    validate_reverification_block_matrix,
)  # Bloque C, C6 -- extraído mecánicamente a reverification_contracts.py; reexportado aquí para preservar el contrato de import existente.





def _closed_contract_reason(exc: Exception) -> tuple[str, Mapping[str, Any]]:
    raw=str(exc).strip()
    known_prefixes=(
        "REVERIFICATION_", "TARGET_ISSUE_", "BASE_", "PROPOSED_",
        "ORIGINAL_", "SECTION_", "CLAIM_", "APPLICATION_",
    )
    if raw and raw.startswith(known_prefixes) and " " not in raw:
        return raw, {"exception_type": type(exc).__name__, "normalized": True}
    return "REVERIFICATION_INPUT_CONTRACT_INVALID", {
        "exception_type": type(exc).__name__,
        "message_fingerprint": _sha256_text(raw),
        "normalized": False,
    }


def _frozen_evidence_rows(context: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in (context.get("authorized_evidence") or ()) if isinstance(row, Mapping))


def _frozen_evidence_texts(context: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        text for text in (canonical_correction_evidence_text(row) for row in _frozen_evidence_rows(context))
        if text
    )


def _contains_delimited(text: str, value: str) -> bool:
    return bool(value.strip()) and re.search(
        r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)", text, re.I | re.UNICODE
    ) is not None


def _tokenize_significant(text: str) -> tuple[str, ...]:
    stop={"el","la","los","las","un","una","de","del","y","o","en","para","por","con","que","se","es","al","a"}
    return tuple(t for t in re.findall(r"[\wáéíóúüñ]+", text.casefold(), re.UNICODE) if len(t)>2 and t not in stop)


def _validate_phase62_evidence(context: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    ids=tuple(context.get("evidence_ids") or ())
    rows=_frozen_evidence_rows(context)
    canonical_order=tuple(str(row.get("evidence_id", "")).strip() for row in rows)
    issues=[]
    if not ids or not rows:
        issues.append("REVERIFICATION_EVIDENCE_REQUIRED")
    if ids != canonical_order:
        if set(ids)==set(canonical_order):
            issues.append("REVERIFICATION_EVIDENCE_ORDER_MISMATCH")
        else:
            issues.append("REVERIFICATION_EVIDENCE_SET_MISMATCH")
    if len(ids)!=len(set(ids)):
        issues.append("REVERIFICATION_EVIDENCE_ID_DUPLICATE")
    for eid,row in zip(canonical_order, rows):
        if not eid:
            issues.append("REVERIFICATION_EVIDENCE_ID_MISSING")
        if not str(row.get("source_filename","")).strip() or not str(row.get("chunk_id","")).strip():
            issues.append("DOCUMENT_IDENTITY_INVALID")
        if row.get("authorized_for_section") is not True:
            issues.append("REVERIFICATION_EVIDENCE_NOT_AUTHORIZED")
        role=str(row.get("usage_role","")).upper()
        if row.get("outside_section_sources") is True or role in {"CORPUS_WIDE","CONTRAST","DISCARDED"}:
            issues.append("REVERIFICATION_CORPUS_WIDE_EVIDENCE_FORBIDDEN")
        if not canonical_correction_evidence_text(row):
            issues.append("REVERIFICATION_EVIDENCE_TEXT_MISSING")
    return (not issues, tuple(dict.fromkeys(issues)))


def _validate_action_target_issue(context: Mapping[str, Any]) -> tuple[str, ...]:
    action=str(context.get("correction_action_type", ""))
    targets=set(context.get("target_issue_codes") or ())
    matrix=dict(context.get("policy", {}).get("reverification_action_target_issue_matrix") or {})
    allowed=set(matrix.get(action) or ())
    return () if targets and targets.issubset(allowed) else ("ACTION_TARGET_ISSUE_MISMATCH",)


def _relation_terms(relation: str) -> tuple[str, ...]:
    return {
        "PROPOSED_BY": ("proposed by", "propuesto por", "propuso"),
        "DEVELOPED_BY": ("developed by", "desarrollado por", "desarrolló"),
        "INTRODUCED_IN": ("introduced in", "introducido en", "presentado en"),
        "REPORTED_BY": ("reported by", "reportado por", "informó"),
        "EVALUATED_BY": ("evaluated by", "evaluado por", "evaluó"),
    }.get(relation, ())


def _citation_markers(ref: Mapping[str, Any]) -> tuple[str, ...]:
    source=str(ref.get("source_filename", "")).strip()
    chunk=str(ref.get("chunk_id", "")).strip()
    markers=[]
    if source:
        base=source.replace("\\", "/").rsplit("/",1)[-1]
        stem=base.rsplit(".",1)[0]
        markers.extend((source,base,stem))
    if chunk:
        markers.append(chunk)
    return tuple(m for m in markers if m)


def _refs_linked(text: str, refs: tuple[Mapping[str, Any], ...]) -> bool:
    return bool(refs) and all(any(marker in text for marker in _citation_markers(ref)) for ref in refs)


def _evidence_supports_proposed_claim(context: Mapping[str, Any], refs: tuple[Mapping[str, Any], ...], virtual_text: str) -> bool:
    rows=_frozen_evidence_rows(context)
    by_pair={(str(r.get("source_filename","")),str(r.get("chunk_id",""))):r for r in rows}
    replacement=str(context.get("replacement_text", ""))
    tokens=set(_tokenize_significant(replacement)) or set(_tokenize_significant(virtual_text))
    for ref in refs:
        row=by_pair.get((str(ref.get("source_filename","")),str(ref.get("chunk_id",""))))
        if not row:
            return False
        evidence_tokens=set(_tokenize_significant(canonical_correction_evidence_text(row)))
        if tokens and not (tokens & evidence_tokens):
            return False
    return True


def _scope_conditions(text: str) -> tuple[str, ...]:
    patterns=(
        r"\bsolo\b[^,.;]*", r"\búnicamente\b[^,.;]*", r"\bsolamente\b[^,.;]*",
        r"\ben (?:el|la|los|las)\b[^,.;]*", r"\bbajo\b[^,.;]*",
        r"\bpara (?:el|la|los|las)\b[^,.;]*", r"\bdurante\b[^,.;]*", r"\bcuando\b[^,.;]*",
    )
    out=[]
    for pattern in patterns:
        out.extend(m.group(0).strip() for m in re.finditer(pattern,text,re.I|re.UNICODE))
    return tuple(dict.fromkeys(out))


def _validate_narrow_scope(context: Mapping[str, Any], virtual_text: str, corpus: str) -> tuple[str, ...]:
    cv=dict(context.get("correction_validation_result") or {})
    original=str(context.get("original_claim_text", ""))
    new_conditions=tuple(str(x).strip() for x in cv.get("new_conditions", ()) if str(x).strip())
    issues=[]
    if not new_conditions:
        issues.append("NARROW_SCOPE_CONDITIONS_REQUIRED")
    for condition in new_conditions:
        if not _contains_delimited(virtual_text, condition) or _contains_delimited(original, condition):
            issues.append("SCOPE_NOT_NARROWED")
        if not _contains_delimited(corpus, condition):
            issues.append("UNSUPPORTED_NEW_INFORMATION")
    expansive=("todos","todas","siempre","cualquier","cualquiera","globalmente","sin restricción","sin restricciones","en general")
    if any(_contains_delimited(virtual_text, term) and not _contains_delimited(original,term) for term in expansive):
        issues.append("SCOPE_EXPANSION_DETECTED")
    original_conditions=tuple(str(x).strip() for x in cv.get("old_conditions", ()) if str(x).strip()) or _scope_conditions(original)
    if any(_contains_delimited(original,c) and not _contains_delimited(virtual_text,c) for c in original_conditions):
        issues.append("ORIGINAL_SCOPE_CONDITION_REMOVED")
    if len(re.findall(r"[.!?]", virtual_text)) > len(re.findall(r"[.!?]", original)):
        issues.append("SCOPE_NOT_NARROWED")
    return tuple(dict.fromkeys(issues))


def _validate_add_qualification(context: Mapping[str, Any], virtual_text: str, corpus: str) -> tuple[str, ...]:
    cv=dict(context.get("correction_validation_result") or {})
    original=str(context.get("original_claim_text", ""))
    added=tuple(str(x).strip() for x in cv.get("new_conditions", ()) if str(x).strip())
    issues=[]
    if not added:
        issues.append("QUALIFICATION_CONDITIONS_REQUIRED")
    generic_qualifiers={"en","para","por","con","de","a","el","la","los","las"}
    for qualifier in added:
        if qualifier.casefold() in generic_qualifiers or not _tokenize_significant(qualifier):
            issues.append("QUALIFICATION_DIFFERENTIAL_INVALID")
        if not _contains_delimited(virtual_text,qualifier) or _contains_delimited(original,qualifier):
            issues.append("QUALIFICATION_DIFFERENTIAL_INVALID")
        if not _contains_delimited(corpus,qualifier):
            issues.append("UNSUPPORTED_NEW_INFORMATION")
    uncertainty=("puede","podría","sugiere","aproximadamente","posiblemente","probablemente","según")
    certainty=("demuestra","garantiza","siempre","definitivamente","sin duda","prueba")
    if any(_contains_delimited(virtual_text,t) and not _contains_delimited(original,t) for t in certainty):
        issues.append("QUALIFICATION_INCREASES_CERTAINTY")
    if any(_contains_delimited(original,t) and not _contains_delimited(virtual_text,t) for t in uncertainty):
        issues.append("QUALIFICATION_INCREASES_CERTAINTY")
    base_tokens=set(_tokenize_significant(original)) - set(sum((_tokenize_significant(q) for q in added),()))
    proposed_tokens=set(_tokenize_significant(virtual_text))
    if base_tokens and len(base_tokens & proposed_tokens)/len(base_tokens) < .75:
        issues.append("SUPPORTED_MEANING_NOT_PRESERVED")
    return tuple(dict.fromkeys(issues))


def _validate_remove_fragment(context: Mapping[str, Any], virtual_text: str) -> tuple[str, ...]:
    original=str(context.get("original_claim_text", ""))
    target=dict(context.get("target_span_in_claim") or {})
    target_text=str(target.get("text", ""))
    cv=dict(context.get("correction_validation_result") or {})
    issues=[]
    if not virtual_text.strip():
        issues.append("EMPTY_PROPOSED_CLAIM")
    if cv.get("unsupported_fragment") not in (None,"",target_text):
        issues.append("REMOVAL_TARGET_HALLUCINATION_MISMATCH")
    protected=(
        "no","nunca","sin","mayor","menor","supera","porque","debido","causa",
        "antes","después","durante","siempre","actualmente","previamente","posteriormente",
    )
    for token in protected:
        if _contains_delimited(target_text,token) or (_contains_delimited(original,token) and not _contains_delimited(virtual_text,token)):
            issues.append("REMOVAL_ALTERS_SUPPORTED_MEANING")
            break
    return tuple(dict.fromkeys(issues))


def _validate_phase62_action(context: Mapping[str, Any], virtual_text: str) -> tuple[bool, tuple[str, ...]]:
    action=str(context.get("correction_action_type", ""))
    cv=dict(context.get("correction_validation_result") or {})
    texts=_frozen_evidence_texts(context)
    corpus="\n".join(texts)
    issues=list(_validate_action_target_issue(context))
    if action == "REPLACE_NUMERIC_VALUE":
        pairs=tuple(tuple(x) for x in cv.get("new_numeric_pairs", ()) if isinstance(x,(list,tuple)) and len(x)==2)
        if not pairs: issues.append("NUMERIC_PAIRS_REQUIRED")
        for pair in pairs:
            if not any(quantitative_pair_supported(t,pair) for t in texts): issues.append("UNSUPPORTED_NEW_NUMERIC_VALUE")
        metric=str(cv.get("metric_context","")).strip()
        if metric and not any(metric_context_supported(t,metric) for t in texts): issues.append("NUMERIC_CONTEXT_MISMATCH")
        declared={(_normalize_decimal_literal(a),str(b).strip().casefold()) for a,b in pairs}
        introduced=set(extract_quantitative_pairs_strict(virtual_text))-set(extract_quantitative_pairs_strict(str(context.get("original_claim_text",""))))
        if any(pair not in declared for pair in introduced): issues.append("UNDECLARED_NEW_NUMERIC_VALUE")
    elif action == "CORRECT_ATTRIBUTION":
        elements=tuple(str(x).strip() for x in cv.get("new_attribution_elements",()) if str(x).strip())
        relation=str(cv.get("attribution_relation","")).strip()
        subject=str(cv.get("attribution_subject", elements[0] if elements else "")).strip()
        obj=str(cv.get("attribution_object", elements[1] if len(elements)>1 else "")).strip()
        if not subject or not relation or not obj: issues.append("ATTRIBUTION_FIELDS_REQUIRED")
        if subject and not _contains_delimited(corpus,subject): issues.append("UNSUPPORTED_NEW_ATTRIBUTION")
        if relation and not any(_contains_delimited(corpus,term) for term in _relation_terms(relation)):
            issues.append("ATTRIBUTION_RELATION_NOT_SUPPORTED")
        if obj and not _contains_delimited(corpus,obj): issues.append("ATTRIBUTION_OBJECT_NOT_SUPPORTED")
    elif action == "REPLACE_CITATION":
        span=cv.get("citation_text_span")
        original=str(context.get("original_claim_text", ""))
        if not isinstance(span,Mapping): issues.append("CITATION_TEXT_SPAN_REQUIRED")
        else:
            st,en=span.get("start"),span.get("end")
            if type(st) is not int or type(en) is not int or not (0<=st<en<=len(original)) or original[st:en]!=span.get("text"):
                issues.append("CITATION_TEXT_SPAN_STALE")
        old_refs=tuple(r for r in cv.get("old_citation_refs",()) if isinstance(r,Mapping))
        new_refs=tuple(r for r in cv.get("new_citation_refs",()) if isinstance(r,Mapping))
        if not isinstance(span,Mapping) or not _refs_linked(str(span.get("text","")),old_refs):
            issues.append("CITATION_TEXT_REFERENCE_MISMATCH")
        if not _refs_linked(virtual_text,new_refs):
            issues.append("NEW_CITATION_MARKER_MISSING")
        allowed={(str(r.get("source_filename","")),str(r.get("chunk_id",""))) for r in _frozen_evidence_rows(context)}
        if any((str(r.get("source_filename","")),str(r.get("chunk_id",""))) not in allowed for r in new_refs):
            issues.append("NEW_CITATION_REFERENCE_MISMATCH")
        if new_refs and not _evidence_supports_proposed_claim(context,new_refs,virtual_text):
            issues.append("NEW_CITATION_DOES_NOT_SUPPORT_PROPOSED_CLAIM")
    elif action == "NARROW_SCOPE":
        issues.extend(_validate_narrow_scope(context,virtual_text,corpus))
    elif action == "ADD_QUALIFICATION":
        issues.extend(_validate_add_qualification(context,virtual_text,corpus))
    elif action == "REMOVE_UNSUPPORTED_FRAGMENT":
        issues.extend(_validate_remove_fragment(context,virtual_text))
    else:
        issues.append("REVERIFICATION_ACTION_UNSUPPORTED")
    result=tuple(dict.fromkeys(issues))
    return (not result,result)


def _recompute_phase5t_proposal_fingerprint(context: Mapping[str, Any]) -> str:
    from src.tools.verification.corrections import compute_correction_proposal_fingerprint, fingerprint_text
    cv=dict(context.get("correction_validation_result") or {})
    prompt_version=str(cv.get("prompt_version") or context.get("policy",{}).get("correction_user_prompt_version") or "")
    return compute_correction_proposal_fingerprint(
        original_claim_fingerprint=str(context.get("original_claim_fingerprint","")),
        original_section_fingerprint=str(context.get("original_section_fingerprint","")),
        target_text_fingerprint=fingerprint_text(str(context.get("target_span_in_claim",{}).get("text",""))),
        claim_id=str(context.get("claim_id","")),
        action_type=str(context.get("correction_action_type","")),
        target_span=dict(context.get("target_span_in_claim") or {}),
        replacement_text=str(context.get("replacement_text","")),
        evidence_ids=tuple(context.get("evidence_ids") or ()),
        prompt_version=prompt_version,
    )


def run_virtual_reverification_prechecks(context: Mapping[str, Any]) -> dict[str, Any]:
    """Fase 6.2R: solo construcción virtual y prechecks deterministas."""
    try:
        value=validate_correction_reverification_input_contract(context)
    except Exception as exc:
        reason,detail=_closed_contract_reason(exc)
        return _precheck_result(context,status="PRECHECK_BLOCKED",reasons=(reason,),diagnostics=(detail,))
    policy=value["policy"]
    for key in ("require_frozen_reverification_evidence","require_same_risk_policy_version","require_virtual_proposed_claim_reconstruction"):
        if policy.get(key) is not True:
            return _precheck_result(value,status="PRECHECK_BLOCKED",reasons=("REVERIFICATION_POLICY_INVARIANT_VIOLATION",),diagnostics=({"policy_key":key},),contract_valid=True)
    section_text=context.get("section_text")
    if not isinstance(section_text,str) or not section_text:
        return _precheck_result(value,status="PRECHECK_BLOCKED",reasons=("SECTION_TEXT_REQUIRED",),contract_valid=True)
    original=value["original_claim_text"]
    claim_fp=_sha256_text(original); section_fp=_sha256_text(section_text)
    if claim_fp!=value["original_claim_fingerprint"] or claim_fp!=value["base_claim_fingerprint"]:
        return _precheck_result(value,status="PRECHECK_BLOCKED",reasons=("ORIGINAL_CLAIM_FINGERPRINT_MISMATCH",),contract_valid=True)
    if section_fp!=value["original_section_fingerprint"] or section_fp!=value["base_section_fingerprint"]:
        return _precheck_result(value,status="PRECHECK_BLOCKED",reasons=("ORIGINAL_SECTION_FINGERPRINT_MISMATCH",),contract_valid=True)
    cspan=value["claim_span_in_section"]; tspan=value["target_span_in_claim"]
    if cspan["end"]>len(section_text) or section_text[cspan["start"]:cspan["end"]]!=cspan["text"] or cspan["text"]!=original:
        return _precheck_result(value,status="PRECHECK_BLOCKED",reasons=("CLAIM_SPAN_TEXT_MISMATCH",),contract_valid=True,base_fingerprints_valid=True)
    if tspan["end"]>len(original) or original[tspan["start"]:tspan["end"]]!=tspan["text"]:
        return _precheck_result(value,status="PRECHECK_BLOCKED",reasons=("TARGET_SPAN_TEXT_MISMATCH",),contract_valid=True,base_fingerprints_valid=True)
    try:
        from src.tools.verification.corrections import build_virtual_corrected_claim
        virtual=build_virtual_corrected_claim(original,tspan,value["replacement_text"])
    except Exception as exc:
        return _precheck_result(value,status="PRECHECK_BLOCKED",reasons=("VIRTUAL_PROPOSED_CLAIM_BUILD_FAILED",),diagnostics=({"exception_type":type(exc).__name__},),contract_valid=True,base_fingerprints_valid=True)
    if virtual!=value["proposed_claim_text"]:
        return _precheck_result(value,status="PRECHECK_REJECTED",virtual_text=virtual,reasons=("PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH",),contract_valid=True,base_fingerprints_valid=True,spans_valid=True)
    proposed_fp_ok=_sha256_text(virtual)==value["proposed_claim_text_fingerprint"]
    if not proposed_fp_ok:
        return _precheck_result(value,status="PRECHECK_BLOCKED",virtual_text=virtual,reasons=("PROPOSED_CLAIM_TEXT_FINGERPRINT_MISMATCH",),contract_valid=True,base_fingerprints_valid=True,spans_valid=True)
    cv=value["correction_validation_result"]
    if cv.get("proposal_status")!="ACCEPTED_FOR_REVERIFICATION" or cv.get("correction_applied") is not False or cv.get("conflict_active") is True or cv.get("invalidated") is True or cv.get("replaced_by_correction_id"):
        return _precheck_result(value,status="PRECHECK_BLOCKED",virtual_text=virtual,reasons=("REVERIFICATION_PROPOSAL_NOT_CURRENT",),contract_valid=True,base_fingerprints_valid=True,proposed_text_fingerprint_valid=True,spans_valid=True)
    try:
        recomputed=_recompute_phase5t_proposal_fingerprint(value)
    except Exception as exc:
        return _precheck_result(value,status="PRECHECK_BLOCKED",virtual_text=virtual,reasons=("PROPOSAL_FINGERPRINT_RECOMPUTATION_FAILED",),diagnostics=({"exception_type":type(exc).__name__},),contract_valid=True,base_fingerprints_valid=True,proposed_text_fingerprint_valid=True,spans_valid=True)
    if recomputed!=value["proposal_fingerprint"]:
        return _precheck_result(value,status="PRECHECK_BLOCKED",virtual_text=virtual,reasons=("PROPOSAL_FINGERPRINT_MISMATCH",),diagnostics=({"recomputed_fingerprint":recomputed},),contract_valid=True,base_fingerprints_valid=True,proposed_text_fingerprint_valid=True,spans_valid=True)
    evidence_ok,evidence_issues=_validate_phase62_evidence(value)
    if not evidence_ok:
        return _precheck_result(value,status="PRECHECK_BLOCKED",virtual_text=virtual,reasons=evidence_issues,contract_valid=True,base_fingerprints_valid=True,proposed_text_fingerprint_valid=True,proposal_fingerprint_valid=True,spans_valid=True)
    if not virtual.strip() or not _phase62_brackets_balanced(virtual) or re.search(r"\s{2,}",virtual):
        return _precheck_result(value,status="PRECHECK_REJECTED",virtual_text=virtual,reasons=("TEXT_INTEGRITY_INVALID",),contract_valid=True,base_fingerprints_valid=True,proposed_text_fingerprint_valid=True,proposal_fingerprint_valid=True,spans_valid=True,evidence_valid=True)
    action_ok,action_issues=_validate_phase62_action(value,virtual)
    if not action_ok:
        return _precheck_result(value,status="PRECHECK_REJECTED",virtual_text=virtual,reasons=action_issues,contract_valid=True,base_fingerprints_valid=True,proposed_text_fingerprint_valid=True,proposal_fingerprint_valid=True,spans_valid=True,evidence_valid=True,textual_integrity_valid=True)
    return _precheck_result(value,status="PRECHECK_PASSED",virtual_text=virtual,contract_valid=True,base_fingerprints_valid=True,proposed_text_fingerprint_valid=True,proposal_fingerprint_valid=True,spans_valid=True,evidence_valid=True,textual_integrity_valid=True,action_validation_valid=True)

# Phase 6.3: reverificación virtual independiente con double.
from typing import Protocol, Sequence

class ReverificationLLM(Protocol):
    def invoke(self, messages: Sequence[Mapping[str, str]]) -> str: ...








def _safe_raw_hash(raw: str) -> str:
    return sha256(str(raw).encode("utf-8")).hexdigest()




# Phase 6.3R: immutable snapshots and scientific coherence.
def _stable_json_fingerprint(payload: Mapping[str, Any]) -> str:
    import json as _json
    return sha256(_json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _selected_evidence_text(row: Mapping[str, Any]) -> str:
    return str(row.get("canonical_text") or row.get("contractual_text") or row.get("text") or "")


def compute_frozen_evidence_snapshot_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = []
    for row in rows:
        payload.append({
            "evidence_id": str(row.get("evidence_id", "")),
            "source_filename": str(row.get("source_filename", "")),
            "chunk_id": str(row.get("chunk_id", "")),
            "authorized_for_section": row.get("authorized_for_section") is True,
            "usage_role": str(row.get("usage_role", "")),
            "text": _selected_evidence_text(row),
        })
    return _stable_json_fingerprint({"evidence": payload})


def compute_reverification_policy_fingerprint(policy: Mapping[str, Any]) -> str:
    keys = (
        "max_reverification_llm_attempts",
        "max_reverification_format_repair_attempts",
        "reverification_confidence_min",
        "reverification_confidence_max",
        "reverification_system_prompt_version",
        "reverification_user_prompt_version",
        "reverification_retrieval_rounds",
        "reverification_risk_policy_version",
        "require_frozen_reverification_evidence",
        "require_same_risk_policy_version",
        "require_virtual_proposed_claim_reconstruction",
        "reverification_observed_issue_codes",
        "reverification_llm_reason_codes",
    )
    return _stable_json_fingerprint({key: policy.get(key) for key in keys})


def compute_reverification_context_fingerprint(context: Mapping[str, Any], *, evidence_snapshot_fingerprint: str, policy_fingerprint: str) -> str:
    policy = context.get("policy") or {}
    payload = {
        "correction_id": str(context.get("correction_id", "")),
        "claim_id": str(context.get("claim_id", "")),
        "section_id": str(context.get("section_id", "")),
        "source_verdict": str(context.get("source_verdict", "")),
        "source_issue_codes": list(context.get("source_issue_codes") or ()),
        "target_issue_codes": list(context.get("target_issue_codes") or ()),
        "correction_action_type": str(context.get("correction_action_type", "")),
        "proposal_fingerprint": str(context.get("proposal_fingerprint", "")),
        "virtual_proposed_claim_text_fingerprint": str(context.get("virtual_proposed_claim_text_fingerprint", context.get("proposed_claim_text_fingerprint", ""))),
        "evidence_snapshot_fingerprint": evidence_snapshot_fingerprint,
        "system_prompt_version": policy.get("reverification_system_prompt_version"),
        "user_prompt_version": policy.get("reverification_user_prompt_version"),
        "risk_policy_version": policy.get("reverification_risk_policy_version"),
        "retrieval_allowed": False,
        "retrieval_rounds": policy.get("reverification_retrieval_rounds"),
        "policy_fingerprint": policy_fingerprint,
    }
    return _stable_json_fingerprint(payload)


# Override precheck result to freeze evidence, policy and full context snapshots.
def _precheck_result(context: Mapping[str, Any], *, status: str, virtual_text: str = "",
                     reasons: tuple[str, ...] = (), technical: tuple[str, ...] = (),
                     diagnostics: tuple[Mapping[str, Any], ...] = (),
                     contract_valid: bool = False,
                     base_fingerprints_valid: bool = False,
                     proposed_text_fingerprint_valid: bool = False,
                     proposal_fingerprint_valid: bool = False,
                     spans_valid: bool = False, evidence_valid: bool = False,
                     textual_integrity_valid: bool = False,
                     action_validation_valid: bool = False) -> dict[str, Any]:
    from src.tools.verification.traceability import CorrectionReverificationPrecheckResult
    rows = tuple(row for row in (context.get("authorized_evidence") or ()) if isinstance(row, Mapping))
    evidence_fp = compute_frozen_evidence_snapshot_fingerprint(rows) if rows else ""
    policy = dict(context.get("policy") or {})
    policy_fp = compute_reverification_policy_fingerprint(policy) if policy else ""
    context_for_fp = dict(context)
    context_for_fp["virtual_proposed_claim_text_fingerprint"] = _sha256_text(virtual_text) if virtual_text else str(context.get("proposed_claim_text_fingerprint", ""))
    context_fp = compute_reverification_context_fingerprint(
        context_for_fp, evidence_snapshot_fingerprint=evidence_fp, policy_fingerprint=policy_fp
    ) if policy else ""
    fingerprints_valid = bool(base_fingerprints_valid and proposed_text_fingerprint_valid and proposal_fingerprint_valid)
    return CorrectionReverificationPrecheckResult(
        correction_id=str(context.get("correction_id", "")), claim_id=str(context.get("claim_id", "")),
        section_id=str(context.get("section_id", "")), virtual_proposed_claim_text=virtual_text,
        precheck_status=status, contract_valid=contract_valid, fingerprints_valid=fingerprints_valid,
        spans_valid=spans_valid, evidence_valid=evidence_valid,
        textual_integrity_valid=textual_integrity_valid, action_validation_valid=action_validation_valid,
        reason_codes=tuple(dict.fromkeys(reasons)), technical_issue_codes=tuple(dict.fromkeys(technical)),
        virtual_proposed_claim_text_fingerprint=_sha256_text(virtual_text) if virtual_text else "",
        proposal_fingerprint=str(context.get("proposal_fingerprint", "")),
        base_claim_fingerprint=str(context.get("base_claim_fingerprint", "")),
        base_section_fingerprint=str(context.get("base_section_fingerprint", "")),
        frozen_evidence_snapshot_fingerprint=evidence_fp,
        reverification_policy_fingerprint=policy_fp,
        reverification_context_fingerprint=context_fp,
        diagnostic_details=tuple(dict(item) for item in diagnostics), llm_calls=0, correction_applied=False,
    ).to_dict()


def build_reverification_claim_context(reverification_input: Mapping[str, Any], precheck_result: Mapping[str, Any]) -> dict[str, Any]:
    if precheck_result.get("precheck_status") != "PRECHECK_PASSED":
        raise ValueError("REVERIFICATION_PRECHECK_NOT_PASSED")
    mismatch_codes = {
        "correction_id": "REVERIFICATION_PRECHECK_CORRECTION_ID_MISMATCH",
        "claim_id": "REVERIFICATION_PRECHECK_CLAIM_ID_MISMATCH",
        "section_id": "REVERIFICATION_PRECHECK_SECTION_ID_MISMATCH",
    }
    for field, code in mismatch_codes.items():
        if str(precheck_result.get(field, "")) != str(reverification_input.get(field, "")):
            raise ValueError(code)
    for field in ("proposal_fingerprint", "base_claim_fingerprint", "base_section_fingerprint"):
        if str(precheck_result.get(field, "")) != str(reverification_input.get(field, "")):
            raise ValueError(f"REVERIFICATION_PRECHECK_CONTEXT_MISMATCH:{field}")
    if str(precheck_result.get("virtual_proposed_claim_text_fingerprint", "")) != str(reverification_input.get("proposed_claim_text_fingerprint", "")):
        raise ValueError("REVERIFICATION_PRECHECK_CONTEXT_MISMATCH:proposed_claim_text_fingerprint")
    from src.config.verification_policy_config import validate_verification_input_policy
    policy = validate_verification_input_policy(dict(reverification_input.get("policy") or {}))
    if policy["reverification_retrieval_rounds"] != 0:
        raise ValueError("REVERIFICATION_RETRIEVAL_FORBIDDEN")
    rows = tuple(dict(x) for x in reverification_input.get("authorized_evidence", ()) if isinstance(x, Mapping))
    evidence_fp = compute_frozen_evidence_snapshot_fingerprint(rows)
    if evidence_fp != precheck_result.get("frozen_evidence_snapshot_fingerprint"):
        raise ValueError("REVERIFICATION_EVIDENCE_SNAPSHOT_MISMATCH")
    policy_fp = compute_reverification_policy_fingerprint(policy)
    if policy_fp != precheck_result.get("reverification_policy_fingerprint"):
        raise ValueError("REVERIFICATION_POLICY_SNAPSHOT_MISMATCH")
    context = {
        "verification_mode": "REVERIFICATION",
        "correction_id": reverification_input["correction_id"], "claim_id": reverification_input["claim_id"],
        "section_id": reverification_input["section_id"], "original_claim_text": reverification_input["original_claim_text"],
        "claim_text": precheck_result["virtual_proposed_claim_text"],
        "virtual_proposed_claim_text_fingerprint": precheck_result["virtual_proposed_claim_text_fingerprint"],
        "proposal_fingerprint": precheck_result["proposal_fingerprint"],
        "base_claim_fingerprint": precheck_result["base_claim_fingerprint"],
        "base_section_fingerprint": precheck_result["base_section_fingerprint"],
        "source_verdict": reverification_input["source_verdict"],
        "source_issue_codes": tuple(reverification_input["source_issue_codes"]),
        "target_issue_codes": tuple(reverification_input["target_issue_codes"]),
        "correction_action_type": reverification_input["correction_action_type"],
        "allowed_evidence_ids": tuple(reverification_input["evidence_ids"]),
        "authorized_evidence": rows, "retrieval_allowed": False, "retrieval_rounds": 0,
        "frozen_evidence_snapshot_fingerprint": evidence_fp,
        "reverification_policy_fingerprint": policy_fp,
        "policy": policy,
    }
    context_fp = compute_reverification_context_fingerprint(context, evidence_snapshot_fingerprint=evidence_fp, policy_fingerprint=policy_fp)
    if context_fp != precheck_result.get("reverification_context_fingerprint"):
        raise ValueError("REVERIFICATION_CONTEXT_FINGERPRINT_MISMATCH")
    context["reverification_context_fingerprint"] = context_fp
    return context


def validate_independent_reverification_response(response: Mapping[str, Any], *, context: Mapping[str, Any]) -> dict[str, Any]:
    from src.config.verification_policy_config import (
        REVERIFICATION_OUTPUT_FIELDS, SCIENTIFIC_VERDICTS, SUPPORT_LEVELS,
        REVERIFICATION_ASSESSMENTS, REVERIFICATION_ACTION_ASSESSMENT_FIELD,
    )
    if not isinstance(response, Mapping) or set(response) != set(REVERIFICATION_OUTPUT_FIELDS):
        raise ValueError("REVERIFICATION_OUTPUT_SCHEMA_INVALID")
    value = dict(response)
    if value["correction_id"] != context["correction_id"]:
        raise ValueError("REVERIFICATION_RESPONSE_CORRECTION_ID_MISMATCH")
    if value["claim_id"] != context["claim_id"]:
        raise ValueError("REVERIFICATION_RESPONSE_CLAIM_ID_MISMATCH")
    if value["proposed_verdict"] not in SCIENTIFIC_VERDICTS or value["support_level"] not in SUPPORT_LEVELS:
        raise ValueError("REVERIFICATION_OUTPUT_SCHEMA_INVALID")
    expected_support = context["policy"]["reverification_support_level_by_verdict"].get(value["proposed_verdict"])
    if expected_support != value["support_level"]:
        raise ValueError("REVERIFICATION_VERDICT_SUPPORT_LEVEL_MISMATCH")
    for field in ("evidence_ids_used", "observed_issue_codes", "target_issues_resolved", "reason_codes"):
        if not isinstance(value[field], list) or any(not isinstance(x, str) or not x.strip() for x in value[field]) or len(value[field]) != len(set(value[field])):
            raise ValueError("REVERIFICATION_OUTPUT_SCHEMA_INVALID")
    allowed_order = tuple(context["allowed_evidence_ids"])
    if not set(value["evidence_ids_used"]).issubset(set(allowed_order)):
        raise ValueError("REVERIFICATION_UNKNOWN_EVIDENCE_ID")
    used = set(value["evidence_ids_used"])
    value["evidence_ids_used"] = tuple(eid for eid in allowed_order if eid in used)
    observed_allowed = set(context["policy"]["reverification_observed_issue_codes"])
    if not set(value["observed_issue_codes"]).issubset(observed_allowed):
        raise ValueError("REVERIFICATION_UNKNOWN_ISSUE_CODE")
    reason_allowed = set(context["policy"]["reverification_llm_reason_codes"])
    if not set(value["reason_codes"]).issubset(reason_allowed):
        raise ValueError("REVERIFICATION_UNKNOWN_REASON_CODE")
    if not set(value["target_issues_resolved"]).issubset(set(context["target_issue_codes"])):
        raise ValueError("REVERIFICATION_UNKNOWN_ISSUE_CODE")
    value["observed_issue_codes"] = tuple(value["observed_issue_codes"])
    value["target_issues_resolved"] = tuple(value["target_issues_resolved"])
    value["reason_codes"] = tuple(value["reason_codes"])
    for field in ("supported_meaning_preserved", "intended_semantic_change_valid", "unintended_semantic_change_absent", "manual_review_recommended"):
        if type(value[field]) is not bool:
            raise ValueError("REVERIFICATION_OUTPUT_SCHEMA_INVALID")
    assessment_fields = ("scope_assessment", "numeric_assessment", "attribution_assessment", "citation_assessment")
    if any(value[field] not in REVERIFICATION_ASSESSMENTS for field in assessment_fields):
        raise ValueError("REVERIFICATION_OUTPUT_SCHEMA_INVALID")
    applicable = REVERIFICATION_ACTION_ASSESSMENT_FIELD.get(context["correction_action_type"])
    for field in assessment_fields:
        if field == applicable:
            if value[field] == "NOT_APPLICABLE":
                raise ValueError("REVERIFICATION_ACTION_ASSESSMENT_MISMATCH")
        elif value[field] != "NOT_APPLICABLE":
            raise ValueError("REVERIFICATION_ACTION_ASSESSMENT_MISMATCH")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not float("-inf") < float(confidence) < float("inf"):
        raise ValueError("REVERIFICATION_CONFIDENCE_INVALID")
    lo, hi = float(context["policy"]["reverification_confidence_min"]), float(context["policy"]["reverification_confidence_max"])
    if not lo <= float(confidence) <= hi:
        raise ValueError("REVERIFICATION_CONFIDENCE_INVALID")
    value["confidence"] = float(confidence)
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise ValueError("REVERIFICATION_OUTPUT_SCHEMA_INVALID")
    value["rationale"] = value["rationale"].strip()
    if value["proposed_verdict"] == "SUPPORTED":
        if not value["evidence_ids_used"]:
            raise ValueError("REVERIFICATION_SUPPORTED_REQUIRES_EVIDENCE")
        if value["support_level"] in {"NONE", "NOT_EVALUATED"}:
            raise ValueError("REVERIFICATION_SUPPORTED_REQUIRES_SUPPORT_LEVEL")
    if set(value["target_issues_resolved"]) & set(value["observed_issue_codes"]):
        raise ValueError("REVERIFICATION_RESOLVED_ISSUE_STILL_OBSERVED")
    if value["proposed_verdict"] == "NOT_EVALUATED" and value["target_issues_resolved"]:
        raise ValueError("REVERIFICATION_NOT_EVALUATED_CANNOT_RESOLVE_TARGET")
    return value


# Override runner so Phase 6.3R fields are persisted; acceptance remains absent.
def run_independent_virtual_reverification(reverification_input: Mapping[str, Any], precheck_result: Mapping[str, Any], *, reverification_llm: ReverificationLLM | None) -> dict[str, Any]:
    from src.tools.verification.prompting import build_reverification_messages, parse_reverification_response
    from src.tools.verification.traceability import CorrectionIndependentReverificationResult
    base = {
        "correction_id": str(reverification_input.get("correction_id", "")),
        "claim_id": str(reverification_input.get("claim_id", "")),
        "section_id": str(reverification_input.get("section_id", "")),
        "proposal_fingerprint": str(precheck_result.get("proposal_fingerprint", "")),
        "virtual_proposed_claim_text_fingerprint": str(precheck_result.get("virtual_proposed_claim_text_fingerprint", "")),
        "frozen_evidence_snapshot_fingerprint": str(precheck_result.get("frozen_evidence_snapshot_fingerprint", "")),
        "reverification_context_fingerprint": str(precheck_result.get("reverification_context_fingerprint", "")),
    }
    def result(status: str, *, technical=(), reason=(), raw=(), trace=(), calls=0, fa=0, fr=0, sa=0, sr=0, data=None):
        data = data or {}
        return CorrectionIndependentReverificationResult(
            **base, reverification_execution_status=status,
            proposed_verdict=str(data.get("proposed_verdict", "NOT_EVALUATED")),
            support_level=str(data.get("support_level", "NOT_EVALUATED")),
            observed_issue_codes=tuple(data.get("observed_issue_codes", ())),
            target_issues_resolved_reported=tuple(data.get("target_issues_resolved", ())),
            evidence_ids_used=tuple(data.get("evidence_ids_used", ())),
            supported_meaning_preserved=bool(data.get("supported_meaning_preserved", False)),
            intended_semantic_change_valid=bool(data.get("intended_semantic_change_valid", False)),
            unintended_semantic_change_absent=bool(data.get("unintended_semantic_change_absent", False)),
            scope_assessment=str(data.get("scope_assessment", "NOT_APPLICABLE")),
            numeric_assessment=str(data.get("numeric_assessment", "NOT_APPLICABLE")),
            attribution_assessment=str(data.get("attribution_assessment", "NOT_APPLICABLE")),
            citation_assessment=str(data.get("citation_assessment", "NOT_APPLICABLE")),
            manual_review_recommended=bool(data.get("manual_review_recommended", False)),
            reason_codes=tuple(reason) + tuple(data.get("reason_codes", ())),
            technical_issue_codes=tuple(technical), rationale=str(data.get("rationale", "")), confidence=data.get("confidence"),
            prompt_version=str((reverification_input.get("policy") or {}).get("reverification_user_prompt_version", "")),
            raw_attempts=tuple(raw), decision_trace=tuple(trace), reverification_llm_calls=calls,
            format_attempts=fa, format_retries=fr, schema_attempts=sa, schema_retries=sr,
            tool_names_selected=("ReverificationLLM",) if calls else (), correction_applied=False,
        ).to_dict()
    if precheck_result.get("precheck_status") != "PRECHECK_PASSED":
        return result("BLOCKED", reason=("REVERIFICATION_PRECHECK_NOT_PASSED",))
    try:
        context = build_reverification_claim_context(reverification_input, precheck_result)
    except (KeyError, TypeError, ValueError) as exc:
        return result("BLOCKED", reason=(str(exc).split(":",1)[0] if str(exc) else "REVERIFICATION_CONTEXT_INCOMPATIBLE",))
    if reverification_llm is None:
        return result("BLOCKED", technical=("REVERIFICATION_DEPENDENCY_UNAVAILABLE",))
    policy=context["policy"]; max_calls=int(policy["max_reverification_llm_attempts"]); max_format_repairs=int(policy["max_reverification_format_repair_attempts"])
    raw_attempts=[]; trace=[]; previous_errors=[]; calls=fa=fr=sa=sr=0; last_error=""
    while calls < max_calls:
        messages=build_reverification_messages(context,previous_errors=previous_errors); calls+=1
        try: raw=reverification_llm.invoke(messages)
        except Exception as exc:
            trace.append({"attempt_number":calls,"parse_status":"NOT_ATTEMPTED","schema_status":"NOT_ATTEMPTED","exception_type":type(exc).__name__})
            return result("FAILED",technical=("REVERIFICATION_LLM_INVOCATION_FAILED",),raw=raw_attempts,trace=trace,calls=calls,fa=fa,fr=fr,sa=sa,sr=sr)
        attempt={"attempt_number":calls,"raw_output_hash":_safe_raw_hash(raw),"parse_status":"PENDING","schema_status":"NOT_ATTEMPTED","exception_type":""}; fa+=1
        try: parsed=parse_reverification_response(raw); attempt["parse_status"]="VALID"
        except ValueError:
            last_error="REVERIFICATION_OUTPUT_INVALID_JSON"; attempt["parse_status"]="INVALID"; raw_attempts.append(attempt); trace.append(dict(attempt))
            if fr>=max_format_repairs or calls>=max_calls: break
            fr+=1; previous_errors=[last_error]; continue
        sa+=1
        try:
            validated=validate_independent_reverification_response(parsed,context=context); attempt["schema_status"]="VALID"; raw_attempts.append(attempt); trace.append(dict(attempt))
            return result("COMPLETED",raw=raw_attempts,trace=trace,calls=calls,fa=fa,fr=fr,sa=sa,sr=sr,data=validated)
        except ValueError as exc:
            last_error=str(exc); attempt["schema_status"]="INVALID"; raw_attempts.append(attempt); trace.append(dict(attempt))
            if calls>=max_calls: break
            sr+=1; previous_errors=[last_error]
    technical=("REVERIFICATION_ATTEMPTS_EXHAUSTED", "REVERIFICATION_OUTPUT_INVALID_JSON" if last_error=="REVERIFICATION_OUTPUT_INVALID_JSON" else "REVERIFICATION_OUTPUT_SCHEMA_INVALID")
    return result("FAILED",technical=technical,raw=raw_attempts,trace=trace,calls=calls,fa=fa,fr=fr,sa=sa,sr=sr)


# Phase 6.4: deterministic before/after comparison, risk and provisional decision.
def _ordered_scientific_issue_codes(values: _Sequence[str], policy: Mapping[str, Any]) -> tuple[str, ...]:
    allowed_order = tuple(policy["reverification_issue_order"])
    allowed = set(allowed_order)
    normalized = {str(item).strip() for item in values if str(item).strip()}
    unknown = normalized - allowed
    if unknown:
        raise ValueError("REVERIFICATION_UNKNOWN_ISSUE_CODE")
    return tuple(code for code in allowed_order if code in normalized)


def _risk_from_validated_scientific_state(
    *, verdict: str, issue_codes: _Sequence[str], policy: Mapping[str, Any]
) -> str | None:
    issues = set(issue_codes)
    critical = set(policy["reverification_critical_new_issue_codes"])
    if verdict in {"NOT_EVALUATED", "AMBIGUOUS"}:
        return None
    if issues & critical:
        return "HIGH"
    if verdict in {"PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "NOT_VERIFIABLE", "CONTRADICTED"}:
        return "MEDIUM"
    if issues:
        return "MEDIUM"
    if verdict in {"SUPPORTED", "NOT_APPLICABLE"}:
        return "LOW"
    return None


def _risk_delta(before: str | None, after: str | None) -> str:
    if before is None or after is None:
        return "NOT_COMPARABLE"
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if rank[after] < rank[before]:
        return "REDUCED"
    if rank[after] > rank[before]:
        return "INCREASED"
    return "UNCHANGED"






def canonicalize_precheck_gate_reason_code(code: str) -> str:
    """Map only explicitly approved parameterized families; never infer by substring."""
    from src.config.verification_policy_config import PRECHECK_PARAMETERIZED_REASON_CODE_FAMILIES
    if not isinstance(code, str) or not code.strip():
        return ""
    normalized = code.strip()
    family = normalized.split(":", 1)[0]
    if family in PRECHECK_PARAMETERIZED_REASON_CODE_FAMILIES:
        return family
    return normalized


def discover_precheck_parameterized_reason_code_families() -> tuple[str, ...]:
    """Return the frozen parameterized families without source introspection.

    Phase 6.7 removes the runtime dependency on ``inspect.getsource``. Coverage
    remains explicit through the centralized contractual catalog.
    """
    from src.config.verification_policy_config import PRECHECK_PARAMETERIZED_REASON_CODE_FAMILIES
    return tuple(sorted(PRECHECK_PARAMETERIZED_REASON_CODE_FAMILIES))


def audit_precheck_gate_reason_code_coverage(
    *, extra_emittable_families: Sequence[str] = (),
) -> dict[str, tuple[str, ...]]:
    """Audit real producer families, the closed catalog, and disjointness."""
    from src.config.verification_policy_config import (
        PRECHECK_RUNTIME_EMITTED_REASON_CODES,
        PRECHECK_PARAMETERIZED_REASON_CODE_FAMILIES,
        PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES,
        PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES,
        PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES,
    )
    temporary = set(PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES)
    contractual = set(PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES)
    scientific = set(PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES)
    union = temporary | contractual | scientific
    overlaps = (temporary & contractual) | (temporary & scientific) | (contractual & scientific)
    runtime = set(PRECHECK_RUNTIME_EMITTED_REASON_CODES)
    discovered_families = set(discover_precheck_parameterized_reason_code_families())
    discovered_families.update(str(x).strip() for x in extra_emittable_families if str(x).strip())
    declared_families = set(PRECHECK_PARAMETERIZED_REASON_CODE_FAMILIES)
    uncovered_families = discovered_families - declared_families
    misclassified_families = discovered_families - union
    uncovered = (runtime - union) | uncovered_families | misclassified_families
    allowed_non_runtime = {
        "REVERIFICATION_DEPENDENCY_UNAVAILABLE",
        "REVERIFICATION_POLICY_UNAVAILABLE",
        *declared_families,
    }
    uncategorized_extras = union - runtime - allowed_non_runtime
    duplicated_emittable = {
        code for code in PRECHECK_RUNTIME_EMITTED_REASON_CODES
        if PRECHECK_RUNTIME_EMITTED_REASON_CODES.count(code) > 1
    }
    return {
        "uncovered": tuple(sorted(uncovered)),
        "uncovered_parameterized_families": tuple(sorted(uncovered_families)),
        "discovered_parameterized_families": tuple(sorted(discovered_families)),
        "category_overlaps": tuple(sorted(overlaps)),
        "uncategorized_extras": tuple(sorted(uncategorized_extras)),
        "duplicated_emittable": tuple(sorted(duplicated_emittable)),
    }



# Phase 6.4R: snapshot revalidation and hardened comparison contract.
def _comparison_ordered_codes(values: _Sequence[str], allowed_order: _Sequence[str], error: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError(error)
    normalized = {str(x).strip() for x in values if str(x).strip()}
    allowed = tuple(allowed_order)
    if normalized - set(allowed):
        raise ValueError(error)
    return tuple(x for x in allowed if x in normalized)







# Phase 6.4S: public official validators reused by origin and comparison flows.
def validate_correction_reverification_precheck_result_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete immutable precheck result contract."""
    required = {
        "correction_id", "claim_id", "section_id", "virtual_proposed_claim_text", "precheck_status",
        "contract_valid", "fingerprints_valid", "spans_valid", "evidence_valid", "textual_integrity_valid",
        "action_validation_valid", "reason_codes", "technical_issue_codes", "virtual_proposed_claim_text_fingerprint",
        "proposal_fingerprint", "base_claim_fingerprint", "base_section_fingerprint",
        "frozen_evidence_snapshot_fingerprint", "reverification_policy_fingerprint",
        "reverification_context_fingerprint", "diagnostic_details", "llm_calls", "correction_applied",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("REVERIFICATION_PRECHECK_CONTRACT_INVALID")
    out = dict(value)
    for field in (
        "correction_id", "claim_id", "section_id", "virtual_proposed_claim_text",
        "proposal_fingerprint", "virtual_proposed_claim_text_fingerprint", "base_claim_fingerprint",
        "base_section_fingerprint", "frozen_evidence_snapshot_fingerprint",
        "reverification_policy_fingerprint", "reverification_context_fingerprint",
    ):
        if not isinstance(out.get(field), str) or not out[field].strip():
            raise ValueError("REVERIFICATION_PRECHECK_CONTRACT_INVALID")
    if out["precheck_status"] not in {"PRECHECK_PASSED", "PRECHECK_BLOCKED", "PRECHECK_REJECTED"}:
        raise ValueError("REVERIFICATION_PRECHECK_CONTRACT_INVALID")
    for field in ("contract_valid", "fingerprints_valid", "spans_valid", "evidence_valid", "textual_integrity_valid", "action_validation_valid", "correction_applied"):
        if type(out.get(field)) is not bool:
            raise ValueError("REVERIFICATION_PRECHECK_CONTRACT_INVALID")
    if out["precheck_status"] == "PRECHECK_PASSED" and not all(out[f] is True for f in ("contract_valid", "fingerprints_valid", "spans_valid", "evidence_valid", "textual_integrity_valid", "action_validation_valid")):
        raise ValueError("REVERIFICATION_PRECHECK_CONTRACT_INVALID")
    if type(out.get("llm_calls")) is not int or out["llm_calls"] != 0 or out["correction_applied"] is not False:
        raise ValueError("REVERIFICATION_PRECHECK_CONTRACT_INVALID")
    for field in ("reason_codes", "technical_issue_codes", "diagnostic_details"):
        if not isinstance(out[field], (list, tuple)):
            raise ValueError("REVERIFICATION_PRECHECK_CONTRACT_INVALID")
    if any(not isinstance(x, str) or not x.strip() for x in out["reason_codes"] + out["technical_issue_codes"]):
        raise ValueError("REVERIFICATION_PRECHECK_CONTRACT_INVALID")
    if any(not isinstance(x, Mapping) for x in out["diagnostic_details"]):
        raise ValueError("REVERIFICATION_PRECHECK_CONTRACT_INVALID")
    out["reason_codes"] = tuple(out["reason_codes"])
    out["technical_issue_codes"] = tuple(out["technical_issue_codes"])
    out["diagnostic_details"] = tuple(dict(x) for x in out["diagnostic_details"])
    return out


def validate_correction_independent_reverification_result_contract(
    value: Mapping[str, Any], *, context: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the complete Phase 6.3 result using its official scientific response validator."""
    required = {
        "correction_id", "claim_id", "section_id", "reverification_execution_status", "proposed_verdict",
        "support_level", "observed_issue_codes", "target_issues_resolved_reported", "evidence_ids_used",
        "supported_meaning_preserved", "intended_semantic_change_valid", "unintended_semantic_change_absent",
        "scope_assessment", "numeric_assessment", "attribution_assessment", "citation_assessment",
        "manual_review_recommended", "reason_codes", "technical_issue_codes", "rationale", "confidence",
        "prompt_version", "raw_attempts", "decision_trace", "reverification_llm_calls", "format_attempts",
        "format_retries", "schema_attempts", "schema_retries", "proposal_fingerprint",
        "virtual_proposed_claim_text_fingerprint", "frozen_evidence_snapshot_fingerprint",
        "reverification_context_fingerprint", "tool_names_considered", "tool_names_selected", "correction_applied",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("REVERIFICATION_RESULT_CONTRACT_INVALID")
    out = dict(value)
    if out.get("reverification_execution_status") not in {"COMPLETED", "FAILED", "BLOCKED"}:
        raise ValueError("REVERIFICATION_RESULT_CONTRACT_INVALID")
    for field in ("correction_id", "claim_id", "section_id", "proposal_fingerprint", "virtual_proposed_claim_text_fingerprint", "frozen_evidence_snapshot_fingerprint", "reverification_context_fingerprint", "prompt_version"):
        if not isinstance(out.get(field), str) or not out[field].strip():
            raise ValueError("REVERIFICATION_RESULT_CONTRACT_INVALID")
    if out["correction_applied"] is not False:
        raise ValueError("REVERIFICATION_RESULT_CONTRACT_INVALID")
    if out["reverification_execution_status"] != "COMPLETED":
        raise ValueError("REVERIFICATION_RESULT_NOT_COMPLETED")
    response = {
        "correction_id": out["correction_id"], "claim_id": out["claim_id"],
        "proposed_verdict": out["proposed_verdict"], "support_level": out["support_level"],
        "evidence_ids_used": list(out["evidence_ids_used"]),
        "observed_issue_codes": list(out["observed_issue_codes"]),
        "target_issues_resolved": list(out["target_issues_resolved_reported"]),
        "supported_meaning_preserved": out["supported_meaning_preserved"],
        "intended_semantic_change_valid": out["intended_semantic_change_valid"],
        "unintended_semantic_change_absent": out["unintended_semantic_change_absent"],
        "scope_assessment": out["scope_assessment"], "numeric_assessment": out["numeric_assessment"],
        "attribution_assessment": out["attribution_assessment"], "citation_assessment": out["citation_assessment"],
        "manual_review_recommended": out["manual_review_recommended"],
        "reason_codes": list(out["reason_codes"]), "rationale": out["rationale"], "confidence": out["confidence"],
    }
    normalized = validate_independent_reverification_response(response, context=context)
    for field, normalized_field in (("evidence_ids_used", "evidence_ids_used"), ("observed_issue_codes", "observed_issue_codes"), ("target_issues_resolved_reported", "target_issues_resolved"), ("reason_codes", "reason_codes")):
        out[field] = tuple(normalized[normalized_field])
    for field in ("proposed_verdict", "support_level", "supported_meaning_preserved", "intended_semantic_change_valid", "unintended_semantic_change_absent", "scope_assessment", "numeric_assessment", "attribution_assessment", "citation_assessment", "manual_review_recommended", "rationale", "confidence"):
        out[field] = normalized[field]
    from src.config.verification_policy_config import REVERIFICATION_TECHNICAL_ISSUE_CODES
    if not isinstance(out["technical_issue_codes"], (list, tuple)) or not set(out["technical_issue_codes"]).issubset(set(REVERIFICATION_TECHNICAL_ISSUE_CODES)):
        raise ValueError("REVERIFICATION_UNKNOWN_TECHNICAL_ISSUE_CODE")
    out["technical_issue_codes"] = tuple(out["technical_issue_codes"])
    for field in ("raw_attempts", "decision_trace"):
        if not isinstance(out[field], (list, tuple)) or any(not isinstance(x, Mapping) for x in out[field]):
            raise ValueError("REVERIFICATION_RESULT_CONTRACT_INVALID")
        out[field] = tuple(dict(x) for x in out[field])
    for field in ("tool_names_considered", "tool_names_selected"):
        if not isinstance(out[field], (list, tuple)) or any(not isinstance(x, str) or not x.strip() for x in out[field]):
            raise ValueError("REVERIFICATION_RESULT_CONTRACT_INVALID")
        out[field] = tuple(out[field])
    for field in ("reverification_llm_calls", "format_attempts", "format_retries", "schema_attempts", "schema_retries"):
        if type(out.get(field)) is not int or out[field] < 0:
            raise ValueError("REVERIFICATION_RESULT_CONTRACT_INVALID")
    return out

def _comparison_failure_result(*, reverification_input: Mapping[str, Any], precheck_result: Mapping[str, Any], reverification_result: Mapping[str, Any], reason: str, decision: str, technical: tuple[str, ...] = (), details: Mapping[str, Any] | None = None, precheck_reason_codes: tuple[str, ...] = (), precheck_technical_issue_codes: tuple[str, ...] = ()) -> dict[str, Any]:
    from src.tools.verification.traceability import CorrectionBeforeAfterComparisonResult
    from src.config.verification_policy_config import (
        REVERIFICATION_COMPARISON_RISK_POLICY_VERSION,
        CORRECTION_ACTION_TYPES,
        COMPARISON_GATE_ACTION_NOT_AVAILABLE,
        REVERIFICATION_COMPARISON_REASON_CODES,
        REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES,
    )
    def ident(field: str) -> str:
        for obj in (reverification_input, precheck_result, reverification_result):
            value = obj.get(field) if isinstance(obj, Mapping) else None
            if isinstance(value, str) and value.strip():
                return value
        return "UNAVAILABLE"
    def safe_verdict(obj: Mapping[str, Any], field: str) -> str:
        value = obj.get(field) if isinstance(obj, Mapping) else None
        return str(value) if value in SUPPORT_LEVEL_BY_VERDICT else "NOT_EVALUATED"
    action = reverification_input.get("correction_action_type") if isinstance(reverification_input, Mapping) else None
    safe_action = str(action) if action in CORRECTION_ACTION_TYPES else COMPARISON_GATE_ACTION_NOT_AVAILABLE
    safe_reason = reason if reason in REVERIFICATION_COMPARISON_REASON_CODES else "COMPARISON_INPUT_INVALID"
    safe_technical = tuple(code for code in technical if code in REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES)
    trace_item = {
        "step": "comparison_input_gate",
        "status": "FAILED",
        "reason_code": safe_reason,
        "diagnostic": dict(details or {}),
        "precheck_reason_codes": tuple(str(x) for x in precheck_reason_codes),
        "precheck_technical_issue_codes": tuple(str(x) for x in precheck_technical_issue_codes),
        "gate_classification": str((details or {}).get("gate_classification") or (
            "TEMPORARY_TECHNICAL"
            if decision == "DEFER_TO_MANUAL_REVIEW"
            else "PERMANENT_CONTRACTUAL"
        )),
    }
    result = CorrectionBeforeAfterComparisonResult(
        correction_id=ident("correction_id"), claim_id=ident("claim_id"), section_id=ident("section_id"),
        correction_action_type=safe_action,
        original_verdict=safe_verdict(reverification_input, "source_verdict"),
        proposed_verdict=safe_verdict(reverification_result, "proposed_verdict"),
        source_issue_codes=(), observed_issue_codes=(), target_issue_codes=(), resolved_issue_codes=(), remaining_issue_codes=(), new_issue_codes=(),
        target_issues_resolved=False, reported_resolution_matches=False,
        hallucination_risk_before="NOT_COMPARABLE", hallucination_risk_after="NOT_COMPARABLE", hallucination_risk_delta="NOT_COMPARABLE",
        risk_policy_version=REVERIFICATION_COMPARISON_RISK_POLICY_VERSION, risk_before_recomputed=False, risk_after_computed=False,
        supported_meaning_preserved=False, intended_semantic_change_valid=False, unintended_semantic_change_absent=False,
        scope_assessment="NOT_APPLICABLE", numeric_assessment="NOT_APPLICABLE", attribution_assessment="NOT_APPLICABLE", citation_assessment="NOT_APPLICABLE",
        acceptance_decision=decision if decision in {"REJECT_PROPOSAL","DEFER_TO_MANUAL_REVIEW"} else "REJECT_PROPOSAL",
        manual_review_required=(decision == "DEFER_TO_MANUAL_REVIEW"),
        reason_codes=(safe_reason,), technical_issue_codes=safe_technical,
        decision_trace=(trace_item,),
        proposal_fingerprint=ident("proposal_fingerprint"), virtual_proposed_claim_text_fingerprint=ident("virtual_proposed_claim_text_fingerprint"),
        frozen_evidence_snapshot_fingerprint=ident("frozen_evidence_snapshot_fingerprint"), reverification_context_fingerprint=ident("reverification_context_fingerprint"),
        result_contract_valid=True, additional_llm_calls=0, retrieval_rounds=0, correction_applied=False,
    ).to_dict()
    return validate_before_after_comparison_result_contract(result)


def validate_before_after_comparison_result_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    from src.config.verification_policy_config import (
        REVERIFICATION_COMPARISON_REASON_CODES,
        REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES,
        REVERIFICATION_COMPARISON_RISK_POLICY_VERSION,
        REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES,
        CORRECTION_ACTION_TYPES,
        COMPARISON_GATE_ACTION_NOT_AVAILABLE,
    )
    required = {
        "correction_id", "claim_id", "section_id", "correction_action_type", "original_verdict", "proposed_verdict", "source_issue_codes",
        "observed_issue_codes", "target_issue_codes", "resolved_issue_codes", "remaining_issue_codes", "new_issue_codes",
        "target_issues_resolved", "reported_resolution_matches", "hallucination_risk_before", "hallucination_risk_after",
        "hallucination_risk_delta", "risk_policy_version", "risk_before_recomputed", "risk_after_computed",
        "supported_meaning_preserved", "intended_semantic_change_valid", "unintended_semantic_change_absent",
        "scope_assessment", "numeric_assessment", "attribution_assessment", "citation_assessment", "acceptance_decision",
        "manual_review_required", "reason_codes", "technical_issue_codes", "decision_trace", "proposal_fingerprint",
        "virtual_proposed_claim_text_fingerprint", "frozen_evidence_snapshot_fingerprint", "reverification_context_fingerprint",
        "result_contract_valid", "additional_llm_calls", "retrieval_rounds", "correction_applied",
    }
    if set(value) != required:
        raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    out = dict(value)
    for field in ("correction_id", "claim_id", "section_id", "correction_action_type", "proposal_fingerprint", "virtual_proposed_claim_text_fingerprint", "frozen_evidence_snapshot_fingerprint", "reverification_context_fingerprint"):
        if not isinstance(out.get(field), str) or not out[field].strip(): raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    if out.get("original_verdict") not in SUPPORT_LEVEL_BY_VERDICT or out.get("proposed_verdict") not in SUPPORT_LEVEL_BY_VERDICT:
        raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    allowed_actions = set(CORRECTION_ACTION_TYPES) | {COMPARISON_GATE_ACTION_NOT_AVAILABLE}
    if out.get("correction_action_type") not in allowed_actions:
        raise ValueError("COMPARISON_CORRECTION_ACTION_INVALID")
    if out["correction_action_type"] == COMPARISON_GATE_ACTION_NOT_AVAILABLE:
        gate_reasons = {
            "COMPARISON_INPUT_INVALID", "COMPARISON_CORRECTION_ACTION_INVALID",
            "COMPARISON_PRECHECK_BLOCKED", "COMPARISON_PRECHECK_REJECTED",
            "COMPARISON_PRECHECK_INVALID", "COMPARISON_REVERIFICATION_RESULT_INVALID",
            "COMPARISON_REQUIRED_IDENTITY_FIELD_MISSING",
        }
        if out.get("acceptance_decision") == "ACCEPT_FOR_07C" or not (set(out.get("reason_codes") or ()) & gate_reasons):
            raise ValueError("COMPARISON_CORRECTION_ACTION_INVALID")
    scientific_order = REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES
    source = _comparison_ordered_codes(out["source_issue_codes"], scientific_order, "COMPARISON_RESULT_SCHEMA_INVALID")
    observed = _comparison_ordered_codes(out["observed_issue_codes"], scientific_order, "COMPARISON_RESULT_SCHEMA_INVALID")
    targets = _comparison_ordered_codes(out["target_issue_codes"], scientific_order, "COMPARISON_RESULT_SCHEMA_INVALID")
    resolved = _comparison_ordered_codes(out["resolved_issue_codes"], scientific_order, "COMPARISON_RESULT_SCHEMA_INVALID")
    remaining = _comparison_ordered_codes(out["remaining_issue_codes"], scientific_order, "COMPARISON_RESULT_SCHEMA_INVALID")
    new = _comparison_ordered_codes(out["new_issue_codes"], scientific_order, "COMPARISON_RESULT_SCHEMA_INVALID")
    if resolved != _comparison_ordered_codes(set(source)-set(observed), scientific_order, "COMPARISON_RESULT_SCHEMA_INVALID") or remaining != _comparison_ordered_codes(set(source)&set(observed), scientific_order, "COMPARISON_RESULT_SCHEMA_INVALID") or new != _comparison_ordered_codes(set(observed)-set(source), scientific_order, "COMPARISON_RESULT_SCHEMA_INVALID"):
        raise ValueError("COMPARISON_ISSUE_PARTITION_INVALID")
    for field in ("target_issues_resolved","reported_resolution_matches","risk_before_recomputed","risk_after_computed","supported_meaning_preserved","intended_semantic_change_valid","unintended_semantic_change_absent","manual_review_required","result_contract_valid","correction_applied"):
        if type(out.get(field)) is not bool: raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    if targets and out["target_issues_resolved"] != set(targets).issubset(set(resolved)):
        raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    risks={"LOW","MEDIUM","HIGH","NOT_COMPARABLE"}; deltas={"REDUCED","UNCHANGED","INCREASED","NOT_COMPARABLE"}
    if out["hallucination_risk_before"] not in risks or out["hallucination_risk_after"] not in risks or out["hallucination_risk_delta"] not in deltas:
        raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    if out["risk_policy_version"] != REVERIFICATION_COMPARISON_RISK_POLICY_VERSION: raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    for field in ("scope_assessment","numeric_assessment","attribution_assessment","citation_assessment"):
        if out[field] not in {"VALID","INVALID","NOT_APPLICABLE"}: raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    out["reason_codes"]=_comparison_ordered_codes(out["reason_codes"], REVERIFICATION_COMPARISON_REASON_CODES, "COMPARISON_UNKNOWN_REASON_CODE")
    out["technical_issue_codes"]=_comparison_ordered_codes(out["technical_issue_codes"], REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES, "COMPARISON_UNKNOWN_TECHNICAL_ISSUE_CODE")
    if not isinstance(out["decision_trace"], (list,tuple)) or any(not isinstance(x, Mapping) for x in out["decision_trace"]): raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    if out["acceptance_decision"] not in {"ACCEPT_FOR_07C","REJECT_PROPOSAL","DEFER_TO_MANUAL_REVIEW"}: raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    if out["result_contract_valid"] is not True or out["correction_applied"] is not False or out["additional_llm_calls"] != 0 or out["retrieval_rounds"] != 0:
        raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    from src.config.verification_policy_config import (
        REVERIFICATION_ACTION_ASSESSMENT_FIELD,
        REVERIFICATION_CRITICAL_NEW_ISSUE_CODES,
    )
    applicable = REVERIFICATION_ACTION_ASSESSMENT_FIELD.get(out["correction_action_type"])
    assessment_values = {
        "scope_assessment": out["scope_assessment"],
        "numeric_assessment": out["numeric_assessment"],
        "attribution_assessment": out["attribution_assessment"],
        "citation_assessment": out["citation_assessment"],
    }
    assessments_contract_valid = (
        applicable is not None
        and assessment_values.get(applicable) == "VALID"
        and all(value == "NOT_APPLICABLE" for field, value in assessment_values.items() if field != applicable)
    )
    if out["correction_action_type"] == COMPARISON_GATE_ACTION_NOT_AVAILABLE:
        assessments_contract_valid = all(value == "NOT_APPLICABLE" for value in assessment_values.values())
    critical_new_present = bool(set(new) & set(REVERIFICATION_CRITICAL_NEW_ISSUE_CODES))
    if out["acceptance_decision"] == "ACCEPT_FOR_07C":
        if (
            not out["target_issues_resolved"]
            or not out["reported_resolution_matches"]
            or out["manual_review_required"]
            or out["hallucination_risk_delta"] in {"INCREASED", "NOT_COMPARABLE"}
            or critical_new_present
            or not assessments_contract_valid
            or not out["supported_meaning_preserved"]
            or not out["intended_semantic_change_valid"]
            or not out["unintended_semantic_change_absent"]
        ):
            raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    if out["acceptance_decision"] == "DEFER_TO_MANUAL_REVIEW" and out["manual_review_required"] is not True:
        raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    if out["acceptance_decision"] == "REJECT_PROPOSAL" and out["manual_review_required"] is not False:
        raise ValueError("COMPARISON_RESULT_SCHEMA_INVALID")
    return out


def compare_virtual_reverification_before_after(reverification_input: Mapping[str, Any], precheck_result: Mapping[str, Any], reverification_result: Mapping[str, Any]) -> dict[str, Any]:
    from src.tools.verification.traceability import CorrectionBeforeAfterComparisonResult
    from src.config.verification_policy_config import (
        get_verification_input_policy, REVERIFICATION_COMPARISON_RISK_POLICY_VERSION,
        REVERIFICATION_ACTION_ASSESSMENT_FIELD, CORRECTION_ACTION_TYPES,
    )
    # Explicit precheck gate: validate minimal closed gate contract before snapshots.
    precheck_status = precheck_result.get("precheck_status") if isinstance(precheck_result, Mapping) else None
    if precheck_status in {"PRECHECK_BLOCKED", "PRECHECK_REJECTED"}:
        from src.config.verification_policy_config import (
            PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES,
            PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES,
            PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES,
            PRECHECK_GATE_TECHNICAL_ISSUE_CODES,
        )
        reasons = precheck_result.get("reason_codes") if isinstance(precheck_result, Mapping) else None
        technical_codes = precheck_result.get("technical_issue_codes") if isinstance(precheck_result, Mapping) else None
        minimal_valid = (
            isinstance(reasons, (list, tuple))
            and isinstance(technical_codes, (list, tuple))
            and all(isinstance(x, str) and x.strip() for x in reasons)
            and all(isinstance(x, str) and x.strip() for x in technical_codes)
            and precheck_result.get("correction_applied") is False
            and type(precheck_result.get("llm_calls")) is int
            and precheck_result.get("llm_calls") == 0
            and set(technical_codes).issubset(set(PRECHECK_GATE_TECHNICAL_ISSUE_CODES))
        )
        if not minimal_valid:
            return _comparison_failure_result(
                reverification_input=reverification_input, precheck_result=precheck_result,
                reverification_result=reverification_result, reason="COMPARISON_PRECHECK_INVALID",
                decision="REJECT_PROPOSAL",
                details={"precheck_status": precheck_status, "gate_classification": "INVALID_GATE_CONTRACT"},
            )
        reasons = tuple(reasons)
        technical_codes = tuple(technical_codes)
        canonical_reasons = tuple(canonicalize_precheck_gate_reason_code(code) for code in reasons)
        known_reason_set = (
            set(PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES)
            | set(PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES)
            | set(PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES)
        )
        if any(code not in known_reason_set for code in canonical_reasons):
            return _comparison_failure_result(
                reverification_input=reverification_input, precheck_result=precheck_result,
                reverification_result=reverification_result, reason="COMPARISON_PRECHECK_INVALID",
                decision="REJECT_PROPOSAL", precheck_reason_codes=reasons,
                precheck_technical_issue_codes=technical_codes,
                details={"precheck_status": precheck_status, "gate_classification": "UNKNOWN_REASON_CODE"},
            )
        if precheck_status == "PRECHECK_REJECTED":
            if not canonical_reasons or not all(
                code in PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES for code in canonical_reasons
            ):
                return _comparison_failure_result(
                    reverification_input=reverification_input, precheck_result=precheck_result,
                    reverification_result=reverification_result, reason="COMPARISON_PRECHECK_INVALID",
                    decision="REJECT_PROPOSAL", precheck_reason_codes=reasons,
                    precheck_technical_issue_codes=technical_codes,
                    details={"precheck_status": precheck_status, "gate_classification": "INVALID_GATE_CONTRACT"},
                )
            return _comparison_failure_result(
                reverification_input=reverification_input, precheck_result=precheck_result,
                reverification_result=reverification_result, reason="COMPARISON_PRECHECK_REJECTED",
                decision="REJECT_PROPOSAL", precheck_reason_codes=reasons,
                precheck_technical_issue_codes=technical_codes,
                details={"precheck_status": precheck_status, "gate_classification": "DETERMINISTIC_SCIENTIFIC_REJECTION"},
            )
        temporary = bool(canonical_reasons) and all(code in PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES for code in canonical_reasons)
        if temporary:
            return _comparison_failure_result(
                reverification_input=reverification_input, precheck_result=precheck_result,
                reverification_result=reverification_result, reason="COMPARISON_PRECHECK_BLOCKED",
                decision="DEFER_TO_MANUAL_REVIEW", technical=technical_codes or ("COMPARISON_DEPENDENCY_UNAVAILABLE",),
                precheck_reason_codes=reasons, precheck_technical_issue_codes=technical_codes,
                details={"precheck_status": precheck_status, "gate_classification": "TEMPORARY_TECHNICAL"},
            )
        return _comparison_failure_result(
            reverification_input=reverification_input, precheck_result=precheck_result,
            reverification_result=reverification_result, reason="COMPARISON_PRECHECK_BLOCKED",
            decision="REJECT_PROPOSAL", precheck_reason_codes=reasons,
            precheck_technical_issue_codes=technical_codes,
            details={"precheck_status": precheck_status, "gate_classification": "PERMANENT_CONTRACTUAL"},
        )
    if precheck_status != "PRECHECK_PASSED":
        return _comparison_failure_result(
            reverification_input=reverification_input, precheck_result=precheck_result,
            reverification_result=reverification_result, reason="COMPARISON_PRECHECK_INVALID",
            decision="REJECT_PROPOSAL",
        )
    if not reverification_result:
        return _comparison_failure_result(
            reverification_input=reverification_input, precheck_result=precheck_result,
            reverification_result=reverification_result, reason="COMPARISON_REVERIFICATION_RESULT_INVALID",
            decision="DEFER_TO_MANUAL_REVIEW", technical=("COMPARISON_RESULT_ABSENT",),
        )
    action_value = reverification_input.get("correction_action_type") if isinstance(reverification_input, Mapping) else None
    if action_value not in CORRECTION_ACTION_TYPES:
        return _comparison_failure_result(
            reverification_input=reverification_input, precheck_result=precheck_result, reverification_result=reverification_result,
            reason="COMPARISON_CORRECTION_ACTION_INVALID", decision="REJECT_PROPOSAL",
            details={"field": "correction_action_type"},
        )

    # Explicit identity-presence matrix; missing fields are classified before contract validation.
    identity_presence_matrix = (
        (("input", reverification_input, "correction_id"), ("precheck", precheck_result, "correction_id"), ("result", reverification_result, "correction_id")),
        (("input", reverification_input, "claim_id"), ("precheck", precheck_result, "claim_id"), ("result", reverification_result, "claim_id")),
        (("input", reverification_input, "section_id"), ("precheck", precheck_result, "section_id"), ("result", reverification_result, "section_id")),
        (("input", reverification_input, "proposal_fingerprint"), ("precheck", precheck_result, "proposal_fingerprint"), ("result", reverification_result, "proposal_fingerprint")),
        (("input", reverification_input, "proposed_claim_text_fingerprint"), ("precheck", precheck_result, "virtual_proposed_claim_text_fingerprint"), ("result", reverification_result, "virtual_proposed_claim_text_fingerprint")),
        (("precheck", precheck_result, "frozen_evidence_snapshot_fingerprint"), ("result", reverification_result, "frozen_evidence_snapshot_fingerprint")),
        (("precheck", precheck_result, "reverification_context_fingerprint"), ("result", reverification_result, "reverification_context_fingerprint")),
    )
    for group in identity_presence_matrix:
        for structure_name, structure, field in group:
            present = isinstance(structure, Mapping) and field in structure
            value = structure.get(field) if present else None
            if not present or not isinstance(value, str) or not value.strip():
                return _comparison_failure_result(
                    reverification_input=reverification_input, precheck_result=precheck_result,
                    reverification_result=reverification_result,
                    reason="COMPARISON_REQUIRED_IDENTITY_FIELD_MISSING", decision="REJECT_PROPOSAL",
                    details={"structure": structure_name, "field": field},
                )
    # Full contract gates; no raw exception escapes.
    try:
        validated_input = validate_correction_reverification_input_contract(reverification_input)
    except Exception as error:
        if "VERIFICATION_POLICY_INVALID" in str(error):
            return _comparison_failure_result(reverification_input=reverification_input, precheck_result=precheck_result, reverification_result=reverification_result, reason="COMPARISON_RISK_POLICY_MISMATCH", decision="DEFER_TO_MANUAL_REVIEW", technical=("COMPARISON_POLICY_UNAVAILABLE",))
        return _comparison_failure_result(reverification_input=reverification_input, precheck_result=precheck_result, reverification_result=reverification_result, reason="COMPARISON_INPUT_INVALID", decision="REJECT_PROPOSAL")
    try:
        validated_precheck = validate_correction_reverification_precheck_result_contract(precheck_result)
    except Exception:
        return _comparison_failure_result(reverification_input=reverification_input, precheck_result=precheck_result, reverification_result=reverification_result, reason="COMPARISON_PRECHECK_INVALID", decision="REJECT_PROPOSAL")
    try:
        policy = get_verification_input_policy(dict(validated_input.get("policy") or {}))
    except Exception:
        return _comparison_failure_result(reverification_input=reverification_input, precheck_result=precheck_result, reverification_result=reverification_result, reason="COMPARISON_RISK_POLICY_MISMATCH", decision="DEFER_TO_MANUAL_REVIEW", technical=("COMPARISON_POLICY_UNAVAILABLE",))
    identity_equality_matrix = (
        ("correction_id", validated_input["correction_id"], validated_precheck["correction_id"], reverification_result["correction_id"]),
        ("claim_id", validated_input["claim_id"], validated_precheck["claim_id"], reverification_result["claim_id"]),
        ("section_id", validated_input["section_id"], validated_precheck["section_id"], reverification_result["section_id"]),
        ("proposal_fingerprint", validated_input["proposal_fingerprint"], validated_precheck["proposal_fingerprint"], reverification_result["proposal_fingerprint"]),
        ("virtual_proposed_claim_text_fingerprint", validated_input["proposed_claim_text_fingerprint"], validated_precheck["virtual_proposed_claim_text_fingerprint"], reverification_result["virtual_proposed_claim_text_fingerprint"]),
        ("frozen_evidence_snapshot_fingerprint", validated_precheck["frozen_evidence_snapshot_fingerprint"], reverification_result["frozen_evidence_snapshot_fingerprint"]),
        ("reverification_context_fingerprint", validated_precheck["reverification_context_fingerprint"], reverification_result["reverification_context_fingerprint"]),
    )
    for field, *values in identity_equality_matrix:
        if len(set(values)) != 1:
            return _comparison_failure_result(
                reverification_input=validated_input, precheck_result=validated_precheck,
                reverification_result=reverification_result, reason="COMPARISON_CONTEXT_MISMATCH",
                decision="REJECT_PROPOSAL", details={"field": field},
            )
    try:
        official_context = {
            "correction_id": validated_input["correction_id"],
            "claim_id": validated_input["claim_id"],
            "target_issue_codes": tuple(validated_input["target_issue_codes"]),
            "allowed_evidence_ids": tuple(validated_input["evidence_ids"]),
            "correction_action_type": validated_input["correction_action_type"],
            "policy": policy,
        }
        validated_result = validate_correction_independent_reverification_result_contract(reverification_result, context=official_context)
    except Exception:
        decision = "DEFER_TO_MANUAL_REVIEW" if not reverification_result else "REJECT_PROPOSAL"
        technical = ("COMPARISON_RESULT_ABSENT",) if not reverification_result else ()
        return _comparison_failure_result(reverification_input=reverification_input, precheck_result=precheck_result, reverification_result=reverification_result, reason="COMPARISON_REVERIFICATION_RESULT_INVALID", decision=decision, technical=technical)

    # Recompute every snapshot from current content.
    try:
        rows=tuple(dict(x) for x in validated_input.get("authorized_evidence",()) if isinstance(x,Mapping))
        evidence_fp=compute_frozen_evidence_snapshot_fingerprint(rows)
        policy_fp=compute_reverification_policy_fingerprint(policy)
        context=build_reverification_claim_context(validated_input, validated_precheck)
        context_fp=compute_reverification_context_fingerprint(context,evidence_snapshot_fingerprint=evidence_fp,policy_fingerprint=policy_fp)
        if evidence_fp != validated_precheck["frozen_evidence_snapshot_fingerprint"] or policy_fp != validated_precheck["reverification_policy_fingerprint"] or context_fp != validated_precheck["reverification_context_fingerprint"] or evidence_fp != validated_result["frozen_evidence_snapshot_fingerprint"] or context_fp != validated_result["reverification_context_fingerprint"]:
            raise ValueError("snapshot")
    except Exception:
        return _comparison_failure_result(reverification_input=validated_input, precheck_result=validated_precheck, reverification_result=validated_result, reason="COMPARISON_CONTEXT_SNAPSHOT_MISMATCH", decision="REJECT_PROPOSAL")

    source=_ordered_scientific_issue_codes(validated_input["source_issue_codes"],policy)
    observed=_ordered_scientific_issue_codes(validated_result["observed_issue_codes"],policy)
    targets=_ordered_scientific_issue_codes(validated_input["target_issue_codes"],policy)
    resolved=_ordered_scientific_issue_codes(set(source)-set(observed),policy)
    remaining=_ordered_scientific_issue_codes(set(source)&set(observed),policy)
    new=_ordered_scientific_issue_codes(set(observed)-set(source),policy)
    targets_resolved=set(targets).issubset(set(resolved))
    reported=set(validated_result["target_issues_resolved_reported"])
    reported_matches=reported==(set(targets)&set(resolved))
    reasons=[]
    if not reported_matches: reasons.append("REPORTED_RESOLUTION_MISMATCH")
    if not targets_resolved: reasons.append("TARGET_ISSUE_NOT_RESOLVED")
    critical=set(new)&set(policy["reverification_critical_new_issue_codes"])
    review=set(observed)&set(policy["reverification_noncritical_review_issue_codes"])
    if critical: reasons.append("CRITICAL_NEW_ISSUE_INTRODUCED")
    if set(new)&set(policy["reverification_noncritical_review_issue_codes"]): reasons.append("NONCRITICAL_NEW_ISSUE_REQUIRES_REVIEW")
    if review: reasons.append("REMAINING_SCIENTIFIC_AMBIGUITY")

    before=_risk_from_validated_scientific_state(verdict=validated_input["source_verdict"],issue_codes=source,policy=policy)
    after=_risk_from_validated_scientific_state(verdict=validated_result["proposed_verdict"],issue_codes=observed,policy=policy)
    delta=_risk_delta(before,after)
    reasons.append({"REDUCED":"RISK_REDUCED","UNCHANGED":"RISK_UNCHANGED","INCREASED":"RISK_INCREASED","NOT_COMPARABLE":"RISK_NOT_COMPARABLE"}[delta])
    action=validated_input["correction_action_type"]; applicable=REVERIFICATION_ACTION_ASSESSMENT_FIELD.get(action)
    assessments={k:validated_result[k] for k in ("scope_assessment","numeric_assessment","attribution_assessment","citation_assessment")}
    applicability_invalid=applicable is None or assessments.get(applicable)=="NOT_APPLICABLE" or any(k!=applicable and v!="NOT_APPLICABLE" for k,v in assessments.items())
    applicable_invalid=not applicability_invalid and assessments[applicable]=="INVALID"
    if applicability_invalid: reasons.append("ACTION_ASSESSMENT_NOT_APPLICABLE")
    if applicable_invalid: reasons.append("ACTION_ASSESSMENT_INVALID")
    meaning=validated_result["supported_meaning_preserved"]; intended=validated_result["intended_semantic_change_valid"]; unintended=validated_result["unintended_semantic_change_absent"]
    if not meaning: reasons.append("SUPPORTED_MEANING_NOT_PRESERVED")
    if not intended: reasons.append("INTENDED_SEMANTIC_CHANGE_INVALID")
    if not unintended: reasons.append("UNINTENDED_SEMANTIC_CHANGE_DETECTED")
    manual=validated_result["manual_review_recommended"]
    if manual: reasons.append("MANUAL_REVIEW_RECOMMENDED")
    reject=(not targets_resolved or bool(critical) or applicability_invalid or applicable_invalid or not meaning or not intended or not unintended or delta=="INCREASED")
    defer=(not reported_matches or manual or delta=="NOT_COMPARABLE" or bool(review))
    if reject: decision="REJECT_PROPOSAL"; manual_required=False
    elif defer: decision="DEFER_TO_MANUAL_REVIEW"; manual_required=True
    else: decision="ACCEPT_FOR_07C"; manual_required=False; reasons.append("COMPARISON_ACCEPTED_FOR_07C")
    trace=(
        {"step":"snapshot_revalidation","status":"VALID","evidence_snapshot_fingerprint":evidence_fp,"policy_fingerprint":policy_fp,"context_fingerprint":context_fp},
        {"step":"issue_partition","source":source,"observed":observed,"resolved":resolved,"remaining":remaining,"new":new},
        {"step":"risk_projection","algorithm":"comparison_projection","version":REVERIFICATION_COMPARISON_RISK_POLICY_VERSION,"before":before or "NOT_COMPARABLE","after":after or "NOT_COMPARABLE","delta":delta},
        {"step":"individual_proposal_decision","acceptance_decision":decision,"meaning":"The individual proposal passed reverification; unresolved non-target claim issues remain for Phase 6.6."},
    )
    result=CorrectionBeforeAfterComparisonResult(
        correction_id=validated_input["correction_id"],claim_id=validated_input["claim_id"],section_id=validated_input["section_id"],
        correction_action_type=validated_input["correction_action_type"],
        original_verdict=validated_input["source_verdict"],proposed_verdict=validated_result["proposed_verdict"],
        source_issue_codes=source,observed_issue_codes=observed,target_issue_codes=targets,resolved_issue_codes=resolved,remaining_issue_codes=remaining,new_issue_codes=new,
        target_issues_resolved=targets_resolved,reported_resolution_matches=reported_matches,
        hallucination_risk_before=before or "NOT_COMPARABLE",hallucination_risk_after=after or "NOT_COMPARABLE",hallucination_risk_delta=delta,
        risk_policy_version=REVERIFICATION_COMPARISON_RISK_POLICY_VERSION,risk_before_recomputed=True,risk_after_computed=True,
        supported_meaning_preserved=meaning,intended_semantic_change_valid=intended,unintended_semantic_change_absent=unintended,
        scope_assessment=assessments["scope_assessment"],numeric_assessment=assessments["numeric_assessment"],attribution_assessment=assessments["attribution_assessment"],citation_assessment=assessments["citation_assessment"],
        acceptance_decision=decision,manual_review_required=manual_required,reason_codes=tuple(reasons),technical_issue_codes=(),decision_trace=trace,
        proposal_fingerprint=validated_precheck["proposal_fingerprint"],virtual_proposed_claim_text_fingerprint=validated_precheck["virtual_proposed_claim_text_fingerprint"],
        frozen_evidence_snapshot_fingerprint=evidence_fp,reverification_context_fingerprint=context_fp,result_contract_valid=True,additional_llm_calls=0,retrieval_rounds=0,correction_applied=False,
    ).to_dict()
    return validate_before_after_comparison_result_contract(result)

# Phase 6.5.0S: validadores terminales históricos.
def _terminal_nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"TERMINAL_CONTRACT_INVALID:{field}")
    return value.strip()


def _terminal_string_seq(value: Any, field: str, *, allowed: set[str] | None = None) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"TERMINAL_CONTRACT_INVALID:{field}")
    out = tuple(_terminal_nonempty(v, f"{field}[]") for v in value)
    if len(out) != len(set(out)):
        raise ValueError(f"TERMINAL_CONTRACT_INVALID:{field}:duplicates")
    if allowed is not None and not set(out).issubset(allowed):
        raise ValueError(f"TERMINAL_CONTRACT_UNKNOWN_CODE:{field}")
    return out


def _validate_terminal_evidence_collections(value: dict[str, Any]) -> None:
    identity: dict[str, tuple[str, str]] = {}
    eligible_ids: set[str] = set()
    for field in ("eligible_evidence", "deterministically_discarded_evidence", "evidence_used", "evidence_rejected"):
        rows = value[field]
        if type(rows) not in (list, tuple):
            raise ValueError(f"CLAIM_VERIFICATION_RESULT_INVALID:{field}")
        normalized=[]
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"CLAIM_VERIFICATION_RESULT_INVALID:{field}:row")
            item=dict(row)
            eid=_terminal_nonempty(item.get("evidence_id"), f"{field}:evidence_id")
            src=_terminal_nonempty(item.get("source_filename"), f"{field}:source_filename")
            chunk=_terminal_nonempty(item.get("chunk_id"), f"{field}:chunk_id")
            pair=(src,chunk)
            if eid in identity and identity[eid] != pair:
                raise ValueError("CLAIM_VERIFICATION_EVIDENCE_IDENTITY_CONFLICT")
            identity[eid]=pair
            normalized.append(item)
            if field == "eligible_evidence": eligible_ids.add(eid)
        value[field]=tuple(normalized)
    for field in ("evidence_used", "evidence_rejected"):
        ids={str(row["evidence_id"]) for row in value[field]}
        if not ids.issubset(eligible_ids):
            raise ValueError(f"CLAIM_VERIFICATION_UNKNOWN_ELIGIBLE_EVIDENCE:{field}")


def validate_claim_verification_result_contract(result: Mapping[str, Any]) -> dict[str, Any]:
    """Valida la serialización terminal real de ClaimVerificationResult."""
    from dataclasses import fields
    from src.agents.verification_agent import ClaimVerificationResult
    from src.config.verification_policy_config import (
        SCIENTIFIC_VERDICTS, SUPPORT_LEVEL_BY_VERDICT, SCIENTIFIC_JUDGMENT_STATUSES,
        CLAIM_EXECUTION_STATUSES, CLAIM_TECHNICAL_STATUSES,
        CLAIM_COMPLETED_UNRESOLVED_TECHNICAL_STATUSES, HALLUCINATION_RISKS,
        DETERMINISTIC_ISSUE_CODES, SEMANTIC_ISSUE_CODES, SEMANTIC_REASON_CODES,
        ADDITIONAL_RETRIEVAL_REASON_CODES, RETRIEVAL_REASON_CODES, TECHNICAL_ISSUE_CODES,
        CORRECTION_ELIGIBILITIES, NUMERIC_ASSESSMENTS, ATTRIBUTION_ASSESSMENTS,
        EXTRAPOLATION_ASSESSMENTS, CONTRADICTION_TYPES,
    )
    if not isinstance(result, Mapping):
        raise ValueError("CLAIM_VERIFICATION_RESULT_NOT_MAPPING")
    names=tuple(f.name for f in fields(ClaimVerificationResult))
    missing=[f for f in names if f not in result]
    if missing: raise ValueError("CLAIM_VERIFICATION_RESULT_FIELDS_MISSING:"+",".join(missing))
    if set(result)-set(names): raise ValueError("CLAIM_VERIFICATION_RESULT_UNKNOWN_FIELDS")
    v=dict(result)
    v["claim_id"]=_terminal_nonempty(v["claim_id"],"claim_id")
    v["claim_type"]=_terminal_nonempty(v["claim_type"],"claim_type")
    for f in ("scientific_judgment_required","llm_correction_recommendation","manual_review_required","result_contract_valid","scientific_validation_ok","validation_ok"):
        if type(v[f]) is not bool: raise ValueError(f"CLAIM_VERIFICATION_RESULT_INVALID_BOOL:{f}")
    if not v["result_contract_valid"]: raise ValueError("CLAIM_VERIFICATION_RESULT_CONTRACT_NOT_VALID")
    if v["execution_status"] not in CLAIM_EXECUTION_STATUSES: raise ValueError("CLAIM_VERIFICATION_EXECUTION_STATUS_UNKNOWN")
    if v["technical_status"] not in CLAIM_TECHNICAL_STATUSES: raise ValueError("CLAIM_VERIFICATION_TECHNICAL_STATUS_UNKNOWN")
    if v["scientific_judgment_status"] not in SCIENTIFIC_JUDGMENT_STATUSES: raise ValueError("CLAIM_VERIFICATION_JUDGMENT_STATUS_UNKNOWN")
    if v["scientific_verdict"] not in SCIENTIFIC_VERDICTS: raise ValueError("CLAIM_VERIFICATION_VERDICT_UNKNOWN")
    if v["support_level"] != SUPPORT_LEVEL_BY_VERDICT[v["scientific_verdict"]]: raise ValueError("CLAIM_VERIFICATION_VERDICT_SUPPORT_MISMATCH")
    v["technical_issue_codes"]=_terminal_string_seq(v["technical_issue_codes"],"technical_issue_codes",allowed=set(TECHNICAL_ISSUE_CODES))
    v["deterministic_issue_codes"]=_terminal_string_seq(v["deterministic_issue_codes"],"deterministic_issue_codes",allowed=set(DETERMINISTIC_ISSUE_CODES))
    v["semantic_issue_codes"]=_terminal_string_seq(v["semantic_issue_codes"],"semantic_issue_codes",allowed=set(SEMANTIC_ISSUE_CODES))
    # Superposición histórica intencional: _terminal(...) usa deterministic_issue_codes como reason_codes.
    reason_allowed=(set(SEMANTIC_REASON_CODES)|set(ADDITIONAL_RETRIEVAL_REASON_CODES)|set(RETRIEVAL_REASON_CODES)|
                    set(TECHNICAL_ISSUE_CODES)|set(DETERMINISTIC_ISSUE_CODES)|
                    {"DETERMINISTIC_TERMINAL","SCIENTIFIC_JUDGMENT_COMPLETED","SCIENTIFIC_JUDGMENT_BLOCKED","NO_SCIENTIFIC_JUDGMENT_REQUIRED"})
    v["reason_codes"]=_terminal_string_seq(v["reason_codes"],"reason_codes",allowed=reason_allowed)
    if v["hallucination_risk"] not in HALLUCINATION_RISKS: raise ValueError("CLAIM_VERIFICATION_RISK_UNKNOWN")
    if v["numeric_assessment"] not in NUMERIC_ASSESSMENTS: raise ValueError("CLAIM_VERIFICATION_NUMERIC_ASSESSMENT_UNKNOWN")
    if v["attribution_assessment"] not in ATTRIBUTION_ASSESSMENTS: raise ValueError("CLAIM_VERIFICATION_ATTRIBUTION_ASSESSMENT_UNKNOWN")
    if v["extrapolation_assessment"] not in EXTRAPOLATION_ASSESSMENTS: raise ValueError("CLAIM_VERIFICATION_EXTRAPOLATION_ASSESSMENT_UNKNOWN")
    if v["final_correction_eligibility"] not in CORRECTION_ELIGIBILITIES: raise ValueError("CLAIM_VERIFICATION_CORRECTION_ELIGIBILITY_UNKNOWN")
    ca=v["contradiction_assessment"]
    if not isinstance(ca,Mapping) or set(ca)!={"type","evidence_ids"} or ca.get("type") not in CONTRADICTION_TYPES:
        raise ValueError("CLAIM_VERIFICATION_CONTRADICTION_ASSESSMENT_INVALID")
    ca=dict(ca); ca["evidence_ids"]=_terminal_string_seq(ca["evidence_ids"],"contradiction_assessment.evidence_ids"); v["contradiction_assessment"]=ca
    _validate_terminal_evidence_collections(v)
    eligible_ids={row["evidence_id"] for row in v["eligible_evidence"]}
    if not set(ca["evidence_ids"]).issubset(eligible_ids): raise ValueError("CLAIM_VERIFICATION_UNKNOWN_CONTRADICTION_EVIDENCE")
    tu=v["tool_usage"]
    if not isinstance(tu,Mapping): raise ValueError("CLAIM_VERIFICATION_TOOL_USAGE_INVALID")
    tu=dict(tu)
    for f in ("llm_calls","retrieval_requested","retrieval_rounds"):
        if type(tu.get(f)) is not int or tu[f] < 0: raise ValueError(f"CLAIM_VERIFICATION_TOOL_USAGE_INVALID:{f}")
    v["tool_usage"]=tu
    if type(v["decision_trace"]) not in (list,tuple) or type(v["raw_attempts"]) not in (list,tuple): raise ValueError("CLAIM_VERIFICATION_AUDIT_FIELDS_INVALID")
    # Matrices observadas en verify_claim(...) y _terminal(...).
    if v["scientific_judgment_status"] == "BLOCKED" and (v["scientific_verdict"],v["support_level"]) != ("NOT_EVALUATED","NONE"):
        raise ValueError("CLAIM_VERIFICATION_BLOCKED_VERDICT_INCOHERENT")
    if v["scientific_judgment_status"] == "NOT_REQUIRED" and (v["scientific_verdict"],v["support_level"]) != ("NOT_APPLICABLE","NONE"):
        raise ValueError("CLAIM_VERIFICATION_NOT_REQUIRED_VERDICT_INCOHERENT")
    if not v["scientific_judgment_required"] and v["scientific_judgment_status"] not in {"NOT_REQUIRED","COMPLETED"}:
        raise ValueError("CLAIM_VERIFICATION_JUDGMENT_REQUIRED_INCOHERENT")
    if v["technical_status"] != "OK" and v["scientific_judgment_status"] == "COMPLETED" and v["scientific_verdict"] not in {"NOT_APPLICABLE"}:
        # A terminal technical exhaustion may legitimately finish execution without a
        # scientific verdict.  It remains usable only as an unresolved/manual row.
        technically_unresolved = (
            v["technical_status"] in CLAIM_COMPLETED_UNRESOLVED_TECHNICAL_STATUSES
            and v["technical_status"] in v["technical_issue_codes"]
            and v["scientific_verdict"] == "NOT_EVALUATED"
            and v["support_level"] == "NONE"
            and v["manual_review_required"] is True
            and v["final_correction_eligibility"] == "MANUAL_REVIEW_REQUIRED"
        )
        if not technically_unresolved:
            raise ValueError("CLAIM_VERIFICATION_TECHNICAL_JUDGMENT_INCOHERENT")
    if v["llm_correction_recommendation"] and v["final_correction_eligibility"] in {"NO_CORRECTION_NEEDED","NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE"}:
        raise ValueError("CLAIM_VERIFICATION_RECOMMENDATION_ELIGIBILITY_INCOHERENT")
    if v["final_correction_eligibility"] == "MANUAL_REVIEW_REQUIRED" and not v["manual_review_required"]:
        raise ValueError("CLAIM_VERIFICATION_MANUAL_REVIEW_INCOHERENT")
    return v


def _validate_proposal_span(span: Any, field: str) -> dict[str, Any]:
    if not isinstance(span, Mapping):
        raise ValueError(f"CORRECTION_PROPOSAL_SPAN_INVALID:{field}")
    out = dict(span)
    for key in ("coordinate_base", "coordinate_system", "base_text_fingerprint", "text"):
        if not isinstance(out.get(key), str) or (key != "text" and not out[key].strip()):
            raise ValueError(f"CORRECTION_PROPOSAL_SPAN_INVALID:{field}")
    if type(out.get("start")) is not int or type(out.get("end")) is not int or not (0 <= out["start"] < out["end"]):
        raise ValueError(f"CORRECTION_PROPOSAL_SPAN_INVALID:{field}")
    if out["coordinate_system"] == "PYTHON_CODEPOINT_OFFSETS" and out["end"] - out["start"] != len(out["text"]):
        raise ValueError(f"CORRECTION_PROPOSAL_SPAN_LENGTH_MISMATCH:{field}")
    return out


def validate_correction_proposal_contract(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Valida integridad terminal de resultados localizados y vacíos de CorrectionProposal."""
    from dataclasses import fields
    from src.config.verification_policy_config import (
        CORRECTION_DECISIONS, CORRECTION_ACTION_TYPES, CORRECTION_PROPOSAL_STATUSES,
        CORRECTION_CHANGE_SCOPES, CORRECTION_SEMANTIC_CHANGE_LEVELS, CORRECTION_REASON_CODES,
        CORRECTION_VALIDATION_ISSUE_CODES, ATTRIBUTION_RELATIONS, CORRECTION_LOCALIZATION_METHODS,
    )
    from src.tools.verification.corrections import (
        CorrectionProposal, build_virtual_corrected_claim,
        compute_correction_proposal_fingerprint,
        compute_empty_correction_proposal_fingerprint, fingerprint_text,
    )
    if not isinstance(proposal, Mapping):
        raise ValueError("CORRECTION_PROPOSAL_CONTRACT_NOT_MAPPING")
    names = tuple(f.name for f in fields(CorrectionProposal))
    missing = [f for f in names if f not in proposal]
    if missing:
        raise ValueError("CORRECTION_PROPOSAL_FIELDS_MISSING:" + ",".join(missing))
    if set(proposal) - set(names):
        raise ValueError("CORRECTION_PROPOSAL_UNKNOWN_FIELDS")
    v = dict(proposal)
    for f in ("correction_id", "claim_id", "section_id", "original_text", "original_claim_fingerprint", "original_section_fingerprint", "target_text_fingerprint", "proposed_claim_text", "proposal_fingerprint", "prompt_version"):
        v[f] = _terminal_nonempty(v[f], f)

    if fingerprint_text(v["original_text"]) != v["original_claim_fingerprint"]:
        raise ValueError("CORRECTION_PROPOSAL_ORIGINAL_TEXT_FINGERPRINT_MISMATCH")

    if v["correction_decision"] not in CORRECTION_DECISIONS:
        raise ValueError("CORRECTION_PROPOSAL_DECISION_UNKNOWN")
    if v["action_type"] is not None and v["action_type"] not in CORRECTION_ACTION_TYPES:
        raise ValueError("CORRECTION_PROPOSAL_ACTION_UNKNOWN")
    if v["proposal_status"] not in CORRECTION_PROPOSAL_STATUSES or v["final_proposal_status"] not in CORRECTION_PROPOSAL_STATUSES:
        raise ValueError("CORRECTION_PROPOSAL_STATUS_UNKNOWN")
    if v["proposal_status"] != v["final_proposal_status"]:
        raise ValueError("CORRECTION_PROPOSAL_STATUS_INCOHERENT")
    decision_status = {
        "PROPOSE_CHANGE": {"ACCEPTED_FOR_REVERIFICATION", "REJECTED"},
        "NO_CORRECTION": {"NOT_PROPOSED"},
        "NOT_CORRECTABLE": {"NOT_PROPOSED"},
        "DEFER_TO_MANUAL_REVIEW": {"DEFERRED", "REJECTED"},
    }
    if v["proposal_status"] not in decision_status[v["correction_decision"]]:
        raise ValueError("CORRECTION_PROPOSAL_DECISION_STATUS_INCOHERENT")
    for f in ("llm_correction_recommendation", "requires_manual_review", "accepted_for_reverification", "correction_applied"):
        if type(v[f]) is not bool:
            raise ValueError(f"CORRECTION_PROPOSAL_INVALID_BOOL:{f}")
    if v["correction_applied"]:
        raise ValueError("CORRECTION_PROPOSAL_ALREADY_APPLIED")
    if v["accepted_for_reverification"] != (v["final_proposal_status"] == "ACCEPTED_FOR_REVERIFICATION"):
        raise ValueError("CORRECTION_PROPOSAL_ACCEPTANCE_STATUS_INCOHERENT")

    localized = v["correction_decision"] == "PROPOSE_CHANGE"
    if localized:
        if v["change_scope"] not in CORRECTION_CHANGE_SCOPES or v["semantic_change_level"] not in CORRECTION_SEMANTIC_CHANGE_LEVELS:
            raise ValueError("CORRECTION_PROPOSAL_CHANGE_CONTRACT_INVALID")
    elif (v["change_scope"], v["semantic_change_level"]) != ("NONE", "NONE"):
        raise ValueError("CORRECTION_PROPOSAL_EMPTY_RESULT_CHANGE_INCOHERENT")

    v["evidence_ids"] = _terminal_string_seq(v["evidence_ids"], "evidence_ids")
    v["reason_codes"] = _terminal_string_seq(v["reason_codes"], "reason_codes", allowed=set(CORRECTION_REASON_CODES))
    v["validation_issue_codes"] = _terminal_string_seq(v["validation_issue_codes"], "validation_issue_codes", allowed=set(CORRECTION_VALIDATION_ISSUE_CODES))
    rm = v["retry_metrics"]
    if not isinstance(rm, Mapping) or type(rm.get("llm_calls")) is not int or rm["llm_calls"] < 0:
        raise ValueError("CORRECTION_PROPOSAL_RETRY_METRICS_INVALID")
    if type(v["raw_attempts"]) not in (list, tuple):
        raise ValueError("CORRECTION_PROPOSAL_RAW_ATTEMPTS_INVALID")

    claim_span = v["claim_span_in_section"]
    if claim_span is not None:
        claim_span = _validate_proposal_span(claim_span, "claim_span_in_section")
        v["claim_span_in_section"] = claim_span
        if claim_span["text"] != v["original_text"]:
            raise ValueError("CORRECTION_PROPOSAL_CLAIM_SPAN_TEXT_MISMATCH")
        if claim_span["base_text_fingerprint"] != v["original_section_fingerprint"]:
            raise ValueError("CORRECTION_PROPOSAL_CLAIM_SPAN_BASE_FINGERPRINT_MISMATCH")

    if localized:
        if v["action_type"] is None:
            raise ValueError("CORRECTION_PROPOSAL_LOCALIZED_ACTION_REQUIRED")
        if claim_span is None:
            raise ValueError("CORRECTION_PROPOSAL_SPAN_INVALID:claim_span_in_section")
        target_span = _validate_proposal_span(v["target_span_in_claim"], "target_span_in_claim")
        v["target_span_in_claim"] = target_span
        if target_span["coordinate_base"] != "CLAIM_TEXT" or target_span["coordinate_system"] != "PYTHON_CODEPOINT_OFFSETS":
            raise ValueError("CORRECTION_PROPOSAL_SPAN_INVALID:target_span_in_claim")
        if claim_span["coordinate_base"] != "SECTION_TEXT" or claim_span["coordinate_system"] != "PYTHON_CODEPOINT_OFFSETS":
            raise ValueError("CORRECTION_PROPOSAL_SPAN_INVALID:claim_span_in_section")
        if v["localization_method"] not in CORRECTION_LOCALIZATION_METHODS:
            raise ValueError("CORRECTION_PROPOSAL_LOCALIZATION_METHOD_INVALID")
        _terminal_nonempty(v["target_text"], "target_text")
        if fingerprint_text(v["target_text"]) != v["target_text_fingerprint"]:
            raise ValueError("CORRECTION_PROPOSAL_TARGET_TEXT_FINGERPRINT_MISMATCH")
        if target_span["text"] != v["target_text"]:
            raise ValueError("CORRECTION_PROPOSAL_TARGET_SPAN_TEXT_MISMATCH")
        if target_span["base_text_fingerprint"] != v["original_claim_fingerprint"]:
            raise ValueError("CORRECTION_PROPOSAL_TARGET_SPAN_BASE_FINGERPRINT_MISMATCH")
        try:
            expected_claim = build_virtual_corrected_claim(v["original_text"], target_span, v["replacement_text"])
        except ValueError as exc:
            raise ValueError("CORRECTION_PROPOSAL_PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH") from exc
        if v["proposed_claim_text"] != expected_claim:
            raise ValueError("CORRECTION_PROPOSAL_PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH")

        expected = compute_correction_proposal_fingerprint(
            original_claim_fingerprint=v["original_claim_fingerprint"],
            original_section_fingerprint=v["original_section_fingerprint"],
            target_text_fingerprint=v["target_text_fingerprint"], claim_id=v["claim_id"],
            action_type=v["action_type"], target_span=target_span,
            replacement_text=v["replacement_text"], evidence_ids=v["evidence_ids"],
            prompt_version=v["prompt_version"],
        )
        action = v["action_type"]
        if action == "REPLACE_NUMERIC_VALUE" and not v["new_numeric_pairs"]:
            raise ValueError("CORRECTION_PROPOSAL_ACTION_FIELDS_INCOMPATIBLE")
        if action == "CORRECT_ATTRIBUTION" and (not v["new_attribution_elements"] or v["attribution_relation"] not in ATTRIBUTION_RELATIONS):
            raise ValueError("CORRECTION_PROPOSAL_ACTION_FIELDS_INCOMPATIBLE")
        if action == "REPLACE_CITATION" and (not v["new_citation_refs"] or not isinstance(v["citation_text_span"], Mapping)):
            raise ValueError("CORRECTION_PROPOSAL_ACTION_FIELDS_INCOMPATIBLE")
        if action in {"NARROW_SCOPE", "ADD_QUALIFICATION"} and not v["new_conditions"]:
            raise ValueError("CORRECTION_PROPOSAL_ACTION_FIELDS_INCOMPATIBLE")
    else:
        if v["action_type"] is not None or v["target_span_in_claim"] is not None or v["localization_method"] is not None:
            raise ValueError("CORRECTION_PROPOSAL_EMPTY_RESULT_LOCALIZATION_INCOHERENT")
        if v["target_text"] != "" or v["replacement_text"] != "" or v["evidence_ids"] != ():
            raise ValueError("CORRECTION_PROPOSAL_EMPTY_RESULT_CONTENT_INCOHERENT")
        if v["target_text_fingerprint"] != fingerprint_text(""):
            raise ValueError("CORRECTION_PROPOSAL_TARGET_TEXT_FINGERPRINT_MISMATCH")
        if v["proposed_claim_text"] != v["original_text"]:
            raise ValueError("CORRECTION_PROPOSAL_PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH")
        if v["change_scope"] != "NONE" or v["semantic_change_level"] != "NONE":
            raise ValueError("CORRECTION_PROPOSAL_EMPTY_RESULT_CHANGE_INCOHERENT")
        expected = compute_empty_correction_proposal_fingerprint(
            claim_id=v["claim_id"], decision=v["correction_decision"],
            status=v["proposal_status"], prompt_version=v["prompt_version"],
        )
    if expected != v["proposal_fingerprint"]:
        raise ValueError("CORRECTION_PROPOSAL_FINGERPRINT_MISMATCH")
    return v

def validate_claim_verification_aggregation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record,Mapping): raise ValueError("CLAIM_AGGREGATION_RECORD_NOT_MAPPING")
    if set(record)!={"section_id","claim_verification_result"}: raise ValueError("CLAIM_AGGREGATION_RECORD_SCHEMA_INVALID")
    section_id=_terminal_nonempty(record.get("section_id"),"section_id")
    result=validate_claim_verification_result_contract(record.get("claim_verification_result"))
    return {"section_id":section_id,"claim_verification_result":result}

# Phase 6.5.1A: structural-only validators. No collection-item validation, joins or metrics computation.
def _phase651a_exact_dataclass_mapping(value: Mapping[str, Any], cls: type, code: str) -> dict[str, Any]:
    from dataclasses import fields
    if not isinstance(value, Mapping):
        raise ValueError(f"{code}:NOT_MAPPING")
    expected = tuple(field.name for field in fields(cls))
    missing = [name for name in expected if name not in value]
    if missing:
        raise ValueError(f"{code}:FIELDS_MISSING:" + ",".join(missing))
    unknown = sorted(set(value) - set(expected))
    if unknown:
        raise ValueError(f"{code}:UNKNOWN_FIELDS:" + ",".join(unknown))
    return dict(value)


def _phase651a_nonempty(value: Any, field: str, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{code}:{field}:NONEMPTY_STRING_REQUIRED")
    return value.strip()


def _phase651a_string_tuple(value: Any, field: str, code: str) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{code}:{field}:SEQUENCE_REQUIRED")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{code}:{field}:NONEMPTY_STRINGS_REQUIRED")
    return result


def _phase651a_mapping(value: Any, field: str, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{code}:{field}:MAPPING_REQUIRED")
    return dict(value)


def _phase651a_bool(value: Any, field: str, code: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{code}:{field}:BOOLEAN_REQUIRED")
    return value


def _phase651a_nonnegative_int(value: Any, field: str, code: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{code}:{field}:NONNEGATIVE_INTEGER_REQUIRED")
    return value










def _validate_evidence_row_common(result: dict[str, Any], code: str, support_field: str) -> dict[str, Any]:
    from src.config.verification_policy_config import EVIDENCE_SUPPORT_STATUSES
    for field in ("claim_id", "section_id", "evidence_id", "source_filename", "chunk_id", "usage_role"):
        result[field] = _phase651a_nonempty(result[field], field, code)
    result["authorized_for_section"] = _phase651a_bool(result["authorized_for_section"], "authorized_for_section", code)
    if result[support_field] not in EVIDENCE_SUPPORT_STATUSES:
        raise ValueError(f"{code}:{support_field}:UNKNOWN")
    return result


def validate_claim_evidence_traceability_row_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    from src.tools.verification.traceability import ClaimEvidenceTraceabilityRow
    code = "CLAIM_EVIDENCE_TRACEABILITY_ROW_INVALID"
    result = _phase651a_exact_dataclass_mapping(value, ClaimEvidenceTraceabilityRow, code)
    _validate_evidence_row_common(result, code, "supports_original_claim")
    validate_sha256_hex(result["text_fingerprint"], field="text_fingerprint")
    result["used_in_original_verification"] = _phase651a_bool(result["used_in_original_verification"], "used_in_original_verification", code)
    return result


def validate_correction_evidence_traceability_row_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    from src.tools.verification.traceability import CorrectionEvidenceTraceabilityRow
    code = "CORRECTION_EVIDENCE_TRACEABILITY_ROW_INVALID"
    result = _phase651a_exact_dataclass_mapping(value, CorrectionEvidenceTraceabilityRow, code)
    _validate_evidence_row_common(result, code, "supports_proposed_claim")
    result["correction_id"] = _phase651a_nonempty(result["correction_id"], "correction_id", code)
    result["frozen_evidence_snapshot_fingerprint"] = _phase651a_nonempty(result["frozen_evidence_snapshot_fingerprint"], "frozen_evidence_snapshot_fingerprint", code)
    result["used_in_correction"] = _phase651a_bool(result["used_in_correction"], "used_in_correction", code)
    result["used_in_reverification"] = _phase651a_bool(result["used_in_reverification"], "used_in_reverification", code)
    return result








# Phase 6.5.1AR: closed-enum and structural-coherence validators.
def _phase651ar_sha256_or_none(value: Any, field: str, code: str, *, allow_none: bool = True) -> str | None:
    import re
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{code}:{field}:SHA256_REQUIRED")
    return value


def _phase651ar_optional_enum(value: Any, allowed: tuple[str, ...], field: str, code: str) -> str | None:
    if value is None:
        return None
    if value not in allowed:
        raise ValueError(f"{code}:{field}:UNKNOWN")
    return value


def _phase651ar_nonempty_version_mapping(value: Any, field: str, code: str) -> dict[str, str]:
    result = _phase651a_mapping(value, field, code)
    for key, item in result.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(item, str) or not item.strip():
            raise ValueError(f"{code}:{field}:NONEMPTY_STRING_MAPPING_REQUIRED")
    return result




def validate_metric_value_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    import math
    from src.config.verification_policy_config import METRIC_COMPUTATION_STATUSES
    from src.tools.verification.traceability import MetricValue
    code = "METRIC_VALUE_INVALID"
    result = _phase651a_exact_dataclass_mapping(value, MetricValue, code)
    numerator = _phase651a_nonnegative_int(result["numerator"], "numerator", code)
    denominator = _phase651a_nonnegative_int(result["denominator"], "denominator", code)
    if numerator > denominator:
        raise ValueError(f"{code}:numerator:EXCEEDS_DENOMINATOR")
    status = result["status"]
    if status not in METRIC_COMPUTATION_STATUSES:
        raise ValueError(f"{code}:status:UNKNOWN")
    _phase651a_nonempty(result["unit_definition"], "unit_definition", code)
    _phase651a_nonempty(result["population_filter"], "population_filter", code)
    if denominator > 0:
        if status != "COMPUTED": raise ValueError(f"{code}:status:MUST_BE_COMPUTED")
        if type(result["value"]) not in (int, float) or isinstance(result["value"], bool):
            raise ValueError(f"{code}:value:NUMBER_REQUIRED")
        numeric = float(result["value"])
        if not math.isfinite(numeric): raise ValueError(f"{code}:value:FINITE_REQUIRED")
        if not 0.0 <= numeric <= 1.0: raise ValueError(f"{code}:value:RATE_RANGE")
        expected = numerator / denominator
        if abs(numeric - expected) > 1e-12: raise ValueError(f"{code}:value:RATIO_MISMATCH")
        result["value"] = numeric
    else:
        if status != "NOT_COMPUTABLE": raise ValueError(f"{code}:status:MUST_BE_NOT_COMPUTABLE")
        if result["value"] is not None: raise ValueError(f"{code}:value:MUST_BE_NULL")
    return result




def validate_correction_traceability_row_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    from src.config.verification_policy_config import (
        TRACE_STAGE_AVAILABILITIES, AGGREGATION_SCIENTIFIC_ACTION_TYPES,
        AGGREGATION_GATE_ACTION_NOT_AVAILABLE, CORRECTION_PROPOSAL_STATUSES,
        AGGREGATION_PRECHECK_STATUSES, REVERIFICATION_EXECUTION_STATUSES,
        REVERIFICATION_ACCEPTANCE_DECISIONS, HALLUCINATION_RISKS,
        REVERIFICATION_RISK_DELTAS, AGGREGATION_GATE_CLASSIFICATIONS,
    )
    from src.tools.verification.traceability import CorrectionTraceabilityRow
    code = "CORRECTION_TRACEABILITY_ROW_INVALID"
    result = _phase651a_exact_dataclass_mapping(value, CorrectionTraceabilityRow, code)
    for field in ("correction_id", "claim_id", "section_id", "action_type"):
        result[field] = _phase651a_nonempty(result[field], field, code)
    for field in ("is_scientific_correction_action", "is_gate_result", "manual_review_required", "correction_applied"):
        result[field] = _phase651a_bool(result[field], field, code)
    availability_fields = ("proposal_stage_availability", "precheck_stage_availability", "reverification_stage_availability", "comparison_stage_availability")
    for field in availability_fields:
        if result[field] not in TRACE_STAGE_AVAILABILITIES: raise ValueError(f"{code}:{field}:UNKNOWN")
    state_specs = (
        ("proposal_stage_availability", "proposal_status", CORRECTION_PROPOSAL_STATUSES),
        ("precheck_stage_availability", "precheck_status", AGGREGATION_PRECHECK_STATUSES),
        ("reverification_stage_availability", "reverification_execution_status", REVERIFICATION_EXECUTION_STATUSES),
        ("comparison_stage_availability", "acceptance_decision", REVERIFICATION_ACCEPTANCE_DECISIONS),
    )
    for availability, state, allowed in state_specs:
        available = result[availability] == "AVAILABLE"
        if available and result[state] is None: raise ValueError(f"{code}:{state}:REQUIRED_WHEN_AVAILABLE")
        if not available and result[state] is not None: raise ValueError(f"{code}:{state}:MUST_BE_NULL_WHEN_UNAVAILABLE")
        result[state] = _phase651ar_optional_enum(result[state], allowed, state, code)
    stages = [result[f] for f in availability_fields]
    for index, stage in enumerate(stages):
        if stage == "BLOCKED_UPSTREAM" and any(later == "AVAILABLE" for later in stages[index + 1:]):
            raise ValueError(f"{code}:UPSTREAM_BLOCKED:LATER_STAGE_AVAILABLE")
    for field in ("target_issue_codes", "resolved_issue_codes", "remaining_issue_codes", "new_issue_codes", "precheck_reason_codes", "precheck_technical_issue_codes", "comparison_reason_codes", "comparison_technical_issue_codes"):
        result[field] = _phase651a_string_tuple(result[field], field, code)
    comparison_available = result["comparison_stage_availability"] == "AVAILABLE"
    for field in ("hallucination_risk_before", "hallucination_risk_after"):
        if comparison_available and result[field] is None: raise ValueError(f"{code}:{field}:REQUIRED_WHEN_COMPARISON_AVAILABLE")
        if not comparison_available and result[field] is not None: raise ValueError(f"{code}:{field}:MUST_BE_NULL_WITHOUT_COMPARISON")
        result[field] = _phase651ar_optional_enum(result[field], HALLUCINATION_RISKS, field, code)
    if comparison_available and result["hallucination_risk_delta"] is None: raise ValueError(f"{code}:hallucination_risk_delta:REQUIRED_WHEN_COMPARISON_AVAILABLE")
    if not comparison_available and result["hallucination_risk_delta"] is not None: raise ValueError(f"{code}:hallucination_risk_delta:MUST_BE_NULL_WITHOUT_COMPARISON")
    result["hallucination_risk_delta"] = _phase651ar_optional_enum(result["hallucination_risk_delta"], REVERIFICATION_RISK_DELTAS, "hallucination_risk_delta", code)
    action = result["action_type"]
    if action == AGGREGATION_GATE_ACTION_NOT_AVAILABLE:
        if result["is_scientific_correction_action"] or not result["is_gate_result"]: raise ValueError(f"{code}:NOT_AVAILABLE:GATE_ONLY")
        if result["acceptance_decision"] == "ACCEPT_FOR_07C": raise ValueError(f"{code}:NOT_AVAILABLE:CANNOT_ACCEPT")
    elif action in AGGREGATION_SCIENTIFIC_ACTION_TYPES:
        if not result["is_scientific_correction_action"] or result["is_gate_result"]:
            raise ValueError(f"{code}:SCIENTIFIC_ACTION:FLAGS_INCOHERENT")
    else: raise ValueError(f"{code}:action_type:UNKNOWN")
    if result["is_gate_result"]:
        if result["gate_classification"] not in AGGREGATION_GATE_CLASSIFICATIONS: raise ValueError(f"{code}:gate_classification:REQUIRED_OR_UNKNOWN")
    elif result["gate_classification"] is not None:
        raise ValueError(f"{code}:gate_classification:GATE_ONLY")
    if result["correction_applied"]: raise ValueError(f"{code}:correction_applied:MUST_BE_FALSE")
    return result


def validate_reverification_traceability_row_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    from src.config.verification_policy_config import (
        REVERIFICATION_EXECUTION_STATUSES, REVERIFICATION_ACCEPTANCE_DECISIONS,
        REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES,
    )
    from src.tools.verification.traceability import ReverificationTraceabilityRow
    code = "REVERIFICATION_TRACEABILITY_ROW_INVALID"
    result = _phase651a_exact_dataclass_mapping(value, ReverificationTraceabilityRow, code)
    for field in ("correction_id", "claim_id", "section_id", "prompt_version", "proposal_fingerprint", "virtual_proposed_claim_text_fingerprint", "frozen_evidence_snapshot_fingerprint", "reverification_context_fingerprint"):
        result[field] = _phase651a_nonempty(result[field], field, code)
    if result["reverification_execution_status"] not in REVERIFICATION_EXECUTION_STATUSES: raise ValueError(f"{code}:reverification_execution_status:UNKNOWN")
    result["acceptance_decision"] = _phase651ar_optional_enum(result["acceptance_decision"], REVERIFICATION_ACCEPTANCE_DECISIONS, "acceptance_decision", code)
    for field in ("reverification_llm_calls", "format_attempts", "format_retries", "schema_attempts", "schema_retries"):
        result[field] = _phase651a_nonnegative_int(result[field], field, code)
    if result["format_retries"] > result["format_attempts"]: raise ValueError(f"{code}:format_retries:EXCEEDS_ATTEMPTS")
    if result["schema_retries"] > result["schema_attempts"]: raise ValueError(f"{code}:schema_retries:EXCEEDS_ATTEMPTS")
    result["evidence_ids_used"] = _phase651a_string_tuple(result["evidence_ids_used"], "evidence_ids_used", code)
    for field in ("observed_issue_codes", "target_issues_resolved_reported"):
        result[field] = _phase651a_string_tuple(result[field], field, code)
        if set(result[field]) - set(REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES): raise ValueError(f"{code}:{field}:UNKNOWN")
    for field in ("reported_resolution_matches", "manual_review_recommended", "correction_applied"):
        result[field] = _phase651a_bool(result[field], field, code)
    if result["reverification_execution_status"] in {"FAILED", "BLOCKED"} and result["acceptance_decision"] == "ACCEPT_FOR_07C":
        raise ValueError(f"{code}:FAILED_OR_BLOCKED:CANNOT_ACCEPT")
    if result["correction_applied"]: raise ValueError(f"{code}:correction_applied:MUST_BE_FALSE")
    return result


def validate_provisional_verification_metrics_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    from dataclasses import fields
    from src.tools.verification.traceability import ProvisionalVerificationMetrics
    code = "PROVISIONAL_VERIFICATION_METRICS_INVALID"
    result = _phase651a_exact_dataclass_mapping(value, ProvisionalVerificationMetrics, code)
    metric_fields = {
        "candidate_issue_resolution_rate", "accepted_issue_resolution_rate", "correction_acceptance_rate",
        "new_issue_rate", "hallucination_risk_reduction_rate", "recommendations_generated",
    }
    for field in fields(ProvisionalVerificationMetrics):
        name = field.name
        if name in metric_fields:
            if result[name] is not None: result[name] = validate_metric_value_contract(result[name])
        else:
            result[name] = _phase651a_nonnegative_int(result[name], name, code)
    recommendations = result["recommendations_generated"]
    if recommendations is not None and not (recommendations["status"] == "NOT_COMPUTABLE" and recommendations["value"] is None and recommendations["denominator"] == 0):
        raise ValueError(f"{code}:recommendations_generated:MUST_BE_NOT_COMPUTABLE")
    return result


def validate_provisional_verification_traceability_bundle_contract(value: Mapping[str, Any], *, allow_unvalidated_flag: bool = False) -> dict[str, Any]:
    from src.config.verification_policy_config import (
        AGGREGATION_STATUSES, METRICS_STATUSES, NORMALIZED_BUNDLE_STATUSES,
        AGGREGATION_PARTIAL_REASON_CODES,
    )
    from src.tools.verification.traceability import ProvisionalVerificationTraceabilityBundle
    code = "PROVISIONAL_TRACEABILITY_BUNDLE_INVALID"
    result = _phase651a_exact_dataclass_mapping(value, ProvisionalVerificationTraceabilityBundle, code)
    for field in ("claim_traceability_rows", "correction_traceability_rows", "claim_evidence_traceability_rows", "correction_evidence_traceability_rows", "reverification_traceability_rows"):
        if type(result[field]) not in (list, tuple): raise ValueError(f"{code}:{field}:SEQUENCE_REQUIRED")
        result[field] = tuple(result[field])
    result["metrics"] = validate_provisional_verification_metrics_contract(result["metrics"])
    if result["aggregation_status"] not in AGGREGATION_STATUSES: raise ValueError(f"{code}:aggregation_status:UNKNOWN")
    if result["metrics_status"] not in METRICS_STATUSES: raise ValueError(f"{code}:metrics_status:UNKNOWN")
    if result["normalized_bundle_status"] not in NORMALIZED_BUNDLE_STATUSES: raise ValueError(f"{code}:normalized_bundle_status:UNKNOWN")
    result["partial_reason_codes"] = _phase651a_string_tuple(result["partial_reason_codes"], "partial_reason_codes", code)
    if any(item not in AGGREGATION_PARTIAL_REASON_CODES for item in result["partial_reason_codes"]): raise ValueError(f"{code}:partial_reason_codes:UNKNOWN")
    for field in ("aggregation_issue_codes", "aggregation_warnings"):
        result[field] = _phase651a_string_tuple(result[field], field, code)
    result["policy_versions"] = _phase651ar_nonempty_version_mapping(result["policy_versions"], "policy_versions", code)
    result["schema_versions"] = _phase651ar_nonempty_version_mapping(result["schema_versions"], "schema_versions", code)
    input_fps = _phase651a_mapping(result["input_collection_fingerprints"], "input_collection_fingerprints", code)
    result["input_collection_fingerprints"] = {str(k): _phase651ar_sha256_or_none(v, f"input_collection_fingerprints.{k}", code) for k, v in input_fps.items() if isinstance(k, str) and k.strip()}
    if len(result["input_collection_fingerprints"]) != len(input_fps): raise ValueError(f"{code}:input_collection_fingerprints:EMPTY_KEY")
    for field in ("correction_applied", "official_artifacts_created"):
        result[field] = _phase651a_bool(result[field], field, code)
        if result[field]: raise ValueError(f"{code}:{field}:MUST_BE_FALSE")
    for field in ("additional_llm_calls", "additional_retrieval_rounds"):
        if _phase651a_nonnegative_int(result[field], field, code) != 0: raise ValueError(f"{code}:{field}:MUST_BE_ZERO")
    normalized_status = result["normalized_bundle_status"]
    if normalized_status == "COMPUTED":
        result["normalized_bundle_fingerprint"] = _phase651ar_sha256_or_none(result["normalized_bundle_fingerprint"], "normalized_bundle_fingerprint", code, allow_none=False)
    else:
        if result["normalized_bundle_fingerprint"] is not None: raise ValueError(f"{code}:normalized_bundle_fingerprint:MUST_BE_NULL")
    result["aggregation_audit_fingerprint"] = _phase651ar_sha256_or_none(result["aggregation_audit_fingerprint"], "aggregation_audit_fingerprint", code)
    if result["aggregation_status"] == "INVALID":
        if result["metrics_status"] != "NOT_COMPUTED": raise ValueError(f"{code}:INVALID:METRICS_MUST_NOT_BE_COMPUTED")
        if normalized_status != "NOT_COMPUTABLE": raise ValueError(f"{code}:INVALID:NORMALIZED_STATUS")
    if result["aggregation_status"] == "PARTIAL" and not result["partial_reason_codes"]: raise ValueError(f"{code}:PARTIAL:REASON_REQUIRED")
    if result["aggregation_status"] != "PARTIAL" and result["partial_reason_codes"]: raise ValueError(f"{code}:partial_reason_codes:ONLY_FOR_PARTIAL")
    rate_names = ("candidate_issue_resolution_rate", "accepted_issue_resolution_rate", "correction_acceptance_rate", "new_issue_rate", "hallucination_risk_reduction_rate")
    rates = [result["metrics"][name] for name in rate_names]
    if result["metrics_status"] == "COMPUTED" and any(rate is None or rate["status"] != "COMPUTED" for rate in rates):
        raise ValueError(f"{code}:metrics_status:COMPUTED_REQUIRES_COMPUTED_RATES")
    if result["metrics_status"] == "PARTIALLY_COMPUTED":
        if result["aggregation_status"] != "PARTIAL" or not result["partial_reason_codes"]:
            raise ValueError(f"{code}:metrics_status:PARTIAL_REQUIRES_AUDIT_CAUSE")
    if result["metrics_status"] == "NOT_COMPUTED" and any(rate is not None and rate["status"] == "COMPUTED" for rate in rates):
        raise ValueError(f"{code}:metrics_status:NOT_COMPUTED_FORBIDS_COMPUTED_RATES")
    valid_flag = _phase651a_bool(result["result_contract_valid"], "result_contract_valid", code)
    if not allow_unvalidated_flag and not valid_flag: raise ValueError(f"{code}:result_contract_valid:MUST_BE_DERIVED_TRUE")
    return result



# Phase 6.5.2: structural closure + collection validation/deduplication.
def _phase652_validate_stage_causality(row: Mapping[str, Any], code: str) -> None:
    stages = (
        row["proposal_stage_availability"],
        row["precheck_stage_availability"],
        row["reverification_stage_availability"],
        row["comparison_stage_availability"],
    )
    allowed_next = {
        "AVAILABLE": {"AVAILABLE", "NOT_PRODUCED", "NOT_APPLICABLE", "BLOCKED_UPSTREAM", "FAILED"},
        "NOT_APPLICABLE": {"NOT_APPLICABLE", "BLOCKED_UPSTREAM"},
        "NOT_PRODUCED": {"NOT_PRODUCED", "BLOCKED_UPSTREAM"},
        "FAILED": {"BLOCKED_UPSTREAM"},
        "BLOCKED_UPSTREAM": {"BLOCKED_UPSTREAM"},
    }
    for index, (current, following) in enumerate(zip(stages, stages[1:])):
        if following not in allowed_next[current]:
            raise ValueError(f"{code}:stage_causality:{index}:{current}->{following}:FORBIDDEN")


_validate_correction_traceability_row_contract_phase651ar = validate_correction_traceability_row_contract

def validate_correction_traceability_row_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _validate_correction_traceability_row_contract_phase651ar(value)
    _phase652_validate_stage_causality(result, "CORRECTION_TRACEABILITY_ROW_INVALID")
    return result


_validate_provisional_verification_traceability_bundle_contract_phase651ar = validate_provisional_verification_traceability_bundle_contract

def validate_provisional_verification_traceability_bundle_contract(value: Mapping[str, Any], *, allow_unvalidated_flag: bool = False) -> dict[str, Any]:
    # Re-run the established validator with a temporary compatibility view so
    # COMPUTED means all mandatory rate contracts are present, including
    # legitimate NOT_COMPUTABLE rates with denominator zero.
    raw = dict(value)
    metrics = dict(raw.get("metrics") or {})
    status = raw.get("metrics_status")
    rate_names = (
        "candidate_issue_resolution_rate", "accepted_issue_resolution_rate",
        "correction_acceptance_rate", "new_issue_rate",
        "hallucination_risk_reduction_rate",
    )
    if status == "COMPUTED":
        missing = [name for name in rate_names if metrics.get(name) is None]
        if missing:
            raise ValueError("PROVISIONAL_TRACEABILITY_BUNDLE_INVALID:metrics_status:COMPUTED_REQUIRES_COMPUTED_RATES")
        # The previous validator required every rate status to be COMPUTED.
        # Feed it a compatibility copy only; return validation is performed on
        # the original payload below.
        compatible = dict(raw)
        compatible_metrics = dict(metrics)
        for name in rate_names:
            rate = dict(compatible_metrics[name])
            if rate.get("status") == "NOT_COMPUTABLE" and rate.get("denominator") == 0 and rate.get("value") is None:
                rate.update({"status": "COMPUTED", "denominator": 1, "numerator": 0, "value": 0.0})
            compatible_metrics[name] = rate
        compatible["metrics"] = compatible_metrics
        _validate_provisional_verification_traceability_bundle_contract_phase651ar(
            compatible, allow_unvalidated_flag=allow_unvalidated_flag
        )
        result = _phase651a_exact_dataclass_mapping(raw, __import__(
            "src.tools.verification.traceability", fromlist=["ProvisionalVerificationTraceabilityBundle"]
        ).ProvisionalVerificationTraceabilityBundle, "PROVISIONAL_TRACEABILITY_BUNDLE_INVALID")
        result["metrics"] = validate_provisional_verification_metrics_contract(metrics)
        # Reuse original validation for every non-metric field by restoring its normalized values.
        normalized_compat = _validate_provisional_verification_traceability_bundle_contract_phase651ar(
            compatible, allow_unvalidated_flag=allow_unvalidated_flag
        )
        for key, val in normalized_compat.items():
            if key != "metrics": result[key] = val
        return result
    return _validate_provisional_verification_traceability_bundle_contract_phase651ar(
        raw, allow_unvalidated_flag=allow_unvalidated_flag
    )


def create_provisional_verification_traceability_bundle(**kwargs: Any):
    from src.tools.verification.traceability import ProvisionalVerificationTraceabilityBundle
    provisional_payload = dict(kwargs)
    provisional_payload["result_contract_valid"] = False
    normalized = validate_provisional_verification_traceability_bundle_contract(
        provisional_payload, allow_unvalidated_flag=True
    )
    final_payload = dict(normalized)
    final_payload["result_contract_valid"] = True
    final = ProvisionalVerificationTraceabilityBundle(**final_payload)
    validate_provisional_verification_traceability_bundle_contract(final.to_dict())
    return final


def _phase652_plain_mapping(value: Any) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, Mapping): return dict(mapped)
    raise ValueError("AGGREGATION_COLLECTION_ELEMENT_INVALID:MAPPING_REQUIRED")


def _phase652_normalize_json_value(value: Any) -> Any:
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value): value = asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _phase652_normalize_json_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return tuple(_phase652_normalize_json_value(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"AGGREGATION_COLLECTION_ELEMENT_INVALID:NON_JSON_VALUE:{type(value).__name__}")


def _phase652_canonical_json(value: Any) -> str:
    import json
    normalized = _phase652_normalize_json_value(value)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)









# Phase 6.5.2 context-free collection adapter for the official reverification validator.
def _phase652_validate_independent_reverification_collection_item(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the official terminal validator without performing cross-collection joins.

    The temporary context is derived solely from the result's closed fields. It
    validates the terminal schema and scientific matrices; evidence authorization,
    target provenance and action identity remain reserved for the join phase.
    """
    from src.config.verification_policy_config import (
        REVERIFICATION_ACTION_ASSESSMENT_FIELD, get_verification_input_policy,
    )
    mapped = _phase652_plain_mapping(value)
    assessment_by_action = {field: action for action, field in REVERIFICATION_ACTION_ASSESSMENT_FIELD.items()}
    applicable_fields = [
        field for field in ("scope_assessment", "numeric_assessment", "attribution_assessment", "citation_assessment")
        if mapped.get(field) != "NOT_APPLICABLE"
    ]
    if len(applicable_fields) != 1:
        raise ValueError("REVERIFICATION_RESULT_CONTRACT_INVALID:ACTION_ASSESSMENT_CONTEXT_UNRESOLVED")
    action = assessment_by_action.get(applicable_fields[0])
    if action is None:
        raise ValueError("REVERIFICATION_RESULT_CONTRACT_INVALID:ACTION_ASSESSMENT_CONTEXT_UNRESOLVED")
    context = {
        "correction_id": mapped.get("correction_id"),
        "claim_id": mapped.get("claim_id"),
        "allowed_evidence_ids": tuple(mapped.get("evidence_ids_used") or ()),
        "target_issue_codes": tuple(mapped.get("target_issues_resolved_reported") or ()),
        "correction_action_type": action,
        "policy": get_verification_input_policy(),
    }
    return validate_correction_independent_reverification_result_contract(mapped, context=context)



# Phase 6.5.3 overrides and referential integrity.
def _phase653_allowed_collection_issue(code: str) -> bool:
    from src.config.verification_policy_config import AGGREGATION_COLLECTION_ISSUE_CODES
    if code in AGGREGATION_COLLECTION_ISSUE_CODES:
        return True
    if code.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID:"):
        parts = code.split(":")
        return len(parts) == 3 and parts[1] in {
            "claim_verification_records", "correction_proposals", "correction_reverification_inputs", "correction_precheck_results",
            "independent_reverification_results", "before_after_comparison_results",
        } and parts[2].isdigit()
    return False


def _phase653_validate_duplicate_record(value: Any) -> dict[str, Any]:
    from src.config.verification_policy_config import AGGREGATION_COLLECTION_NAMES, AGGREGATION_DUPLICATE_TYPES
    row = _phase651a_mapping(value, "duplicate_record", "PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID")
    common = {"collection", "primary_key", "duplicate_type"}
    dtype = row.get("duplicate_type")
    if dtype not in AGGREGATION_DUPLICATE_TYPES:
        raise ValueError("PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:duplicate_type:UNKNOWN")
    if row.get("collection") not in AGGREGATION_COLLECTION_NAMES or not str(row.get("primary_key") or ""):
        raise ValueError("PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:duplicate_record:IDENTITY_INVALID")
    expected = common | ({"duplicate_count", "canonical_record"} if dtype == "IDENTICAL" else {"conflicting_records"})
    if set(row) != expected:
        raise ValueError("PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:duplicate_record:SCHEMA_MISMATCH")
    out = {k: _phase652_normalize_json_value(v) for k, v in row.items()}
    if dtype == "IDENTICAL":
        if type(row["duplicate_count"]) is not int or row["duplicate_count"] < 2:
            raise ValueError("PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:duplicate_count:INVALID")
        if not isinstance(row["canonical_record"], Mapping):
            raise ValueError("PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:canonical_record:MAPPING_REQUIRED")
    else:
        if type(row["conflicting_records"]) not in (list, tuple) or len(row["conflicting_records"]) < 2:
            raise ValueError("PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:conflicting_records:INVALID")
        records = tuple(_phase652_normalize_json_value(x) for x in row["conflicting_records"])
        if tuple(sorted(records, key=_phase652_canonical_json)) != records or len({_phase652_canonical_json(x) for x in records}) < 2:
            raise ValueError("PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:conflicting_records:NOT_CANONICAL")
        out["conflicting_records"] = records
    return out


def _phase653_validate_indexes(result: dict[str, Any]) -> None:
    specs = {
        "claim_verification_records": ("normalized_claim_verification_records", lambda x: x["claim_verification_result"]["claim_id"]),
        "correction_proposals": ("normalized_correction_proposals", lambda x: x["correction_id"]),
        "correction_precheck_results": ("normalized_correction_precheck_results", lambda x: x["correction_id"]),
        "independent_reverification_results": ("normalized_independent_reverification_results", lambda x: x["correction_id"]),
        "before_after_comparison_results": ("normalized_before_after_comparison_results", lambda x: x["correction_id"]),
    }
    for collection, (field, getter) in specs.items():
        expected = {str(getter(row)): row for row in result[field]}
        actual = result["primary_indexes"][collection]
        if set(actual) != set(expected):
            raise ValueError(f"PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:primary_indexes:{collection}:KEY_MISMATCH")
        for key in expected:
            if _phase652_canonical_json(actual[key]) != _phase652_canonical_json(expected[key]):
                raise ValueError(f"PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:primary_indexes:{collection}:VALUE_MISMATCH")













# Phase 6.5.4 overrides: full referential chain and provisional row construction.
def validate_provisional_verification_aggregation_input_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    from src.tools.verification.traceability import ProvisionalVerificationAggregationInput
    code="PROVISIONAL_VERIFICATION_AGGREGATION_INPUT_INVALID"
    raw=dict(value)
    raw.setdefault("correction_reverification_inputs", ())
    result=_phase651a_exact_dataclass_mapping(raw,ProvisionalVerificationAggregationInput,code)
    for field in ("claim_verification_records","correction_proposals","correction_reverification_inputs","correction_precheck_results","independent_reverification_results","before_after_comparison_results"):
        if type(result[field]) not in (list,tuple): raise ValueError(f"{code}:{field}:SEQUENCE_REQUIRED")
        result[field]=tuple(result[field])
    for field in ("policy_versions","schema_versions"):
        result[field]=_phase651a_mapping(result[field],field,code)
        for k,v in result[field].items():
            if not isinstance(k,str) or not k.strip() or not isinstance(v,str) or not v.strip(): raise ValueError(f"{code}:{field}:NONEMPTY_STRING_MAPPING_REQUIRED")
    for field in ("additional_llm_calls","additional_retrieval_rounds"):
        if type(result[field]) is not int or result[field]!=0: raise ValueError(f"{code}:{field}:MUST_BE_ZERO")
    for field in ("correction_applied","official_artifacts_created"):
        if result[field] is not False: raise ValueError(f"{code}:{field}:MUST_BE_FALSE")
    return result


def _phase652_collection_specifications():
    return (
        ("claim_verification_records", validate_claim_verification_aggregation_record, lambda x:x["claim_verification_result"]["claim_id"]),
        ("correction_proposals", validate_correction_proposal_contract, lambda x:x["correction_id"]),
        ("correction_reverification_inputs", validate_correction_reverification_input_contract, lambda x:x["correction_id"]),
        ("correction_precheck_results", validate_correction_reverification_precheck_result_contract, lambda x:x["correction_id"]),
        ("independent_reverification_results", _phase652_validate_independent_reverification_collection_item, lambda x:x["correction_id"]),
        ("before_after_comparison_results", validate_before_after_comparison_result_contract, lambda x:x["correction_id"]),
    )






def _phase654_orphan(value:Any)->dict[str,Any]:
    code="PROVISIONAL_REFERENTIAL_INTEGRITY_RESULT_INVALID:orphan_records";m=_phase651a_mapping(value,"orphan",code);allowed={"collection","primary_id","reason_code","claim_id","correction_id"}
    if set(m)-allowed or not {"collection","primary_id","reason_code"}.issubset(m): raise ValueError(f"{code}:SCHEMA")
    return _phase652_normalize_json_value(m)

def _phase654_conflict(value:Any)->dict[str,Any]:
    code="PROVISIONAL_REFERENTIAL_INTEGRITY_RESULT_INVALID:identity_conflicts";m=_phase651a_mapping(value,"conflict",code)
    if set(m)!={"reason_code","correction_id","field","observed_values"}: raise ValueError(f"{code}:SCHEMA")
    return _phase652_normalize_json_value(m)


def _phase653_validate_referential_result(value: Mapping[str, Any], *, allow_unvalidated_flag=False)->dict[str,Any]:
    from src.tools.verification.traceability import ProvisionalReferentialIntegrityResult
    from src.config.verification_policy_config import AGGREGATION_REFERENTIAL_VALIDATION_STATUSES,AGGREGATION_REFERENTIAL_ISSUE_CODES,AGGREGATION_REFERENTIAL_WARNING_CODES
    code="PROVISIONAL_REFERENTIAL_INTEGRITY_RESULT_INVALID";r=_phase651a_exact_dataclass_mapping(value,ProvisionalReferentialIntegrityResult,code)
    for f in ("joined_claim_records","joined_correction_records","rejected_join_candidates"):
        if type(r[f]) not in (list,tuple): raise ValueError(f"{code}:{f}:SEQUENCE")
        r[f]=tuple(_phase652_normalize_json_value(x) for x in r[f])
    r["orphan_records"]=tuple(_phase654_orphan(x) for x in r["orphan_records"]);r["identity_conflicts"]=tuple(_phase654_conflict(x) for x in r["identity_conflicts"])
    for f in ("referential_issue_codes","referential_warnings"):r[f]=_phase651a_string_tuple(r[f],f,code)
    if not set(r["referential_issue_codes"]).issubset(set(AGGREGATION_REFERENTIAL_ISSUE_CODES)):raise ValueError(f"{code}:ISSUE_UNKNOWN")
    if not set(r["referential_warnings"]).issubset(set(AGGREGATION_REFERENTIAL_WARNING_CODES)):raise ValueError(f"{code}:WARNING_UNKNOWN")
    if r["referential_validation_status"] not in AGGREGATION_REFERENTIAL_VALIDATION_STATUSES:raise ValueError(f"{code}:STATUS")
    if r["orphan_records"] and not any(x["reason_code"] in r["referential_issue_codes"] for x in r["orphan_records"]):raise ValueError(f"{code}:ORPHAN_WITHOUT_ISSUE")
    if r["identity_conflicts"] and not all(x["reason_code"] in r["referential_issue_codes"] for x in r["identity_conflicts"]):raise ValueError(f"{code}:CONFLICT_WITHOUT_ISSUE")
    if r["referential_validation_status"]=="VALID" and (r["orphan_records"] or r["identity_conflicts"]):raise ValueError(f"{code}:VALID_WITH_AUDIT_ERRORS")
    if r["referential_validation_status"]=="INVALID" and not (r["referential_issue_codes"] or r["identity_conflicts"] or r["orphan_records"]):raise ValueError(f"{code}:INVALID_WITHOUT_CAUSE")
    if r["referential_validation_status"]=="INVALID" and (r["aggregation_status"]!="INVALID" or r["metrics_status"]!="NOT_COMPUTED"):raise ValueError(f"{code}:INVALID_STATUS")
    flag=_phase651a_bool(r["result_contract_valid"],"result_contract_valid",code)
    if not allow_unvalidated_flag and not flag:raise ValueError(f"{code}:result_contract_valid:MUST_BE_DERIVED_TRUE")
    return r

def validate_provisional_referential_integrity_result_contract(value: Mapping[str, Any])->dict[str,Any]:
    return _phase653_validate_referential_result(value)


def validate_provisional_referential_integrity(collection_result: Any):
    from src.tools.verification.traceability import ProvisionalReferentialIntegrityResult
    from src.tools.verification.corrections import fingerprint_text
    raw=_phase652_plain_mapping(collection_result);cv=validate_provisional_collection_validation_result_contract(raw)
    empty=dict(joined_claim_records=(),joined_correction_records=(),rejected_join_candidates=(),referential_issue_codes=(),referential_warnings=(),orphan_records=(),identity_conflicts=(),referential_validation_status="INVALID",aggregation_status="INVALID",metrics_status="NOT_COMPUTED",result_contract_valid=False)
    if cv["collection_validation_status"]!="VALID":
        empty["referential_issue_codes"]=("AGGREGATION_COLLECTION_ELEMENT_INVALID",) if "AGGREGATION_COLLECTION_ELEMENT_INVALID" in __import__('src.config.verification_policy_config',fromlist=['AGGREGATION_REFERENTIAL_ISSUE_CODES']).AGGREGATION_REFERENTIAL_ISSUE_CODES else ()
        # collection invalid is already audited upstream; keep a structurally valid invalid result
        empty["referential_issue_codes"]=("AGGREGATION_UNKNOWN_CLAIM_ID",)
        n=_phase653_validate_referential_result(empty,allow_unvalidated_flag=True);n["result_contract_valid"]=True;return ProvisionalReferentialIntegrityResult(**n)
    idx=cv["primary_indexes"];claims=idx["claim_verification_records"];props=idx["correction_proposals"];inputs=idx["correction_reverification_inputs"];pcs=idx["correction_precheck_results"];revs=idx["independent_reverification_results"];comps=idx["before_after_comparison_results"]
    issues=[];warnings=[];orphans=[];conflicts=[];joined=[];rejected=[]
    def add_orphan(collection,cid,reason,claim_id=None):
        issues.append(reason);o={"collection":collection,"primary_id":cid,"reason_code":reason,"correction_id":cid};
        if claim_id:o["claim_id"]=claim_id
        orphans.append(o)
    def add_conflict(cid,reason,field,vals):
        issues.append(reason);conflicts.append({"reason_code":reason,"correction_id":cid,"field":field,"observed_values":tuple(vals)})
    byclaim={}
    for cid,p in props.items():byclaim.setdefault(p["claim_id"],[]).append(cid)
    joined_claims=tuple({"claim_id":cid,"section_id":claims[cid]["section_id"],"claim_verification_record":claims[cid],"correction_ids":tuple(sorted(byclaim.get(cid,())))} for cid in sorted(claims))
    if any(not x["correction_ids"] for x in joined_claims): warnings.append("AGGREGATION_CLAIM_WITHOUT_PROPOSAL")
    for cid in sorted(set(props)|set(inputs)|set(pcs)|set(revs)|set(comps)):
        p=props.get(cid);ri=inputs.get(cid);pc=pcs.get(cid);rv=revs.get(cid);cp=comps.get(cid);before=len(issues)
        if p is None:
            if ri:add_orphan("correction_reverification_inputs",cid,"AGGREGATION_ORPHAN_REVERIFICATION_INPUT",ri.get("claim_id"))
            if pc:add_orphan("correction_precheck_results",cid,"AGGREGATION_ORPHAN_PRECHECK_RESULT",pc.get("claim_id"))
            if rv:add_orphan("independent_reverification_results",cid,"AGGREGATION_ORPHAN_REVERIFICATION_RESULT",rv.get("claim_id"))
            if cp:add_orphan("before_after_comparison_results",cid,"AGGREGATION_ORPHAN_COMPARISON_RESULT",cp.get("claim_id"))
            continue
        claim=claims.get(p["claim_id"])
        if claim is None:add_orphan("correction_proposals",cid,"AGGREGATION_UNKNOWN_CLAIM_ID",p["claim_id"]);continue
        if claim["section_id"]!=p["section_id"]:add_conflict(cid,"AGGREGATION_SECTION_ID_MISMATCH","section_id",(claim["section_id"],p["section_id"]))
        if ri is None:
            if p.get("accepted_for_reverification") is True:warnings.append("AGGREGATION_ACCEPTED_PROPOSAL_INPUT_NOT_PRODUCED")
            elif pc:add_orphan("correction_precheck_results",cid,"AGGREGATION_PRECHECK_WITHOUT_REVERIFICATION_INPUT",pc.get("claim_id"))
        else:
            for f,code in (("claim_id","AGGREGATION_CORRECTION_ID_CLAIM_CONFLICT"),("section_id","AGGREGATION_SECTION_ID_MISMATCH"),("proposal_fingerprint","AGGREGATION_PROPOSAL_FINGERPRINT_MISMATCH")):
                if ri[f]!=p[f]:add_conflict(cid,code,f,(p[f],ri[f]))
            if ri["correction_action_type"]!=p["action_type"]:add_conflict(cid,"AGGREGATION_CORRECTION_ACTION_MISMATCH","correction_action_type",(p["action_type"],ri["correction_action_type"]))
            expected=fingerprint_text(p["proposed_claim_text"])
            if ri["proposed_claim_text_fingerprint"]!=expected:add_conflict(cid,"AGGREGATION_PROPOSED_TEXT_FINGERPRINT_MISMATCH","proposed_claim_text_fingerprint",(expected,ri["proposed_claim_text_fingerprint"]))
            source=set(claim["claim_verification_result"].get("deterministic_issue_codes",()))|set(claim["claim_verification_result"].get("semantic_issue_codes",()))
            if not set(ri["target_issue_codes"]).issubset(source):add_conflict(cid,"AGGREGATION_TARGET_ISSUE_WITHOUT_PROVENANCE","target_issue_codes",(tuple(sorted(source)),tuple(ri["target_issue_codes"])))
        if pc is not None:
            if ri is None:add_orphan("correction_precheck_results",cid,"AGGREGATION_PRECHECK_WITHOUT_REVERIFICATION_INPUT",pc.get("claim_id"))
            else:
                for left,right,code in (("claim_id","claim_id","AGGREGATION_CORRECTION_ID_CLAIM_CONFLICT"),("section_id","section_id","AGGREGATION_SECTION_ID_MISMATCH"),("proposal_fingerprint","proposal_fingerprint","AGGREGATION_PROPOSAL_FINGERPRINT_MISMATCH"),("proposed_claim_text_fingerprint","virtual_proposed_claim_text_fingerprint","AGGREGATION_PROPOSED_TEXT_FINGERPRINT_MISMATCH")):
                    if ri[left]!=pc[right]:add_conflict(cid,code,right,(ri[left],pc[right]))
                try:
                    rows=tuple(ri["authorized_evidence"]); efp=compute_frozen_evidence_snapshot_fingerprint(rows)
                    pfp=compute_reverification_policy_fingerprint(ri["policy"])
                    ctx=build_reverification_claim_context(ri,pc); cfp=compute_reverification_context_fingerprint(ctx,evidence_snapshot_fingerprint=efp,policy_fingerprint=pfp)
                    if efp!=pc["frozen_evidence_snapshot_fingerprint"]:add_conflict(cid,"AGGREGATION_EVIDENCE_SNAPSHOT_FINGERPRINT_MISMATCH","frozen_evidence_snapshot_fingerprint",(efp,pc["frozen_evidence_snapshot_fingerprint"]))
                    if pfp!=pc["reverification_policy_fingerprint"]:add_conflict(cid,"AGGREGATION_REVERIFICATION_POLICY_FINGERPRINT_MISMATCH","reverification_policy_fingerprint",(pfp,pc["reverification_policy_fingerprint"]))
                    if cfp!=pc["reverification_context_fingerprint"]:add_conflict(cid,"AGGREGATION_REVERIFICATION_CONTEXT_FINGERPRINT_MISMATCH","reverification_context_fingerprint",(cfp,pc["reverification_context_fingerprint"]))
                except Exception as exc:add_conflict(cid,"AGGREGATION_AUTHORIZED_EVIDENCE_CONTENT_MISMATCH","authorized_evidence",(type(exc).__name__,str(exc)))
        if rv is not None:
            if pc is None:add_orphan("independent_reverification_results",cid,"AGGREGATION_ORPHAN_REVERIFICATION_RESULT",rv.get("claim_id"))
            else:
                for f,code in (("claim_id","AGGREGATION_CORRECTION_ID_CLAIM_CONFLICT"),("section_id","AGGREGATION_SECTION_ID_MISMATCH"),("proposal_fingerprint","AGGREGATION_PROPOSAL_FINGERPRINT_MISMATCH"),("virtual_proposed_claim_text_fingerprint","AGGREGATION_PROPOSED_TEXT_FINGERPRINT_MISMATCH"),("frozen_evidence_snapshot_fingerprint","AGGREGATION_EVIDENCE_SNAPSHOT_FINGERPRINT_MISMATCH"),("reverification_context_fingerprint","AGGREGATION_REVERIFICATION_CONTEXT_FINGERPRINT_MISMATCH")):
                    if rv[f]!=pc[f]:add_conflict(cid,code,f,(pc[f],rv[f]))
                auth={str(e.get("evidence_id")) for e in (ri or {}).get("authorized_evidence",())}
                if not set(rv["evidence_ids_used"]).issubset(auth):add_conflict(cid,"AGGREGATION_UNAUTHORIZED_REVERIFICATION_EVIDENCE","evidence_ids_used",(tuple(sorted(auth)),tuple(rv["evidence_ids_used"])))
        if cp is not None:
            if rv is None:add_orphan("before_after_comparison_results",cid,"AGGREGATION_ORPHAN_COMPARISON_RESULT",cp.get("claim_id"))
            else:
                for f,code in (("claim_id","AGGREGATION_CORRECTION_ID_CLAIM_CONFLICT"),("section_id","AGGREGATION_SECTION_ID_MISMATCH"),("proposal_fingerprint","AGGREGATION_PROPOSAL_FINGERPRINT_MISMATCH"),("virtual_proposed_claim_text_fingerprint","AGGREGATION_PROPOSED_TEXT_FINGERPRINT_MISMATCH"),("frozen_evidence_snapshot_fingerprint","AGGREGATION_EVIDENCE_SNAPSHOT_FINGERPRINT_MISMATCH"),("reverification_context_fingerprint","AGGREGATION_REVERIFICATION_CONTEXT_FINGERPRINT_MISMATCH")):
                    if cp[f]!=rv[f]:add_conflict(cid,code,f,(rv[f],cp[f]))
                if ri and tuple(cp["target_issue_codes"])!=tuple(ri["target_issue_codes"]):add_conflict(cid,"AGGREGATION_TARGET_ISSUE_WITHOUT_PROVENANCE","target_issue_codes",(tuple(ri["target_issue_codes"]),tuple(cp["target_issue_codes"])))
                if cp["correction_action_type"]!=p["action_type"]:add_conflict(cid,"AGGREGATION_CORRECTION_ACTION_MISMATCH","correction_action_type",(p["action_type"],cp["correction_action_type"]))
        if pc is None and ri is not None:warnings.append("AGGREGATION_PROPOSAL_NOT_REVERIFIED")
        elif pc and rv is None:
            warnings.append("AGGREGATION_PRECHECK_TERMINAL_WITHOUT_REVERIFICATION" if pc["precheck_status"]!="PRECHECK_PASSED" else "AGGREGATION_PROPOSAL_NOT_REVERIFIED")
        elif rv and cp is None and rv["reverification_execution_status"] in ("FAILED","BLOCKED"):warnings.append("AGGREGATION_REVERIFICATION_TERMINAL_WITHOUT_COMPARISON")
        record={"correction_id":cid,"claim_id":p["claim_id"],"section_id":p["section_id"],"proposal":p,"reverification_input":ri,"precheck":pc,"reverification":rv,"comparison":cp,"join_status":"INVALID" if len(issues)>before else ("PARTIAL" if None in (ri,pc,rv,cp) else "VALID")}
        (rejected if len(issues)>before else joined).append(record)
    invalid=bool(issues);status="INVALID" if invalid else ("PARTIAL" if warnings else "VALID")
    payload=dict(joined_claim_records=joined_claims,joined_correction_records=tuple(sorted(joined,key=lambda x:(x["section_id"],x["claim_id"],x["correction_id"]))),rejected_join_candidates=tuple(sorted(rejected,key=_phase652_canonical_json)),referential_issue_codes=tuple(sorted(set(issues))),referential_warnings=tuple(sorted(set(warnings))),orphan_records=tuple(sorted(orphans,key=_phase652_canonical_json)),identity_conflicts=tuple(sorted(conflicts,key=_phase652_canonical_json)),referential_validation_status=status,aggregation_status="INVALID" if invalid else ("PARTIAL" if warnings else "VALID"),metrics_status="NOT_COMPUTED",result_contract_valid=False)
    n=_phase653_validate_referential_result(payload,allow_unvalidated_flag=True);n["result_contract_valid"]=True;r=ProvisionalReferentialIntegrityResult(**n);validate_provisional_referential_integrity_result_contract(r.to_dict());return r

def validate_claim_traceability_row_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    from src.config.verification_policy_config import AGGREGATION_CLAIM_TYPES,SCIENTIFIC_VERDICTS,DETERMINISTIC_ISSUE_CODES,SEMANTIC_ISSUE_CODES,HALLUCINATION_RISKS,REVERIFICATION_ACCEPTANCE_DECISIONS
    from src.tools.verification.traceability import ClaimTraceabilityRow
    code="CLAIM_TRACEABILITY_ROW_INVALID";r=_phase651a_exact_dataclass_mapping(value,ClaimTraceabilityRow,code)
    for f in ("claim_id","section_id"):r[f]=_phase651a_nonempty(r[f],f,code)
    if not isinstance(r["original_claim_text"],str):raise ValueError(f"{code}:original_claim_text:STRING_REQUIRED")
    if r["claim_type"] not in AGGREGATION_CLAIM_TYPES: raise ValueError(f"{code}:claim_type:UNKNOWN")
    if r["source_verdict"] not in SCIENTIFIC_VERDICTS: raise ValueError(f"{code}:source_verdict:UNKNOWN")
    if r["source_hallucination_risk"] not in HALLUCINATION_RISKS: raise ValueError(f"{code}:source_hallucination_risk:UNKNOWN")
    confidence=r["source_verification_confidence"]
    status=r["source_confidence_status"]
    if status not in ("AVAILABLE","NOT_AVAILABLE_IN_SOURCE_CONTRACT"):
        raise ValueError(f"{code}:source_confidence_status:UNKNOWN")
    if status=="AVAILABLE":
        if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not 0.0<=float(confidence)<=1.0:
            raise ValueError(f"{code}:source_verification_confidence:INVALID")
        r["source_verification_confidence"]=float(confidence)
    elif confidence is not None:
        raise ValueError(f"{code}:source_verification_confidence:MUST_BE_NULL")
    allowed=set(DETERMINISTIC_ISSUE_CODES)|set(SEMANTIC_ISSUE_CODES)
    for f in ("source_issue_codes","provisional_remaining_issue_codes"):r[f]=_phase651a_string_tuple(r[f],f,code)
    if set(r["source_issue_codes"]+r["provisional_remaining_issue_codes"])-allowed:raise ValueError(f"{code}:ISSUE_UNKNOWN")
    for f in ("correction_ids","individual_accepted_correction_ids","individual_rejected_correction_ids","individual_deferred_correction_ids","individual_proposal_decisions"):r[f]=_phase651a_string_tuple(r[f],f,code)
    if set(r["individual_proposal_decisions"])-set(REVERIFICATION_ACCEPTANCE_DECISIONS):raise ValueError(f"{code}:DECISION_UNKNOWN")
    for f in ("terminal_correction_recommendation","has_correction_proposal","manual_review_required","correction_applied"):r[f]=_phase651a_bool(r[f],f,code)
    if r["correction_applied"]:raise ValueError(f"{code}:APPLIED")
    ids=set(r["correction_ids"]);groups=[set(r[x]) for x in ("individual_accepted_correction_ids","individual_rejected_correction_ids","individual_deferred_correction_ids")]
    if not r["has_correction_proposal"] and (ids or r["individual_proposal_decisions"] or any(groups)):raise ValueError(f"{code}:NO_PROPOSAL:RELATED_FIELDS_MUST_BE_EMPTY")
    if r["has_correction_proposal"] and not ids:raise ValueError(f"{code}:IDS_REQUIRED")
    if any(g-ids for g in groups): raise ValueError(f"{code}:DECISION_IDS:NOT_SUBSET")
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)): raise ValueError(f"{code}:DECISION_IDS:MUST_BE_DISJOINT")
    return r


def _phase654_validate_rows_result(value:Mapping[str,Any],*,allow_unvalidated_flag=False)->dict[str,Any]:
    from src.tools.verification.traceability import ProvisionalTraceabilityRowsResult
    from src.config.verification_policy_config import AGGREGATION_ROW_BUILD_STATUSES,AGGREGATION_ROW_ISSUE_CODES,AGGREGATION_ROW_WARNING_CODES
    code="PROVISIONAL_TRACEABILITY_ROWS_RESULT_INVALID";r=_phase651a_exact_dataclass_mapping(value,ProvisionalTraceabilityRowsResult,code)
    validators=(("claim_traceability_rows",validate_claim_traceability_row_contract),("correction_traceability_rows",validate_correction_traceability_row_contract),("claim_evidence_traceability_rows",validate_claim_evidence_traceability_row_contract),("correction_evidence_traceability_rows",validate_correction_evidence_traceability_row_contract),("reverification_traceability_rows",validate_reverification_traceability_row_contract))
    for f,v in validators:
        if type(r[f]) not in (list,tuple):raise ValueError(f"{code}:{f}:SEQUENCE")
        r[f]=tuple(v(_phase652_plain_mapping(x)) for x in r[f])
    for f in ("row_issue_codes","row_warnings"):r[f]=_phase651a_string_tuple(r[f],f,code)
    if set(r["row_issue_codes"])-set(AGGREGATION_ROW_ISSUE_CODES) or set(r["row_warnings"])-set(AGGREGATION_ROW_WARNING_CODES):raise ValueError(f"{code}:CODE_UNKNOWN")
    if r["row_build_status"] not in AGGREGATION_ROW_BUILD_STATUSES:raise ValueError(f"{code}:STATUS")
    flag=_phase651a_bool(r["result_contract_valid"],"result_contract_valid",code)
    if not allow_unvalidated_flag and not flag:raise ValueError(f"{code}:result_contract_valid")
    return r





# ---------------------------------------------------------------------------
# Phase 6.5.5 -- row closure and aggregate metrics.
# ---------------------------------------------------------------------------
def _phase655_canonical_key(record: Mapping[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(record.get(f) for f in fields)


def _phase655_require_unique_sorted(rows: tuple[Mapping[str, Any], ...], fields: tuple[str, ...], code: str) -> None:
    keys=[_phase655_canonical_key(r,fields) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError(f"{code}:DUPLICATE_KEY")
    if keys!=sorted(keys): raise ValueError(f"{code}:ORDER_INVALID")


_validate_correction_traceability_row_contract_phase654 = validate_correction_traceability_row_contract

def validate_correction_traceability_row_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    raw=_phase652_plain_mapping(value)
    action=raw.get("action_type")
    if action is None:
        # Terminal empty proposal: no scientific action and no gate semantics.
        if raw.get("is_scientific_correction_action") or raw.get("is_gate_result"):
            raise ValueError("CORRECTION_TRACEABILITY_ROW_INVALID:NULL_ACTION_FLAGS")
        if raw.get("proposal_status") not in ("NOT_PROPOSED","DEFERRED","REJECTED"):
            raise ValueError("CORRECTION_TRACEABILITY_ROW_INVALID:NULL_ACTION_STATUS")
        compatible=dict(raw); compatible["action_type"]="REMOVE_UNSUPPORTED_FRAGMENT"
        compatible["is_scientific_correction_action"]=True
        normalized=_validate_correction_traceability_row_contract_phase654(compatible)
        normalized["action_type"]=None; normalized["is_scientific_correction_action"]=False
        return normalized
    return _validate_correction_traceability_row_contract_phase654(raw)


_validate_rows_result_phase654 = _phase654_validate_rows_result

def _phase655_validate_rows_result(value: Mapping[str, Any], *, allow_unvalidated_flag: bool=False) -> dict[str, Any]:
    r=_validate_rows_result_phase654(value,allow_unvalidated_flag=allow_unvalidated_flag)
    code="PROVISIONAL_TRACEABILITY_ROWS_RESULT_INVALID"
    claims=r["claim_traceability_rows"]; corrections=r["correction_traceability_rows"]
    claim_ev=r["claim_evidence_traceability_rows"]; corr_ev=r["correction_evidence_traceability_rows"]; rev=r["reverification_traceability_rows"]
    _phase655_require_unique_sorted(claims,("section_id","claim_id"),code)
    if len({x["claim_id"] for x in claims})!=len(claims): raise ValueError(f"{code}:CLAIM_ID_DUPLICATE")
    _phase655_require_unique_sorted(corrections,("section_id","claim_id","correction_id"),code)
    if len({x["correction_id"] for x in corrections})!=len(corrections): raise ValueError(f"{code}:CORRECTION_ID_DUPLICATE")
    _phase655_require_unique_sorted(claim_ev,("section_id","claim_id","evidence_id"),code)
    _phase655_require_unique_sorted(corr_ev,("section_id","claim_id","correction_id","evidence_id"),code)
    _phase655_require_unique_sorted(rev,("section_id","claim_id","correction_id"),code)
    if len({x["correction_id"] for x in rev})!=len(rev): raise ValueError(f"{code}:REVERIFICATION_DUPLICATE")
    claim_ids={x["claim_id"] for x in claims}; correction_ids={x["correction_id"] for x in corrections}
    if any(x["claim_id"] not in claim_ids for x in corrections): raise ValueError(f"{code}:CORRECTION_WITHOUT_CLAIM")
    if any(x["correction_id"] not in correction_ids for x in corr_ev): raise ValueError(f"{code}:EVIDENCE_WITHOUT_CORRECTION")
    if any(x["correction_id"] not in correction_ids for x in rev): raise ValueError(f"{code}:REVERIFICATION_WITHOUT_CORRECTION")
    invalidating=bool(r["row_issue_codes"])
    if r["row_build_status"]=="VALID" and invalidating: raise ValueError(f"{code}:VALID_WITH_ISSUES")
    if r["row_build_status"]=="INVALID":
        if not invalidating: raise ValueError(f"{code}:INVALID_WITHOUT_ISSUE")
        if r["metrics_status"]!="NOT_COMPUTED": raise ValueError(f"{code}:INVALID_METRICS")
    return r


def validate_provisional_traceability_rows_result_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    return _phase655_validate_rows_result(value)


def _phase655_metric_value(numerator:int, denominator:int, unit:str, population:str):
    from src.tools.verification.traceability import MetricValue
    if denominator==0:
        return MetricValue(None,numerator,0,"NOT_COMPUTABLE",unit,population).to_dict()
    return MetricValue(numerator/denominator,numerator,denominator,"COMPUTED",unit,population).to_dict()


def _phase655_stage_action(proposal: Mapping[str,Any], comparison: Mapping[str,Any]|None):
    if comparison and comparison.get("correction_action_type")=="NOT_AVAILABLE": return "NOT_AVAILABLE",False,True
    action=proposal.get("action_type")
    return action, bool(action), False


def build_provisional_traceability_rows(referential_result:Any):
    from src.tools.verification.traceability import ProvisionalTraceabilityRowsResult,ClaimTraceabilityRow,CorrectionTraceabilityRow,ClaimEvidenceTraceabilityRow,CorrectionEvidenceTraceabilityRow,ReverificationTraceabilityRow
    rr=validate_provisional_referential_integrity_result_contract(_phase652_plain_mapping(referential_result))
    if rr["referential_validation_status"]=="INVALID":
        payload=dict(claim_traceability_rows=(),correction_traceability_rows=(),claim_evidence_traceability_rows=(),correction_evidence_traceability_rows=(),reverification_traceability_rows=(),row_issue_codes=("AGGREGATION_ROW_CONTRACT_INVALID",),row_warnings=(),row_build_status="INVALID",aggregation_status="INVALID",metrics_status="NOT_COMPUTED",result_contract_valid=False)
        n=_phase655_validate_rows_result(payload,allow_unvalidated_flag=True);n["result_contract_valid"]=True;return ProvisionalTraceabilityRowsResult(**n)
    corrections={x["correction_id"]:x for x in rr["joined_correction_records"]}; claim_rows=[];corr_rows=[];claim_ev=[];corr_ev=[];rev_rows=[];warnings=[];issues=[];invalid_claim_ids=set()
    for jc in rr["joined_claim_records"]:
        rec=jc["claim_verification_record"];vr=rec["claim_verification_result"];cids=tuple(sorted(x for x in jc["correction_ids"] if x in corrections));linked=[corrections[x] for x in cids]
        originals={(x["proposal"].get("original_claim_fingerprint"),x["proposal"].get("original_text")) for x in linked}
        if len(originals)>1:
            issues.append("AGGREGATION_ROW_SOURCE_CLAIM_CONFLICT"); invalid_claim_ids.add(vr["claim_id"]); continue
        original=next(iter(originals))[1] if originals else ""
        if not original: warnings.append("AGGREGATION_ROW_SOURCE_CLAIM_TEXT_UNAVAILABLE")
        comparisons=[x["comparison"] for x in linked if x.get("comparison")]
        source=tuple(vr.get("deterministic_issue_codes",()))+tuple(vr.get("semantic_issue_codes",()))
        remaining=set(source) if not comparisons else set().union(*(set(x.get("remaining_issue_codes",())) for x in comparisons))
        decisions=tuple(x["acceptance_decision"] for x in comparisons)
        acc=tuple(x["correction_id"] for x in linked if x.get("comparison") and x["comparison"]["acceptance_decision"]=="ACCEPT_FOR_07C")
        rej=tuple(x["correction_id"] for x in linked if x.get("comparison") and x["comparison"]["acceptance_decision"]=="REJECT_PROPOSAL")
        deferred=tuple(x["correction_id"] for x in linked if x.get("comparison") and x["comparison"]["acceptance_decision"]=="DEFER_TO_MANUAL_REVIEW")
        manual_review = bool(vr["manual_review_required"] or deferred)
        if manual_review or vr["scientific_verdict"] == "NOT_EVALUATED":
            warnings.append("AGGREGATION_ROW_MANUAL_REVIEW_REQUIRED")
        row=ClaimTraceabilityRow(vr["claim_id"],rec["section_id"],vr["claim_type"],original,vr["scientific_verdict"],source,vr["hallucination_risk"],bool(vr["llm_correction_recommendation"]),bool(cids),cids,decisions,acc,rej,deferred,tuple(sorted(remaining)),manual_review,False,source_verification_confidence=vr.get("confidence"),source_confidence_status="AVAILABLE" if vr.get("confidence") is not None else "NOT_AVAILABLE_IN_SOURCE_CONTRACT",claim_uid=vr.get("claim_uid") or "")
        claim_rows.append(validate_claim_traceability_row_contract(row.to_dict()))
        eligible={e["evidence_id"]:e for e in vr.get("eligible_evidence",())};used={e["evidence_id"] for e in vr.get("evidence_used",())};rejected={e["evidence_id"] for e in vr.get("evidence_rejected",())}
        for eid in sorted(used|rejected):
            e=eligible.get(eid) or next((z for z in tuple(vr.get("evidence_used",()))+tuple(vr.get("evidence_rejected",())) if z["evidence_id"]==eid),None)
            if not e or not e.get("source_filename") or not e.get("chunk_id"): warnings.append("AGGREGATION_ROW_EVIDENCE_IDENTITY_UNAVAILABLE"); continue
            claim_ev.append(validate_claim_evidence_traceability_row_contract(ClaimEvidenceTraceabilityRow(vr["claim_id"],rec["section_id"],eid,e["source_filename"],e["chunk_id"],sha256(str(e.get("canonical_text",e.get("text",""))).encode("utf-8")).hexdigest(),str(e.get("usage_role","NOT_EVALUATED")),bool(e.get("authorized_for_section",False)),eid in used,"NOT_EVALUATED").to_dict()))
    for x in sorted(corrections.values(),key=lambda z:(z["section_id"],z["claim_id"],z["correction_id"])):
        if x["claim_id"] in invalid_claim_ids: continue
        p,ri,pc,rv,cp=x["proposal"],x.get("reverification_input"),x.get("precheck"),x.get("reverification"),x.get("comparison")
        auth={e["evidence_id"]:e for e in (ri or {}).get("authorized_evidence",())}
        if ri and set(p.get("evidence_ids",()))-set(auth): issues.append("AGGREGATION_ROW_EVIDENCE_IDENTITY_UNAVAILABLE"); continue
        def avail(obj,blocked=False): return "AVAILABLE" if obj is not None else ("BLOCKED_UPSTREAM" if blocked else "NOT_PRODUCED")
        action,is_scientific,is_gate=_phase655_stage_action(p,cp)
        cr=CorrectionTraceabilityRow(p["correction_id"],p["claim_id"],p["section_id"],action,is_scientific,is_gate,"AVAILABLE",avail(pc),avail(rv,pc is not None and pc["precheck_status"]!="PRECHECK_PASSED"),avail(cp,rv is None or rv.get("reverification_execution_status")!="COMPLETED"),p.get("final_proposal_status"),pc.get("precheck_status") if pc else None,rv.get("reverification_execution_status") if rv else None,cp.get("acceptance_decision") if cp else None,tuple((ri or {}).get("target_issue_codes",())),tuple((cp or {}).get("resolved_issue_codes",())),tuple((cp or {}).get("remaining_issue_codes",())),tuple((cp or {}).get("new_issue_codes",())),(cp or {}).get("hallucination_risk_before"),(cp or {}).get("hallucination_risk_after"),(cp or {}).get("hallucination_risk_delta"),p.get("proposal_fingerprint"),(pc or {}).get("virtual_proposed_claim_text_fingerprint"),(pc or {}).get("frozen_evidence_snapshot_fingerprint"),(pc or {}).get("reverification_context_fingerprint"),tuple((pc or {}).get("reason_codes",())),tuple((pc or {}).get("technical_issue_codes",())),tuple((cp or {}).get("reason_codes",())),tuple((cp or {}).get("technical_issue_codes",())),next((d.get("gate_classification") for d in (cp or {}).get("decision_trace",()) if d.get("gate_classification")),None),bool((cp or {}).get("manual_review_required",False) or (rv or {}).get("manual_review_recommended",False)),False,
            # Trazabilidad terminal completa de la CorrectionProposal (ver
            # CorrectionTraceabilityRow en traceability.py) -- copiada
            # directamente de p (CorrectionProposal.to_dict()), nunca
            # inventada. Antes se descartaba por completo al construir
            # esta fila, dejando proposal_status=DEFERRED sin ninguna
            # causa auditable.
            correction_decision=p.get("correction_decision"),
            final_proposal_status=p.get("final_proposal_status"),
            requires_manual_review=bool(p.get("requires_manual_review",False)),
            accepted_for_reverification=bool(p.get("accepted_for_reverification",False)),
            reason_codes=tuple(p.get("reason_codes") or ()),
            validation_issue_codes=tuple(p.get("validation_issue_codes") or ()),
            decision_path=tuple(p.get("decision_path") or ()),
            retry_metrics=dict(p["retry_metrics"]) if p.get("retry_metrics") else None,
            raw_attempts_count=len(p.get("raw_attempts") or ()),
            raw_attempts_fingerprint=(
                sha256(json.dumps(list(p["raw_attempts"]),ensure_ascii=False,sort_keys=True,default=str).encode("utf-8")).hexdigest()
                if p.get("raw_attempts") else None
            ),
        )
        corr_rows.append(validate_correction_traceability_row_contract(cr.to_dict()))
        ids=set(p.get("evidence_ids",()))|set(auth)|set((rv or {}).get("evidence_ids_used",()))
        for eid in sorted(ids):
            e=auth.get(eid)
            if not e or not e.get("source_filename") or not e.get("chunk_id"): warnings.append("AGGREGATION_ROW_EVIDENCE_IDENTITY_UNAVAILABLE"); continue
            corr_ev.append(validate_correction_evidence_traceability_row_contract(CorrectionEvidenceTraceabilityRow(p["claim_id"],p["correction_id"],p["section_id"],eid,e["source_filename"],e["chunk_id"],e["usage_role"],bool(e["authorized_for_section"]),eid in set(p.get("evidence_ids",())),eid in set((rv or {}).get("evidence_ids_used",())),"NOT_EVALUATED",str((pc or {}).get("frozen_evidence_snapshot_fingerprint","")).to_dict() if False else str((pc or {}).get("frozen_evidence_snapshot_fingerprint",""))).to_dict()))
        if rv:
            rev_rows.append(validate_reverification_traceability_row_contract(ReverificationTraceabilityRow(rv["correction_id"],rv["claim_id"],rv["section_id"],rv["prompt_version"],rv["reverification_execution_status"],rv["reverification_llm_calls"],rv["format_attempts"],rv["format_retries"],rv["schema_attempts"],rv["schema_retries"],tuple(rv["evidence_ids_used"]),tuple(rv["observed_issue_codes"]),tuple(rv["target_issues_resolved_reported"]),bool(cp["reported_resolution_matches"]) if cp else False,bool(rv["manual_review_recommended"]),(cp or {}).get("acceptance_decision"),rv["proposal_fingerprint"],rv["virtual_proposed_claim_text_fingerprint"],rv["frozen_evidence_snapshot_fingerprint"],rv["reverification_context_fingerprint"],False).to_dict()))
    claim_rows.sort(key=lambda x:(x["section_id"],x["claim_id"]));corr_rows.sort(key=lambda x:(x["section_id"],x["claim_id"],x["correction_id"]));claim_ev.sort(key=lambda x:(x["section_id"],x["claim_id"],x["evidence_id"]));corr_ev.sort(key=lambda x:(x["section_id"],x["claim_id"],x["correction_id"],x["evidence_id"]));rev_rows.sort(key=lambda x:(x["section_id"],x["claim_id"],x["correction_id"]))
    invalid=bool(issues); status="INVALID" if invalid else ("PARTIAL" if rr["referential_validation_status"]=="PARTIAL" or warnings else "VALID")
    payload=dict(claim_traceability_rows=tuple(claim_rows),correction_traceability_rows=tuple(corr_rows),claim_evidence_traceability_rows=tuple(claim_ev),correction_evidence_traceability_rows=tuple(corr_ev),reverification_traceability_rows=tuple(rev_rows),row_issue_codes=tuple(sorted(set(issues))),row_warnings=tuple(sorted(set(warnings))),row_build_status=status,aggregation_status=status,metrics_status="NOT_COMPUTED",result_contract_valid=False)
    n=_phase655_validate_rows_result(payload,allow_unvalidated_flag=True);n["result_contract_valid"]=True;r=ProvisionalTraceabilityRowsResult(**n);validate_provisional_traceability_rows_result_contract(r.to_dict());return r


def _phase655_validate_metrics_result(value: Mapping[str,Any], *, allow_unvalidated_flag=False)->dict[str,Any]:
    from src.tools.verification.traceability import ProvisionalMetricsAggregationResult
    from src.config.verification_policy_config import AGGREGATION_METRIC_ISSUE_CODES,AGGREGATION_METRIC_WARNING_CODES,AGGREGATION_METRICS_STATUSES
    code="PROVISIONAL_METRICS_AGGREGATION_RESULT_INVALID";r=_phase651a_exact_dataclass_mapping(value,ProvisionalMetricsAggregationResult,code)
    r["metrics"]=validate_provisional_verification_metrics_contract(r["metrics"])
    for f in ("metric_issue_codes","metric_warnings"):r[f]=_phase651a_string_tuple(r[f],f,code)
    if set(r["metric_issue_codes"])-set(AGGREGATION_METRIC_ISSUE_CODES):raise ValueError(f"{code}:ISSUE_UNKNOWN")
    if set(r["metric_warnings"])-set(AGGREGATION_METRIC_WARNING_CODES):raise ValueError(f"{code}:WARNING_UNKNOWN")
    if r["metrics_status"] not in AGGREGATION_METRICS_STATUSES:raise ValueError(f"{code}:STATUS")
    if r["aggregation_status"] not in ("VALID","PARTIAL","INVALID"):raise ValueError(f"{code}:AGGREGATION_STATUS")
    flag=_phase651a_bool(r["result_contract_valid"],"result_contract_valid",code)
    if not allow_unvalidated_flag and not flag:raise ValueError(f"{code}:result_contract_valid")
    return r


def validate_provisional_metrics_aggregation_result_contract(value:Mapping[str,Any])->dict[str,Any]: return _phase655_validate_metrics_result(value)



# ---------------------------------------------------------------------------
# Phase 6.5.6 — closed metric populations, fingerprints and provisional bundle
# ---------------------------------------------------------------------------

def _phase656_sha256_payload(value: Any) -> str:
    import hashlib
    payload = _phase652_canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()




def compute_provisional_collection_fingerprints(collection_result: Any) -> dict[str, str]:
    """Fingerprint normalized, deduplicated collections only; never raw input order."""
    from src.config.verification_policy_config import (
        PROVISIONAL_COLLECTION_FINGERPRINT_FIELDS,
        PROVISIONAL_COLLECTION_FINGERPRINT_VERSION,
    )
    cv = validate_provisional_collection_validation_result_contract(_phase652_plain_mapping(collection_result))
    source_fields = {
        "claim_verification_records": "normalized_claim_verification_records",
        "correction_proposals": "normalized_correction_proposals",
        "correction_reverification_inputs": "normalized_correction_reverification_inputs",
        "correction_precheck_results": "normalized_correction_precheck_results",
        "independent_reverification_results": "normalized_independent_reverification_results",
        "before_after_comparison_results": "normalized_before_after_comparison_results",
    }
    out: dict[str, str] = {}
    for public_name in PROVISIONAL_COLLECTION_FINGERPRINT_FIELDS:
        normalized_name = source_fields[public_name]
        records = tuple(cv[normalized_name])
        out[public_name] = _phase656_sha256_payload({
            "version": PROVISIONAL_COLLECTION_FINGERPRINT_VERSION,
            "collection": public_name,
            "records": records,
        })
    return out


def _phase656_validate_metric_formula(metrics: Mapping[str, Any]) -> None:
    code = "PROVISIONAL_METRICS_AGGREGATION_RESULT_INVALID"
    expected_total = (
        metrics["verification_llm_calls"] + metrics["correction_llm_calls"]
        + metrics["reverification_llm_calls"] + metrics["additional_llm_calls"]
    )
    if metrics["total_llm_calls"] != expected_total:
        raise ValueError(f"{code}:TOTAL_LLM_CALLS_MISMATCH")
    formulas = {
        "candidate_issue_resolution_rate": (
            metrics["candidate_claim_issues_resolved"], metrics["issues_before"]
        ),
        "accepted_issue_resolution_rate": (
            metrics["accepted_claim_issues_resolved"], metrics["issues_before"]
        ),
        "correction_acceptance_rate": (
            metrics["corrections_accepted_for_07c"], metrics["corrections_reverified"]
        ),
        "new_issue_rate": (
            metrics["corrections_with_new_issues"], metrics["corrections_reverified"]
        ),
        "hallucination_risk_reduction_rate": (
            metrics["risk_reduced"],
            metrics["risk_reduced"] + metrics["risk_unchanged"] + metrics["risk_increased"],
        ),
    }
    for field, (numerator, denominator) in formulas.items():
        rate = metrics.get(field)
        if rate is None:
            continue
        if rate["numerator"] != numerator:
            raise ValueError(f"{code}:{field}:NUMERATOR_MISMATCH")
        if rate["denominator"] != denominator:
            raise ValueError(f"{code}:{field}:DENOMINATOR_MISMATCH")
    if not (metrics["accepted_claim_issues_resolved"] <= metrics["candidate_claim_issues_resolved"] <= metrics["issues_before"]):
        raise ValueError(f"{code}:ISSUE_COUNT_COHERENCE")
    if metrics["corrections_accepted_for_07c"] > metrics["corrections_reverified"]:
        raise ValueError(f"{code}:ACCEPTED_CORRECTIONS_EXCEED_REVERIFIED")
    if metrics["claims_with_accepted_proposals"] > metrics["claims_with_correction_proposals"] or metrics["claims_with_correction_proposals"] > metrics["claims_verified"]:
        raise ValueError(f"{code}:CLAIM_COUNT_COHERENCE")


_validate_provisional_metrics_aggregation_result_contract_phase655 = validate_provisional_metrics_aggregation_result_contract

def validate_provisional_metrics_aggregation_result_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _validate_provisional_metrics_aggregation_result_contract_phase655(value)
    if result["metrics_status"] == "COMPUTED":
        _phase656_validate_metric_formula(result["metrics"])
    return result


def _phase656_authorized_ids(rows: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    claim_ids = {row["claim_id"] for row in rows["claim_traceability_rows"]}
    correction_ids = {row["correction_id"] for row in rows["correction_traceability_rows"]}
    comparison_ids = {
        row["correction_id"] for row in rows["correction_traceability_rows"]
        if row.get("comparison_stage_availability") == "AVAILABLE"
    }
    return claim_ids, correction_ids, comparison_ids


def aggregate_provisional_verification_metrics(rows_result: Any, collection_result: Any):
    """Aggregate scientific metrics only from valid rows; counters use that same authorized population."""
    from src.tools.verification.traceability import ProvisionalVerificationMetrics, ProvisionalMetricsAggregationResult
    rows = validate_provisional_traceability_rows_result_contract(_phase652_plain_mapping(rows_result))
    cv = validate_provisional_collection_validation_result_contract(_phase652_plain_mapping(collection_result))
    if rows["row_build_status"] == "INVALID" or cv["collection_validation_status"] == "INVALID":
        metrics = ProvisionalVerificationMetrics(
            recommendations_generated=_phase655_metric_value(0, 0, "recommendation", "identity unavailable")
        ).to_dict()
        payload = dict(
            metrics=metrics,
            metric_issue_codes=("AGGREGATION_METRICS_INPUT_INVALID",),
            metric_warnings=("AGGREGATION_RECOMMENDATIONS_NOT_COMPUTABLE",),
            metrics_status="NOT_COMPUTED",
            aggregation_status="INVALID",
            result_contract_valid=False,
        )
    else:
        claims = tuple(rows["claim_traceability_rows"])
        corrections = tuple(rows["correction_traceability_rows"])
        claim_ids, correction_ids, comparison_ids = _phase656_authorized_ids(rows)
        proposals_by_id = {x["correction_id"]: x for x in cv["normalized_correction_proposals"] if x["correction_id"] in correction_ids}
        reverifications_by_id = {x["correction_id"]: x for x in cv["normalized_independent_reverification_results"] if x["correction_id"] in correction_ids}
        comparisons_by_id = {x["correction_id"]: x for x in cv["normalized_before_after_comparison_results"] if x["correction_id"] in comparison_ids}
        verification_by_id = {
            x["claim_verification_result"]["claim_id"]: x
            for x in cv["normalized_claim_verification_records"]
            if x["claim_verification_result"]["claim_id"] in claim_ids
        }
        source_issues = {(x["claim_id"], issue) for x in claims for issue in x["source_issue_codes"]}
        candidate = {(x["claim_id"], issue) for x in corrections for issue in x["resolved_issue_codes"]}
        accepted = {
            (x["claim_id"], issue) for x in corrections
            if x.get("acceptance_decision") == "ACCEPT_FOR_07C"
            for issue in x["resolved_issue_codes"]
        }
        remaining = {(x["claim_id"], issue) for x in claims for issue in x["provisional_remaining_issue_codes"]}
        new_issue_pairs = {(x["correction_id"], issue) for x in corrections for issue in x["new_issue_codes"]}
        corrections_with_new = {correction_id for correction_id, _ in new_issue_pairs}
        reverified_ids = {
            x["correction_id"] for x in corrections
            if x.get("reverification_stage_availability") == "AVAILABLE"
            and x.get("reverification_execution_status") == "COMPLETED"
        }
        failed_ids = {
            x["correction_id"] for x in corrections
            if x.get("reverification_stage_availability") == "AVAILABLE"
            and x.get("reverification_execution_status") == "FAILED"
        }
        accepted_ids = {x["correction_id"] for x in corrections if x.get("acceptance_decision") == "ACCEPT_FOR_07C"}
        risk_counts = {delta: sum(1 for x in corrections if x.get("comparison_stage_availability") == "AVAILABLE" and x.get("hallucination_risk_delta") == delta) for delta in ("REDUCED", "UNCHANGED", "INCREASED", "NOT_COMPARABLE")}
        verification_llm = sum(x["claim_verification_result"]["tool_usage"]["llm_calls"] for x in verification_by_id.values())
        correction_llm = sum(x["retry_metrics"]["llm_calls"] for x in proposals_by_id.values())
        reverification_llm = sum(x["reverification_llm_calls"] for x in reverifications_by_id.values())
        comparison_llm = sum(x["additional_llm_calls"] for x in comparisons_by_id.values())
        verification_rounds = sum(x["claim_verification_result"]["tool_usage"]["retrieval_rounds"] for x in verification_by_id.values())
        incremental_requests = sum(x["claim_verification_result"]["tool_usage"]["retrieval_requested"] for x in verification_by_id.values())
        comparison_rounds = sum(x["retrieval_rounds"] for x in comparisons_by_id.values())
        gate_count = lambda code: sum(1 for x in corrections if x.get("gate_classification") == code)
        unknown_reason_count = sum(len(x.get("precheck_reason_codes", ())) for x in corrections if x.get("gate_classification") == "UNKNOWN_REASON_CODE")
        metrics = ProvisionalVerificationMetrics(
            claims_verified=len(claim_ids),
            claims_with_terminal_correction_recommendation=sum(1 for x in claims if x["terminal_correction_recommendation"]),
            claims_with_correction_proposals=sum(1 for x in claims if x["has_correction_proposal"]),
            claims_with_accepted_proposals=sum(1 for x in claims if x["individual_accepted_correction_ids"]),
            claims_requiring_manual_review=sum(1 for x in claims if x["manual_review_required"]),
            corrections_proposed=sum(1 for x in proposals_by_id.values() if x.get("correction_decision") == "PROPOSE_CHANGE"),
            corrections_precheck_passed=sum(1 for x in corrections if x.get("precheck_status") == "PRECHECK_PASSED"),
            corrections_precheck_blocked=sum(1 for x in corrections if x.get("precheck_status") == "PRECHECK_BLOCKED"),
            corrections_precheck_rejected=sum(1 for x in corrections if x.get("precheck_status") == "PRECHECK_REJECTED"),
            corrections_reverified=len(reverified_ids),
            corrections_failed_reverification=len(failed_ids),
            corrections_accepted_for_07c=len(accepted_ids),
            corrections_rejected=sum(1 for x in corrections if x.get("acceptance_decision") == "REJECT_PROPOSAL"),
            corrections_deferred=sum(1 for x in corrections if x.get("acceptance_decision") == "DEFER_TO_MANUAL_REVIEW"),
            issues_before=len(source_issues),
            candidate_claim_issues_resolved=len(candidate),
            accepted_claim_issues_resolved=len(accepted),
            issues_remaining=len(remaining),
            new_issues_introduced=len(new_issue_pairs),
            corrections_with_new_issues=len(corrections_with_new),
            risk_reduced=risk_counts["REDUCED"], risk_unchanged=risk_counts["UNCHANGED"],
            risk_increased=risk_counts["INCREASED"], risk_not_comparable=risk_counts["NOT_COMPARABLE"],
            verification_llm_calls=verification_llm, correction_llm_calls=correction_llm,
            reverification_llm_calls=reverification_llm, additional_llm_calls=comparison_llm,
            total_llm_calls=verification_llm + correction_llm + reverification_llm + comparison_llm,
            verification_retrieval_rounds=verification_rounds,
            incremental_retrieval_requests=incremental_requests,
            correction_retrieval_rounds=0, reverification_retrieval_rounds=0,
            comparison_retrieval_rounds=comparison_rounds, additional_retrieval_rounds=0,
            invalid_gate_results=gate_count("INVALID_GATE_CONTRACT") + gate_count("UNKNOWN_REASON_CODE"),
            temporary_technical_blocks=gate_count("TEMPORARY_TECHNICAL"),
            permanent_contractual_blocks=gate_count("PERMANENT_CONTRACTUAL"),
            deterministic_scientific_rejections=gate_count("DETERMINISTIC_SCIENTIFIC_REJECTION"),
            unknown_precheck_reason_codes=unknown_reason_count,
            not_available_action_results=sum(1 for x in corrections if x.get("action_type") == "NOT_AVAILABLE"),
            candidate_issue_resolution_rate=_phase655_metric_value(len(candidate), len(source_issues), "(claim_id,issue_code)", "authorized comparison rows"),
            accepted_issue_resolution_rate=_phase655_metric_value(len(accepted), len(source_issues), "(claim_id,issue_code)", "ACCEPT_FOR_07C rows"),
            correction_acceptance_rate=_phase655_metric_value(len(accepted_ids), len(reverified_ids), "correction_id", "completed authorized reverifications"),
            new_issue_rate=_phase655_metric_value(len(corrections_with_new), len(reverified_ids), "correction_id", "completed authorized reverifications"),
            hallucination_risk_reduction_rate=_phase655_metric_value(risk_counts["REDUCED"], risk_counts["REDUCED"] + risk_counts["UNCHANGED"] + risk_counts["INCREASED"], "correction_id", "authorized comparable comparisons"),
            recommendations_generated=_phase655_metric_value(0, 0, "recommendation", "no contractual identity"),
        ).to_dict()
        payload = dict(
            metrics=metrics,
            metric_issue_codes=(),
            metric_warnings=("AGGREGATION_RECOMMENDATIONS_NOT_COMPUTABLE",),
            metrics_status="COMPUTED",
            aggregation_status=rows["aggregation_status"],
            result_contract_valid=False,
        )
    normalized = _phase655_validate_metrics_result(payload, allow_unvalidated_flag=True)
    normalized["result_contract_valid"] = True
    result = ProvisionalMetricsAggregationResult(**normalized)
    validate_provisional_metrics_aggregation_result_contract(result.to_dict())
    return result


def _phase656_partial_reasons(rows: Mapping[str, Any], referential: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: set[str] = set()
    correction_rows = tuple(rows.get("correction_traceability_rows", ()))
    avail_fields = (
        "proposal_stage_availability", "precheck_stage_availability",
        "reverification_stage_availability", "comparison_stage_availability",
    )
    values = {row.get(field) for row in correction_rows for field in avail_fields}
    if "BLOCKED_UPSTREAM" in values:
        reasons.add("PARTIAL_UPSTREAM_BLOCKED")
    if "FAILED" in values or any(row.get("reverification_execution_status") == "FAILED" for row in correction_rows):
        reasons.add("PARTIAL_STAGE_FAILED")
    if "NOT_PRODUCED" in values:
        reasons.add("PARTIAL_STAGE_NOT_PRODUCED")
    if any(row.get("manual_review_required") for row in rows.get("claim_traceability_rows", ())):
        reasons.add("PARTIAL_MANUAL_REVIEW_REQUIRED")
    if "NOT_APPLICABLE" in values or any("WITHOUT_PROPOSAL" in warning for warning in referential.get("referential_warnings", ())):
        reasons.add("PARTIAL_EXPECTED")
    return tuple(code for code in (
        "PARTIAL_EXPECTED", "PARTIAL_UPSTREAM_BLOCKED", "PARTIAL_STAGE_FAILED", "PARTIAL_STAGE_NOT_PRODUCED",
        "PARTIAL_MANUAL_REVIEW_REQUIRED"
    ) if code in reasons)


def _phase656_normalized_bundle_payload(rows: Mapping[str, Any], metrics: Mapping[str, Any], aggregation_status: str, metrics_status: str, partial_reason_codes: tuple[str, ...], policy_versions: Mapping[str, str], schema_versions: Mapping[str, str]) -> dict[str, Any]:
    from src.config.verification_policy_config import PROVISIONAL_BUNDLE_FINGERPRINT_VERSION
    return {
        "version": PROVISIONAL_BUNDLE_FINGERPRINT_VERSION,
        "claim_traceability_rows": tuple(rows["claim_traceability_rows"]),
        "correction_traceability_rows": tuple(rows["correction_traceability_rows"]),
        "claim_evidence_traceability_rows": tuple(rows["claim_evidence_traceability_rows"]),
        "correction_evidence_traceability_rows": tuple(rows["correction_evidence_traceability_rows"]),
        "reverification_traceability_rows": tuple(rows["reverification_traceability_rows"]),
        "metrics": metrics,
        "aggregation_status": aggregation_status,
        "metrics_status": metrics_status,
        "partial_reason_codes": partial_reason_codes,
        "policy_versions": dict(sorted(policy_versions.items())),
        "schema_versions": dict(sorted(schema_versions.items())),
        "invariants": {
            "correction_applied": False,
            "official_artifacts_created": False,
            "additional_llm_calls": 0,
            "additional_retrieval_rounds": 0,
        },
    }





# Phase 6.5.7: integral audit and provisional-bundle freeze.
def _phase657_safe_audit_shape(value: Any, _seen: set[int] | None = None) -> Any:
    """Return a deterministic, non-retained structural representation for hashing.

    The representation is used only as input to SHA-256. Raw content is never
    placed in collection results or bundles. Unsupported objects are represented
    by stable type information and safe public state when available.
    """
    from dataclasses import asdict, is_dataclass
    import math

    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "NaN"}
        if math.isinf(value):
            return {"__float__": "+Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, bytes):
        import hashlib
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest(), "length": len(value)}
    marker = id(value)
    if marker in _seen:
        return {"__cycle__": f"{type(value).__module__}.{type(value).__qualname__}"}
    _seen.add(marker)
    try:
        if is_dataclass(value):
            return _phase657_safe_audit_shape(asdict(value), _seen)
        if isinstance(value, Mapping):
            pairs = []
            for key, item in value.items():
                safe_key = _phase657_safe_audit_shape(key, _seen)
                safe_value = _phase657_safe_audit_shape(item, _seen)
                pairs.append((safe_key, safe_value))
            pairs.sort(key=lambda pair: _phase652_canonical_json(pair[0]))
            return {"__mapping__": tuple({"key": key, "value": item} for key, item in pairs)}
        if isinstance(value, (list, tuple)):
            return {"__sequence__": tuple(_phase657_safe_audit_shape(item, _seen) for item in value)}
        if isinstance(value, (set, frozenset)):
            items = [_phase657_safe_audit_shape(item, _seen) for item in value]
            items.sort(key=_phase652_canonical_json)
            return {"__set__": tuple(items)}
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                mapped = to_dict()
            except Exception:
                mapped = None
            if isinstance(mapped, Mapping):
                return {
                    "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                    "state": _phase657_safe_audit_shape(mapped, _seen),
                }
        state = getattr(value, "__dict__", None)
        if isinstance(state, Mapping):
            public_state = {str(k): v for k, v in state.items() if not str(k).startswith("_")}
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": _phase657_safe_audit_shape(public_state, _seen),
            }
        return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}"}
    finally:
        _seen.discard(marker)


def _phase657_raw_element_fingerprint(value: Any) -> str:
    return _phase656_sha256_payload({"audit_shape": _phase657_safe_audit_shape(value)})


def _phase657_validate_invalid_element_record(value: Any) -> dict[str, Any]:
    from src.config.verification_policy_config import AGGREGATION_COLLECTION_NAMES
    code = "PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID:invalid_element_records"
    row = _phase651a_mapping(value, "invalid_element_record", code)
    expected = {"collection", "position", "reason_code", "raw_element_fingerprint"}
    if set(row) != expected:
        raise ValueError(f"{code}:SCHEMA_MISMATCH")
    collection = row["collection"]
    if collection not in AGGREGATION_COLLECTION_NAMES:
        raise ValueError(f"{code}:collection:UNKNOWN")
    if type(row["position"]) is not int or row["position"] < 0:
        raise ValueError(f"{code}:position:INVALID")
    reason = row["reason_code"]
    expected_reason = f"AGGREGATION_COLLECTION_ELEMENT_INVALID:{collection}:{row['position']}"
    if reason != expected_reason:
        raise ValueError(f"{code}:reason_code:MISMATCH")
    fingerprint = row["raw_element_fingerprint"]
    if not isinstance(fingerprint, str) or not __import__('re').fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError(f"{code}:raw_element_fingerprint:INVALID_SHA256")
    return dict(row)


# Effective 6.5.7 collection-result validator.
def _phase652_validate_collection_result_contract(value: Mapping[str, Any], *, allow_unvalidated_flag: bool = False) -> dict[str, Any]:
    from src.config.verification_policy_config import AGGREGATION_COLLECTION_NAMES, COLLECTION_VALIDATION_STATUSES, AGGREGATION_STATUSES, METRICS_STATUSES
    from src.tools.verification.traceability import ProvisionalCollectionValidationResult
    code = "PROVISIONAL_COLLECTION_VALIDATION_RESULT_INVALID"
    result = _phase651a_exact_dataclass_mapping(value, ProvisionalCollectionValidationResult, code)
    fields = {
        "claim_verification_records": "normalized_claim_verification_records",
        "correction_proposals": "normalized_correction_proposals",
        "correction_reverification_inputs": "normalized_correction_reverification_inputs",
        "correction_precheck_results": "normalized_correction_precheck_results",
        "independent_reverification_results": "normalized_independent_reverification_results",
        "before_after_comparison_results": "normalized_before_after_comparison_results",
    }
    for field in fields.values():
        if type(result[field]) not in (list, tuple):
            raise ValueError(f"{code}:{field}:SEQUENCE_REQUIRED")
        result[field] = tuple(_phase652_normalize_json_value(item) for item in result[field])
    indexes = _phase651a_mapping(result["primary_indexes"], "primary_indexes", code)
    if set(indexes) != set(AGGREGATION_COLLECTION_NAMES):
        raise ValueError(f"{code}:primary_indexes:EXACT_COLLECTIONS_REQUIRED")
    result["primary_indexes"] = {
        collection: {
            str(key): _phase652_normalize_json_value(item)
            for key, item in _phase651a_mapping(indexes[collection], collection, code).items()
        }
        for collection in AGGREGATION_COLLECTION_NAMES
    }
    if type(result["duplicate_records"]) not in (list, tuple):
        raise ValueError(f"{code}:duplicate_records:SEQUENCE_REQUIRED")
    result["duplicate_records"] = tuple(_phase653_validate_duplicate_record(item) for item in result["duplicate_records"])
    if type(result["invalid_element_records"]) not in (list, tuple):
        raise ValueError(f"{code}:invalid_element_records:SEQUENCE_REQUIRED")
    result["invalid_element_records"] = tuple(
        _phase657_validate_invalid_element_record(item) for item in result["invalid_element_records"]
    )
    if tuple(sorted(result["invalid_element_records"], key=_phase652_canonical_json)) != result["invalid_element_records"]:
        raise ValueError(f"{code}:invalid_element_records:NOT_CANONICAL")
    for field in ("collection_issue_codes", "collection_warnings"):
        result[field] = _phase651a_string_tuple(result[field], field, code)
    if any(not _phase653_allowed_collection_issue(item) for item in result["collection_issue_codes"]):
        raise ValueError(f"{code}:collection_issue_codes:UNKNOWN")
    duplicate_types = {item["duplicate_type"] for item in result["duplicate_records"]}
    if "CONFLICTING" in duplicate_types and "AGGREGATION_CONFLICTING_DUPLICATE" not in result["collection_issue_codes"]:
        raise ValueError(f"{code}:CONFLICTING_DUPLICATE:ISSUE_REQUIRED")
    if "IDENTICAL" in duplicate_types and "AGGREGATION_IDENTICAL_DUPLICATE_DEDUPLICATED" not in result["collection_issue_codes"]:
        raise ValueError(f"{code}:IDENTICAL_DUPLICATE:ISSUE_REQUIRED")
    invalid_reasons = {item["reason_code"] for item in result["invalid_element_records"]}
    if not invalid_reasons.issubset(set(result["collection_issue_codes"])):
        raise ValueError(f"{code}:invalid_element_records:ISSUE_REQUIRED")
    positional_issues = {item for item in result["collection_issue_codes"] if item.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID:")}
    if positional_issues != invalid_reasons:
        raise ValueError(f"{code}:invalid_element_records:POSITIONAL_ISSUE_MISMATCH")
    _phase653_validate_indexes(result)
    if result["collection_validation_status"] not in COLLECTION_VALIDATION_STATUSES:
        raise ValueError(f"{code}:collection_validation_status:UNKNOWN")
    if result["aggregation_status"] not in AGGREGATION_STATUSES or result["metrics_status"] not in METRICS_STATUSES:
        raise ValueError(f"{code}:STATUS_UNKNOWN")
    invalid = result["collection_validation_status"] == "INVALID"
    if invalid and (result["aggregation_status"] != "INVALID" or result["metrics_status"] != "NOT_COMPUTED"):
        raise ValueError(f"{code}:INVALID:STATUS_COHERENCE")
    if not invalid and result["aggregation_status"] == "INVALID":
        raise ValueError(f"{code}:VALID:CANNOT_AGGREGATION_INVALID")
    if bool(result["invalid_element_records"]) != bool(positional_issues):
        raise ValueError(f"{code}:invalid_element_records:COHERENCE")
    flag = _phase651a_bool(result["result_contract_valid"], "result_contract_valid", code)
    if not allow_unvalidated_flag and not flag:
        raise ValueError(f"{code}:result_contract_valid:MUST_BE_DERIVED_TRUE")
    return result


def validate_provisional_collection_validation_result_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    return _phase652_validate_collection_result_contract(value)


_AGGREGATION_DIAGNOSTIC_SINK: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_AGGREGATION_DIAGNOSTIC_SINK", default=None
)


@contextmanager
def aggregation_diagnostic_sink(path: Path | None):
    """Instrumentación de diagnóstico TEMPORAL y estrictamente opt-in.

    Mientras el contexto está activo, si ``validate_and_normalize_
    provisional_collections`` (la función activa, más abajo) encuentra
    algún elemento inválido en cualquier colección de agregación,
    además de su comportamiento normal (idéntico, sin cambios), escribe
    un archivo de diagnóstico de solo lectura en ``path`` con el
    registro crudo completo, el tipo y mensaje reales de la excepción
    -- información que el comportamiento normal descarta
    deliberadamente (solo conserva ``type(exc).__name__`` en los
    ``warnings`` de agregación, nunca ``str(exc)``).

    Cuando ``path`` es ``None`` (el valor por defecto en todo el resto
    del código, fuera de este contexto), la función se comporta
    EXACTAMENTE igual que antes de esta instrumentación -- no se activa
    ningún camino nuevo, no se escribe ningún archivo. Reversible
    quitando este bloque y su uso en ``execute_prepared_agent07``."""

    token = _AGGREGATION_DIAGNOSTIC_SINK.set(path)
    try:
        yield
    finally:
        _AGGREGATION_DIAGNOSTIC_SINK.reset(token)


# Effective 6.5.7 collection normalization with safe invalid-element audit records.
def validate_and_normalize_provisional_collections(value: Mapping[str, Any]):
    from src.config.verification_policy_config import AGGREGATION_COLLECTION_NAMES
    from src.tools.verification.traceability import ProvisionalCollectionValidationResult
    inp = validate_provisional_verification_aggregation_input_contract(value)
    groups: dict[str, tuple[Mapping[str, Any], ...]] = {}
    issues: list[str] = []
    warnings: list[str] = []
    duplicates: list[dict[str, Any]] = []
    invalid_elements: list[dict[str, Any]] = []
    invalid = False
    # Instrumentación de diagnóstico opt-in (ver aggregation_diagnostic_sink
    # más arriba): NUNCA afecta issues/warnings/invalid/invalid_elements --
    # es una lista aparte, solo poblada dentro del except de abajo, y solo
    # volcada a disco si el sink está activo. Con el sink en None (el
    # comportamiento por defecto en cualquier llamada fuera de este
    # contexto), esta lista se calcula igual pero nunca se escribe --
    # ningún costo ni cambio de comportamiento observable.
    _diagnostic_sink_path = _AGGREGATION_DIAGNOSTIC_SINK.get()
    _diagnostic_records: list[dict[str, Any]] = []
    for collection, validator, key_getter in _phase652_collection_specifications():
        by_key: dict[str, dict[str, dict[str, Any]]] = {}
        for position, item in enumerate(inp[collection]):
            try:
                normalized = _phase652_normalize_json_value(validator(_phase652_plain_mapping(item)))
                key = str(key_getter(normalized))
                canonical = _phase652_canonical_json(normalized)
                by_key.setdefault(key, {}).setdefault(canonical, {"record": normalized, "positions": []})["positions"].append(position)
            except Exception as exc:
                invalid = True
                reason = f"AGGREGATION_COLLECTION_ELEMENT_INVALID:{collection}:{position}"
                issues.append(reason)
                # The warning retains only exception class and a closed positional reason; no raw exception message.
                warnings.append(f"{reason}:{type(exc).__name__}")
                invalid_elements.append({
                    "collection": collection,
                    "position": position,
                    "reason_code": reason,
                    "raw_element_fingerprint": _phase657_raw_element_fingerprint(item),
                })
                if _diagnostic_sink_path is not None:
                    # Captura de solo lectura/diagnóstico -- nunca se
                    # relanza ni se suprime la excepción original (ya se
                    # manejó arriba exactamente igual que siempre); esto
                    # solo AGREGA información a un archivo aparte, nunca
                    # oficial, nunca leído por el resto del pipeline.
                    try:
                        _raw_element_json_safe = json.loads(json.dumps(item, default=str))
                    except Exception:
                        _raw_element_json_safe = repr(item)
                    _diagnostic_records.append({
                        "collection": collection,
                        "position": position,
                        "reason_code": reason,
                        "raw_element_fingerprint": _phase657_raw_element_fingerprint(item),
                        "raw_element": _raw_element_json_safe,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    })
        retained: dict[str, Mapping[str, Any]] = {}
        for key, variants in by_key.items():
            if len(variants) > 1:
                invalid = True
                issues.append("AGGREGATION_CONFLICTING_DUPLICATE")
                duplicates.append({
                    "collection": collection,
                    "primary_key": key,
                    "duplicate_type": "CONFLICTING",
                    "conflicting_records": tuple(item["record"] for _, item in sorted(variants.items())),
                })
            else:
                only = next(iter(variants.values()))
                retained[key] = only["record"]
                if len(only["positions"]) > 1:
                    issues.append("AGGREGATION_IDENTICAL_DUPLICATE_DEDUPLICATED")
                    duplicates.append({
                        "collection": collection,
                        "primary_key": key,
                        "duplicate_type": "IDENTICAL",
                        "duplicate_count": len(only["positions"]),
                        "canonical_record": only["record"],
                    })
        groups[collection] = tuple(retained[key] for key in sorted(retained))
    indexes = {
        collection: {
            str(record["claim_verification_result"]["claim_id"] if collection == "claim_verification_records" else record["correction_id"]): record
            for record in groups[collection]
        }
        for collection in AGGREGATION_COLLECTION_NAMES
    }
    payload = dict(
        normalized_claim_verification_records=groups["claim_verification_records"],
        normalized_correction_proposals=groups["correction_proposals"],
        normalized_correction_reverification_inputs=groups["correction_reverification_inputs"],
        normalized_correction_precheck_results=groups["correction_precheck_results"],
        normalized_independent_reverification_results=groups["independent_reverification_results"],
        normalized_before_after_comparison_results=groups["before_after_comparison_results"],
        primary_indexes=indexes,
        duplicate_records=tuple(sorted(duplicates, key=_phase652_canonical_json)),
        invalid_element_records=tuple(sorted(invalid_elements, key=_phase652_canonical_json)),
        collection_issue_codes=tuple(sorted(set(issues))),
        collection_warnings=tuple(sorted(set(warnings))),
        collection_validation_status="INVALID" if invalid else "VALID",
        aggregation_status="INVALID" if invalid else "VALID",
        metrics_status="NOT_COMPUTED",
        result_contract_valid=False,
    )
    normalized_payload = _phase652_validate_collection_result_contract(payload, allow_unvalidated_flag=True)
    normalized_payload["result_contract_valid"] = True
    result = ProvisionalCollectionValidationResult(**normalized_payload)
    validate_provisional_collection_validation_result_contract(result.to_dict())
    if _diagnostic_sink_path is not None and _diagnostic_records:
        # Escritura de diagnóstico, estrictamente adicional: ocurre DESPUÉS
        # de que "result" ya quedó construido y validado exactamente como
        # siempre -- esta escritura no puede alterar aggregation_status,
        # resolution_status, ningún reason_code ni ninguna decisión
        # científica, porque "result" ya está cerrado en este punto. Si la
        # escritura misma fallara (permiso, disco lleno, etc.), se ignora
        # silenciosamente: un fallo de diagnóstico nunca debe convertirse
        # en un fallo del pipeline real.
        try:
            _diagnostic_sink_path.parent.mkdir(parents=True, exist_ok=True)
            _diagnostic_sink_path.write_text(
                json.dumps({"invalid_elements": _diagnostic_records}, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass
    return result


# Effective 6.5.7 audit payload: versions and invalid-element fingerprints are audit-only.
def _phase656_audit_payload(
    collection: Mapping[str, Any],
    referential: Mapping[str, Any],
    rows: Mapping[str, Any],
    metric_result: Mapping[str, Any],
    input_fingerprints: Mapping[str, str],
    aggregation_status: str,
    metrics_status: str,
    partial_reason_codes: tuple[str, ...],
    policy_versions: Mapping[str, str] | None = None,
    schema_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    from src.config.verification_policy_config import PROVISIONAL_AUDIT_FINGERPRINT_VERSION
    return {
        "version": PROVISIONAL_AUDIT_FINGERPRINT_VERSION,
        "policy_versions": dict(sorted((policy_versions or {}).items())),
        "schema_versions": dict(sorted((schema_versions or {}).items())),
        "input_collection_fingerprints": dict(sorted(input_fingerprints.items())),
        "invalid_element_records": tuple(collection.get("invalid_element_records", ())),
        "duplicate_records": tuple(collection.get("duplicate_records", ())),
        "collection_issue_codes": tuple(collection.get("collection_issue_codes", ())),
        "collection_warnings": tuple(collection.get("collection_warnings", ())),
        "referential_issue_codes": tuple(referential.get("referential_issue_codes", ())),
        "referential_warnings": tuple(referential.get("referential_warnings", ())),
        "orphan_records": tuple(referential.get("orphan_records", ())),
        "identity_conflicts": tuple(referential.get("identity_conflicts", ())),
        "rejected_join_candidates": tuple(referential.get("rejected_join_candidates", ())),
        "row_issue_codes": tuple(rows.get("row_issue_codes", ())),
        "row_warnings": tuple(rows.get("row_warnings", ())),
        "metric_issue_codes": tuple(metric_result.get("metric_issue_codes", ())),
        "metric_warnings": tuple(metric_result.get("metric_warnings", ())),
        "statuses": {
            "collection_validation_status": collection.get("collection_validation_status"),
            "referential_validation_status": referential.get("referential_validation_status"),
            "row_build_status": rows.get("row_build_status"),
            "aggregation_status": aggregation_status,
            "metrics_status": metrics_status,
        },
        "partial_reason_codes": partial_reason_codes,
    }


# Effective 6.5.7 bundle builder: pass contract versions into the audit payload.
def build_provisional_verification_traceability_bundle(aggregation_input: Any):
    """Build the frozen provisional bundle without artifacts, application, LLM, or retrieval."""
    raw_input = validate_provisional_verification_aggregation_input_contract(_phase652_plain_mapping(aggregation_input))
    collection = validate_and_normalize_provisional_collections(raw_input)
    collection_map = collection.to_dict()
    input_fingerprints = compute_provisional_collection_fingerprints(collection)
    if collection.collection_validation_status == "INVALID":
        referential_map = {
            "referential_issue_codes": (), "referential_warnings": (), "orphan_records": (),
            "identity_conflicts": (), "rejected_join_candidates": (), "referential_validation_status": "INVALID",
        }
        rows_map = {
            "claim_traceability_rows": (), "correction_traceability_rows": (),
            "claim_evidence_traceability_rows": (), "correction_evidence_traceability_rows": (),
            "reverification_traceability_rows": (), "row_issue_codes": (), "row_warnings": (),
            "row_build_status": "INVALID",
        }
        from src.tools.verification.traceability import ProvisionalVerificationMetrics, ProvisionalMetricsAggregationResult
        invalid_metrics = ProvisionalVerificationMetrics(
            recommendations_generated=_phase655_metric_value(0, 0, "recommendation", "identity unavailable")
        ).to_dict()
        invalid_metric_payload = dict(
            metrics=invalid_metrics,
            metric_issue_codes=("AGGREGATION_METRICS_INPUT_INVALID",),
            metric_warnings=("AGGREGATION_RECOMMENDATIONS_NOT_COMPUTABLE",),
            metrics_status="NOT_COMPUTED", aggregation_status="INVALID", result_contract_valid=False,
        )
        normalized_metric_payload = _phase655_validate_metrics_result(invalid_metric_payload, allow_unvalidated_flag=True)
        normalized_metric_payload["result_contract_valid"] = True
        metric_result = ProvisionalMetricsAggregationResult(**normalized_metric_payload)
        aggregation_status = "INVALID"
        metrics_status = "NOT_COMPUTED"
        partial_reasons: tuple[str, ...] = ()
    else:
        referential = validate_provisional_referential_integrity(collection)
        referential_map = referential.to_dict()
        rows = build_provisional_traceability_rows(referential)
        rows_map = rows.to_dict()
        metric_result = aggregate_provisional_verification_metrics(rows, collection)
        if referential.referential_validation_status == "INVALID" or rows.row_build_status == "INVALID":
            aggregation_status = "INVALID"
            metrics_status = "NOT_COMPUTED"
            partial_reasons = ()
        else:
            aggregation_status = "PARTIAL" if (
                referential.referential_validation_status == "PARTIAL" or rows.row_build_status == "PARTIAL"
            ) else "VALID"
            metrics_status = metric_result.metrics_status
            partial_reasons = _phase656_partial_reasons(rows_map, referential_map) if aggregation_status == "PARTIAL" else ()
    metric_map = metric_result.to_dict()
    normalized_status = "NOT_COMPUTABLE" if aggregation_status == "INVALID" else "COMPUTED"
    normalized_fingerprint = None
    if normalized_status == "COMPUTED":
        normalized_fingerprint = _phase656_sha256_payload(_phase656_normalized_bundle_payload(
            rows_map, metric_map["metrics"], aggregation_status, metrics_status, partial_reasons,
            raw_input["policy_versions"], raw_input["schema_versions"],
        ))
    audit_fingerprint = _phase656_sha256_payload(_phase656_audit_payload(
        collection_map, referential_map, rows_map, metric_map, input_fingerprints,
        aggregation_status, metrics_status, partial_reasons,
        raw_input["policy_versions"], raw_input["schema_versions"],
    ))
    aggregation_issues = tuple(sorted(set(
        tuple(collection_map.get("collection_issue_codes", ()))
        + tuple(referential_map.get("referential_issue_codes", ()))
        + tuple(rows_map.get("row_issue_codes", ()))
        + tuple(metric_map.get("metric_issue_codes", ()))
    )))
    aggregation_warnings = tuple(sorted(set(
        tuple(collection_map.get("collection_warnings", ()))
        + tuple(referential_map.get("referential_warnings", ()))
        + tuple(rows_map.get("row_warnings", ()))
        + tuple(metric_map.get("metric_warnings", ()))
    )))
    return create_provisional_verification_traceability_bundle(
        claim_traceability_rows=tuple(rows_map["claim_traceability_rows"]),
        correction_traceability_rows=tuple(rows_map["correction_traceability_rows"]),
        claim_evidence_traceability_rows=tuple(rows_map["claim_evidence_traceability_rows"]),
        correction_evidence_traceability_rows=tuple(rows_map["correction_evidence_traceability_rows"]),
        reverification_traceability_rows=tuple(rows_map["reverification_traceability_rows"]),
        metrics=metric_map["metrics"],
        aggregation_status=aggregation_status,
        metrics_status=metrics_status,
        partial_reason_codes=partial_reasons,
        aggregation_issue_codes=aggregation_issues,
        aggregation_warnings=aggregation_warnings,
        normalized_bundle_status=normalized_status,
        normalized_bundle_fingerprint=normalized_fingerprint,
        aggregation_audit_fingerprint=audit_fingerprint,
        input_collection_fingerprints=input_fingerprints,
        policy_versions=raw_input["policy_versions"],
        schema_versions=raw_input["schema_versions"],
        correction_applied=False,
        official_artifacts_created=False,
        additional_llm_calls=0,
        additional_retrieval_rounds=0,
    )

# Phase 6.5.7 final bundle audit-catalog and observable-state closure.
_phase657_bundle_validator_base = validate_provisional_verification_traceability_bundle_contract


def _phase657_allowed_aggregation_issue(code: str) -> bool:
    from src.config.verification_policy_config import (
        AGGREGATION_COLLECTION_ISSUE_CODES,
        AGGREGATION_REFERENTIAL_ISSUE_CODES,
        AGGREGATION_ROW_ISSUE_CODES,
        AGGREGATION_METRIC_ISSUE_CODES,
        AGGREGATION_COLLECTION_NAMES,
    )
    approved = set(AGGREGATION_COLLECTION_ISSUE_CODES) | set(AGGREGATION_REFERENTIAL_ISSUE_CODES) | set(AGGREGATION_ROW_ISSUE_CODES) | set(AGGREGATION_METRIC_ISSUE_CODES)
    if code in approved:
        return True
    if code.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID:"):
        parts = code.split(":")
        return len(parts) == 3 and parts[1] in AGGREGATION_COLLECTION_NAMES and parts[2].isdigit()
    return False


def _phase657_allowed_aggregation_warning(code: str) -> bool:
    from src.config.verification_policy_config import (
        AGGREGATION_REFERENTIAL_WARNING_CODES,
        AGGREGATION_ROW_WARNING_CODES,
        AGGREGATION_METRIC_WARNING_CODES,
        AGGREGATION_COLLECTION_NAMES,
    )
    approved = set(AGGREGATION_REFERENTIAL_WARNING_CODES) | set(AGGREGATION_ROW_WARNING_CODES) | set(AGGREGATION_METRIC_WARNING_CODES)
    if code in approved:
        return True
    if code.startswith("AGGREGATION_COLLECTION_ELEMENT_INVALID:"):
        parts = code.split(":")
        return len(parts) == 4 and parts[1] in AGGREGATION_COLLECTION_NAMES and parts[2].isdigit() and bool(parts[3])
    return False


def _phase657_observable_partial_reasons(bundle: Mapping[str, Any]) -> set[str]:
    reasons: set[str] = set()
    correction_rows = tuple(bundle.get("correction_traceability_rows") or ())
    availability_fields = (
        "proposal_stage_availability", "precheck_stage_availability",
        "reverification_stage_availability", "comparison_stage_availability",
    )
    values = {row.get(field) for row in correction_rows if isinstance(row, Mapping) for field in availability_fields}
    if "NOT_APPLICABLE" in values or any(
        isinstance(row, Mapping) and not row.get("has_correction_proposal", False)
        for row in tuple(bundle.get("claim_traceability_rows") or ())
    ):
        reasons.add("PARTIAL_EXPECTED")
    if "BLOCKED_UPSTREAM" in values:
        reasons.add("PARTIAL_UPSTREAM_BLOCKED")
    if "FAILED" in values or any(
        isinstance(row, Mapping) and row.get("reverification_execution_status") == "FAILED"
        for row in correction_rows
    ):
        reasons.add("PARTIAL_STAGE_FAILED")
    if "NOT_PRODUCED" in values:
        reasons.add("PARTIAL_STAGE_NOT_PRODUCED")
    if any(
        isinstance(row, Mapping) and row.get("manual_review_required") is True
        for row in tuple(bundle.get("claim_traceability_rows") or ())
    ):
        reasons.add("PARTIAL_MANUAL_REVIEW_REQUIRED")
    warnings = set(bundle.get("aggregation_warnings") or ())
    if "AGGREGATION_CLAIM_WITHOUT_PROPOSAL" in warnings:
        reasons.add("PARTIAL_EXPECTED")
    if warnings & {"AGGREGATION_ACCEPTED_PROPOSAL_INPUT_NOT_PRODUCED", "AGGREGATION_PROPOSAL_NOT_REVERIFIED"}:
        reasons.add("PARTIAL_STAGE_NOT_PRODUCED")
    if "AGGREGATION_PRECHECK_TERMINAL_WITHOUT_REVERIFICATION" in warnings:
        reasons.add("PARTIAL_UPSTREAM_BLOCKED")
    if "AGGREGATION_REVERIFICATION_TERMINAL_WITHOUT_COMPARISON" in warnings:
        reasons.add("PARTIAL_STAGE_FAILED")
    return reasons


def validate_provisional_verification_traceability_bundle_contract(value: Mapping[str, Any], *, allow_unvalidated_flag: bool = False) -> dict[str, Any]:
    result = _phase657_bundle_validator_base(value, allow_unvalidated_flag=allow_unvalidated_flag)
    code = "PROVISIONAL_TRACEABILITY_BUNDLE_INVALID"
    if any(not _phase657_allowed_aggregation_issue(item) for item in result["aggregation_issue_codes"]):
        raise ValueError(f"{code}:aggregation_issue_codes:UNKNOWN")
    if any(not _phase657_allowed_aggregation_warning(item) for item in result["aggregation_warnings"]):
        raise ValueError(f"{code}:aggregation_warnings:UNKNOWN")
    status = result["aggregation_status"]
    if status == "VALID" and result["partial_reason_codes"]:
        raise ValueError(f"{code}:VALID:CANNOT_HAVE_PARTIAL_REASONS")
    if status == "INVALID":
        if result["normalized_bundle_fingerprint"] is not None:
            raise ValueError(f"{code}:INVALID:CANNOT_HAVE_NORMALIZED_FINGERPRINT")
        if result["metrics_status"] != "NOT_COMPUTED":
            raise ValueError(f"{code}:INVALID:METRICS_MUST_NOT_BE_COMPUTED")
    if status == "PARTIAL":
        observable = _phase657_observable_partial_reasons(result)
        if not observable or not set(result["partial_reason_codes"]).issubset(observable):
            raise ValueError(f"{code}:PARTIAL:CAUSE_NOT_OBSERVABLE")
    return result

# ---------------------------------------------------------------------------
# Phase 6.6 blocking context extension.
# The provisional bundle lacked the localized proposal fields required to
# resolve multiple accepted proposals. This additive row context is copied
# only from already validated, referentially accepted CorrectionProposal data.
# ---------------------------------------------------------------------------
_phase66_build_rows_base = build_provisional_traceability_rows
_phase66_validate_correction_row_base = validate_correction_traceability_row_contract


def validate_correction_traceability_row_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _phase652_plain_mapping(value)
    expected_extra = {
        "original_claim_fingerprint", "target_span_in_claim",
        "replacement_text", "proposed_claim_text",
    }
    # Older internal callers are normalized to the additive schema.
    for field in expected_extra:
        raw.setdefault(field, None)
    normalized = _phase66_validate_correction_row_base(raw)
    scientific = bool(normalized["is_scientific_correction_action"])
    values = {field: raw[field] for field in expected_extra}
    if scientific:
        # Legacy structural fixtures may omit the additive 6.6 context. Productive
        # rows populate all four fields atomically; partial population is invalid.
        if any(v is not None for v in values.values()):
            if not all(v is not None for v in values.values()):
                raise ValueError("CORRECTION_TRACEABILITY_ROW_INVALID:PARTIAL_PATCH_CONTEXT")
            if not isinstance(values["original_claim_fingerprint"], str) or len(values["original_claim_fingerprint"]) != 64:
                raise ValueError("CORRECTION_TRACEABILITY_ROW_INVALID:ORIGINAL_FINGERPRINT")
            if not isinstance(values["target_span_in_claim"], Mapping):
                raise ValueError("CORRECTION_TRACEABILITY_ROW_INVALID:TARGET_SPAN")
            if not isinstance(values["replacement_text"], str) or not isinstance(values["proposed_claim_text"], str):
                raise ValueError("CORRECTION_TRACEABILITY_ROW_INVALID:PROPOSAL_TEXT")
    else:
        # Empty terminal proposals and gates do not expose a scientific patch to
        # Phase 6.6, even when a legacy structural fixture retained stale fields.
        values = {field: None for field in expected_extra}
    normalized.update(values)
    return normalized


def build_provisional_traceability_rows(referential_result: Any):
    from src.tools.verification.traceability import ProvisionalTraceabilityRowsResult
    base = _phase66_build_rows_base(referential_result)
    if base.row_build_status == "INVALID":
        return base
    rr = validate_provisional_referential_integrity_result_contract(_phase652_plain_mapping(referential_result))
    contexts = {x["correction_id"]: x["proposal"] for x in rr["joined_correction_records"]}
    rows = []
    for row in base.correction_traceability_rows:
        item = dict(row)
        proposal = contexts.get(item["correction_id"], {})
        if item["is_scientific_correction_action"]:
            item.update(
                original_claim_fingerprint=proposal.get("original_claim_fingerprint"),
                target_span_in_claim=proposal.get("target_span_in_claim"),
                replacement_text=proposal.get("replacement_text"),
                proposed_claim_text=proposal.get("proposed_claim_text"),
            )
        else:
            item.update(
                original_claim_fingerprint=None,
                target_span_in_claim=None,
                replacement_text=None,
                proposed_claim_text=None,
            )
        rows.append(validate_correction_traceability_row_contract(item))
    payload = base.to_dict()
    payload["correction_traceability_rows"] = tuple(rows)
    normalized = _phase655_validate_rows_result(payload, allow_unvalidated_flag=True)
    normalized["result_contract_valid"] = True
    result = ProvisionalTraceabilityRowsResult(**normalized)
    validate_provisional_traceability_rows_result_contract(result.to_dict())
    return result
