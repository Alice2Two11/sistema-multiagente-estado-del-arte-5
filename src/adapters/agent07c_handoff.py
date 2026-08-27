"""Strict compatibility adapter from committed Agent 07 contracts to the original 07C notebook inputs."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import csv, hashlib, io, json, math
from typing import Any, Mapping, Sequence

from src.tools.verification.validation import validate_provisional_verification_traceability_bundle_contract
from src.tools.verification.resolution import validate_provisional_multi_proposal_resolution_result
from src.tools.verification.corrections import fingerprint_text
from src.adapters.verification_runtime import validate_agent07_runtime_result_contract
from src.adapters.agent06_verification_handoff import validate_agent06_verification_handoff_contract, Agent07RetrieverBinding

HISTORICAL_COMPATIBILITY_MATRIX={
 "verification_report.csv":"bundle.claim_traceability_rows V3 with source confidence status",
 "hallucination_report.csv":"claim rows.source_hallucination_risk + correction risk before/after/delta",
 "citation_check.csv":"claim_evidence_traceability_rows",
 "claim_traceability_matrix.csv":"claim/correction/evidence/reverification rows",
 "auto_corrections_log.csv":"resolution plans.selected_patch_records + virtual_result_text",
 "claim_atomization_log.csv":"claim rows preserving claim_id, section_id, original_claim_text",
}

AGENT07C_REQUIRED_ARTIFACTS=(
 "verified_state_of_art.json","verified_state_of_art.md","verification_report.csv",
 "claim_traceability_matrix.csv","auto_corrections_log.csv",
 "verification_validation_report.json","verification_traceability_manifest.json",
)
VERIFICATION_REPORT_COLUMNS=("claim_id","section_id","claim","verdict","confidence","hallucination_risk","correction_needed","evidence_used_ids","confidence_status")
CORRECTION_LOG_COLUMNS=("claim_id","section_id","status","application_scope","original_draft_modified","action","old_fragment","new_fragment","correction_id")
SCIENTIFIC_HANDOFF_CHECK_NAMES=("claim_coverage_ok","section_identity_ok","authorized_evidence_ok","eligible_manual_disjoint_ok","source_draft_fingerprint_ok","patch_application_ok","json_markdown_consistency_ok","correction_log_consistency_ok","claim_traceability_consistency_ok")

REQUIRED_SAFETY_POLICY={
 "uses_ground_truth":False,"uses_external_knowledge":False,"uses_section_evidence_fallback":False,
 "uses_fuzzy_citation_repair":False,"uses_chunks_clean_for_rag":True,
 "performs_independent_rag_per_claim":True,"restricts_retrieval_to_outline_sources":True,
 "validates_all_claims_returned":True,"validates_llm_evidence_against_claim_candidates":True,
}

@dataclass(frozen=True, slots=True)
class Agent07CPreparedInput:
    experiment_id: str
    verified_state_of_art: Mapping[str,Any]
    eligible_claim_ids: tuple[str,...]
    manual_review_claim_ids: tuple[str,...]
    post_correction_reverification_claim_ids: tuple[str,...]
    artifact_payloads: Mapping[str,bytes]
    artifact_hashes: Mapping[str,str]
    optional_artifact_payloads: Mapping[str,bytes]
    optional_artifact_hashes: Mapping[str,str]
    source_draft_fingerprint: str
    prepared_draft_fingerprint: str
    correction_applied_to_copy: bool
    original_draft_modified: bool
    evaluation_ready_emitted: bool
    result_contract_valid: bool
    def to_dict(self): return asdict(self)

def _canonical_json_bytes(v:Any)->bytes:
    return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def _pretty_json_bytes(v:Any)->bytes:return json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode("utf-8")
def _sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def _csv_bytes(rows:Sequence[Mapping[str,Any]],fields:Sequence[str])->bytes:
    s=io.StringIO(newline="");w=csv.DictWriter(s,fieldnames=list(fields),extrasaction="ignore");w.writeheader();w.writerows(rows);return s.getvalue().encode("utf-8")
def _csv_rows(raw:bytes)->tuple[list[str],list[dict[str,str]]]:
    reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")));return list(reader.fieldnames or ()),[dict(r) for r in reader]
def _exact_keys(value:Mapping[str,Any],expected:set[str],code:str)->None:
    if set(value)!=expected:raise ValueError(f"{code}:SCHEMA")
def _section_text_key(section:Mapping[str,Any])->str:
    keys=[k for k in ("section_text","text","content") if isinstance(section.get(k),str)]
    if len(keys)!=1:raise ValueError("AGENT07C_SECTION_TEXT_FIELD_AMBIGUOUS")
    return keys[0]
def _sections_by_id(draft:Mapping[str,Any])->dict[str,dict[str,Any]]:
    sections=draft.get("sections")
    if not isinstance(sections,list) or not sections:raise ValueError("AGENT07C_DRAFT_SECTIONS_MISSING")
    out={}
    for section in sections:
        if not isinstance(section,dict):raise ValueError("AGENT07C_DRAFT_SECTION_INVALID")
        sid=str(section.get("section_id") or section.get("id") or "").strip()
        if not sid or sid in out:raise ValueError("AGENT07C_DRAFT_SECTION_ID_INVALID")
        _section_text_key(section);out[sid]=section
    return out

def _normalize_verdict(value:str)->str:
    mapping={"SUPPORTED":"supported","PARTIALLY_SUPPORTED":"partially_supported","UNSUPPORTED":"unsupported","UNCLEAR":"unclear","NOT_EVALUATED":"unclear","NOT_APPLICABLE":"unclear"}
    return mapping.get(str(value).upper(),str(value).strip().lower())
def _normalize_risk(value:str)->str:
    mapping={"NONE":"none","LOW":"low","MEDIUM":"medium","HIGH":"high","NOT_COMPARABLE":"high","NOT_EVALUATED":"high"}
    return mapping.get(str(value).upper(),str(value).strip().lower())

def _apply_section_claim_replacements(*, section_text:str, contexts:Sequence[Mapping[str,Any]], plans:Sequence[Mapping[str,Any]])->str:
    by_claim={str(c["claim_id"]):c for c in contexts}
    replacements=[]
    expected_section_fp=fingerprint_text(section_text)
    for plan in plans:
        cid=str(plan["claim_id"]);ctx=by_claim.get(cid)
        if ctx is None:raise ValueError(f"AGENT07C_CLAIM_SOURCE_CONTEXT_MISSING:{cid}")
        if str(ctx.get("section_id"))!=str(plan["section_id"]):raise ValueError("AGENT07C_CLAIM_CONTEXT_SECTION_MISMATCH")
        if ctx.get("section_fingerprint")!=expected_section_fp:raise ValueError("AGENT07C_SECTION_FINGERPRINT_MISMATCH")
        original=str(plan.get("original_claim_text") or "")
        if ctx.get("claim_fingerprint")!=fingerprint_text(original):raise ValueError("AGENT07C_CLAIM_FINGERPRINT_MISMATCH")
        span=ctx.get("claim_span_in_section")
        if not isinstance(span,Mapping):raise ValueError("AGENT07C_CLAIM_SPAN_MISSING")
        start,end=span.get("start"),span.get("end")
        if isinstance(start,bool) or isinstance(end,bool) or not isinstance(start,int) or not isinstance(end,int) or not 0<=start<=end<=len(section_text):raise ValueError("AGENT07C_CLAIM_SPAN_INVALID")
        if span.get("base_text_fingerprint")!=expected_section_fp:raise ValueError("AGENT07C_CLAIM_SPAN_BASE_MISMATCH")
        if span.get("text")!=original or section_text[start:end]!=original:raise ValueError("AGENT07C_CLAIM_SPAN_TEXT_MISMATCH")
        virtual=plan.get("virtual_result_text")
        if not isinstance(virtual,str):raise ValueError("AGENT07C_VIRTUAL_RESULT_TEXT_MISSING")
        replacements.append((start,end,virtual,cid))
    ordered=sorted(replacements,key=lambda x:(-x[0],-x[1],x[3]))
    asc=sorted(replacements)
    for left,right in zip(asc,asc[1:]):
        if right[0]<left[1]:raise ValueError("AGENT07C_SECTION_CLAIM_SPANS_OVERLAP")
    result=section_text
    for start,end,new,_ in ordered:
        if result[start:end]!=section_text[start:end]:raise ValueError("AGENT07C_SECTION_OFFSET_DRIFT")
        result=result[:start]+new+result[end:]
    return result

def _apply_markdown_sections(*, source_markdown:str, original_sections:Mapping[str,str], corrected_sections:Mapping[str,str])->str:
    replacements=[]
    for sid,old in original_sections.items():
        new=corrected_sections[sid]
        if old==new:continue
        starts=[];pos=0
        while True:
            idx=source_markdown.find(old,pos)
            if idx<0:break
            starts.append(idx);pos=idx+1
        if len(starts)!=1:raise ValueError(f"AGENT07C_MARKDOWN_SECTION_LOCATION_AMBIGUOUS:{sid}")
        replacements.append((starts[0],starts[0]+len(old),new,sid))
    result=source_markdown
    for start,end,new,_ in sorted(replacements,key=lambda x:(-x[0],-x[1],x[3])):
        if result[start:end]!=source_markdown[start:end]:raise ValueError("AGENT07C_MARKDOWN_OFFSET_DRIFT")
        result=result[:start]+new+result[end:]
    return result

def _validate_safety(policy:Mapping[str,Any])->None:
    if not isinstance(policy,Mapping):raise ValueError("AGENT07C_SAFETY_POLICY_INVALID")
    for key,expected in REQUIRED_SAFETY_POLICY.items():
        if policy.get(key) is not expected:raise ValueError(f"AGENT07C_SAFETY_POLICY_MISMATCH:{key}")


def _authorized_terminal_evidence(*, handoff: Mapping[str, Any], rag_records: Sequence[Mapping[str, Any]]) -> dict[str, set[tuple[str,str,str,bool,str]]]:
    def inherited_identity(e: Mapping[str, Any]) -> tuple[str,str,str,bool,str]:
        text=str(e.get("canonical_text",e.get("text",""))).strip()
        return (str(e.get("evidence_id","")),str(e.get("source_filename","")),str(e.get("chunk_id","")),bool(e.get("authorized_for_section") is True),hashlib.sha256(text.encode("utf-8")).hexdigest())
    result={str(c["claim_id"]):set() for c in handoff["claim_verification_contexts"]}
    for c in handoff["claim_verification_contexts"]:
        result[str(c["claim_id"])].update(inherited_identity(e) for e in c.get("eligible_evidence",()))
    for record in rag_records:
        cid=str(record["claim_id"])
        if cid not in result: raise ValueError("AGENT07C_SAFETY_EVIDENCE_PROVENANCE_UNPROVEN")
        for candidate in record["retrieved_candidate_records"]:
            if cid not in tuple(str(q) for q in candidate["query_ids"]): raise ValueError("AGENT07C_SAFETY_EVIDENCE_PROVENANCE_UNPROVEN")
            result[cid].add((str(candidate["evidence_id"]),str(candidate["source_filename"]),str(candidate["chunk_id"]),True,str(candidate["text_fingerprint"])))
    return result

def _derive_and_validate_safety(*, attestation:Mapping[str,Any], runtime_result:Mapping[str,Any], retriever_binding:Agent07RetrieverBinding|Mapping[str,Any], agent06_handoff:Mapping[str,Any], claim_rows:Sequence[Mapping[str,Any]], claim_evidence:Sequence[Mapping[str,Any]], expected_claim_ids:set[str], provisional_bundle:Mapping[str,Any], resolution_result:Mapping[str,Any])->dict[str,bool]:
    _validate_safety(attestation)
    runtime=validate_agent07_runtime_result_contract(runtime_result)
    handoff=validate_agent06_verification_handoff_contract(agent06_handoff)
    binding=asdict(retriever_binding) if isinstance(retriever_binding,Agent07RetrieverBinding) else dict(retriever_binding)
    if set(binding)!=set(Agent07RetrieverBinding.__dataclass_fields__) or any(not isinstance(binding[k],str) or not binding[k] for k in binding):
        raise ValueError("AGENT07C_SAFETY_RETRIEVER_BINDING_UNPROVEN")
    if binding["experiment_id"]!=handoff["experiment_id"]: raise ValueError("AGENT07C_SAFETY_RETRIEVER_EXPERIMENT_MISMATCH")
    if runtime["provisional_bundle"] != dict(provisional_bundle) or runtime["multi_proposal_resolution_result"] != dict(resolution_result):
        raise ValueError("AGENT07C_SAFETY_RUNTIME_CONTRACT_MISMATCH")
    metrics=runtime["execution_metrics"]
    expected_count=len(expected_claim_ids)
    binding_fp=hashlib.sha256(json.dumps(binding,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
    rag_records=tuple(metrics.get("independent_rag_claim_records",()))
    rag_ids={(str(x["section_id"]),str(x["claim_id"])) for x in rag_records}
    expected_rag_ids={(str(c["section_id"]),str(c["claim_id"])) for c in handoff["claim_verification_contexts"]}
    authorized_sources={str(c["claim_id"]):set(str(x) for x in c.get("authorized_source_filenames",())) for c in handoff["claim_verification_contexts"]}
    records_valid=(
        len(rag_records)==expected_count and rag_ids==expected_rag_ids and
        all(
            int(x["retrieval_requested"])>=1 and int(x["retrieval_rounds"])>=1 and
            x["retriever_binding_fingerprint"]==binding_fp and
            x["retrieval_status"] in {"COMPLETED_WITH_RESULTS","COMPLETED_NO_RESULTS"} and
            ((x["retrieval_status"]=="COMPLETED_WITH_RESULTS") == bool(x["retrieved_candidate_ids"])) and
            all(
                c["source_filename"] in authorized_sources.get(str(x["claim_id"]),set()) and
                str(x["claim_id"]) in tuple(str(q) for q in c["query_ids"]) and
                bool(c["chunk_id"])
                for c in x["retrieved_candidate_records"]
            )
            for x in rag_records
        )
    )
    if metrics["claims_processed"]!=expected_count or metrics["independent_rag_claims"]!=expected_count or metrics["independent_rag_claims_with_results"] + metrics["independent_rag_claims_without_results"] != expected_count or not records_valid:
        raise ValueError("AGENT07C_SAFETY_INDEPENDENT_RAG_UNPROVEN")
    if metrics["evidence_candidate_validation_claims"]!=expected_count:
        raise ValueError("AGENT07C_SAFETY_EVIDENCE_VALIDATION_UNPROVEN")
    actual_claim_ids={str(r["claim_id"]) for r in claim_rows}
    if actual_claim_ids!=expected_claim_ids or set(handoff["expected_claim_ids"])!=expected_claim_ids:
        raise ValueError("AGENT07C_SAFETY_CLAIM_COVERAGE_UNPROVEN")
    authorized_terminal_evidence=_authorized_terminal_evidence(handoff=handoff,rag_records=rag_records)
    for e in claim_evidence:
        if e.get("authorized_for_section") is not True: raise ValueError("AGENT07C_SAFETY_OUTLINE_AUTHORIZATION_UNPROVEN")
        identity=(str(e["evidence_id"]),str(e["source_filename"]),str(e["chunk_id"]),True,str(e["text_fingerprint"]))
        if identity not in authorized_terminal_evidence.get(str(e["claim_id"]),set()): raise ValueError("AGENT07C_SAFETY_EVIDENCE_PROVENANCE_UNPROVEN")
    if not handoff.get("outline_mapping_fingerprint"): raise ValueError("AGENT07C_SAFETY_OUTLINE_MAPPING_UNPROVEN")
    return dict(REQUIRED_SAFETY_POLICY)

def validate_original_agent07c_input_artifacts(*, artifact_payloads:Mapping[str,bytes], experiment_id:str)->dict[str,Any]:
    """Executable extraction of the original notebook 07C input gate (cell 3)."""
    missing=sorted(set(AGENT07C_REQUIRED_ARTIFACTS)-set(artifact_payloads))
    if missing:raise ValueError("AGENT07C_REQUIRED_INPUT_MISSING:"+",".join(missing))
    if set(artifact_payloads)!=set(AGENT07C_REQUIRED_ARTIFACTS):raise ValueError("AGENT07C_ARTIFACT_SET_INVALID")
    try:
        verified=json.loads(artifact_payloads["verified_state_of_art.json"])
        report=json.loads(artifact_payloads["verification_validation_report.json"])
        manifest=json.loads(artifact_payloads["verification_traceability_manifest.json"])
    except Exception as exc:raise ValueError("AGENT07C_JSON_INPUT_INVALID") from exc
    for name,obj in (("verified",verified),("validation_report",report),("manifest",manifest)):
        if not isinstance(obj,dict):raise ValueError(f"AGENT07C_{name.upper()}_NOT_OBJECT")
    if report.get("experiment_id")!=experiment_id or manifest.get("experiment_id")!=experiment_id:raise ValueError("AGENT07C_EXPERIMENT_MISMATCH")
    checks=report.get("scientific_handoff_checks")
    if not isinstance(checks,dict) or set(checks)!=set(SCIENTIFIC_HANDOFF_CHECK_NAMES) or any(type(v) is not bool for v in checks.values()):raise ValueError("AGENT07C_SCIENTIFIC_HANDOFF_CHECKS_INVALID")
    if report.get("scientific_handoff_validation_ok") is not all(checks.values()):raise ValueError("AGENT07C_SCIENTIFIC_HANDOFF_GLOBAL_MISMATCH")
    expected_validation=bool(report.get("structural_validation_ok") and report.get("scientific_handoff_validation_ok") and report.get("original_07c_artifact_gate_ok"))
    if report.get("validation_ok") is not expected_validation:raise ValueError("AGENT07C_VALIDATION_REPORT_GLOBAL_MISMATCH")
    if report.get("validation_ok") is not True:raise ValueError("AGENT07C_VALIDATION_REPORT_NOT_OK")
    if not isinstance(report.get("validation_errors"),list) or not isinstance(report.get("validation_warnings"),list):raise ValueError("AGENT07C_VALIDATION_REPORT_SCHEMA_INVALID")
    if manifest.get("validation_report",{}).get("validation_ok") is not True:raise ValueError("AGENT07C_MANIFEST_VALIDATION_NOT_OK")
    workflow=manifest.get("workflow_state")
    if not isinstance(workflow,dict) or workflow.get("verification_completed") is not True or not isinstance(workflow.get("post_correction_recheck_required"),bool):raise ValueError("AGENT07C_MANIFEST_WORKFLOW_INVALID")
    _validate_safety(manifest.get("safety_policy",{}))
    vcols,vrows=_csv_rows(artifact_payloads["verification_report.csv"])
    required={"claim_id","section_id","claim","verdict","confidence","hallucination_risk","correction_needed","evidence_used_ids"}
    if not required.issubset(vcols):raise ValueError("AGENT07C_VERIFICATION_REPORT_COLUMNS_INVALID")
    if not vrows:raise ValueError("AGENT07C_VERIFICATION_REPORT_EMPTY")
    ids=[r["claim_id"].strip() for r in vrows]
    if not all(ids) or len(ids)!=len(set(ids)):raise ValueError("AGENT07C_VERIFICATION_REPORT_CLAIM_IDS_INVALID")
    ccols,crows=_csv_rows(artifact_payloads["auto_corrections_log.csv"])
    if not {"claim_id","section_id","status","action","old_fragment","new_fragment"}.issubset(ccols):raise ValueError("AGENT07C_CORRECTION_LOG_COLUMNS_INVALID")
    unknown=sorted({r["claim_id"].strip() for r in crows if r["claim_id"].strip()}-set(ids))
    if unknown:raise ValueError("AGENT07C_CORRECTION_LOG_UNKNOWN_CLAIM")
    applied=[r for r in crows if r["status"].strip()=="applied"]
    if len([r["claim_id"] for r in applied])!=len({r["claim_id"] for r in applied}):raise ValueError("AGENT07C_MULTIPLE_APPLIED_PER_CLAIM")
    if workflow["post_correction_recheck_required"] != bool(applied):raise ValueError("AGENT07C_RECHECK_FLAG_MISMATCH")
    sections=verified.get("sections")
    if not isinstance(sections,list) or not sections:raise ValueError("AGENT07C_VERIFIED_SECTIONS_INVALID")
    section_ids=[str(s.get("section_id") or s.get("id") or "") for s in sections if isinstance(s,dict)]
    if len(section_ids)!=len(sections) or not all(section_ids) or len(section_ids)!=len(set(section_ids)):raise ValueError("AGENT07C_VERIFIED_SECTION_IDS_INVALID")
    md=artifact_payloads["verified_state_of_art.md"].decode("utf-8")
    for section in sections:
        text=section[_section_text_key(section)]
        if text not in md:raise ValueError("AGENT07C_JSON_MARKDOWN_DIVERGENCE")
    return {"validation_ok":True,"verification_rows":len(vrows),"applied_corrections":len(applied)}

def _validate_prepared_payload(value:Mapping[str,Any],*,allow_unvalidated:bool=False)->dict[str,Any]:
    expected=set(Agent07CPreparedInput.__dataclass_fields__)
    _exact_keys(value,expected,"AGENT07C_PREPARED_INPUT_INVALID")
    if not isinstance(value["experiment_id"],str) or not value["experiment_id"].strip():raise ValueError("AGENT07C_EXPERIMENT_ID_INVALID")
    payloads=value["artifact_payloads"]
    if not isinstance(payloads,Mapping) or set(payloads)!=set(AGENT07C_REQUIRED_ARTIFACTS) or any(not isinstance(v,bytes) for v in payloads.values()):raise ValueError("AGENT07C_ARTIFACT_PAYLOADS_INVALID")
    hashes=value["artifact_hashes"]
    if not isinstance(hashes,Mapping) or set(hashes)!=set(payloads):raise ValueError("AGENT07C_ARTIFACT_HASHES_INVALID")
    for name,raw in payloads.items():
        if hashes[name]!=_sha(raw):raise ValueError("AGENT07C_ARTIFACT_HASH_MISMATCH")
    optional=value["optional_artifact_payloads"]; optional_hashes=value["optional_artifact_hashes"]
    allowed_optional={"hallucination_report.csv","citation_check.csv","claim_atomization_log.csv","manual_review_queue.csv"}
    if not isinstance(optional,Mapping) or set(optional)-allowed_optional or any(not isinstance(v,bytes) for v in optional.values()): raise ValueError("AGENT07C_OPTIONAL_ARTIFACT_PAYLOADS_INVALID")
    if not isinstance(optional_hashes,Mapping) or set(optional_hashes)!=set(optional): raise ValueError("AGENT07C_OPTIONAL_ARTIFACT_HASHES_INVALID")
    for name,raw in optional.items():
        if optional_hashes[name]!=_sha(raw): raise ValueError("AGENT07C_OPTIONAL_ARTIFACT_HASH_MISMATCH")
    validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id=value["experiment_id"])
    for name in ("eligible_claim_ids","manual_review_claim_ids","post_correction_reverification_claim_ids"):
        seq=value[name]
        if type(seq) not in (tuple,list) or any(not isinstance(x,str) or not x for x in seq) or len(seq)!=len(set(seq)):raise ValueError(f"AGENT07C_{name.upper()}_INVALID")
    if set(value["eligible_claim_ids"])!=set(value["post_correction_reverification_claim_ids"]):raise ValueError("AGENT07C_REVERIFICATION_CLAIM_SET_MISMATCH")
    manifest=json.loads(payloads["verification_traceability_manifest.json"])
    workflow=manifest.get("workflow_state",{})
    expected_manual=tuple(sorted(value["manual_review_claim_ids"]))
    manifest_manual=tuple(sorted(str(x) for x in workflow.get("manual_review_claim_ids",())))
    if manifest_manual!=expected_manual:raise ValueError("AGENT07C_MANUAL_REVIEW_MANIFEST_MISMATCH")
    if bool(expected_manual) is not bool(workflow.get("pending_manual_review",False)):raise ValueError("AGENT07C_MANUAL_REVIEW_FLAG_MISMATCH")
    for name in ("source_draft_fingerprint","prepared_draft_fingerprint"):
        fp=value[name]
        if not isinstance(fp,str) or len(fp)!=64 or any(c not in "0123456789abcdef" for c in fp):raise ValueError(f"AGENT07C_{name.upper()}_INVALID")
    if value["original_draft_modified"] is not False or value["evaluation_ready_emitted"] is not False:raise ValueError("AGENT07C_ISOLATION_INVALID")
    expected_applied=bool(value["eligible_claim_ids"])
    if value["correction_applied_to_copy"] is not expected_applied:raise ValueError("AGENT07C_COPY_APPLICATION_FLAG_MISMATCH")
    if not allow_unvalidated and value["result_contract_valid"] is not True:raise ValueError("AGENT07C_RESULT_CONTRACT_NOT_DERIVED")
    return dict(value)

def validate_agent07c_prepared_input_contract(value:Agent07CPreparedInput|Mapping[str,Any])->dict[str,Any]:
    return _validate_prepared_payload(value.to_dict() if isinstance(value,Agent07CPreparedInput) else value)

def create_agent07c_prepared_input(**kwargs:Any)->Agent07CPreparedInput:
    if "result_contract_valid" in kwargs:raise TypeError("result_contract_valid is derived")
    provisional=Agent07CPreparedInput(result_contract_valid=False,**kwargs)
    normalized=_validate_prepared_payload(provisional.to_dict(),allow_unvalidated=True)
    normalized["result_contract_valid"]=True
    final=Agent07CPreparedInput(**normalized)
    validate_agent07c_prepared_input_contract(final)
    return final

def prepare_agent07c_input_from_agent07(*, provisional_bundle:Mapping[str,Any], resolution_result:Mapping[str,Any], source_draft:Mapping[str,Any], source_draft_markdown:str, experiment_id:str, committed_source_draft_fingerprint:str, claim_source_contexts:Sequence[Mapping[str,Any]], safety_policy:Mapping[str,Any], runtime_result:Mapping[str,Any], retriever_binding:Agent07RetrieverBinding|Mapping[str,Any], agent06_handoff:Mapping[str,Any]) -> Agent07CPreparedInput:
    bundle=validate_provisional_verification_traceability_bundle_contract(provisional_bundle)
    resolution=validate_provisional_multi_proposal_resolution_result(resolution_result)
    if resolution["source_bundle_audit_fingerprint"]!=bundle["aggregation_audit_fingerprint"]:raise ValueError("AGENT07C_SOURCE_BUNDLE_MISMATCH")
    original=deepcopy(dict(source_draft));copy_draft=deepcopy(original)
    canonical_source_fp=_sha(_canonical_json_bytes(original))
    if committed_source_draft_fingerprint!=canonical_source_fp:raise ValueError("AGENT07C_SOURCE_DRAFT_FINGERPRINT_MISMATCH")
    contexts=[deepcopy(dict(c)) for c in claim_source_contexts]
    for c in contexts:
        if c.get("source_draft_fingerprint")!=committed_source_draft_fingerprint:raise ValueError("AGENT07C_CONTEXT_SOURCE_DRAFT_FINGERPRINT_MISMATCH")
    context_ids={(str(c.get("section_id")),str(c.get("claim_id"))) for c in contexts}
    if len(context_ids)!=len(contexts):raise ValueError("AGENT07C_DUPLICATE_SOURCE_CONTEXT")
    source_sections=_sections_by_id(original);target_sections=_sections_by_id(copy_draft)
    plans=list(resolution["claim_resolution_plans"])
    eligible_plans=[p for p in plans if p.get("eligible_for_07c")]
    claim_level_manual={
        str(row["claim_id"])
        for row in bundle["claim_traceability_rows"]
        if row.get("manual_review_required")
    }
    resolution_level_manual={
        str(plan["claim_id"])
        for plan in plans
        if plan.get("manual_review_required") or plan.get("blocks_07c")
    }
    manual=sorted(claim_level_manual | resolution_level_manual)
    plans_by_section:dict[str,list[Mapping[str,Any]]]={}
    for p in eligible_plans:plans_by_section.setdefault(str(p["section_id"]),[]).append(p)
    original_section_texts={};corrected_section_texts={}
    for sid,section in source_sections.items():
        key=_section_text_key(section);old=section[key];original_section_texts[sid]=old
        if sid in plans_by_section:
            relevant_contexts=[c for c in contexts if str(c.get("section_id"))==sid]
            new=_apply_section_claim_replacements(section_text=old,contexts=relevant_contexts,plans=plans_by_section[sid])
        else:new=old
        corrected_section_texts[sid]=new;target_sections[sid][_section_text_key(target_sections[sid])]=new
    corrected_md=_apply_markdown_sections(source_markdown=source_draft_markdown,original_sections=original_section_texts,corrected_sections=corrected_section_texts)
    claim_rows=list(bundle["claim_traceability_rows"]);claim_evidence=list(bundle["claim_evidence_traceability_rows"])
    expected_claim_ids={str(c.get("claim_id") or "") for c in contexts}
    if "" in expected_claim_ids or len(expected_claim_ids)!=len(contexts): raise ValueError("AGENT07C_EXPECTED_CLAIM_IDS_INVALID")
    bundle_claim_ids={str(r["claim_id"]) for r in claim_rows}
    if bundle_claim_ids!=expected_claim_ids: raise ValueError("AGENT07C_CLAIM_COVERAGE_MISMATCH")
    derived_safety=_derive_and_validate_safety(attestation=safety_policy,runtime_result=runtime_result,retriever_binding=retriever_binding,agent06_handoff=agent06_handoff,claim_rows=claim_rows,claim_evidence=claim_evidence,expected_claim_ids=expected_claim_ids,provisional_bundle=bundle,resolution_result=resolution)
    verification=[];matrix=[]
    for row in claim_rows:
        used=sorted(e["evidence_id"] for e in claim_evidence if e["claim_id"]==row["claim_id"] and e.get("used_in_original_verification"))
        verification.append({
          "claim_id":row["claim_id"],"section_id":row["section_id"],"claim":row["original_claim_text"],
          "verdict":_normalize_verdict(row["source_verdict"]),
          "confidence":"" if row.get("source_verification_confidence") is None else str(row["source_verification_confidence"]),
          "hallucination_risk":_normalize_risk(row["source_hallucination_risk"]),
          "correction_needed":str(bool(row["terminal_correction_recommendation"] or row["has_correction_proposal"])).lower(),
          "evidence_used_ids":"; ".join(used),"confidence_status":row.get("source_confidence_status","NOT_AVAILABLE_IN_SOURCE_CONTRACT"),
        })
        matrix.append({"claim_id":row["claim_id"],"section_id":row["section_id"],"claim":row["original_claim_text"],"verdict":_normalize_verdict(row["source_verdict"]),"evidence_used_ids":"; ".join(used),"correction_ids":"; ".join(row["correction_ids"]),"remaining_issue_codes":"; ".join(row["provisional_remaining_issue_codes"]),"manual_review_required":str(bool(row["manual_review_required"])).lower()})
    logs=[]
    for plan in sorted(eligible_plans,key=lambda p:(p["section_id"],p["claim_id"])):
        # Original 07C allows one applied row per claim; preserve the full selected correction identities in one canonical row.
        patches=sorted(plan.get("selected_patch_records",()),key=lambda p:(-p["target_span_in_claim"]["start"],-p["target_span_in_claim"]["end"],p["correction_id"]))
        logs.append({"claim_id":plan["claim_id"],"section_id":plan["section_id"],"status":"applied","application_scope":"COPY_ONLY","original_draft_modified":"false","action":"MULTI_PATCH" if len(patches)>1 else ("PATCH" if patches else "NO_OP"),"old_fragment":plan["original_claim_text"],"new_fragment":plan["virtual_result_text"],"correction_id":";".join(p["correction_id"] for p in patches)})
    # Build provisional artifacts with validation_ok false; derive it only after structural checks.
    unavailable_confidence=any(r.get("confidence_status")=="NOT_AVAILABLE_IN_SOURCE_CONTRACT" for r in verification)
    validation_report={"experiment_id":experiment_id,"structural_validation_ok":False,"scientific_handoff_checks":{name:False for name in SCIENTIFIC_HANDOFF_CHECK_NAMES},"scientific_handoff_validation_ok":False,"original_07c_artifact_gate_ok":False,"validation_ok":False,"validation_errors":[],"validation_warnings":["source confidence unavailable in source contract" ] if unavailable_confidence else []}
    manifest={
      "stage":"07_agente_verificador_trazabilidad_adapter","experiment_id":experiment_id,
      "fingerprint":resolution["multi_proposal_resolution_fingerprint"] or resolution["multi_proposal_audit_fingerprint"],
      "validation_report":validation_report,
      "workflow_state":{
        "verification_completed":True,
        "post_correction_recheck_required":bool(logs),
        "pending_manual_review":bool(manual),
        "manual_review_claim_ids":manual,
        "claim_level_manual_review_count":len(claim_level_manual),
        "resolution_level_manual_review_count":len(resolution_level_manual),
      },
      "safety_policy":derived_safety,
      "source_contracts":{"bundle_fingerprint":bundle["normalized_bundle_fingerprint"],"bundle_audit_fingerprint":bundle["aggregation_audit_fingerprint"],"resolution_fingerprint":resolution["multi_proposal_resolution_fingerprint"]},
      "correction_application":{"application_scope":"COPY_ONLY","original_draft_modified":False,"evaluation_ready_emitted":False},
    }
    payloads={
      "verified_state_of_art.json":_pretty_json_bytes(copy_draft),"verified_state_of_art.md":corrected_md.encode("utf-8"),
      "verification_report.csv":_csv_bytes(verification,VERIFICATION_REPORT_COLUMNS),
      "claim_traceability_matrix.csv":_csv_bytes(matrix,("claim_id","section_id","claim","verdict","evidence_used_ids","correction_ids","remaining_issue_codes","manual_review_required")),
      "auto_corrections_log.csv":_csv_bytes(logs,CORRECTION_LOG_COLUMNS),
      "verification_validation_report.json":_pretty_json_bytes(validation_report),
      "verification_traceability_manifest.json":_pretty_json_bytes(manifest),
    }
    # Derive the three validation layers independently. The seven-artifact gate is only
    # the structural notebook gate; it is not the complete scientific validation of 07C.
    eligible_ids={str(p["claim_id"]) for p in eligible_plans}
    context_by_claim={str(c["claim_id"]):c for c in contexts}
    authorized_terminal_evidence=_authorized_terminal_evidence(handoff=agent06_handoff,rag_records=tuple(runtime_result["execution_metrics"]["independent_rag_claim_records"]))
    matrix_ids={str(r["claim_id"]) for r in matrix}
    log_ids={str(r["claim_id"]) for r in logs}
    checks={
      "claim_coverage_ok": bundle_claim_ids==expected_claim_ids==set(agent06_handoff["expected_claim_ids"]),
      "section_identity_ok": all(str(context_by_claim.get(str(r["claim_id"]),{}).get("section_id"))==str(r["section_id"]) for r in claim_rows),
      "authorized_evidence_ok": all(e.get("authorized_for_section") is True and (str(e["evidence_id"]),str(e["source_filename"]),str(e["chunk_id"]),True,str(e["text_fingerprint"])) in authorized_terminal_evidence.get(str(e["claim_id"]),set()) for e in claim_evidence),
      "eligible_manual_disjoint_ok": not bool(set(manual)&eligible_ids),
      "source_draft_fingerprint_ok": canonical_source_fp==committed_source_draft_fingerprint and all(c.get("source_draft_fingerprint")==committed_source_draft_fingerprint for c in contexts),
      "patch_application_ok": log_ids==eligible_ids and all(r["status"]=="applied" and r["application_scope"]=="COPY_ONLY" and r["original_draft_modified"]=="false" for r in logs),
      "json_markdown_consistency_ok": all(text in corrected_md for text in corrected_section_texts.values()),
      "correction_log_consistency_ok": len(logs)==len(log_ids)==len(eligible_ids),
      "claim_traceability_consistency_ok": matrix_ids==expected_claim_ids and len(matrix)==len(expected_claim_ids),
    }
    validation_report["structural_validation_ok"]=True
    validation_report["scientific_handoff_checks"]=checks
    validation_report["scientific_handoff_validation_ok"]=all(checks.values())
    validation_report["original_07c_artifact_gate_ok"]=True
    validation_report["validation_ok"]=validation_report["structural_validation_ok"] and validation_report["scientific_handoff_validation_ok"] and validation_report["original_07c_artifact_gate_ok"]
    manifest["validation_report"]=validation_report
    payloads["verification_validation_report.json"]=_pretty_json_bytes(validation_report)
    payloads["verification_traceability_manifest.json"]=_pretty_json_bytes(manifest)
    try:validate_original_agent07c_input_artifacts(artifact_payloads=payloads,experiment_id=experiment_id)
    except Exception as exc:
        validation_report["original_07c_artifact_gate_ok"]=False;validation_report["validation_ok"]=False;validation_report["validation_errors"]=[type(exc).__name__]
        raise
    hashes={name:_sha(raw) for name,raw in payloads.items()}
    hallucination=[{"claim_id":r["claim_id"],"section_id":r["section_id"],"hallucination_risk":r["hallucination_risk"]} for r in verification]
    citations=[{"claim_id":e["claim_id"],"section_id":e["section_id"],"evidence_id":e["evidence_id"],"source_filename":e["source_filename"],"chunk_id":e["chunk_id"],"text_fingerprint":e["text_fingerprint"],"authorized_for_section":str(bool(e["authorized_for_section"])).lower(),"used_in_original_verification":str(bool(e["used_in_original_verification"])).lower()} for e in claim_evidence]
    atoms=[{"claim_id":r["claim_id"],"section_id":r["section_id"],"claim":r["original_claim_text"],"claim_type":r["claim_type"]} for r in claim_rows]
    manual_queue=[{
      "claim_id":row["claim_id"],
      "section_id":row["section_id"],
      "claim":row["original_claim_text"],
      "verdict":_normalize_verdict(row["source_verdict"]),
      "hallucination_risk":_normalize_risk(row["source_hallucination_risk"]),
      "source_issue_codes":"; ".join(row["source_issue_codes"]),
      "remaining_issue_codes":"; ".join(row["provisional_remaining_issue_codes"]),
    } for row in claim_rows if row.get("manual_review_required")]
    optional={
      "hallucination_report.csv":_csv_bytes(hallucination,("claim_id","section_id","hallucination_risk")),
      "citation_check.csv":_csv_bytes(citations,("claim_id","section_id","evidence_id","source_filename","chunk_id","authorized_for_section","used_in_original_verification")),
      "claim_atomization_log.csv":_csv_bytes(atoms,("claim_id","section_id","claim","claim_type")),
    }
    if manual_queue:
        optional["manual_review_queue.csv"]=_csv_bytes(
            manual_queue,
            ("claim_id","section_id","claim","verdict","hallucination_risk","source_issue_codes","remaining_issue_codes"),
        )
    return create_agent07c_prepared_input(
      experiment_id=experiment_id,verified_state_of_art=copy_draft,
      eligible_claim_ids=tuple(sorted(str(p["claim_id"]) for p in eligible_plans)),manual_review_claim_ids=tuple(manual),
      post_correction_reverification_claim_ids=tuple(sorted(str(p["claim_id"]) for p in eligible_plans)),artifact_payloads=payloads,artifact_hashes=hashes,optional_artifact_payloads=optional,optional_artifact_hashes={name:_sha(raw) for name,raw in optional.items()},
      source_draft_fingerprint=canonical_source_fp,prepared_draft_fingerprint=_sha(_canonical_json_bytes(copy_draft)),
      correction_applied_to_copy=bool(eligible_plans),original_draft_modified=False,evaluation_ready_emitted=False,
    )
