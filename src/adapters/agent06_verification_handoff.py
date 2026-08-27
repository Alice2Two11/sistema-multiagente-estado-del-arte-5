"""Contractual hand-off from committed Agent 06 artifacts into Agent 07.

This adapter does not alter Agent 06 artifacts and does not infer scientific
outputs that belong to Agent 07.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import csv, hashlib, io, json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.contracts.agent_result import AgentResult, ExecutionStatus
from src.orchestration.decision_log_frontier import committed_predecessor_for_stage
from src.state.fingerprints import sha256_file
from src.state.state_store import StateStore
from src.tools.verification.corrections import fingerprint_text

AGENT06_REQUIRED_ARTIFACTS = (
    "state_of_art_draft.json", "state_of_art_draft.md", "draft_sections.csv",
    "draft_rag_evidence.csv", "draft_claim_evidence.csv",
    "numeric_hallucination_check.csv", "draft_validation_report.json",
    "draft_generation_manifest.json",
)

@dataclass(frozen=True, slots=True)
class Agent07RetrieverBinding:
    experiment_id: str
    collection_name: str
    embedding_model: str
    chroma_manifest_fingerprint: str
    chunks_manifest_fingerprint: str

    def to_dict(self) -> dict[str, Any]: return asdict(self)

AGENT06_HANDOFF_REQUIRED_FIELDS = {
    "commit_status", "run_id", "experiment_id", "artifact_identity", "schema_version",
    "source_draft_fingerprint", "agent06_manifest_source_draft_fingerprint",
    "claim_verification_contexts", "expected_claim_ids", "claim_inventory_fingerprint",
    "agent06_decision_id", "outline_mapping_fingerprint", "claim_identity_migration_signal",
    "integration_metadata",
}

def validate_agent06_verification_handoff_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != AGENT06_HANDOFF_REQUIRED_FIELDS:
        raise ValueError("AGENT07_AGENT06_HANDOFF_SCHEMA_INVALID")
    out=deepcopy(dict(value))
    for key in ("run_id","experiment_id","artifact_identity","schema_version","agent06_decision_id"):
        if not isinstance(out[key],str) or not out[key].strip(): raise ValueError(f"AGENT07_AGENT06_HANDOFF_FIELD_INVALID:{key}")
    if out["commit_status"] != "COMMITTED": raise ValueError("AGENT07_AGENT06_HANDOFF_NOT_COMMITTED")
    if not isinstance(out["claim_identity_migration_signal"], bool): raise ValueError("AGENT07_AGENT06_HANDOFF_FIELD_INVALID:claim_identity_migration_signal")
    for key in ("source_draft_fingerprint","claim_inventory_fingerprint","outline_mapping_fingerprint"):
        v=out[key]
        if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v): raise ValueError(f"AGENT07_AGENT06_HANDOFF_FINGERPRINT_INVALID:{key}")
    contexts=out["claim_verification_contexts"]
    expected=tuple(out["expected_claim_ids"])
    if not isinstance(contexts,(tuple,list)) or not contexts: raise ValueError("AGENT07_AGENT06_HANDOFF_CONTEXTS_INVALID")
    ids=[]; pairs=set()
    for ctx in contexts:
        if not isinstance(ctx,Mapping): raise ValueError("AGENT07_AGENT06_HANDOFF_CONTEXT_INVALID")
        cid=str(ctx.get("claim_id") or ""); sid=str(ctx.get("section_id") or "")
        if not cid or not sid: raise ValueError("AGENT07_AGENT06_HANDOFF_CONTEXT_IDENTITY_INVALID")
        if cid in ids: raise ValueError(f"AGENT07_AGENT06_GLOBAL_CLAIM_ID_DUPLICATE:{cid}")
        if (sid,cid) in pairs: raise ValueError("AGENT07_AGENT06_HANDOFF_CONTEXT_DUPLICATE")
        ids.append(cid); pairs.add((sid,cid))
    for ctx in contexts:
        authorized=ctx.get("authorized_source_filenames")
        if not isinstance(authorized,(tuple,list)) or any(not isinstance(x,str) or not x.strip() for x in authorized):
            raise ValueError("AGENT07_AGENT06_AUTHORIZED_SOURCES_INVALID")
        if len(set(authorized)) != len(tuple(authorized)) or tuple(authorized) != tuple(sorted(authorized)):
            raise ValueError("AGENT07_AGENT06_AUTHORIZED_SOURCES_AMBIGUOUS")
        for evidence in ctx.get("eligible_evidence",()):
            if evidence.get("authorized_for_section") is True and str(evidence.get("source_filename") or "") not in set(authorized):
                raise ValueError("AGENT07_AGENT06_AUTHORIZED_SOURCE_MISMATCH")
    if tuple(sorted(ids)) != tuple(sorted(expected)): raise ValueError("AGENT07_AGENT06_HANDOFF_CLAIM_COVERAGE_MISMATCH")
    if not isinstance(out["integration_metadata"],Mapping): raise ValueError("AGENT07_AGENT06_HANDOFF_INTEGRATION_METADATA_INVALID")
    return out


def _sha256_json(value: Any) -> str:
    raw=json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str,str]]:
    with path.open("r",encoding="utf-8-sig",newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _artifact_basename(ref: Any) -> str:
    return Path(ref.path).name


def resolve_committed_agent06_artifacts(
    *, store: StateStore, stage_name: str, target_stage: str = "07_agente_verificador"
) -> tuple[Any, AgentResult, dict[str, Path], str]:
    """Resolve only the exact AgentResult committed for the exact Agent 06 stage.

    Antes: leía ``state.stages[stage_name]`` (el ``StageState`` VIGENTE)
    y ``[e for e in decision_log if e.stage==stage_name][-1]`` (la
    última entrada CRONOLÓGICA para esa etapa) -- ambos pueden reflejar
    una ejecución espuria posterior a un ``HALT_STAGE`` terminal ya
    comprometido en OTRA etapa (ej. 07), que sobrescribe el
    ``StageState`` de 06 y aparece como su entrada más reciente en el
    log, aunque nunca fue una continuación causal legítima de nada.

    Corrección posterior (parche 11): se reemplazó esa lectura por
    ``authoritative_decision_log_entry_for_stage`` -- pero esa función
    responde una pregunta DISTINTA ("¿cuál es el estado terminal
    VIGENTE de 06, para reconocer un restart?"), no la que hace falta
    aquí. Si 06 tuvo una ejecución espuria de OTRA naturaleza (ej.
    ``06/HALT_STAGE`` que aparece después de que 07 YA había alcanzado
    su propio ``HALT_STAGE`` terminal, sin ninguna transición que
    señalara esa reejecución) esa función seguía devolviendo el
    espurio como "vigente" -- correcto para reconocer un restart, pero
    equivocado para resolver la dependencia real que habilitó a 07.

    Ahora: se usa ``committed_predecessor_for_stage`` -- responde la
    pregunta correcta para un hand-off upstream: "¿cuál fue el ÚLTIMO
    commit de ``stage_name`` que REALMENTE avanzó
    (``COMPLETED``/``ADVANCE``) hacia ``target_stage``?" -- excluye por
    diseño cualquier entrada posterior que no esté precedida por esa
    transición causal exacta, sin importar si es "la más reciente"
    cronológicamente."""

    state=store.load()
    if stage_name not in state.stages:
        raise ValueError("AGENT07_AGENT06_STAGE_NOT_FOUND")
    log = committed_predecessor_for_stage(state.decision_log, predecessor=stage_name, target=target_stage)
    if log is None:
        raise ValueError("AGENT07_AGENT06_COMMIT_LOG_NOT_FOUND")
    result=AgentResult.from_dict(log.result)
    if result.execution_status != ExecutionStatus.COMPLETED:
        raise ValueError("AGENT07_AGENT06_RESULT_NOT_COMPLETED")
    refs={Path(ref.path).name: ref for ref in result.output_artifacts.values()}
    missing=sorted(set(AGENT06_REQUIRED_ARTIFACTS)-set(refs))
    if missing:
        raise ValueError("AGENT07_AGENT06_REQUIRED_ARTIFACTS_MISSING:"+",".join(missing))
    paths: dict[str,Path]={}
    for name in AGENT06_REQUIRED_ARTIFACTS:
        ref=refs[name]; path=Path(ref.path)
        if not path.is_file(): raise ValueError(f"AGENT07_AGENT06_ARTIFACT_MISSING:{name}")
        if sha256_file(path)!=ref.hash: raise ValueError(f"AGENT07_AGENT06_ARTIFACT_HASH_MISMATCH:{name}")
        paths[name]=path
    return state,result,paths,log.decision_id


def validate_agent07_experiment_compatibility(*, active_config: Mapping[str,Any], agent07_config: Mapping[str,Any], experiment_paths: Mapping[str,str]) -> None:
    if not isinstance(active_config,Mapping) or not isinstance(agent07_config,Mapping):
        raise ValueError("AGENT07_GLOBAL_CONFIG_INVALID")
    required_paths={"code_root":"/content/tesis_codigo","project_root":"/content/proyecto_estado_arte","experiment_root":"/content/proyecto_estado_arte/experimento_paper_02"}
    for key,expected in required_paths.items():
        actual=experiment_paths.get(key)
        if actual != expected: raise ValueError(f"AGENT07_EXPERIMENT_PATH_MISMATCH:{key}")
    aliases={
      "verification_policy":("verification_policy",),
      "verification_prompt_version":("verification_prompt_version",),
      "verification_budgets":("verification_budgets",),
      "verification_model":("verification_model","openai_model"),
      "correction_model":("correction_model","openai_model"),
    }
    for out_key,candidates in aliases.items():
        expected=next((active_config.get(k) for k in candidates if active_config.get(k) is not None),None)
        actual=agent07_config.get(out_key)
        if expected is not None and actual != expected:
            raise ValueError(f"AGENT07_GLOBAL_CONFIG_MISMATCH:{out_key}")


def validate_productive_retriever_binding(*, binding: Agent07RetrieverBinding|Mapping[str,Any], active_config: Mapping[str,Any], chroma_manifest_path: str|Path, chunks_manifest_path: str|Path, committed_experiment_id: str) -> dict[str,Any]:
    p=asdict(binding) if isinstance(binding,Agent07RetrieverBinding) else dict(binding)
    required=set(Agent07RetrieverBinding.__dataclass_fields__)
    if set(p)!=required: raise ValueError("AGENT07_RETRIEVER_BINDING_SCHEMA_INVALID")
    cm_path, ch_path=Path(chroma_manifest_path),Path(chunks_manifest_path)
    if not cm_path.is_file() or not ch_path.is_file(): raise ValueError("AGENT07_RETRIEVER_MANIFEST_MISSING")
    cm=_read_json(cm_path)
    if p["experiment_id"] != committed_experiment_id or cm.get("experiment_id") != committed_experiment_id:
        raise ValueError("AGENT07_RETRIEVER_EXPERIMENT_MISMATCH")
    expected_collection=active_config.get("chroma_collection_name") or active_config.get("collection_name")
    expected_embedding=active_config.get("embedding_model")
    if p["collection_name"]!=expected_collection or cm.get("collection_name")!=expected_collection:
        raise ValueError("AGENT07_RETRIEVER_COLLECTION_MISMATCH")
    if p["embedding_model"]!=expected_embedding or cm.get("embedding_model")!=expected_embedding:
        raise ValueError("AGENT07_RETRIEVER_EMBEDDING_MODEL_MISMATCH")
    if p["chroma_manifest_fingerprint"]!=sha256_file(cm_path): raise ValueError("AGENT07_CHROMA_MANIFEST_FINGERPRINT_MISMATCH")
    if p["chunks_manifest_fingerprint"]!=sha256_file(ch_path): raise ValueError("AGENT07_CHUNKS_MANIFEST_FINGERPRINT_MISMATCH")
    return p


def _section_map(draft_json: Any, section_rows: Sequence[Mapping[str,str]]) -> dict[str,str]:
    result={}
    for row in section_rows:
        sid=str(row.get("section_id") or "").strip(); text=row.get("section_text") or row.get("text") or row.get("content") or row.get("draft_text")
        if sid and isinstance(text,str): result[sid]=text
    sections=draft_json.get("sections",[]) if isinstance(draft_json,Mapping) else []
    for sec in sections:
        if isinstance(sec,Mapping):
            sid=str(sec.get("section_id") or sec.get("id") or "").strip(); text=sec.get("section_text") or sec.get("text") or sec.get("content")
            if sid and isinstance(text,str): result.setdefault(sid,text)
    return result


def build_agent07_input_from_committed_agent06(*, store: StateStore, stage_name: str, agent07_config: Mapping[str,Any], policy_versions: Mapping[str,str], schema_versions: Mapping[str,str], experiment_paths: Mapping[str,str], outline_paper_mapping_path: str|Path) -> dict[str,Any]:
    """Build Agent 07 contexts only from the exact committed Agent 06 result and the committed outline authorization mapping."""
    for name,value in (("agent07_config",agent07_config),("policy_versions",policy_versions),("schema_versions",schema_versions),("experiment_paths",experiment_paths)):
        if not isinstance(value,Mapping): raise ValueError(f"AGENT07_AGENT06_{name.upper()}_INVALID")
    state,result,paths,agent06_decision=resolve_committed_agent06_artifacts(store=store,stage_name=stage_name)
    mapping_path=Path(outline_paper_mapping_path)
    if not mapping_path.is_file(): raise ValueError("AGENT07_OUTLINE_MAPPING_MISSING")
    mapping_rows=_read_csv(mapping_path)
    allowed_by_section: dict[str,set[str]]={}
    for row in mapping_rows:
        sid=str(row.get("section_id") or "").strip(); source=str(row.get("source_filename") or "").strip()
        if not sid or not source: raise ValueError("AGENT07_OUTLINE_MAPPING_ROW_INVALID")
        allowed_by_section.setdefault(sid,set()).add(source)
    draft=_read_json(paths["state_of_art_draft.json"]); sections=_read_csv(paths["draft_sections.csv"])
    claim_rows=_read_csv(paths["draft_claim_evidence.csv"]); rag_rows=_read_csv(paths["draft_rag_evidence.csv"])
    numeric_rows=_read_csv(paths["numeric_hallucination_check.csv"]); manifest=_read_json(paths["draft_generation_manifest.json"])
    section_texts=_section_map(draft,sections)
    raw_by_claim: dict[tuple[str,str],dict[str,dict[str,Any]]]={}
    for row in [*claim_rows,*rag_rows]:
        cid=str(row.get("claim_id") or "").strip(); sid=str(row.get("section_id") or "").strip()
        eid=str(row.get("evidence_id") or row.get("chunk_id") or "").strip()
        if not cid or not sid or not eid: continue
        source=str(row.get("source_filename") or "").strip(); chunk=str(row.get("chunk_id") or "").strip(); text=str(row.get("text") or row.get("chunk_text") or row.get("evidence_text") or "")
        if not source or not chunk or not text: raise ValueError(f"AGENT07_AGENT06_EVIDENCE_PROVENANCE_MISSING:{sid}:{cid}:{eid}")
        authorized=source in allowed_by_section.get(sid,set())
        ev={"evidence_id":eid,"source_filename":source,"chunk_id":chunk,"text":text,"usage_role":str(row.get("usage_role") or "ELIGIBLE"),"authorized_for_section":authorized}
        bucket=raw_by_claim.setdefault((sid,cid),{})
        previous=bucket.get(eid)
        if previous is None: bucket[eid]=ev
        elif _sha256_json(previous)!=_sha256_json(ev): raise ValueError(f"AGENT07_AGENT06_EVIDENCE_CONFLICTING_DUPLICATE:{sid}:{cid}:{eid}")
    numeric={}
    for r in numeric_rows:
        key=(str(r.get("section_id") or "").strip(),str(r.get("claim_id") or "").strip())
        value=r.get("numeric_risk") or r.get("risk")
        if key[0] and key[1] and value not in (None,""): numeric[key]=str(value)
    # Authoritative claim inventory: Agent 06 committed draft sections[].claims, with the
    # same deterministic IDs used by Agent 06 when it writes draft_claim_evidence.csv.
    inventory: dict[tuple[str,str],dict[str,Any]]={}
    global_claim_ids: set[str]=set()
    draft_sections=draft.get("sections",()) if isinstance(draft,Mapping) else ()
    if not isinstance(draft_sections,list): raise ValueError("AGENT07_AGENT06_CLAIM_INVENTORY_INVALID")
    for section_record in draft_sections:
        if not isinstance(section_record,Mapping): raise ValueError("AGENT07_AGENT06_CLAIM_INVENTORY_INVALID")
        sid=str(section_record.get("section_id") or section_record.get("id") or "").strip()
        claims=section_record.get("claims",())
        if not sid or not isinstance(claims,list): raise ValueError("AGENT07_AGENT06_CLAIM_INVENTORY_INVALID")
        for index,claim in enumerate(claims,start=1):
            if not isinstance(claim,Mapping): raise ValueError(f"AGENT07_AGENT06_CLAIM_INVENTORY_ITEM_INVALID:{sid}:{index}")
            cid=str(claim.get("claim_id") or f"{sid}_C{index}").strip()
            text=str(claim.get("claim") or claim.get("claim_text") or claim.get("original_claim_text") or "").strip()
            if not cid or not text: raise ValueError(f"AGENT07_AGENT06_CLAIM_INVENTORY_ITEM_INVALID:{sid}:{index}")
            key=(sid,cid)
            if key in inventory: raise ValueError(f"AGENT07_AGENT06_CLAIM_ID_DUPLICATE:{sid}:{cid}")
            if cid in global_claim_ids: raise ValueError(f"AGENT07_AGENT06_GLOBAL_CLAIM_ID_DUPLICATE:{cid}")
            global_claim_ids.add(cid)
            inventory[key]={"claim_id":cid,"claim_uid":str(claim.get("claim_uid") or ""),"claim_version":claim.get("claim_version"),"section_id":sid,"claim_text":text,"inventory_position":index,"section_title":str(section_record.get("section_title") or "").strip(),"source_free_organizational_section":bool(section_record.get("source_free_organizational_section") is True or (isinstance(section_record.get("section_validation"),Mapping) and section_record.get("section_validation",{}).get("source_free_organizational_section") is True)),"supporting_citations":tuple(claim.get("supporting_citations",()) or ())}
    if not inventory: raise ValueError("AGENT07_AGENT06_NO_CLAIM_INVENTORY")
    evidence_claim_keys=set(raw_by_claim)
    unknown_evidence=sorted(evidence_claim_keys-set(inventory))
    if unknown_evidence:
        sid,cid=unknown_evidence[0]; raise ValueError(f"AGENT07_AGENT06_UNKNOWN_EVIDENCE_CLAIM:{sid}:{cid}")
    contexts=[]
    canonical_draft_fingerprint=_sha256_json(draft)
    declared=manifest.get("source_draft_fingerprint")
    source_draft_fingerprint=canonical_draft_fingerprint
    for (sid,cid),item in sorted(inventory.items()):
        text=item["claim_text"];section=section_texts.get(sid)
        if not isinstance(section,str): raise ValueError(f"AGENT07_AGENT06_SECTION_TEXT_MISSING:{sid}")
        claim_rows_for_id=[r for r in claim_rows if str(r.get("section_id") or "").strip()==sid and str(r.get("claim_id") or "").strip()==cid]
        explicit={(r.get("claim_start") or r.get("start"),r.get("claim_end") or r.get("end")) for r in claim_rows_for_id if (r.get("claim_start") or r.get("start")) not in (None,"") and (r.get("claim_end") or r.get("end")) not in (None,"")}
        if len(explicit)>1: raise ValueError(f"AGENT07_AGENT06_CLAIM_SPAN_CONFLICT:{sid}:{cid}")
        if explicit:
            start_raw,end_raw=next(iter(explicit));start,end=int(start_raw),int(end_raw)
        else:
            if section.count(text)!=1: raise ValueError(f"AGENT07_AGENT06_CLAIM_SPAN_AMBIGUOUS:{sid}:{cid}")
            start=section.index(text);end=start+len(text)
        if section[start:end]!=text: raise ValueError(f"AGENT07_AGENT06_CLAIM_SPAN_MISMATCH:{sid}:{cid}")
        cfp=fingerprint_text(text);sfp=fingerprint_text(section)
        evidence=tuple(sorted(raw_by_claim.get((sid,cid),{}).values(),key=lambda x:x["evidence_id"]))
        contexts.append({
          "claim_id":cid,"claim_uid":item.get("claim_uid") or "","claim_id_origin":"inherited_agent06","section_id":sid,"section_title":item["section_title"],"original_claim_text":text,"section_text":section,"supporting_citations":item["supporting_citations"],"source_free_organizational_section":item["source_free_organizational_section"],
          "claim_span_in_section":{"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS","base_text_fingerprint":sfp,"start":start,"end":end,"text":text},
          "claim_fingerprint":cfp,"section_fingerprint":sfp,"eligible_evidence":evidence,
          "authorized_source_filenames":tuple(sorted(allowed_by_section.get(sid,set()))),
          "source_draft_fingerprint":source_draft_fingerprint,
          "numeric_risk":numeric.get((sid,cid)),"numeric_risk_status":"EVALUATED" if (sid,cid) in numeric else "NOT_AVAILABLE",
          "field_provenance":{"claim_id":"state_of_art_draft.json:sections[].claims","claim_uid":"state_of_art_draft.json:sections[].claims[].claim_uid","claim_id_origin":"committed Agent06 inventory","section_id":"state_of_art_draft.json:sections[].section_id","section_title":"state_of_art_draft.json:sections[].section_title","original_claim_text":"state_of_art_draft.json:sections[].claims","supporting_citations":"state_of_art_draft.json:sections[].claims[].supporting_citations","source_free_organizational_section":"state_of_art_draft.json:section_validation","section_text":"draft_sections.csv|state_of_art_draft.json","claim_span_in_section":"explicit artifact coordinates|unique exact location","eligible_evidence":"draft_claim_evidence.csv|draft_rag_evidence.csv + outline_paper_mapping.csv authorization","authorized_source_filenames":"outline_paper_mapping.csv","numeric_risk":"numeric_hallucination_check.csv" if (sid,cid) in numeric else "ABSENT","source_draft_fingerprint":"draft_generation_manifest.json|canonical draft fingerprint"},
        })
    if not contexts: raise ValueError("AGENT07_AGENT06_NO_CLAIM_CONTEXTS")
    # claim_identity_migration_signal: EXCLUSIVAMENTE automático, nunca
    # configurado a mano por experimento. True únicamente cuando las
    # tres condiciones se cumplen a la vez:
    #   1) el ciclo escrito tiene declarado el contrato LEGACY (no
    #      None, no ya STABLE_UID_V1 -- máximo una vez, nunca se vuelve
    #      a emitir tras la migración real).
    #   2) TODOS los claims publicados en este draft comprometido tienen
    #      claim_version válido (>=1) -- confirma que se produjeron
    #      realmente vía el mecanismo de identidad (resolve_claim_
    #      identity), no que alguien les puso un claim_uid a mano.
    #   3) TODOS tienen además claim_uid no vacío.
    # Ninguna reconstrucción por similitud, ninguna configuración manual.
    declared_cycle = state.cycles.get("writer_verifier")
    declared_contract_version = declared_cycle.claim_identity_contract_version if declared_cycle is not None else None
    all_claims_stable = bool(inventory) and all(
        bool(item.get("claim_uid"))
        and isinstance(item.get("claim_version"), int)
        and item.get("claim_version") >= 1
        for item in inventory.values()
    )
    claim_identity_migration_signal = declared_contract_version == "LEGACY" and all_claims_stable
    handoff={"commit_status":"COMMITTED","run_id":state.identity.run_id,"experiment_id":state.identity.experiment_id,"artifact_identity":str(manifest.get("artifact_identity") or agent06_decision),"schema_version":state.identity.schema_version,"source_draft_fingerprint":source_draft_fingerprint,"agent06_manifest_source_draft_fingerprint":str(declared) if declared else None,"claim_verification_contexts":tuple(contexts),"expected_claim_ids":tuple(cid for _,cid in sorted(inventory)),"claim_inventory_fingerprint":_sha256_json(tuple(inventory[k] for k in sorted(inventory))),"agent06_decision_id":agent06_decision,"outline_mapping_fingerprint":sha256_file(mapping_path),"claim_identity_migration_signal":claim_identity_migration_signal,"integration_metadata":{"agent07_config_fingerprint":_sha256_json(agent07_config),"policy_versions":dict(policy_versions),"schema_versions":dict(schema_versions),"experiment_paths":dict(experiment_paths)}}
    return validate_agent06_verification_handoff_contract(handoff)
