from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import pandas as pd
from src.contracts.agent_input import AgentInput

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

@dataclass(frozen=True)
class ThematicDependencyBundle:
    kb_csv: Path; kb_jsonl: Path; manifest03: Path; quantitative_files: dict; quantitative_manifest: dict|None

def _check_ref(agent_input,name,required=True):
    ref=agent_input.dependencies.get(name)
    if ref is None:
        if required: raise FileNotFoundError(f"{name} no fue proporcionado")
        return None
    p=Path(ref.path)
    if not p.is_file(): raise FileNotFoundError(f"{name} no existe")
    if sha256_file(p)!=ref.hash: raise ValueError(f"DEPENDENCY_HASH_MISMATCH:{name}")
    return p

def _forbid_gt(agent_input):
    text=json.dumps(agent_input.to_dict(),ensure_ascii=False).casefold()
    forbidden=['ground_truth_path','ground_truth_file','ground_truth_hash','ground_truth_content','reference_state_of_art','published_review_text']
    if any(x in text for x in forbidden): raise ValueError('GROUND_TRUTH_POLICY_VIOLATION')

def validate_dependencies(agent_input:AgentInput):
    _forbid_gt(agent_input)
    kb=_check_ref(agent_input,'scientific_knowledge_base_csv')
    kbj=_check_ref(agent_input,'scientific_knowledge_base_jsonl')
    man=_check_ref(agent_input,'scientific_extraction_manifest')
    m=json.loads(man.read_text(encoding='utf-8'))
    if str(m.get('experiment_id'))!=agent_input.experiment_id: raise ValueError('SCIENTIFIC_EXTRACTION_MANIFEST_MISMATCH')
    if m.get('run_id') not in (None,'',agent_input.run_id): raise ValueError('SCIENTIFIC_EXTRACTION_MANIFEST_MISMATCH')
    if str(m.get('stage','')).startswith('03') is False: raise ValueError('SCIENTIFIC_EXTRACTION_MANIFEST_MISMATCH')
    safety=m.get('safety_policy',{})
    if any(bool(safety.get(k)) for k in ['uses_ground_truth','uses_external_knowledge','uses_review_sections','uses_bibliography']): raise ValueError('GROUND_TRUTH_POLICY_VIOLATION')
    df=pd.read_csv(kb)
    req={'source_filename','title','include_in_state_of_art'}
    if not req.issubset(df.columns): raise ValueError('INVALID_KB_SCHEMA')
    qnames=['quantitative_comparative_table','quantitative_datasets_table','quantitative_techniques_table','dataset_technique_summary','quantitative_extraction_manifest']
    present={n:agent_input.dependencies.get(n) for n in qnames}
    count=sum(v is not None for v in present.values())
    qfiles={}; qmanifest=None
    if count not in (0,len(qnames)): raise FileNotFoundError('QUANTITATIVE_ARTIFACT_NOT_FOUND')
    if count:
        for n in qnames: qfiles[n]=_check_ref(agent_input,n)
        qmanifest=json.loads(qfiles['quantitative_extraction_manifest'].read_text(encoding='utf-8'))
        if str(qmanifest.get('experiment_id'))!=agent_input.experiment_id: raise ValueError('QUANTITATIVE_MANIFEST_MISMATCH')
        if qmanifest.get('run_id') not in (None,'',agent_input.run_id): raise ValueError('QUANTITATIVE_MANIFEST_MISMATCH')
        qs=qmanifest.get('safety_policy',{})
        if any(bool(qs.get(k)) for k in ['uses_ground_truth','uses_external_knowledge','uses_review_sections','uses_bibliography']): raise ValueError('GROUND_TRUTH_POLICY_VIOLATION')
    return ThematicDependencyBundle(kb,kbj,man,qfiles,qmanifest),df,m
