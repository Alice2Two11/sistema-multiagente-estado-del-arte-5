from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
from src.state.fingerprints import sha256_file

REQUIRED_KB_COLUMNS={"source_filename","title","research_problem","objective","task_type","target_domain","target_variable_or_object","temporal_horizon_or_scope","methods_or_models","method_families","datasets_or_case_study","input_variables_or_data_sources","evaluation_metrics","main_results","reported_best_method_or_model","limitations_or_gaps","contribution","relevance_for_state_of_art","domain_specific_notes","include_in_state_of_art","retrieved_chunk_ids"}
REQUIRED_CHUNK_COLUMNS={"chunk_id","source_filename","text"}
FORBIDDEN_GT_KEYS=("ground_truth_path","ground_truth_file","ground_truth_content","ground_truth_hash","ground_truth_document")

def to_bool(value):
    if isinstance(value,bool): return value
    return str(value).strip().lower() in {"true","1","yes","sí","si"}

def safe_str(value):
    if value is None: return ""
    try:
        if pd.isna(value): return ""
    except Exception: pass
    return str(value)

def parse_delimited_values(value):
    text=safe_str(value).strip()
    if not text: return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed=json.loads(text)
            if isinstance(parsed,list): return [safe_str(x).strip() for x in parsed if safe_str(x).strip()]
        except Exception: pass
    return [x.strip() for x in re.split(r"[;\n|]+",text) if x.strip()]

def reject_ground_truth_payload(value: Any, location="input"):
    if isinstance(value, Mapping):
        for key,nested in value.items():
            if str(key).casefold() in FORBIDDEN_GT_KEYS: raise ValueError(f"GROUND_TRUTH_POLICY_VIOLATION en {location}.{key}")
            reject_ground_truth_payload(nested,f"{location}.{key}")
    elif isinstance(value,(list,tuple)):
        for i,nested in enumerate(value): reject_ground_truth_payload(nested,f"{location}[{i}]")

def load_and_validate_inputs(*, experiment_id:str, run_id:str, dependencies:Mapping[str,Any], policy:Mapping[str,Any]):
    required=["scientific_knowledge_base_csv","scientific_knowledge_base_jsonl","scientific_extraction_manifest"]
    if policy["verify_values_against_source_chunks"]: required.append("chunks_clean_for_rag_csv")
    for name in required:
        if name not in dependencies: raise FileNotFoundError(f"DEPENDENCY_NOT_FOUND: {name}")
        ref=dependencies[name]; path=Path(ref.path)
        if not path.is_file(): raise FileNotFoundError(f"DEPENDENCY_NOT_FOUND: {name}: {path}")
        if sha256_file(path) != ref.hash: raise ValueError(f"DEPENDENCY_HASH_MISMATCH: {name}")
    kb_path=Path(dependencies["scientific_knowledge_base_csv"].path)
    kb_jsonl=Path(dependencies["scientific_knowledge_base_jsonl"].path)
    manifest_path=Path(dependencies["scientific_extraction_manifest"].path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != experiment_id: raise ValueError("KB_MANIFEST_MISMATCH: experiment_id")
    if manifest.get("run_id") not in {None, run_id}: raise ValueError("KB_MANIFEST_MISMATCH: run_id")
    if manifest.get("stage") not in {None,"03_agente_extraccion_kb","03_extraccion_kb"}: raise ValueError("KB_MANIFEST_MISMATCH: producer_stage")
    safety=manifest.get("safety_policy",{})
    for key in ("ground_truth_used","uses_ground_truth","review_sections_used","uses_review_sections","bibliography_used","uses_bibliography"):
        if safety.get(key,False): raise ValueError(f"GROUND_TRUTH_POLICY_VIOLATION: {key}")
    df=pd.read_csv(kb_path)
    if df.empty: raise ValueError("INVALID_KB_SCHEMA: KB vacía")
    missing=sorted(REQUIRED_KB_COLUMNS-set(df.columns))
    if missing: raise ValueError(f"INVALID_KB_SCHEMA: faltan {missing}")
    if df["source_filename"].duplicated().any(): raise ValueError("DUPLICATE_KB_SOURCE")
    if df["source_filename"].fillna("").astype(str).str.casefold().str.contains(r"ground[_\s-]*truth|gt_",regex=True).any(): raise ValueError("GROUND_TRUTH_POLICY_VIOLATION")
    if policy["only_include_state_of_art_papers"]: df=df[df["include_in_state_of_art"].map(to_bool)].copy()
    if df.empty: raise ValueError("NO_ELIGIBLE_PAPERS")
    chunks=None
    if policy["verify_values_against_source_chunks"]:
        chunks=pd.read_csv(Path(dependencies["chunks_clean_for_rag_csv"].path))
        missing=sorted(REQUIRED_CHUNK_COLUMNS-set(chunks.columns))
        if missing: raise ValueError(f"SOURCE_CHUNKS_SCHEMA_INVALID: faltan {missing}")
        for col in ("is_review_section_chunk","is_bibliography_chunk","excluded_from_rag"):
            if col in chunks and chunks[col].map(to_bool).any(): raise ValueError(f"GROUND_TRUTH_POLICY_VIOLATION: unsafe chunks {col}")
    return df, chunks, manifest
