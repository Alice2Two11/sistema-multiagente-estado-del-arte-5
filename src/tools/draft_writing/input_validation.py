from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd


def _load_json(path):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj,dict):
        raise ValueError(f"INVALID_JSON_OBJECT:{path}")
    return obj


def _to_bool(value):
    if isinstance(value,bool): return value
    return str(value).strip().lower() in {"true","1","yes","si","sí"}


def _safe(value):
    return "" if value is None else str(value).strip()


def validate_draft_dependencies(agent_input):
    required=("outline_json","outline_mapping","outline_validation","outline_manifest","kb_final","thematic_manifest","thematic_validation","chunks_clean")
    for name in required:
        if name not in agent_input.dependencies or not Path(agent_input.dependencies[name].path).is_file():
            raise FileNotFoundError(f"DRAFT_INPUT_NOT_FOUND:{name}")
    outline=_load_json(agent_input.dependencies["outline_json"].path)
    oval=_load_json(agent_input.dependencies["outline_validation"].path)
    oman=_load_json(agent_input.dependencies["outline_manifest"].path)
    tman=_load_json(agent_input.dependencies["thematic_manifest"].path)
    tval=_load_json(agent_input.dependencies["thematic_validation"].path)
    cman=_load_json(agent_input.dependencies["chroma_manifest"].path) if "chroma_manifest" in agent_input.dependencies else None
    manifest_pairs=[("outline",oman),("thematic",tman)]
    if cman is not None: manifest_pairs.append(("chroma",cman))
    for name,payload in manifest_pairs:
        if payload.get("experiment_id") not in (None,agent_input.experiment_id):
            raise ValueError(f"{name.upper()}_MANIFEST_MISMATCH")
    if not oval.get("validation_ok",False):
        raise ValueError("OUTLINE_NOT_APPROVED")
    if tval.get("validation_ok") is False:
        raise ValueError("THEMATIC_NOT_APPROVED")
    if oman.get("validation_ok") is False:
        raise ValueError("OUTLINE_MANIFEST_NOT_APPROVED")
    if tman.get("validation_ok") is False:
        raise ValueError("THEMATIC_MANIFEST_NOT_APPROVED")
    safety_pairs=[("outline",oman),("thematic",tman)]
    if cman is not None: safety_pairs.append(("chroma",cman))
    for stage_name,manifest in safety_pairs:
        safety=manifest.get("safety_policy",{})
        if not isinstance(safety,dict):
            raise ValueError(f"INVALID_{stage_name.upper()}_SAFETY_POLICY")
        for key in ("uses_ground_truth","ground_truth_used","uses_external_knowledge"):
            if safety.get(key) is True:
                raise ValueError("GROUND_TRUTH_POLICY_VIOLATION" if "ground_truth" in key else "EXTERNAL_KNOWLEDGE_POLICY_VIOLATION")
    outline_safety=oman.get("safety_policy",{})
    source_flag=outline_safety.get("source_filenames_validated")
    if source_flag is None:
        source_flag=outline_safety.get("source_filenames_validated_and_repaired")
    if source_flag is None:
        source_flag=(oval.get("validation_ok",False) and not oval.get("unresolved_references",[]) and not oval.get("papers_missing_from_coverage",[]) and not oval.get("coverage_entries_not_used",[]) and not oval.get("duplicate_coverage_sources",[]) and not oval.get("coverage_used_sections_mismatches",[]))
    if not bool(source_flag):
        raise ValueError("OUTLINE_SOURCES_NOT_VALIDATED")
    title_flag=outline_safety.get("titles_validated")
    if title_flag is None:
        title_flag=oval.get("validation_ok",False) and not oval.get("unresolved_references",[])
    if not bool(title_flag):
        raise ValueError("OUTLINE_TITLES_NOT_VALIDATED")
    expected_collection=agent_input.agent_context.runtime_resources.get("chroma_collection_name")
    expected_embedding=agent_input.agent_context.runtime_resources.get("embedding_model_name")
    if cman is not None:
        if expected_collection is not None and cman.get("collection_name") != expected_collection:
            raise ValueError("CHROMA_COLLECTION_MISMATCH")
        if expected_embedding is not None and cman.get("embedding_model") != expected_embedding:
            raise ValueError("CHROMA_EMBEDDING_MODEL_MISMATCH")
        for flag in ("ground_truth_indexed","review_sections_indexed","bibliography_indexed","excluded_chunks_indexed"):
            if cman.get(flag,False):
                raise ValueError(f"UNSAFE_CHROMA_INDEX:{flag}")
    kb=pd.read_csv(agent_input.dependencies["kb_final"].path)
    chunks=pd.read_csv(agent_input.dependencies["chunks_clean"].path)
    mapping=pd.read_csv(agent_input.dependencies["outline_mapping"].path)
    if kb.empty or not {"source_filename","title"}.issubset(kb.columns):
        raise ValueError("INVALID_DRAFT_KB_SCHEMA")
    if chunks.empty or not {"source_filename","chunk_id","text"}.issubset(chunks.columns):
        raise ValueError("INVALID_CHUNKS_SCHEMA")
    if kb["source_filename"].astype(str).duplicated().any():
        raise ValueError("DUPLICATE_KB_SOURCE")
    if chunks["chunk_id"].astype(str).duplicated().any():
        raise ValueError("DUPLICATE_CHUNK_ID")
    ground_truth_mask=chunks["source_filename"].fillna("").astype(str).str.lower().str.contains(r"ground[_\s-]*truth|gt_",regex=True)
    if ground_truth_mask.any():
        raise ValueError("GROUND_TRUTH_POLICY_VIOLATION")
    for column in ("is_review_section_chunk","is_bibliography_chunk","excluded_from_rag"):
        if column in chunks.columns and chunks[column].apply(_to_bool).any():
            raise ValueError(f"UNSAFE_CHUNKS:{column}")
    if cman is not None:
        expected_count=int(cman.get("num_chunks_indexed",-1))
        if expected_count != len(chunks):
            raise ValueError("CHROMA_CHUNK_COUNT_MISMATCH")
    sections=outline.get("sections",[])
    if not isinstance(sections,list) or not sections:
        raise ValueError("INVALID_OUTLINE_SCHEMA")
    section_ids=[_safe(s.get("section_id")) for s in sections]
    if not all(section_ids) or len(section_ids)!=len(set(section_ids)):
        raise ValueError("INVALID_OUTLINE_SECTION_IDS")
    required_mapping={"section_id","source_filename","title"}
    if not required_mapping.issubset(mapping.columns):
        raise ValueError("INVALID_OUTLINE_MAPPING_SCHEMA")
    valid_sources=set(kb["source_filename"].astype(str).str.strip())
    source_to_title={_safe(r["source_filename"]):_safe(r["title"]) for _,r in kb.iterrows()}
    mapping_errors=[]; mapping_sources=defaultdict(set); outline_sources=defaultdict(set)
    valid_section_ids=set(section_ids)
    for idx,row in mapping.iterrows():
        sid=_safe(row.get("section_id")); source=_safe(row.get("source_filename")); title=_safe(row.get("title"))
        if sid not in valid_section_ids: mapping_errors.append({"row":int(idx),"reason":"invalid_section_id"}); continue
        if source not in valid_sources: mapping_errors.append({"row":int(idx),"reason":"invalid_source_filename"}); continue
        if title and title!=source_to_title[source]: mapping_errors.append({"row":int(idx),"reason":"title_source_mismatch"}); continue
        mapping_sources[sid].add(source)
    for section in sections:
        sid=_safe(section.get("section_id"))
        for paper in section.get("papers_to_use") or []:
            if not isinstance(paper,dict): mapping_errors.append({"section_id":sid,"reason":"paper_reference_not_object"}); continue
            source=_safe(paper.get("source_filename")); title=_safe(paper.get("title"))
            if source not in valid_sources: mapping_errors.append({"section_id":sid,"reason":"invalid_outline_source"}); continue
            if title!=source_to_title[source]: mapping_errors.append({"section_id":sid,"reason":"outline_title_source_mismatch"}); continue
            outline_sources[sid].add(source)
    for sid in section_ids:
        if mapping_sources[sid] != outline_sources[sid]:
            mapping_errors.append({"section_id":sid,"reason":"mapping_outline_source_set_mismatch"})
    if mapping_errors:
        raise ValueError("OUTLINE_MAPPING_INCONSISTENT:"+json.dumps(mapping_errors,ensure_ascii=False))
    quant=pd.DataFrame(); dataset=pd.DataFrame()
    optional=("quantitative_table","dataset_summary","quantitative_manifest")
    present=[name for name in optional if name in agent_input.dependencies]
    if present and set(present)!=set(optional):
        raise ValueError("INVALID_QUANTITATIVE_CONTEXT")
    if present:
        qman=_load_json(agent_input.dependencies["quantitative_manifest"].path)
        if qman.get("experiment_id") != agent_input.experiment_id:
            raise ValueError("QUANTITATIVE_MANIFEST_MISMATCH")
        qs=qman.get("safety_policy",{})
        for key in ("uses_ground_truth","uses_review_sections","uses_bibliography","uses_external_knowledge"):
            if qs.get(key,False): raise ValueError("INVALID_QUANTITATIVE_CONTEXT")
        quant=pd.read_csv(agent_input.dependencies["quantitative_table"].path)
        dataset=pd.read_csv(agent_input.dependencies["dataset_summary"].path)
        required_quant={"source_filename","metric","value","model_or_method","dataset_or_case","evaluation_scope","data_resolution","verification_status","value_found_in_source_chunk"}
        if not required_quant.issubset(quant.columns):
            raise ValueError("INVALID_QUANTITATIVE_CONTEXT")
        quant=quant[quant["value_found_in_source_chunk"].astype(str).str.lower().isin(["true","1","yes"])].copy()
    return {"outline":outline,"outline_validation":oval,"outline_manifest":oman,"kb":kb,"chunks":chunks,"mapping":mapping,"chroma_manifest":cman,"quantitative":quant,"dataset_summary":dataset}
