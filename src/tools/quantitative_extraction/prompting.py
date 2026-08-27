from __future__ import annotations
import json
from .input_validation import safe_str

def compact_row_text(row):
    fields=("source_filename","title","paper_type","research_problem","objective","task_type","target_domain","target_variable_or_object","temporal_horizon_or_scope","methods_or_models","method_families","datasets_or_case_study","input_variables_or_data_sources","evaluation_metrics","main_results","reported_best_method_or_model","limitations_or_gaps","contribution","relevance_for_state_of_art","domain_specific_notes")
    return json.dumps({k:safe_str(row.get(k,"")) for k in fields},ensure_ascii=False,indent=2)

def build_quant_prompt(row):
    return ("Eres un extractor de información científica estructurada.\n"
            "Normaliza únicamente información explícita. No inventes ni uses conocimiento externo. "
            "Conserva valores exactamente y evidencia literal. Devuelve solo JSON válido con "
            "source_filename, paper_title, techniques, datasets, quantitative_results y notes.\nFicha:\n"
            + compact_row_text(row))
