"""Policies exclusive to the invocable 03B quantitative capability."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping

STAGE_NAME = "03B_extraccion_cuantitativa_kb"
QUANT_PROMPT_VERSION = "v3_domain_agnostic_canonical_kb"
QUANT_SCHEMA_VERSION = "v3_scope_resolution_evidence"
QUANT_FLATTENING_VERSION = "v3_dataset_descriptive_metadata_preservation"
QUANT_STAGE_VERSION = "03B_CAPABILITY_V16_DATASET_NORMALIZATION_REPAIR_CANDIDATE"
ARTIFACT_FILENAMES = (
    "structured_quantitative_extraction.json",
    "structured_quantitative_extraction_raw.jsonl",
    "quantitative_extraction_errors.csv",
    "quantitative_comparative_table.csv",
    "quantitative_datasets_table.csv",
    "quantitative_techniques_table.csv",
    "dataset_technique_summary.csv",
    "quantitative_extraction_report.md",
    "quantitative_extraction_manifest.json",
)
PROVISIONAL_DIAGNOSTIC_THRESHOLDS = {
    "status": "PROVISIONAL_NOT_SCIENTIFICALLY_VALIDATED",
    "paper_quantitative_coverage_min": None,
    "source_chunk_confirmation_rate_min": None,
    "unconfirmed_value_rate_max": None,
    "successful_extraction_rate_min": None,
    "dataset_coverage_min": None,
    "technique_coverage_min": None,
    "minimum_usable_quality": None,
}
DEFAULT_QUANTITATIVE_EXTRACTION_POLICY = {
    "temperature": 0.1,
    "auto_rebuild": True,
    "force_rebuild": False,
    "only_include_state_of_art_papers": True,
    "verify_values_against_source_chunks": True,
    "allow_all_clean_chunks_fallback": True,
    "max_attempts": 1,
    "deterministic_flattening_repair": False,
    "diagnostic_thresholds": deepcopy(PROVISIONAL_DIAGNOSTIC_THRESHOLDS),
}

# CONFIG-B (Stage 03B): campos que son responsabilidad de
# 00_setup_config.ipynb (coinciden exactamente con
# FIXED_QUANTITATIVE_EXTRACTION_POLICY del notebook). max_attempts,
# deterministic_flattening_repair y diagnostic_thresholds quedan
# fuera deliberadamente: son contrato interno de 03B, nunca
# responsabilidad de 00.
_REQUIRED_FROM_00 = {
    "temperature",
    "auto_rebuild",
    "force_rebuild",
    "only_include_state_of_art_papers",
    "verify_values_against_source_chunks",
    "allow_all_clean_chunks_fallback",
}

def validate_quantitative_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping): raise TypeError("La política 03B debe ser un mapping.")
    # CONFIG-B (Stage 03B): los 6 campos de _REQUIRED_FROM_00 coinciden
    # exactamente con FIXED_QUANTITATIVE_EXTRACTION_POLICY del notebook
    # 00 -- deben venir de active_experiment.json["quantitative_
    # extraction_policy"], nunca rellenarse en silencio desde
    # DEFAULT_QUANTITATIVE_EXTRACTION_POLICY. El chequeo se hace ANTES
    # de aplicar el default: en particular, "temperature" tiene hoy
    # valores distintos en 00 (0.0) y en el default interno (0.1) --
    # así que sin este chequeo, una ausencia silenciosa de 00 haría
    # que 03B corriera con temperature=0.1 en vez de 0.0, sin ningún
    # aviso. max_attempts/deterministic_flattening_repair/
    # diagnostic_thresholds siguen siendo contrato interno de 03B y
    # SÍ toman su valor de DEFAULT_QUANTITATIVE_EXTRACTION_POLICY si
    # 00 no los proporciona -- 00 nunca tuvo esa responsabilidad.
    missing_from_00 = sorted(_REQUIRED_FROM_00 - set(value))
    if missing_from_00:
        raise ValueError(
            "quantitative_extraction_policy: faltan campos obligatorios "
            "que 00_setup_config.ipynb debe proporcionar (sin default "
            f"interno): {missing_from_00}"
        )
    merged=deepcopy(DEFAULT_QUANTITATIVE_EXTRACTION_POLICY); merged.update(dict(value))
    required=set(DEFAULT_QUANTITATIVE_EXTRACTION_POLICY)-{"diagnostic_thresholds"}
    missing=sorted(required-set(merged))
    if missing: raise ValueError(f"Política 03B incompleta: {missing}")
    temperature=float(merged["temperature"])
    if not 0.0 <= temperature <= 2.0: raise ValueError("temperature debe estar entre 0 y 2.")
    merged["temperature"]=temperature
    for key in ("auto_rebuild","force_rebuild","only_include_state_of_art_papers","verify_values_against_source_chunks","allow_all_clean_chunks_fallback","deterministic_flattening_repair"):
        if not isinstance(merged[key], bool): raise TypeError(f"{key} debe ser bool.")
    if merged["max_attempts"] != 1: raise ValueError("La primera candidata 03B admite únicamente max_attempts=1.")
    thresholds=merged.get("diagnostic_thresholds",{})
    if not isinstance(thresholds, Mapping): raise TypeError("diagnostic_thresholds debe ser mapping.")
    merged["diagnostic_thresholds"]=deepcopy(dict(thresholds))
    return merged
