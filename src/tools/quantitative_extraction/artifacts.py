from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.io.atomic_write import atomic_write_csv, atomic_write_json, atomic_write_jsonl, atomic_write_text
from src.config.quantitative_extraction_policy_config import ARTIFACT_FILENAMES
from src.state.fingerprints import sha256_file


def artifact_paths(output_dir):
    root = Path(output_dir)
    return {name: root / name for name in ARTIFACT_FILENAMES}


def _existing_result(path: Path):
    from src.io.atomic_write import AtomicWriteResult
    return AtomicWriteResult(path=str(path), hash=sha256_file(path), size_bytes=path.stat().st_size)


def write_quantitative_artifacts(*, output_dir, results, raw_records, errors, quantitative_rows, dataset_rows, technique_rows, metrics, manifest_base, preserve_structured_sources=False):
    paths = artifact_paths(output_dir)
    written = {}
    if preserve_structured_sources:
        for name in ARTIFACT_FILENAMES[:2]:
            if not paths[name].is_file():
                raise FileNotFoundError(f"DEPENDENCY_NOT_FOUND: falta artefacto fuente preservado {paths[name]}")
            written[name] = _existing_result(paths[name])
    else:
        written[ARTIFACT_FILENAMES[0]] = atomic_write_json(paths[ARTIFACT_FILENAMES[0]], results)
        written[ARTIFACT_FILENAMES[1]] = atomic_write_jsonl(paths[ARTIFACT_FILENAMES[1]], raw_records)

    error_fields = (
        "source_filename", "error_type", "error_code", "error_message",
        "raw_path", "raw_value", "discarded", "created_at",
    )
    normalized_errors = []
    for row in errors:
        item = {key: row.get(key, "") for key in error_fields}
        if not item["created_at"]:
            item["created_at"] = datetime.now(timezone.utc).isoformat()
        normalized_errors.append(item)
    written[ARTIFACT_FILENAMES[2]] = atomic_write_csv(paths[ARTIFACT_FILENAMES[2]], normalized_errors, fieldnames=error_fields)

    quant_fields = (
        "source_filename", "paper_title", "model_or_method", "metric", "value",
        "numeric_value", "unit", "dataset_or_case", "evaluation_scope",
        "data_resolution", "condition", "source_text_evidence",
        "value_found_in_kb_text", "value_found_in_source_chunk",
        "source_chunk_scope", "source_chunk_ids_checked", "verification_status",
        "raw_path", "raw_value",
    )
    written[ARTIFACT_FILENAMES[3]] = atomic_write_csv(paths[ARTIFACT_FILENAMES[3]], quantitative_rows, fieldnames=quant_fields)
    dataset_fields = (
        "source_filename", "paper_title", "dataset_name", "description",
        "case_study", "data_type", "temporal_resolution", "spatial_resolution",
        "analysis_scope", "coordinates", "altitude_masl", "data_split",
        "source_text_evidence", "raw_path", "raw_value",
    )
    written[ARTIFACT_FILENAMES[4]] = atomic_write_csv(paths[ARTIFACT_FILENAMES[4]], dataset_rows, fieldnames=dataset_fields)
    technique_fields = (
        "source_filename", "paper_title", "technique_name", "technique_family",
        "role", "source_text_evidence", "raw_path", "raw_value",
    )
    written[ARTIFACT_FILENAMES[5]] = atomic_write_csv(paths[ARTIFACT_FILENAMES[5]], technique_rows, fieldnames=technique_fields)

    summaries = []
    sources = sorted({r["source_filename"] for r in dataset_rows + technique_rows})
    for source in sources:
        titles = [r["paper_title"] for r in dataset_rows + technique_rows if r["source_filename"] == source]
        summaries.append({
            "source_filename": source,
            "paper_title": titles[0] if titles else "",
            "techniques": "; ".join(sorted({r["technique_name"] for r in technique_rows if r["source_filename"] == source and r["technique_name"]})),
            "datasets": "; ".join(sorted({r["dataset_name"] for r in dataset_rows if r["source_filename"] == source and r["dataset_name"]})),
        })
    written[ARTIFACT_FILENAMES[6]] = atomic_write_csv(paths[ARTIFACT_FILENAMES[6]], summaries, fieldnames=("source_filename", "paper_title", "techniques", "datasets"))

    lines = [
        "# Reporte de extracción cuantitativa estructurada", "",
        f"Fecha: {datetime.now(timezone.utc).isoformat()}", "",
        f"- Papers procesados: {metrics['counts']['papers_processed']}",
        f"- Candidatos cuantitativos crudos: {metrics.get('raw_quantitative_candidate_count', 0)}",
        f"- Filas cuantitativas: {metrics.get('flattened_quantitative_rows', metrics['counts']['quantitative_rows'])}",
        f"- Candidatos de técnicas crudos: {metrics.get('raw_technique_candidate_count', 0)}",
        f"- Filas de técnicas: {metrics.get('flattened_technique_rows', metrics['counts']['technique_rows'])}",
        f"- Filas de datasets: {metrics.get('flattened_dataset_rows', metrics['counts']['dataset_rows'])}",
        f"- Registros descartados: {metrics.get('discarded_record_count', 0)}",
        f"- Advertencias de normalización: {metrics.get('normalization_warning_count', 0)}",
        f"- Confirmados en chunks: {metrics['counts']['confirmed_in_source_chunks']}",
        f"- Solo en KB: {metrics['counts']['found_only_in_kb_text']}",
        f"- No confirmados: {metrics['counts']['not_confirmed']}", "",
        "Los valores encontrados solo en la KB no se presentan como confirmados en el paper.",
    ]
    written[ARTIFACT_FILENAMES[7]] = atomic_write_text(paths[ARTIFACT_FILENAMES[7]], "\n".join(lines) + "\n")

    manifest = dict(manifest_base)
    manifest.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "repair": {
            "deterministic_flattening_repair": bool(preserve_structured_sources),
            "openai_called": False if preserve_structured_sources else None,
            "structured_json_preserved": bool(preserve_structured_sources),
            "raw_jsonl_preserved": bool(preserve_structured_sources),
        },
        "safety_policy": {
            "uses_scientific_knowledge_base": True,
            "uses_ground_truth": False,
            "uses_external_knowledge": False,
            "uses_review_sections": False,
            "uses_bibliography": False,
        },
        "outputs": {k: {"path": v.path, "sha256": v.hash} for k, v in written.items()},
    })
    written[ARTIFACT_FILENAMES[8]] = atomic_write_json(paths[ARTIFACT_FILENAMES[8]], manifest)
    return written, manifest
