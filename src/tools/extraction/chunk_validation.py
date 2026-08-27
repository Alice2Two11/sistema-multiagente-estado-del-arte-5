"""Chunk validation extracted from notebook 03 without behavioral changes.

This module contains only the deterministic validation block originally used
for ``chunks_clean_for_rag.csv``. It does not read or write files and does not
perform retrieval.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


RAG_CLEAN_VALIDATION_VERSION = "v3_no_gt_no_review_no_bibliography_chroma"

REQUIRED_CHUNK_COLUMNS = {
    "chunk_id",
    "source_filename",
    "source_pdf_path",
    "chunk_index",
    "text",
    "chars",
    "is_review_section_chunk",
    "is_bibliography_chunk",
    "excluded_from_rag",
}


def to_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "sí", "si"])
    )


def validate_chunks_dataframe(
    dataframe: pd.DataFrame,
    *,
    experiment_id: str,
    chunks_file: str,
    created_at: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if created_at is None:
        created_at = datetime.now().isoformat()

    missing_columns = sorted(REQUIRED_CHUNK_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "chunks_clean_for_rag.csv no tiene las columnas "
            f"esperadas. Faltan: {missing_columns}"
        )

    df = dataframe.copy()
    for column in [
        "is_review_section_chunk",
        "is_bibliography_chunk",
        "excluded_from_rag",
    ]:
        df[column] = to_bool_series(df[column])

    df["chunk_index"] = pd.to_numeric(df["chunk_index"], errors="coerce")
    if df["chunk_index"].isna().any():
        raise ValueError("Hay valores inválidos en chunk_index.")
    df["chunk_index"] = df["chunk_index"].astype(int)
    df = (
        df.sort_values(["source_filename", "chunk_index"], kind="stable")
        .reset_index(drop=True)
    )

    empty_text_mask = (
        df["text"].fillna("").astype(str).str.strip().eq("")
    )
    review_mask = df["is_review_section_chunk"]
    bibliography_mask = df["is_bibliography_chunk"]
    excluded_mask = df["excluded_from_rag"]
    ground_truth_mask = (
        df["source_filename"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(r"ground[_\s-]*truth|gt_", regex=True)
    )

    validation_errors = []
    if empty_text_mask.any():
        validation_errors.append("Hay chunks sin texto.")
    if review_mask.any():
        validation_errors.append(
            "Hay chunks de secciones de revisión dentro del archivo limpio."
        )
    if bibliography_mask.any():
        validation_errors.append(
            "Hay chunks de bibliografía dentro del archivo limpio."
        )
    if excluded_mask.any():
        validation_errors.append(
            "Hay chunks con excluded_from_rag=True dentro del archivo limpio."
        )
    if ground_truth_mask.any():
        validation_errors.append(
            "El archivo limpio parece contener Ground Truth."
        )

    report = {
        "stage": "03_agente_extraccion_kb",
        "experiment_id": experiment_id,
        "created_at": created_at,
        "chunks_file": str(chunks_file),
        "num_chunks": int(len(df)),
        "num_papers": int(df["source_filename"].nunique()),
        "empty_text_chunks": int(empty_text_mask.sum()),
        "review_section_chunks": int(review_mask.sum()),
        "bibliography_chunks": int(bibliography_mask.sum()),
        "excluded_from_rag_chunks": int(excluded_mask.sum()),
        "possible_ground_truth_chunks": int(ground_truth_mask.sum()),
        "validation_version": RAG_CLEAN_VALIDATION_VERSION,
        "valid_for_extraction": len(validation_errors) == 0,
        "errors": validation_errors,
    }
    return df, report
