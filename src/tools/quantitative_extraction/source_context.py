from __future__ import annotations
from typing import Any
import pandas as pd
from .input_validation import parse_delimited_values, safe_str

def get_source_chunk_context(row: Any, chunks: pd.DataFrame|None, *, allow_all_clean_chunks_fallback: bool):
    if chunks is None: return {"text":"","chunk_ids":[],"scope":"disabled"}
    source=safe_str(row.get("source_filename", ""))
    source_rows=chunks[chunks["source_filename"].astype(str)==source].copy()
    requested=parse_delimited_values(row.get("retrieved_chunk_ids", ""))
    selected=source_rows[source_rows["chunk_id"].astype(str).isin(requested)] if requested else source_rows.iloc[0:0]
    scope="retrieved_by_03"
    if selected.empty and allow_all_clean_chunks_fallback:
        selected=source_rows; scope="all_clean_chunks_fallback"
    selected=selected.sort_values("chunk_id")
    return {"text":"\n\n".join(selected["text"].fillna("").astype(str).tolist()),"chunk_ids":selected["chunk_id"].astype(str).tolist(),"scope":scope if not selected.empty else "no_chunks"}
