"""Retrieval helpers extracted from notebook 03 without behavioral changes.

The module receives the validated chunks DataFrame and a collection-like
object by parameter. It does not connect to Chroma, construct embeddings,
read files, write trace files, or load global configuration.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd


def retrieve_chunks_for_paper(
    source_filename: str,
    max_chunks: int,
    *,
    df_chunks_clean: pd.DataFrame,
    collection: Any,
    retrieval_queries: Sequence[str],
    retrieval_profile: str,
    retrieval_profile_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = df_chunks_clean[
        df_chunks_clean["source_filename"]
        == source_filename
    ]

    source_chunk_count = len(source_rows)

    if source_chunk_count == 0:
        raise ValueError(
            f"No hay chunks para {source_filename}"
        )

    fetch_k = min(
        int(
            retrieval_profile_config[
                "fetch_k"
            ]
        ),
        source_chunk_count,
    )

    candidates = {}
    trace_rows = []

    for retrieval_query in retrieval_queries:
        result = collection.query(
            query_texts=[retrieval_query],
            n_results=fetch_k,
            where={
                "source_filename": source_filename
            },
        )

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            chunk_id = str(
                metadata.get("chunk_id", "")
            )

            score = 1 - float(distance)

            trace_rows.append({
                "source_filename": source_filename,
                "retrieval_query": retrieval_query,
                "chunk_id": chunk_id,
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
                "score": score,
                "retrieval_profile": (
                    retrieval_profile
                ),
                "retrieval_mode": "chroma",
            })

            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "source_filename": source_filename,
                    "source_pdf_path": metadata.get(
                        "source_pdf_path"
                    ),
                    "chunk_index": int(
                        metadata.get("chunk_index", 0)
                    ),
                    "text": document,
                    "score": score,
                    "matched_queries": {
                        retrieval_query
                    },
                    "retrieval_mode": "chroma",
                }
            else:
                candidate = candidates[chunk_id]
                candidate["score"] = max(
                    candidate["score"],
                    score,
                )
                candidate["matched_queries"].add(
                    retrieval_query
                )

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda item: (
            -len(item["matched_queries"]),
            -item["score"],
            item["chunk_index"],
        ),
    )

    selected = ranked_candidates[:max_chunks]
    selected_ids = {
        item["chunk_id"]
        for item in selected
    }

    if len(selected) < min(
        max_chunks,
        source_chunk_count,
    ):
        for _, row in source_rows.iterrows():
            chunk_id = str(row["chunk_id"])

            if chunk_id in selected_ids:
                continue

            selected.append({
                "chunk_id": chunk_id,
                "source_filename": (
                    source_filename
                ),
                "source_pdf_path": (
                    row["source_pdf_path"]
                ),
                "chunk_index": int(
                    row["chunk_index"]
                ),
                "text": str(row["text"]),
                "score": None,
                "matched_queries": set(),
                "retrieval_mode": (
                    "ordered_clean_fallback"
                ),
            })

            selected_ids.add(chunk_id)

            trace_rows.append({
                "source_filename": (
                    source_filename
                ),
                "retrieval_query": "",
                "chunk_id": chunk_id,
                "chunk_index": int(
                    row["chunk_index"]
                ),
                "score": None,
                "retrieval_profile": (
                    retrieval_profile
                ),
                "retrieval_mode": (
                    "ordered_clean_fallback"
                ),
            })

            if len(selected) >= max_chunks:
                break

    selected = sorted(
        selected,
        key=lambda item: item["chunk_index"],
    )

    return selected, trace_rows


def build_context_from_chunks(
    selected_chunks: Sequence[Mapping[str, Any]],
    max_context_chars: int,
) -> str:
    context_parts = [
        f"[chunk_id: {chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in selected_chunks
    ]

    context = "\n\n---\n\n".join(
        context_parts
    )

    return context[:max_context_chars]


def get_first_chunks_context(
    source_filename: str,
    n_chunks: int,
    *,
    df_chunks_clean: pd.DataFrame,
) -> str:
    group = (
        df_chunks_clean[
            df_chunks_clean["source_filename"]
            == source_filename
        ]
        .sort_values("chunk_index")
        .head(n_chunks)
    )

    context_parts = []

    for _, row in group.iterrows():
        context_parts.append(
            f"[CHUNK_ID: {row['chunk_id']}]\n"
            f"{row['text']}"
        )

    return "\n\n---\n\n".join(
        context_parts
    )
