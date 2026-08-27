from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Any
import math
import re
import unicodedata

from .retrieval import is_non_substantive_evidence, safe_str

Candidate = dict[str, Any]
ChunkKey = tuple[str, str]


def _candidate_key(row: Candidate) -> ChunkKey:
    return (safe_str(row.get("source_filename")), safe_str(row.get("chunk_id")))


def _normalize_tokens(text: Any) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKD", safe_str(text).lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return tuple(
        token
        for token in re.findall(r"[a-z0-9_]+", normalized)
        if len(token) > 2
    )


def _validate_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"HYBRID_RETRIEVAL_INVALID_TYPE:{name}:expected_integer")
    if value <= 0:
        raise ValueError(f"HYBRID_RETRIEVAL_INVALID:{name}:must_be_greater_than_0")
    return value


def _validate_non_negative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"HYBRID_RETRIEVAL_INVALID_TYPE:{name}:expected_integer")
    if value < 0:
        raise ValueError(f"HYBRID_RETRIEVAL_INVALID:{name}:must_be_greater_than_or_equal_to_0")
    return value


def _valid_pairs_from_chunks(chunks_df) -> set[ChunkKey]:
    if chunks_df is None or getattr(chunks_df, "empty", True):
        return set()
    return {
        (safe_str(row["source_filename"]), safe_str(row["chunk_id"]))
        for _, row in chunks_df.iterrows()
        if safe_str(row.get("source_filename")) and safe_str(row.get("chunk_id"))
    }


def _apply_per_source_limit(rows: Sequence[Candidate], limit: int) -> list[Candidate]:
    counts: dict[str, int] = {}
    output: list[Candidate] = []
    for original in rows:
        row = deepcopy(original)
        source = safe_str(row.get("source_filename"))
        if counts.get(source, 0) >= limit:
            continue
        counts[source] = counts.get(source, 0) + 1
        output.append(row)
    return output


def deduplicate_candidates(
    rows: Iterable[Candidate],
    *,
    allowed_papers: Iterable[str] | None = None,
    valid_source_chunk_pairs: Iterable[ChunkKey] | None = None,
) -> list[Candidate]:
    """Deduplicate by the contractual identity ``(source_filename, chunk_id)``.

    Incomplete identities, unauthorized papers, invalid pairs and non-substantive
    chunks are rejected. For duplicate rows, the row with the highest component
    score is retained. Retrieval provenance is merged deterministically.
    """

    allowed = {safe_str(item) for item in (allowed_papers or ()) if safe_str(item)}
    valid_pairs = set(valid_source_chunk_pairs or ())
    best: dict[ChunkKey, Candidate] = {}

    for original in rows:
        row = deepcopy(dict(original))
        source, chunk_id = _candidate_key(row)
        text = safe_str(row.get("text") or row.get("chunk_text"))
        if not source or not chunk_id or not text:
            continue
        if allowed and source not in allowed:
            continue
        if valid_pairs and (source, chunk_id) not in valid_pairs:
            continue
        if is_non_substantive_evidence(text):
            continue

        row["source_filename"] = source
        row["chunk_id"] = chunk_id
        row["text"] = text
        row["score"] = float(row.get("score", 0.0) or 0.0)

        sources = row.get("retrieval_sources") or [
            row.get("retrieval_source") or row.get("retrieval_method")
        ]
        row["retrieval_sources"] = sorted(
            {safe_str(item) for item in sources if safe_str(item)}
        )

        key = (source, chunk_id)
        current = best.get(key)
        if current is None:
            best[key] = row
            continue

        merged_sources = sorted(
            set(current.get("retrieval_sources") or ())
            | set(row.get("retrieval_sources") or ())
        )
        current_score = float(current.get("score", 0.0) or 0.0)
        new_score = float(row.get("score", 0.0) or 0.0)
        chosen = row if new_score > current_score else current
        chosen = deepcopy(chosen)
        chosen["retrieval_sources"] = merged_sources
        best[key] = chosen

    return sorted(
        best.values(),
        key=lambda row: (
            -float(row.get("score", 0.0) or 0.0),
            row["source_filename"],
            row["chunk_id"],
        ),
    )


def query_chroma_candidates(
    collection,
    chunks_df,
    query: str,
    allowed_papers: Sequence[str],
    *,
    candidate_multiplier: int,
    top_k_evidence_per_section: int,
    max_candidates_per_source: int,
    valid_source_chunk_pairs: Iterable[ChunkKey] | None = None,
) -> list[Candidate]:
    candidate_multiplier = _validate_positive_integer("candidate_multiplier", candidate_multiplier)
    top_k = _validate_positive_integer("top_k_evidence_per_section", top_k_evidence_per_section)
    per_source_limit = _validate_positive_integer("max_candidates_per_source", max_candidates_per_source)

    allowed = sorted({safe_str(item) for item in allowed_papers if safe_str(item)})
    if not allowed or collection is None:
        return []

    candidate_k = top_k * candidate_multiplier
    rows: list[Candidate] = []
    for source in allowed:
        source_count = int((chunks_df["source_filename"].astype(str) == source).sum())
        if source_count <= 0:
            continue
        n_results = min(candidate_k, per_source_limit, source_count)
        result = collection.query(
            query_texts=[safe_str(query)],
            n_results=n_results,
            where={"source_filename": source},
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            returned_source = safe_str(metadata.get("source_filename"))
            chunk_id = safe_str(metadata.get("chunk_id"))
            if returned_source != source:
                continue
            rows.append(
                {
                    "source_filename": returned_source,
                    "chunk_id": chunk_id,
                    "text": safe_str(document),
                    "score": float(1.0 - float(distance)),
                    "retrieval_source": "chroma",
                    "retrieval_sources": ["chroma"],
                }
            )

    deduped = deduplicate_candidates(
        rows,
        allowed_papers=allowed,
        valid_source_chunk_pairs=valid_source_chunk_pairs,
    )
    deduped = _apply_per_source_limit(deduped, per_source_limit)
    deduped = deduped[:candidate_k]
    for rank, row in enumerate(deduped, start=1):
        row["chroma_rank"] = rank
        row["csv_rank"] = None
    return deduped


def query_csv_ranked_candidates(
    chunks_df,
    query: str,
    allowed_papers: Sequence[str],
    *,
    candidate_multiplier: int,
    top_k_evidence_per_section: int,
    max_candidates_per_source: int,
    valid_source_chunk_pairs: Iterable[ChunkKey] | None = None,
) -> list[Candidate]:
    candidate_multiplier = _validate_positive_integer("candidate_multiplier", candidate_multiplier)
    top_k = _validate_positive_integer("top_k_evidence_per_section", top_k_evidence_per_section)
    per_source_limit = _validate_positive_integer("max_candidates_per_source", max_candidates_per_source)

    allowed = sorted({safe_str(item) for item in allowed_papers if safe_str(item)})
    query_tokens = set(_normalize_tokens(query))
    if not allowed or not query_tokens or chunks_df is None or chunks_df.empty:
        return []

    candidate_k = top_k * candidate_multiplier
    rows: list[Candidate] = []
    subset = chunks_df[chunks_df["source_filename"].astype(str).isin(allowed)]
    for _, source_row in subset.iterrows():
        source = safe_str(source_row.get("source_filename"))
        chunk_id = safe_str(source_row.get("chunk_id"))
        text = safe_str(source_row.get("text"))
        if not source or not chunk_id or not text:
            continue
        text_tokens = set(_normalize_tokens(text))
        overlap = len(query_tokens & text_tokens)
        if overlap <= 0:
            continue
        score = overlap / len(query_tokens)
        rows.append(
            {
                "source_filename": source,
                "chunk_id": chunk_id,
                "text": text,
                "score": float(score),
                "lexical_overlap": overlap,
                "retrieval_source": "csv",
                "retrieval_sources": ["csv"],
            }
        )

    deduped = deduplicate_candidates(
        rows,
        allowed_papers=allowed,
        valid_source_chunk_pairs=valid_source_chunk_pairs,
    )
    deduped = _apply_per_source_limit(deduped, per_source_limit)
    deduped = deduped[:candidate_k]
    for rank, row in enumerate(deduped, start=1):
        row["csv_rank"] = rank
        row["chroma_rank"] = None
    return deduped


def reciprocal_rank_fusion(
    chroma_rows: Sequence[Candidate],
    csv_rows: Sequence[Candidate],
    *,
    rrf_k: int,
) -> list[Candidate]:
    rrf_k = _validate_positive_integer("rrf_k", rrf_k)
    fused: dict[ChunkKey, Candidate] = {}

    for source_name, rank_field, rows in (
        ("chroma", "chroma_rank", chroma_rows),
        ("csv", "csv_rank", csv_rows),
    ):
        for fallback_rank, original in enumerate(rows, start=1):
            row = deepcopy(dict(original))
            key = _candidate_key(row)
            if not key[0] or not key[1]:
                continue
            rank = row.get(rank_field)
            if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                rank = fallback_rank
            if key not in fused:
                fused[key] = {
                    "source_filename": key[0],
                    "chunk_id": key[1],
                    "text": safe_str(row.get("text")),
                    "score": 0.0,
                    "rrf_score": 0.0,
                    "chroma_rank": None,
                    "csv_rank": None,
                    "retrieval_source": "rrf",
                    "retrieval_sources": [],
                }
            item = fused[key]
            item[rank_field] = rank
            item["rrf_score"] += 1.0 / (rrf_k + rank)
            item["score"] = item["rrf_score"]
            item["retrieval_sources"] = sorted(
                set(item["retrieval_sources"]) | {source_name}
            )
            if not item["text"]:
                item["text"] = safe_str(row.get("text"))

    def best_rank(row: Candidate) -> float:
        ranks = [row.get("chroma_rank"), row.get("csv_rank")]
        valid = [rank for rank in ranks if isinstance(rank, int) and rank > 0]
        return min(valid) if valid else math.inf

    return sorted(
        fused.values(),
        key=lambda row: (
            -float(row.get("rrf_score", 0.0) or 0.0),
            best_rank(row),
            row["source_filename"],
            row["chunk_id"],
        ),
    )


def balanced_hybrid_selection(
    chroma_rows: Sequence[Candidate],
    csv_rows: Sequence[Candidate],
    fused_rows: Sequence[Candidate],
    *,
    chroma_quota: int,
    csv_quota: int,
    rrf_quota: int,
    top_k_evidence_per_section: int,
    max_candidates_per_source: int,
    max_evidence_chars: int,
    allowed_papers: Sequence[str],
    valid_source_chunk_pairs: Iterable[ChunkKey] | None = None,
) -> list[Candidate]:
    chroma_quota = _validate_non_negative_integer("chroma_quota", chroma_quota)
    csv_quota = _validate_non_negative_integer("csv_quota", csv_quota)
    rrf_quota = _validate_non_negative_integer("rrf_quota", rrf_quota)
    top_k = _validate_positive_integer("top_k_evidence_per_section", top_k_evidence_per_section)
    per_source_limit = _validate_positive_integer("max_candidates_per_source", max_candidates_per_source)
    char_limit = _validate_positive_integer("max_evidence_chars", max_evidence_chars)
    if chroma_quota + csv_quota + rrf_quota != top_k:
        raise ValueError(
            "HYBRID_RETRIEVAL_INVALID:retrieval_quotas:"
            "chroma_quota_plus_csv_quota_plus_rrf_quota_must_equal_top_k_evidence_per_section"
        )

    allowed = {safe_str(item) for item in allowed_papers if safe_str(item)}
    valid_pairs = set(valid_source_chunk_pairs or ())
    selected: list[Candidate] = []
    selected_keys: set[ChunkKey] = set()
    source_counts: dict[str, int] = {}
    used_chars = 0
    fused_by_key = {_candidate_key(row): row for row in fused_rows}

    def try_add(original: Candidate, bucket: str) -> bool:
        nonlocal used_chars
        row = deepcopy(dict(original))
        fused_metadata = fused_by_key.get(_candidate_key(row))
        if fused_metadata is not None:
            row["chroma_rank"] = fused_metadata.get("chroma_rank")
            row["csv_rank"] = fused_metadata.get("csv_rank")
            row["rrf_score"] = fused_metadata.get("rrf_score", 0.0)
            row["retrieval_sources"] = list(
                fused_metadata.get("retrieval_sources") or row.get("retrieval_sources") or ()
            )
        source, chunk_id = _candidate_key(row)
        text = safe_str(row.get("text"))
        key = (source, chunk_id)
        if not source or not chunk_id or not text or key in selected_keys:
            return False
        if source not in allowed:
            return False
        if valid_pairs and key not in valid_pairs:
            return False
        if source_counts.get(source, 0) >= per_source_limit:
            return False
        if used_chars + len(text) > char_limit:
            return False
        row["source_filename"] = source
        row["chunk_id"] = chunk_id
        row["text"] = text
        row["selection_bucket"] = bucket
        row["selection_order"] = len(selected) + 1
        row["retrieval_source"] = safe_str(row.get("retrieval_source")) or bucket
        row["retrieval_sources"] = sorted(
            {safe_str(item) for item in (row.get("retrieval_sources") or [row["retrieval_source"]]) if safe_str(item)}
        )
        selected.append(row)
        selected_keys.add(key)
        source_counts[source] = source_counts.get(source, 0) + 1
        used_chars += len(text)
        return True

    def consume(rows: Sequence[Candidate], desired: int, bucket: str) -> int:
        added = 0
        for row in rows:
            if len(selected) >= top_k or added >= desired:
                break
            if try_add(row, bucket):
                added += 1
        return added

    chroma_added = consume(chroma_rows, chroma_quota, "chroma_quota")
    csv_added = consume(csv_rows, csv_quota, "csv_quota")
    rrf_target = rrf_quota + (chroma_quota - chroma_added) + (csv_quota - csv_added)
    consume(fused_rows, rrf_target, "rrf_quota_or_redistribution")

    if len(selected) < top_k:
        remaining_pool = list(fused_rows) + list(chroma_rows) + list(csv_rows)
        consume(remaining_pool, top_k - len(selected), "valid_remaining_completion")

    return selected[:top_k]


def retrieve_section_evidence_hybrid(
    section: dict[str, Any],
    collection,
    chunks_df,
    *,
    candidate_multiplier: int,
    chroma_quota: int,
    csv_quota: int,
    rrf_quota: int,
    rrf_k: int,
    top_k_evidence_per_section: int,
    max_evidence_chars: int,
    max_candidates_per_source: int,
) -> list[Candidate]:
    papers = section.get("papers_to_use") or []
    allowed_papers = [
        safe_str(paper.get("source_filename") if isinstance(paper, dict) else paper)
        for paper in papers
    ]
    allowed_papers = [item for item in allowed_papers if item]
    if not allowed_papers:
        return []

    query_parts = [section.get("section_title"), section.get("purpose")]
    query_parts.extend(section.get("key_arguments") or [])
    query_parts.extend(section.get("evidence_needs") or [])
    query = " ".join(safe_str(item) for item in query_parts if safe_str(item))

    valid_pairs = _valid_pairs_from_chunks(chunks_df)
    chroma_rows = query_chroma_candidates(
        collection,
        chunks_df,
        query,
        allowed_papers,
        candidate_multiplier=candidate_multiplier,
        top_k_evidence_per_section=top_k_evidence_per_section,
        max_candidates_per_source=max_candidates_per_source,
        valid_source_chunk_pairs=valid_pairs,
    )
    csv_rows = query_csv_ranked_candidates(
        chunks_df,
        query,
        allowed_papers,
        candidate_multiplier=candidate_multiplier,
        top_k_evidence_per_section=top_k_evidence_per_section,
        max_candidates_per_source=max_candidates_per_source,
        valid_source_chunk_pairs=valid_pairs,
    )
    fused_rows = reciprocal_rank_fusion(chroma_rows, csv_rows, rrf_k=rrf_k)
    return balanced_hybrid_selection(
        chroma_rows,
        csv_rows,
        fused_rows,
        chroma_quota=chroma_quota,
        csv_quota=csv_quota,
        rrf_quota=rrf_quota,
        top_k_evidence_per_section=top_k_evidence_per_section,
        max_candidates_per_source=max_candidates_per_source,
        max_evidence_chars=max_evidence_chars,
        allowed_papers=allowed_papers,
        valid_source_chunk_pairs=valid_pairs,
    )
