from __future__ import annotations
import re


def safe_str(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def tokenize_for_overlap(text):
    return {
        token
        for token in re.findall(r"[\wáéíóúüñ]+", safe_str(text).lower())
        if len(token) > 2
    }


def is_non_substantive_evidence(text):
    low = safe_str(text).lower()
    blocked = (
        "author contributions",
        "funding",
        "acknowledgments",
        "acknowledgements",
        "conflicts of interest",
        "publisher's note",
        "publisher note",
    )
    return not low or any(item in low for item in blocked)


def _valid_pairs_from_chunks(chunks_df):
    if chunks_df is None or chunks_df.empty:
        return set()
    return {
        (safe_str(row["source_filename"]), safe_str(row["chunk_id"]))
        for _, row in chunks_df.iterrows()
    }


def dedupe_evidence(rows, valid_source_chunk_pairs=None):
    """Preserva literalmente la selección original por par fuente–chunk."""
    valid_pairs = set(valid_source_chunk_pairs or ())
    best_by_pair = {}

    for raw_row in rows:
        row = dict(raw_row)
        row["source_filename"] = safe_str(row.get("source_filename"))
        row["chunk_id"] = safe_str(row.get("chunk_id"))
        row["text"] = safe_str(row.get("text") or row.get("chunk_text"))
        row["score"] = float(row.get("score", 0.0) or 0.0)
        pair = (row["source_filename"], row["chunk_id"])

        if valid_pairs and pair not in valid_pairs:
            continue
        if is_non_substantive_evidence(row["text"]):
            continue
        if pair not in best_by_pair or row["score"] > best_by_pair[pair]["score"]:
            best_by_pair[pair] = row

    return sorted(
        best_by_pair.values(),
        key=lambda row: (
            -row["score"],
            row["source_filename"],
            row["chunk_id"],
        ),
    )


def build_section_query(section):
    parts = [section.get("section_title"), section.get("purpose")]
    parts += list(section.get("key_arguments") or [])
    parts += list(section.get("evidence_needs") or [])
    return " ".join(safe_str(item) for item in parts if safe_str(item))


def query_chroma_restricted(
    collection,
    chunks_df,
    query,
    source_filenames,
    top_k,
    max_evidence_chars=18000,
    valid_source_chunk_pairs=None,
):
    if not source_filenames:
        return []

    per_source_k = max(1, top_k // len(source_filenames) + 1)
    rows = []

    for source in source_filenames:
        source_chunk_count = int(
            (chunks_df["source_filename"].astype(str) == source).sum()
        )
        result = collection.query(
            query_texts=[query],
            n_results=min(per_source_k, max(1, source_chunk_count)),
            where={"source_filename": source},
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            chunk_id = safe_str(metadata.get("chunk_id"))
            returned_source = safe_str(metadata.get("source_filename"))
            if returned_source != source:
                continue

            rows.append({
                "source_filename": returned_source,
                "chunk_id": chunk_id,
                "text": safe_str(document)[:max_evidence_chars],
                "score": float(1.0 - float(distance)),
                "retrieval_method": "chroma_restricted",
            })

    return dedupe_evidence(rows, valid_source_chunk_pairs)[:top_k]


def query_csv_restricted(
    chunks_df,
    query,
    source_filenames,
    top_k,
    max_evidence_chars=18000,
    valid_source_chunk_pairs=None,
):
    if not source_filenames:
        return []

    query_tokens = tokenize_for_overlap(query)
    rows = []
    subset = chunks_df[
        chunks_df["source_filename"].astype(str).isin(source_filenames)
    ]

    for _, row in subset.iterrows():
        text = safe_str(row["text"])
        text_tokens = tokenize_for_overlap(text)
        overlap = len(query_tokens & text_tokens)
        score = overlap / max(len(query_tokens), 1)
        rows.append({
            "source_filename": safe_str(row["source_filename"]),
            "chunk_id": safe_str(row["chunk_id"]),
            "text": text[:max_evidence_chars],
            "score": float(score),
            "retrieval_method": "csv_lexical_restricted",
        })

    return dedupe_evidence(rows, valid_source_chunk_pairs)[:top_k]


def retrieve_section_evidence(
    section,
    collection,
    chunks_df,
    top_k,
    max_evidence_chars=18000,
):
    source_filenames = [
        safe_str(paper.get("source_filename") if isinstance(paper, dict) else paper)
        for paper in (section.get("papers_to_use") or [])
    ]
    source_filenames = [source for source in source_filenames if source]
    if not source_filenames:
        return []

    valid_source_chunk_pairs = _valid_pairs_from_chunks(chunks_df)
    query = build_section_query(section)
    rows = query_chroma_restricted(
        collection,
        chunks_df,
        query,
        source_filenames,
        top_k,
        max_evidence_chars=max_evidence_chars,
        valid_source_chunk_pairs=valid_source_chunk_pairs,
    )

    if len(rows) < top_k:
        rows.extend(
            query_csv_restricted(
                chunks_df,
                query,
                source_filenames,
                top_k,
                max_evidence_chars=max_evidence_chars,
                valid_source_chunk_pairs=valid_source_chunk_pairs,
            )
        )

    return dedupe_evidence(rows, valid_source_chunk_pairs)[:top_k]
