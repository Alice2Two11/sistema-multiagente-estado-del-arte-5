"""Stage-03 artifact, manifest, and final-audit helpers.

This module extracts only the local behavior characterized from notebook 03,
cells 4, 5, 13, 14, and 15. It does not coordinate the complete extraction
agent, modify PipelineState, or implement PREPARE/COMMIT/RESUME.

Global rebuild orchestration remains outside this module. Paths, timestamps,
data, and filesystem-facing dependencies are received as parameters.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from src.io.atomic_write import atomic_write_text
from src.state.fingerprints import sha256_file, sha256_text


TRACKED_STAGE_OUTPUT_KEYS = [
    "CARDS_JSONL_PATH",
    "CARDS_SUMMARY_CSV_PATH",
    "CARDS_ERRORS_CSV_PATH",
    "CARDS_QUALITY_CSV_PATH",
    "RETRIEVAL_TRACE_CSV_PATH",
    "KB_CSV_PATH",
    "KB_JSONL_PATH",
    "EXTRACTION_MANIFEST_PATH",
]

REQUIRED_STAGE_OUTPUT_KEYS = [
    "CARDS_JSONL_PATH",
    "CARDS_SUMMARY_CSV_PATH",
    "CARDS_ERRORS_CSV_PATH",
    "CARDS_QUALITY_CSV_PATH",
    "RETRIEVAL_TRACE_CSV_PATH",
    "KB_CSV_PATH",
    "KB_JSONL_PATH",
]

FINAL_OUTPUT_KEYS = [
    "CARDS_JSONL_PATH",
    "CARDS_SUMMARY_CSV_PATH",
    "CARDS_ERRORS_CSV_PATH",
    "CARDS_QUALITY_CSV_PATH",
    "RETRIEVAL_TRACE_CSV_PATH",
    "KB_CSV_PATH",
    "KB_JSONL_PATH",
    "EXTRACTION_MANIFEST_PATH",
    "CHUNKS_VALIDATION_REPORT_PATH",
]

CURRENT_EXTRACTION_SIGNATURE_KEYS = [
    "stage",
    "experiment_id",
    "experiment_dir",
    "chunks_clean_path",
    "chunks_clean_sha256",
    "chroma_manifest_sha256",
    "chroma_collection_name",
    "openai_model",
    "embedding_model",
    "experiment_profile",
    "topic_profile",
    "generation_profile",
    "rag_policy",
    "extraction_policy",
    "card_required_fields",
    "card_list_fields",
    "classification_fields",
    "extraction_prompt_version",
    "relevance_prompt_version",
    "kb_schema_version",
    "rag_clean_validation_version",
]

EXTRACTION_MANIFEST_TOP_LEVEL_KEYS = [
    "stage",
    "experiment_id",
    "created_or_checked_at",
    "fingerprint",
    "signature",
    "automatic_decision",
    "retrieval",
    "safety_policy",
    "outputs",
    "counts",
]

EXTRACTION_MANIFEST_AUTOMATIC_DECISION_KEYS = [
    "status",
    "auto_rebuild_enabled",
    "force_rebuild",
    "rebuild_executed",
    "reclassified_relevance",
    "kb_recreated",
    "backup_dir_created",
]

EXTRACTION_MANIFEST_RETRIEVAL_KEYS = [
    "chroma_used",
    "collection_name",
    "embedding_model",
    "retrieval_profile",
    "retrieval_profile_config",
    "num_retrieval_queries",
    "num_trace_rows",
    "trace_file",
]

EXTRACTION_MANIFEST_SAFETY_KEYS = [
    "uses_chunks_clean_for_rag",
    "ground_truth_used",
    "review_sections_used",
    "bibliography_used",
    "rag_clean_validated",
]

EXTRACTION_MANIFEST_OUTPUT_KEYS = [
    "scientific_cards_jsonl",
    "scientific_cards_summary_csv",
    "scientific_cards_errors_csv",
    "scientific_cards_quality_csv",
    "extraction_retrieval_trace_csv",
    "scientific_knowledge_base_csv",
    "scientific_knowledge_base_jsonl",
    "manifest",
    "chunks_validation_report",
]

EXTRACTION_MANIFEST_COUNT_KEYS = [
    "num_cards",
    "num_kb_rows",
    "num_included_in_state_of_art",
    "num_chunks_clean",
    "num_source_papers",
    "num_extraction_errors",
]


def load_json_file(path: str | Path) -> Any:
    """Load JSON or return None when the path does not exist."""

    source = Path(path)

    if not source.exists():
        return None

    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_file(
    data: Any,
    path: str | Path,
) -> Any:
    """Atomically write JSON using the notebook's serialization settings."""

    serialized = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    return atomic_write_text(
        path,
        serialized,
        encoding="utf-8",
    )


def _sha256_file_or_none(
    path: str | Path,
) -> str | None:
    source = Path(path)

    if not source.exists():
        return None

    return sha256_file(source)


def stable_hash_dict(data: Any) -> str:
    """Preserve the notebook JSON representation while reusing SHA helpers."""

    serialized = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    return sha256_text(serialized)


def any_stage_outputs_exist(
    tracked_outputs: Sequence[str | Path],
) -> bool:
    return any(
        Path(path).exists()
        for path in tracked_outputs
    )


def required_stage_outputs_exist(
    required_stage_outputs: Sequence[
        str | Path
    ],
) -> bool:
    return all(
        Path(path).exists()
        for path in required_stage_outputs
    )


def backup_stage_outputs(
    *,
    outputs_dir: str | Path,
    dir_extraction: str | Path,
    dir_kb: str | Path,
    experiment_id: str,
    reason: str = "auto",
    now_callable: Callable[
        [], datetime
    ] = datetime.now,
) -> Path:
    timestamp = now_callable().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_dir = (
        Path(outputs_dir)
        / (
            "03_extraccion_kb_BACKUP_"
            f"{timestamp}_{reason}"
        )
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = []

    if Path(dir_extraction).exists():
        destination = (
            backup_dir
            / Path(dir_extraction).name
        )

        shutil.copytree(
            dir_extraction,
            destination,
            dirs_exist_ok=True,
        )

        copied.append(
            str(destination)
        )

    if Path(dir_kb).exists():
        destination = (
            backup_dir
            / Path(dir_kb).name
        )

        shutil.copytree(
            dir_kb,
            destination,
            dirs_exist_ok=True,
        )

        copied.append(
            str(destination)
        )

    metadata = {
        "created_at": (
            now_callable().isoformat()
        ),
        "reason": reason,
        "copied": copied,
        "experiment_id": experiment_id,
    }

    save_json_file(
        metadata,
        backup_dir
        / "backup_metadata.json",
    )

    return backup_dir


def reset_stage_outputs(
    dir_extraction: str | Path,
    dir_kb: str | Path,
) -> None:
    for target in [
        Path(dir_extraction),
        Path(dir_kb),
    ]:
        if target.exists():
            shutil.rmtree(target)

        target.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_dataframe_even_if_empty(
    rows: Any,
    path: str | Path,
    columns: Sequence[str],
) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )

    atomic_write_text(
        path,
        dataframe.to_csv(
            index=False
        ),
        encoding="utf-8",
    )

    return dataframe


def build_current_extraction_signature(
    *,
    experiment_id: str,
    experiment_dir: str | Path,
    chunks_clean_path: str | Path,
    chroma_manifest_path: str | Path,
    chroma_collection_name: str,
    openai_model: str,
    embedding_model_name: str,
    experiment_profile: Any,
    topic_profile: Any,
    generation_profile: Any,
    rag_policy: Any,
    extraction_policy: Any,
    card_required_fields: Any,
    card_list_fields: Any,
    classification_fields: Any,
    extraction_prompt_version: str,
    relevance_prompt_version: str,
    kb_schema_version: str,
    rag_clean_validation_version: str,
) -> dict[str, Any]:
    return {
        "stage": (
            "03_agente_extraccion_kb"
        ),
        "experiment_id": experiment_id,
        "experiment_dir": str(
            experiment_dir
        ),
        "chunks_clean_path": str(
            chunks_clean_path
        ),
        "chunks_clean_sha256": (
            _sha256_file_or_none(
                chunks_clean_path
            )
        ),
        "chroma_manifest_sha256": (
            _sha256_file_or_none(
                chroma_manifest_path
            )
        ),
        "chroma_collection_name": (
            chroma_collection_name
        ),
        "openai_model": openai_model,
        "embedding_model": (
            embedding_model_name
        ),
        "experiment_profile": (
            experiment_profile
        ),
        "topic_profile": topic_profile,
        "generation_profile": (
            generation_profile
        ),
        "rag_policy": rag_policy,
        "extraction_policy": (
            extraction_policy
        ),
        "card_required_fields": (
            card_required_fields
        ),
        "card_list_fields": (
            card_list_fields
        ),
        "classification_fields": (
            classification_fields
        ),
        "extraction_prompt_version": (
            extraction_prompt_version
        ),
        "relevance_prompt_version": (
            relevance_prompt_version
        ),
        "kb_schema_version": (
            kb_schema_version
        ),
        "rag_clean_validation_version": (
            rag_clean_validation_version
        ),
    }


def decide_extraction_rebuild(
    *,
    force_rebuild_extraction: Any,
    previous_manifest: Mapping[
        str, Any
    ] | None,
    stage_outputs_exist: Any,
    current_fingerprint: str,
    auto_rebuild_extraction: Any,
) -> dict[str, Any]:
    """Calculate only the notebook's rebuild status and flags."""

    if force_rebuild_extraction:
        extraction_status = (
            "force_rebuild_requested"
        )
        rebuild_required = True

    elif previous_manifest is None:
        extraction_status = (
            "no_previous_manifest"
        )
        rebuild_required = True

    elif not stage_outputs_exist:
        extraction_status = (
            "missing_outputs"
        )
        rebuild_required = True

    elif (
        previous_manifest.get(
            "fingerprint"
        )
        != current_fingerprint
    ):
        extraction_status = (
            "stale_outputs_config_changed"
        )
        rebuild_required = True

    else:
        extraction_status = (
            "outputs_are_current"
        )
        rebuild_required = False

    if (
        rebuild_required
        and not auto_rebuild_extraction
    ):
        raise RuntimeError(
            "La extracción necesita regenerarse, "
            "pero EXTRACTION_POLICY"
            "['auto_rebuild'] es False."
        )

    should_rebuild_extraction = (
        rebuild_required
        and auto_rebuild_extraction
    ) or force_rebuild_extraction

    return {
        "extraction_status": (
            extraction_status
        ),
        "rebuild_required": (
            rebuild_required
        ),
        "should_rebuild_extraction": (
            should_rebuild_extraction
        ),
    }


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in [
            "true",
            "1",
            "yes",
            "sí",
            "si",
        ]
    )


def build_extraction_manifest(
    *,
    cards: Sequence[Mapping[str, Any]],
    df_kb: pd.DataFrame,
    df_chunks_clean: pd.DataFrame,
    trace_dataframe: pd.DataFrame,
    errors_dataframe: pd.DataFrame,
    experiment_id: str,
    created_at: str,
    current_fingerprint: str,
    current_extraction_signature: Mapping[
        str, Any
    ],
    extraction_status: str,
    auto_rebuild_extraction: Any,
    force_rebuild_extraction: Any,
    should_rebuild_extraction: Any,
    reclassify_relevance: Any,
    kb_should_recreate: Any,
    backup_dir_created: str | Path | None,
    chroma_collection_name: str,
    embedding_model_name: str,
    extraction_retrieval_profile: str,
    extraction_retrieval_profile_config: Any,
    extraction_retrieval_queries: Sequence[
        Any
    ],
    retrieval_trace_csv_path: str | Path,
    validation_report: Mapping[str, Any],
    paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    if (
        "include_in_state_of_art"
        in df_kb.columns
    ):
        included_count = int(
            df_kb[
                "include_in_state_of_art"
            ]
            .apply(_to_bool)
            .sum()
        )
    else:
        included_count = None

    return {
        "stage": (
            "03_agente_extraccion_kb"
        ),
        "experiment_id": experiment_id,
        "created_or_checked_at": (
            created_at
        ),
        "fingerprint": (
            current_fingerprint
        ),
        "signature": (
            current_extraction_signature
        ),
        "automatic_decision": {
            "status": extraction_status,
            "auto_rebuild_enabled": (
                auto_rebuild_extraction
            ),
            "force_rebuild": (
                force_rebuild_extraction
            ),
            "rebuild_executed": (
                should_rebuild_extraction
            ),
            "reclassified_relevance": (
                reclassify_relevance
            ),
            "kb_recreated": (
                kb_should_recreate
            ),
            "backup_dir_created": (
                str(
                    backup_dir_created
                )
                if backup_dir_created
                else None
            ),
        },
        "retrieval": {
            "chroma_used": True,
            "collection_name": (
                chroma_collection_name
            ),
            "embedding_model": (
                embedding_model_name
            ),
            "retrieval_profile": (
                extraction_retrieval_profile
            ),
            "retrieval_profile_config": (
                extraction_retrieval_profile_config
            ),
            "num_retrieval_queries": len(
                extraction_retrieval_queries
            ),
            "num_trace_rows": int(
                len(trace_dataframe)
            ),
            "trace_file": str(
                retrieval_trace_csv_path
            ),
        },
        "safety_policy": {
            "uses_chunks_clean_for_rag": (
                True
            ),
            "ground_truth_used": False,
            "review_sections_used": False,
            "bibliography_used": False,
            "rag_clean_validated": (
                validation_report.get(
                    "valid_for_extraction",
                    False,
                )
            ),
        },
        "outputs": {
            "scientific_cards_jsonl": str(
                paths[
                    "CARDS_JSONL_PATH"
                ]
            ),
            "scientific_cards_summary_csv": str(
                paths[
                    "CARDS_SUMMARY_CSV_PATH"
                ]
            ),
            "scientific_cards_errors_csv": str(
                paths[
                    "CARDS_ERRORS_CSV_PATH"
                ]
            ),
            "scientific_cards_quality_csv": str(
                paths[
                    "CARDS_QUALITY_CSV_PATH"
                ]
            ),
            "extraction_retrieval_trace_csv": str(
                paths[
                    "RETRIEVAL_TRACE_CSV_PATH"
                ]
            ),
            "scientific_knowledge_base_csv": str(
                paths["KB_CSV_PATH"]
            ),
            "scientific_knowledge_base_jsonl": str(
                paths["KB_JSONL_PATH"]
            ),
            "manifest": str(
                paths[
                    "EXTRACTION_MANIFEST_PATH"
                ]
            ),
            "chunks_validation_report": str(
                paths[
                    "CHUNKS_VALIDATION_REPORT_PATH"
                ]
            ),
        },
        "counts": {
            "num_cards": int(
                len(cards)
            ),
            "num_kb_rows": int(
                len(df_kb)
            ),
            "num_included_in_state_of_art": (
                included_count
            ),
            "num_chunks_clean": int(
                len(df_chunks_clean)
            ),
            "num_source_papers": int(
                df_chunks_clean[
                    "source_filename"
                ].nunique()
            ),
            "num_extraction_errors": int(
                len(errors_dataframe)
            ),
        },
    }


def report_output_status(
    outputs: Sequence[str | Path],
    *,
    print_fn: Callable[..., Any] = print,
) -> None:
    for output_path in outputs:
        print_fn(
            "OK"
            if Path(
                output_path
            ).exists()
            else "FALTA",
            "→",
            output_path,
        )


def restore_validation_report(
    validation_report: Any,
    validation_report_path: str | Path,
    *,
    save_json_file_fn: Callable[
        [Any, str | Path], Any
    ] = save_json_file,
    print_fn: Callable[..., Any] = print,
) -> None:
    save_json_file_fn(
        validation_report,
        validation_report_path,
    )

    if not Path(
        validation_report_path
    ).exists():
        raise RuntimeError(
            "No se pudo restaurar el reporte "
            "de validación."
        )

    print_fn(
        "Reporte restaurado:",
        validation_report_path,
    )


def audit_final_consistency(
    *,
    chunks_validation_report_path: (
        str | Path
    ),
    extraction_manifest_path: (
        str | Path
    ),
    df_chunks_clean: pd.DataFrame,
    cards: Sequence[
        Mapping[str, Any]
    ],
    retrieval_trace_csv_path: (
        str | Path
    ),
    load_json_file_fn: Callable[
        [str | Path], Any
    ] = load_json_file,
    read_csv_fn: Callable[
        [str | Path], pd.DataFrame
    ] = pd.read_csv,
    print_fn: Callable[..., Any] = print,
) -> dict[str, Any]:
    final_validation_report = (
        load_json_file_fn(
            chunks_validation_report_path
        )
    )

    final_manifest = load_json_file_fn(
        extraction_manifest_path
    )

    if not final_validation_report:
        raise RuntimeError(
            "No existe el reporte de "
            "validación de chunks limpios."
        )

    if not final_validation_report.get(
        "valid_for_extraction",
        False,
    ):
        raise RuntimeError(
            "El reporte final indica que "
            "los chunks no son seguros "
            "para extracción."
        )

    if not final_manifest:
        raise RuntimeError(
            "No existe el manifiesto final "
            "del agente 03."
        )

    if not final_manifest.get(
        "retrieval",
        {},
    ).get(
        "chroma_used",
        False,
    ):
        raise RuntimeError(
            "El manifiesto no confirma "
            "el uso de Chroma."
        )

    expected_sources = set(
        df_chunks_clean[
            "source_filename"
        ].astype(str)
    )

    card_sources = {
        str(
            card.get(
                "source_filename",
                "",
            )
        )
        for card in cards
    }

    missing_card_sources = sorted(
        expected_sources
        - card_sources
    )

    if missing_card_sources:
        raise RuntimeError(
            "Faltan fichas para estos "
            "papers: "
            + ", ".join(
                missing_card_sources
            )
        )

    print_fn(
        "Auditoría final 03: OK"
    )
    print_fn(
        "Papers procesados:",
        len(expected_sources),
    )
    print_fn(
        "Fichas generadas:",
        len(cards),
    )

    trace_dataframe = read_csv_fn(
        retrieval_trace_csv_path
    )

    print_fn(
        "Filas de trazabilidad RAG:",
        len(trace_dataframe),
    )
    print_fn(
        "Ground Truth usado:",
        final_manifest[
            "safety_policy"
        ]["ground_truth_used"],
    )
    print_fn(
        "Secciones de revisión usadas:",
        final_manifest[
            "safety_policy"
        ]["review_sections_used"],
    )

    return {
        "expected_sources": (
            expected_sources
        ),
        "card_sources": (
            card_sources
        ),
        "missing_card_sources": (
            missing_card_sources
        ),
        "num_trace_rows": int(
            len(trace_dataframe)
        ),
    }
