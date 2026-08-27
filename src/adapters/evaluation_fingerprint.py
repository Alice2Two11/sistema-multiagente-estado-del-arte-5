"""Fingerprint y política de reconstrucción de 08 (celda 15). Reproduce
``evaluation_signature``/``evaluation_fingerprint``/``SHOULD_REBUILD_EVALUATION``
sin la parte de backup en disco (que vive en ``evaluation_persistence.py``).

``stable_hash_dict``/``sha256_file`` no estaban en ningún bloque anterior
— se portan aquí, literales de notebook 08 celda 1.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.state.fingerprints import sha256_file

__all__ = [
    "stable_hash_dict",
    "sha256_file",
    "sha256_text",
    "build_evaluation_signature",
    "compute_evaluation_fingerprint",
    "resolve_rebuild_decision",
]


def stable_hash_dict(value: dict[str, Any]) -> str:
    """Copia literal de notebook 08, celda 1: hash estable (orden de claves
    fijo) de un diccionario JSON-serializable."""

    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_evaluation_signature(
    *,
    experiment_id: str,
    evaluation_policy: dict[str, Any],
    openai_model: str,
    evaluation_ready_json_path: str | Path,
    upstream_fingerprint: str | None,
    ground_truth_text: str,
    chunks: list[dict[str, Any]],
    traceability_rows: list[dict[str, Any]],
    llm_judge_prompt_version: str,
) -> dict[str, Any]:
    """``evaluation_signature`` (celda 15) — fuente ÚNICA de fingerprint,
    usada tanto para el fingerprint transaccional de ``StateStore``
    (decide ``SKIPPED_FRESH``) como para ``evaluation_manifest.json``. No
    hay una segunda representación "débil": ambos consumen exactamente
    este mismo diccionario.

    Incluye TODAS las dependencias que pueden cambiar el resultado
    (pedido A1): estado del arte evaluado, fingerprint upstream de 07,
    Ground Truth, ``chunks_clean_for_rag.csv``, trazabilidad,
    ``evaluation_policy`` completa (ya incluye ``evaluation_embedding_model``
    y ``bertscore_model`` como claves propias — no se duplican sueltas),
    modelo OpenAI, experimento activo, y la versión del prompt del LLM
    Judge (``llm_judge.PROMPT_VERSION`` — la única "versión" que el
    notebook real trae consigo; no hay otro prompt versionado en 08)."""

    return {
        "stage": "08_evaluacion_experimental",
        "experiment_id": experiment_id,
        "evaluation_policy": evaluation_policy,
        "openai_model": openai_model,
        "llm_judge_prompt_version": llm_judge_prompt_version,
        "generated_source": str(evaluation_ready_json_path),
        "generated_sha256": sha256_file(evaluation_ready_json_path),
        "upstream_fingerprint": upstream_fingerprint,
        "ground_truth_sha256": sha256_text(ground_truth_text),
        "chunks_fingerprint": stable_hash_dict({"chunks": chunks}),
        "traceability_fingerprint": stable_hash_dict({"traceability_rows": traceability_rows}),
    }


def compute_evaluation_fingerprint(evaluation_signature: dict[str, Any]) -> str:
    return stable_hash_dict(evaluation_signature)


def resolve_rebuild_decision(
    *,
    force_rebuild: bool,
    auto_rebuild: bool,
    previous_manifest: dict[str, Any] | None,
    current_fingerprint: str,
    missing_outputs: list[str],
) -> dict[str, Any]:
    """Reproduce la rama de decisión real de la celda 15 (ya caracterizada
    en ``tests/orchestration/test_evaluation_characterization_initial.py``,
    T11-T15): ``force_rebuild`` > ausencia de manifiesto previo > fingerprint
    obsoleto > outputs faltantes > vigente. Si hace falta reconstruir y
    ``auto_rebuild`` es False (y no es ``force_rebuild``), lanza
    ``RuntimeError`` con el mismo mensaje real."""

    if force_rebuild:
        status, rebuild_required = "force_rebuild", True
    elif previous_manifest is None:
        status, rebuild_required = "no_previous_manifest", True
    elif previous_manifest.get("fingerprint") != current_fingerprint:
        status, rebuild_required = "stale_outputs_dependency_changed", True
    elif missing_outputs:
        status, rebuild_required = "missing_outputs", True
    else:
        status, rebuild_required = "outputs_are_current", False

    if rebuild_required and not auto_rebuild and not force_rebuild:
        raise RuntimeError(
            "La evaluación necesita regenerarse, pero "
            "EVALUATION_POLICY['auto_rebuild'] es False."
        )

    should_rebuild = force_rebuild or (rebuild_required and auto_rebuild)
    return {"status": status, "should_rebuild": should_rebuild}
