"""Persistencia de los 15 outputs obligatorios de 08 (celda 23,
``required_outputs``) + intermedios. Imports pesados (pandas, para
``to_markdown``) diferidos dentro de las funciones.

Lista exacta de los 15 outputs, obtenida de ``required_outputs`` (celda 23,
verificada por grep exhaustivo en una ronda anterior — 15 entradas, ninguna
es ``numeric_hallucination_check.csv`` ni
``agent08_upstream_numeric_check.csv``, que NO participan de esta lista):

1. automatic_metrics.csv
2. semantic_chunk_alignment.csv
3. bertscore_chunk_alignment.csv
4. factual_metrics.csv
5. final_citation_check.csv
6. final_claim_audit.csv
7. llm_judge_evaluation.json
8. llm_judge_scores.csv
9. corpus_gap_suggestions.csv
10. corpus_gap_suggestions.md
11. final_selected_metrics.csv
12. evaluation_summary.json
13. final_evaluation_report.md
14. evaluation_validation_report.json
15. evaluation_manifest.json

``agent08_upstream_numeric_check.csv`` (equivalente en este código a
``RECHECK_NUMERIC_CSV_PATH``) se escribe como artefacto INTERMEDIO, en una
función separada, explícitamente fuera de la lista de 15.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_OUTPUT_FILENAMES = [
    "automatic_metrics.csv",
    "semantic_chunk_alignment.csv",
    "bertscore_chunk_alignment.csv",
    "factual_metrics.csv",
    "final_citation_check.csv",
    "final_claim_audit.csv",
    "llm_judge_evaluation.json",
    "llm_judge_scores.csv",
    "corpus_gap_suggestions.csv",
    "corpus_gap_suggestions.md",
    "final_selected_metrics.csv",
    "evaluation_summary.json",
    "final_evaluation_report.md",
    "evaluation_validation_report.json",
    "evaluation_manifest.json",
]

INTERMEDIATE_NUMERIC_CHECK_FILENAME = "agent08_upstream_numeric_check.csv"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    import pandas as pd

    return pd.DataFrame(rows).to_markdown(index=False)


def persist_intermediate_numeric_check(*, output_dir: Path, numeric_rows: list[dict[str, Any]]) -> Path:
    """Escribe el artefacto INTERMEDIO (no uno de los 15) — equivalente a
    ``RECHECK_NUMERIC_CSV_PATH`` sobrescrito por 08 en la celda 19."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / INTERMEDIATE_NUMERIC_CHECK_FILENAME
    _write_csv(path, numeric_rows)
    return path


def build_final_evaluation_report_markdown(
    *,
    experiment_id: str,
    topic_name: str,
    source_stage: str,
    reverification_performed: bool,
    reverification_reason: str | None,
    upstream_runtime_status: str,
    claims_verified: int,
    claims_requiring_manual_review: int,
    manual_review_claim_ids: list[str],
    evaluation_ready_json_path: str,
    ground_truth_source_path: str,
    automatic_metric_rows: list[dict[str, Any]],
    judge_score_rows: list[dict[str, Any]],
    overall_assessment: str,
    factual_metric_rows: list[dict[str, Any]],
    final_selected_metrics: list[dict[str, Any]],
    corpus_gap_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Reporte final de evaluación",
        "",
        "**Experimento:** " + experiment_id,
        "",
        "**Tema:** " + str(topic_name or ""),
        "",
        "## Procedencia del texto evaluado",
        "",
        "**Etapa fuente:** " + source_stage,
        "",
        "**Reverificación ejecutada:** " + str(reverification_performed),
        "",
        "**Motivo de reverificación:** " + str(reverification_reason),
        "",
        "**Estado upstream:** " + upstream_runtime_status,
        "",
        "**Claims verificados:** " + str(claims_verified),
        "",
        "**Claims con revisión manual pendiente:** " + str(claims_requiring_manual_review),
        "",
        "**IDs pendientes:** "
        + (", ".join(manual_review_claim_ids) if manual_review_claim_ids else "Ninguno"),
        "",
        "**Texto evaluado:** " + str(evaluation_ready_json_path),
        "",
        "**Ground Truth:** revisión de literatura explícita de " + str(ground_truth_source_path),
        "",
        "## 1. Métricas automáticas",
        "",
        _to_markdown(automatic_metric_rows),
        "",
        "## 2. LLM Judge",
        "",
        _to_markdown(judge_score_rows),
        "",
        "**Evaluación general:** " + str(overall_assessment),
        "",
        "## 3. Métricas factuales y de trazabilidad",
        "",
        _to_markdown(factual_metric_rows),
        "",
        "## Métricas seleccionadas para el reporte",
        "",
        _to_markdown(final_selected_metrics),
        "",
        "## Brechas frente al Ground Truth",
        "",
    ]

    if not corpus_gap_rows:
        lines.append("No se identificaron brechas temáticas claras.")
    else:
        lines.append(_to_markdown(corpus_gap_rows))

    return "\n".join(lines)


def persist_evaluation_outputs(
    *,
    output_dir: str | Path,
    automatic_metric_rows: list[dict[str, Any]],
    semantic_alignment_rows: list[dict[str, Any]],
    bertscore_pair_metadata: list[dict[str, Any]],
    factual_metric_rows: list[dict[str, Any]],
    citation_rows: list[dict[str, Any]],
    claim_audit_rows: list[dict[str, Any]],
    llm_judge_result: dict[str, Any],
    judge_score_rows: list[dict[str, Any]],
    corpus_gap_rows: list[dict[str, Any]],
    corpus_gap_markdown: str,
    final_selected_metrics: list[dict[str, Any]],
    evaluation_summary: dict[str, Any],
    final_evaluation_report_markdown: str,
    evaluation_validation_report: dict[str, Any],
    evaluation_manifest: dict[str, Any],
) -> dict[str, Path]:
    """Escribe los 15 outputs obligatorios. Devuelve
    ``{filename: path}`` para los 15 — el llamador (contrato transaccional)
    construye los ``ArtifactReference`` con hash a partir de esto."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    def out(name: str) -> Path:
        path = output_dir / name
        paths[name] = path
        return path

    _write_csv(out("automatic_metrics.csv"), automatic_metric_rows)
    _write_csv(out("semantic_chunk_alignment.csv"), semantic_alignment_rows)
    _write_csv(out("bertscore_chunk_alignment.csv"), bertscore_pair_metadata)
    _write_csv(out("factual_metrics.csv"), factual_metric_rows)
    _write_csv(out("final_citation_check.csv"), citation_rows)
    _write_csv(out("final_claim_audit.csv"), claim_audit_rows)
    _write_json(llm_judge_result, out("llm_judge_evaluation.json"))
    _write_csv(out("llm_judge_scores.csv"), judge_score_rows)
    _write_csv(out("corpus_gap_suggestions.csv"), corpus_gap_rows)
    out("corpus_gap_suggestions.md").write_text(corpus_gap_markdown, encoding="utf-8")
    _write_csv(out("final_selected_metrics.csv"), final_selected_metrics)
    _write_json(evaluation_summary, out("evaluation_summary.json"))
    out("final_evaluation_report.md").write_text(final_evaluation_report_markdown, encoding="utf-8")
    _write_json(evaluation_validation_report, out("evaluation_validation_report.json"))
    _write_json(evaluation_manifest, out("evaluation_manifest.json"))

    return paths


def find_missing_outputs(*, output_dir: str | Path) -> list[str]:
    output_dir = Path(output_dir)
    return [name for name in REQUIRED_OUTPUT_FILENAMES if not (output_dir / name).is_file()]


def backup_existing_outputs(*, output_dir: str | Path, backup_root: str | Path) -> Path | None:
    """Reproduce ``backup_evaluation_outputs`` (celda 1): si ya existen
    outputs previos, los copia a un directorio de backup con timestamp
    antes de sobrescribir. Devuelve la ruta del backup, o ``None`` si no
    había nada que respaldar."""

    import shutil
    from datetime import datetime

    output_dir = Path(output_dir)
    existing = [name for name in REQUIRED_OUTPUT_FILENAMES if (output_dir / name).is_file()]
    if not existing:
        return None

    backup_dir = Path(backup_root) / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in existing:
        shutil.copy2(output_dir / name, backup_dir / name)
    return backup_dir
