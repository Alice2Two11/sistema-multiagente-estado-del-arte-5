"""Construcción del reporte final de 08 (celda 23): métricas seleccionadas
(15 filas), sugerencias de ampliación de corpus, y ``evaluation_summary``.
Solo construcción en memoria — persistencia queda en el adaptador.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.tools.evaluation.text_normalization import safe_str

# (metric, label) — orden real exacto de la celda 23.
_AUTOMATIC_SELECTED = [
    ("rougeL_fmeasure", "ROUGE-L F1"),
    ("bertscore_f1", "BERTScore F1"),
    ("semantic_f1", "Similitud semántica F1"),
    ("global_semantic_similarity", "Similitud semántica global"),
]
_FACTUAL_SELECTED = [
    ("factual_precision", "Precisión factual"),
    ("citation_error_rate", "Error de cita"),
    ("hallucination_rate", "Tasa de alucinación"),
    ("evidence_coverage", "Cobertura de evidencia"),
    ("traceability_text_coverage", "Cobertura de trazabilidad"),
    ("numeric_error_rate", "Error numérico"),
]


def _metric_value(rows: list[dict[str, Any]], metric_name: str) -> Any:
    for row in rows:
        if str(row["metric"]) == metric_name:
            value = row["value"]
            try:
                return float(value)
            except Exception:
                return value
    return None


def build_final_selected_metrics(
    *,
    automatic_metric_rows: list[dict[str, Any]],
    judge_score_rows: list[dict[str, Any]],
    factual_metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """4 automáticas + 5 del Judge + 6 factuales = 15 filas, orden exacto."""

    rows: list[dict[str, Any]] = []

    for metric, label in _AUTOMATIC_SELECTED:
        rows.append(
            {
                "section": "Métricas automáticas",
                "criterion": label,
                "metric": metric,
                "value": _metric_value(automatic_metric_rows, metric),
            }
        )

    for row in judge_score_rows:
        rows.append(
            {
                "section": "LLM Judge",
                "criterion": row["metric"],
                "metric": "llm_judge_" + row["metric"],
                "value": int(row["score_1_to_5"]),
            }
        )

    for metric, label in _FACTUAL_SELECTED:
        rows.append(
            {
                "section": "Métricas factuales",
                "criterion": label,
                "metric": metric,
                "value": _metric_value(factual_metric_rows, metric),
            }
        )

    return rows


def build_corpus_gap_rows(
    *, missing_topics_or_omissions: list[dict[str, Any]], create_corpus_gap_suggestions: bool
) -> list[dict[str, Any]]:
    if not create_corpus_gap_suggestions:
        return []

    gap_rows = []
    for omission in missing_topics_or_omissions:
        if not isinstance(omission, dict):
            continue
        gap_rows.append(
            {
                "topic": safe_str(omission.get("topic")),
                "ground_truth_basis": safe_str(omission.get("ground_truth_basis")),
                "importance": safe_str(omission.get("importance")),
                "search_keywords": "; ".join(
                    safe_str(keyword)
                    for keyword in omission.get("search_keywords", [])
                    if safe_str(keyword)
                ),
            }
        )
    return gap_rows


def build_corpus_gap_markdown(gap_rows: list[dict[str, Any]]) -> str:
    lines = ["# Sugerencias de ampliación del corpus", ""]

    if not gap_rows:
        lines.append(
            "El LLM Judge no identificó brechas temáticas "
            "suficientemente claras frente al Ground Truth."
        )
    else:
        lines.append(
            "Las siguientes brechas proceden de la comparación "
            "con la revisión de literatura del Ground Truth. "
            "Son orientaciones para búsqueda manual, no papers inventados."
        )
        lines.append("")
        for row in gap_rows:
            lines.extend(
                [
                    "## " + safe_str(row["topic"]),
                    "",
                    "**Base en Ground Truth:** " + safe_str(row["ground_truth_basis"]),
                    "",
                    "**Importancia:** " + safe_str(row["importance"]),
                    "",
                    "**Palabras clave:** " + safe_str(row["search_keywords"]),
                    "",
                ]
            )

    return "\n".join(lines)


def build_evaluation_summary(
    *,
    experiment_id: str,
    topic_name: str,
    evaluation_ready_json_path: str,
    source_stage: str,
    reverification_performed: bool,
    reverification_reason: str | None,
    upstream_runtime_status: str,
    claims_verified: int,
    claims_requiring_manual_review: int,
    manual_review_claim_ids: list[str],
    generated_status: str | None,
    ground_truth_source_path: str,
    generated_plain_text: str,
    ground_truth_plain_text: str,
    generated_language: str,
    ground_truth_language: str,
    translation_mode: str,
    automatic_metric_rows: list[dict[str, Any]],
    judge_score_rows: list[dict[str, Any]],
    factual_metric_rows: list[dict[str, Any]],
    factual_consistency_status: str,
    overall_assessment: str,
    corpus_gap_count: int,
) -> dict[str, Any]:
    automatic_metrics_dict = {row["metric"]: float(row["value"]) for row in automatic_metric_rows}
    factual_metrics_dict = {row["metric"]: row["value"] for row in factual_metric_rows}
    llm_judge_scores_dict = {row["metric"]: int(row["score_1_to_5"]) for row in judge_score_rows}

    return {
        "stage": "08_evaluacion_experimental",
        "experiment_id": experiment_id,
        "topic": topic_name,
        "created_at": datetime.now().isoformat(),
        "generated_source": str(evaluation_ready_json_path),
        "source_stage": source_stage,
        "reverification_performed": reverification_performed,
        "reverification_reason": reverification_reason,
        "upstream_runtime_status": upstream_runtime_status,
        "claims_verified": claims_verified,
        "claims_requiring_manual_review": claims_requiring_manual_review,
        "manual_review_claim_ids": manual_review_claim_ids,
        "generated_status": generated_status,
        "ground_truth_source": str(ground_truth_source_path),
        "ground_truth_scope": "explicit_literature_review_section_only",
        "generated_words": len(generated_plain_text.split()),
        "ground_truth_words": len(ground_truth_plain_text.split()),
        "languages": {
            "generated": generated_language,
            "ground_truth": ground_truth_language,
            "rouge_translation_mode": translation_mode,
        },
        "automatic_metrics": automatic_metrics_dict,
        "llm_judge_scores": llm_judge_scores_dict,
        "factual_metrics": factual_metrics_dict,
        "factual_consistency_status": factual_consistency_status,
        "overall_assessment": overall_assessment,
        "corpus_gap_count": int(corpus_gap_count),
    }
