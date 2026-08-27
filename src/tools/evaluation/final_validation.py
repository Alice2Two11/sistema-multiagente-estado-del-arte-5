"""Validación final de la etapa 08 (celda 23): ``evaluation_validation_ok``.

Distinta de ``factual_consistency_ok`` (celda 19, ``factual_assembly.py``).
Esta SÍ está gobernada por ``EVALUATION_POLICY["fail_on_invalid_evaluation"]``.

Reglas preservadas exactamente:
- Si ``factual_consistency_ok`` es False: agrega un WARNING (no bloqueante)
  si ``source_stage == "AGENT07" and upstream_runtime_status == "PARTIAL"``;
  en cualquier otro caso agrega un ERROR.
- ``llm_judge_invalid`` si ``judge_errors`` no está vacío.
- ``unexpected_final_metric_count`` si las métricas seleccionadas no son
  exactamente 15.
- ``missing_metric:<nombre>`` por cada una de las 7 métricas obligatorias
  ausentes: rougeL_fmeasure, bertscore_f1, semantic_f1, factual_precision,
  citation_error_rate, hallucination_rate, evidence_coverage.
- ``evaluation_validation_ok = len(validation_errors) == 0`` — nótese que
  el WARNING de arriba NO cuenta como error: la evaluación puede quedar
  ``validation_ok=True`` incluso con ``factual_consistency_ok=False``, si
  fue el caso PARTIAL/AGENT07 y no hay ningún otro error.
- Bloqueo real: ``if not evaluation_validation_ok and
  FAIL_ON_INVALID_EVALUATION: raise ValueError(...)`` — el mensaje es
  literal del notebook.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

REQUIRED_FINAL_METRICS = [
    "rougeL_fmeasure",
    "bertscore_f1",
    "semantic_f1",
    "factual_precision",
    "citation_error_rate",
    "hallucination_rate",
    "evidence_coverage",
]


def evaluate_final_validation(
    *,
    factual_consistency_ok: bool,
    factual_consistency_status: str,
    source_stage: str,
    upstream_runtime_status: str,
    judge_errors: list[str],
    final_selected_metrics: list[dict[str, Any]],
    experiment_id: str,
    reverification_performed: bool,
    reverification_reason: str | None,
    claims_requiring_manual_review: int,
    manual_review_claim_ids: list[str],
) -> dict[str, Any]:
    validation_errors: list[str] = []
    validation_warnings: list[str] = []

    if not factual_consistency_ok:
        if source_stage == "AGENT07" and upstream_runtime_status == "PARTIAL":
            validation_warnings.append("upstream_partial_factual_consistency_not_approved")
        else:
            validation_errors.append("factual_consistency_not_approved")

    if judge_errors:
        validation_errors.append("llm_judge_invalid")

    if len(final_selected_metrics) != 15:
        validation_errors.append("unexpected_final_metric_count")

    present_metrics = {row["metric"] for row in final_selected_metrics}
    for required_metric in REQUIRED_FINAL_METRICS:
        if required_metric not in present_metrics:
            validation_errors.append(f"missing_metric:{required_metric}")

    evaluation_validation_ok = len(validation_errors) == 0

    evaluation_validation_report = {
        "stage": "08_evaluacion_experimental",
        "validation_version": "v5_three_sections_complete",
        "created_at": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "validation_ok": evaluation_validation_ok,
        "errors": validation_errors,
        "warnings": validation_warnings,
        "source_stage": source_stage,
        "reverification_performed": reverification_performed,
        "reverification_reason": reverification_reason,
        "upstream_runtime_status": upstream_runtime_status,
        "claims_requiring_manual_review": claims_requiring_manual_review,
        "manual_review_claim_ids": manual_review_claim_ids,
        "evaluated_only_07c_evaluation_ready": source_stage == "AGENT07C",
        "ground_truth_used_only_in_evaluation": True,
        "ground_truth_scope": "explicit_literature_review_section_only",
        "factual_consistency_ok": factual_consistency_ok,
        "factual_consistency_status": factual_consistency_status,
        "llm_judge_valid": not judge_errors,
        "final_metric_count": int(len(final_selected_metrics)),
    }

    return {
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "evaluation_validation_ok": evaluation_validation_ok,
        "evaluation_validation_report": evaluation_validation_report,
    }


def resolve_final_validation_gate(
    *, final_validation_result: dict[str, Any], fail_on_invalid_evaluation: bool, validation_report_path: str
) -> None:
    """Reproduce EXACTAMENTE el bloqueo real: ``if not evaluation_validation_ok
    and FAIL_ON_INVALID_EVALUATION: raise ValueError(...)``. Mensaje literal
    del notebook."""

    if not final_validation_result["evaluation_validation_ok"] and fail_on_invalid_evaluation:
        raise ValueError(
            "La evaluación final no superó su validación. "
            f"Revisa {validation_report_path}."
        )
