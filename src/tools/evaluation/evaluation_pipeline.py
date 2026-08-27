"""Ensamblador integral de la etapa 08: conecta TODOS los bloques ya
migrados y probados, en el orden real del notebook. No reimplementa
ninguna fórmula — solo llama a los módulos ya aprobados. Sin persistencia
(eso vive en ``src/adapters/evaluation_persistence.py``).

Orden real reproducido (celdas 13/15/17/19/21/23):
1. Ground Truth (Bloque 2).
2. Normalización de textos (Bloque 1, ya aplicada por el llamador al texto
   generado — ver nota abajo).
3. Idioma + traducción (Bloque 3) — dentro de ``build_automatic_metrics``.
4. Métricas automáticas: ROUGE-L/semántica/BERTScore (Bloques 4A-4C,
   ensambladas por ``build_automatic_metrics``).
5. Auditoría factual 5A+5B (``factual_assembly.build_factual_audit``).
6. ``factual_consistency_ok`` y su puerta (``factual_assembly``).
7. LLM Judge.
8. Métricas seleccionadas (15 filas).
9. ``evaluation_validation_ok`` y su puerta (``final_validation``).
10. Resumen, sugerencias de corpus.

Nota sobre el texto generado: el notebook real normaliza el texto generado
(``normalize_content_text``) ANTES de este bloque, como parte de la
resolución del draft (fuera de alcance de 08 en sí — pertenece a la
resolución del upstream de 07, ``src/adapters/evaluation_upstream.py``, ya
existente). Este ensamblador recibe ``generated_plain_text`` YA
normalizado — no lo normaliza de nuevo.
"""

from __future__ import annotations

from typing import Any, Callable

from src.tools.evaluation.automatic_metrics import build_automatic_metrics
from src.tools.evaluation.factual_assembly import (
    build_factual_audit,
    evaluate_factual_consistency,
    resolve_factual_gate,
)
from src.tools.evaluation.final_report import (
    build_corpus_gap_markdown,
    build_corpus_gap_rows,
    build_evaluation_summary,
    build_final_selected_metrics,
)
from src.tools.evaluation.final_validation import (
    evaluate_final_validation,
    resolve_final_validation_gate,
)
from src.tools.evaluation.ground_truth import resolve_ground_truth_comparable_text
from src.tools.evaluation.language_preprocessing import detect_language_code
from src.tools.evaluation.llm_judge import build_judge_score_rows, run_llm_judge


def run_evaluation_pipeline(
    *,
    # --- resueltos por el llamador (evaluation_upstream.py, ya existente) ---
    generated_plain_text: str,
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    traceability_rows: list[dict[str, Any]],
    source_stage: str,
    upstream_runtime_status: str,
    reverification_performed: bool,
    reverification_reason: str | None,
    claims_verified: int,
    claims_requiring_manual_review: int,
    manual_review_claim_ids: list[str],
    generated_status: str | None,
    evaluation_ready_json_path: str,
    experiment_id: str,
    topic_name: str,
    # --- Ground Truth (Bloque 2) ---
    ground_truth_dir: str,
    # --- política ya resuelta (sin defaults) ---
    evaluation_policy: dict[str, Any],
    # --- dependencias inyectables ---
    translation_llm_factory: Callable[[], Any],
    embedding_model_factory: Callable[[str], Any] | None,
    bertscore_score_fn: Callable[..., Any] | None,
    judge_llm_factory: Callable[[], Any],
) -> dict[str, Any]:
    # --- 1. Ground Truth ---
    ground_truth_plain_text, ground_truth_metadata, ground_truth_source_path = (
        resolve_ground_truth_comparable_text(
            ground_truth_dir=ground_truth_dir,
            minimum_words=evaluation_policy["minimum_ground_truth_words"],
            require_explicit_end_heading=evaluation_policy[
                "require_explicit_ground_truth_end_heading"
            ],
        )
    )

    if len(generated_plain_text.split()) < evaluation_policy["minimum_generated_words"]:
        raise ValueError(
            "El texto generado es demasiado corto para una evaluación válida."
        )

    # Confirmado en celda 21 (no es un "saltar el Judge silenciosamente" —
    # es un ValueError bloqueante real si la política desactiva el Judge).
    if not evaluation_policy["run_llm_judge"]:
        raise ValueError(
            "La política de evaluación desactivó el LLM Judge, "
            "pero esta tesis requiere la rúbrica cualitativa."
        )

    # --- 3. Idioma ---
    generated_language = detect_language_code(generated_plain_text)
    ground_truth_language = detect_language_code(ground_truth_plain_text)

    # --- 3-4. Traducción + métricas automáticas (Bloques 3, 4A-4C) ---
    automatic_metrics_result = build_automatic_metrics(
        generated_plain_text=generated_plain_text,
        ground_truth_plain_text=ground_truth_plain_text,
        generated_language=generated_language,
        ground_truth_language=ground_truth_language,
        evaluation_policy=evaluation_policy,
        translation_llm_factory=translation_llm_factory,
        embedding_model_factory=embedding_model_factory,
        bertscore_score_fn=bertscore_score_fn,
    )

    # --- 5-6. Auditoría factual + factual_consistency_ok (Bloques 5A+5B) ---
    factual_audit = build_factual_audit(
        sections=sections,
        chunks=chunks,
        traceability_rows=traceability_rows,
        generated_content_text=generated_plain_text,
    )
    factual_consistency_result = evaluate_factual_consistency(factual_audit)
    resolve_factual_gate(
        factual_consistency_result=factual_consistency_result,
        source_stage=source_stage,
        upstream_runtime_status=upstream_runtime_status,
    )

    # --- 7. LLM Judge ---
    automatic_metrics_dict = {
        row["metric"]: float(row["value"]) for row in automatic_metrics_result.automatic_metric_rows
    }
    factual_metrics_dict = {
        row["metric"]: row["value"] for row in factual_audit["factual_metric_rows"]
    }
    judge_run = run_llm_judge(
        topic_name=topic_name,
        source_stage=source_stage,
        automatic_metrics=automatic_metrics_dict,
        factual_metrics=factual_metrics_dict,
        generated_plain_text=generated_plain_text,
        ground_truth_plain_text=ground_truth_plain_text,
        max_generated_chars=evaluation_policy["llm_judge_max_generated_chars"],
        max_ground_truth_chars=evaluation_policy["llm_judge_max_ground_truth_chars"],
        max_attempts=evaluation_policy["llm_judge_max_attempts"],
        llm_factory=judge_llm_factory,
    )
    llm_judge_result = judge_run["result"]
    judge_score_rows = build_judge_score_rows(llm_judge_result)
    from src.tools.evaluation.llm_judge import validate_judge_result

    judge_errors = validate_judge_result(llm_judge_result)

    # --- 8. Métricas seleccionadas (15 filas) ---
    final_selected_metrics = build_final_selected_metrics(
        automatic_metric_rows=automatic_metrics_result.automatic_metric_rows,
        judge_score_rows=judge_score_rows,
        factual_metric_rows=factual_audit["factual_metric_rows"],
    )

    # --- 9. evaluation_validation_ok (celda 23) ---
    final_validation = evaluate_final_validation(
        factual_consistency_ok=factual_consistency_result["factual_consistency_ok"],
        factual_consistency_status=factual_consistency_result["factual_consistency_status"],
        source_stage=source_stage,
        upstream_runtime_status=upstream_runtime_status,
        judge_errors=judge_errors,
        final_selected_metrics=final_selected_metrics,
        experiment_id=experiment_id,
        reverification_performed=reverification_performed,
        reverification_reason=reverification_reason,
        claims_requiring_manual_review=claims_requiring_manual_review,
        manual_review_claim_ids=manual_review_claim_ids,
    )

    # --- 10. Resumen + sugerencias de corpus ---
    corpus_gap_rows = build_corpus_gap_rows(
        missing_topics_or_omissions=llm_judge_result.get("missing_topics_or_omissions", []),
        create_corpus_gap_suggestions=evaluation_policy["create_corpus_gap_suggestions"],
    )
    corpus_gap_markdown = build_corpus_gap_markdown(corpus_gap_rows)

    evaluation_summary = build_evaluation_summary(
        experiment_id=experiment_id,
        topic_name=topic_name,
        evaluation_ready_json_path=evaluation_ready_json_path,
        source_stage=source_stage,
        reverification_performed=reverification_performed,
        reverification_reason=reverification_reason,
        upstream_runtime_status=upstream_runtime_status,
        claims_verified=claims_verified,
        claims_requiring_manual_review=claims_requiring_manual_review,
        manual_review_claim_ids=manual_review_claim_ids,
        generated_status=generated_status,
        ground_truth_source_path=str(ground_truth_source_path),
        generated_plain_text=generated_plain_text,
        ground_truth_plain_text=ground_truth_plain_text,
        generated_language=generated_language,
        ground_truth_language=ground_truth_language,
        translation_mode=automatic_metrics_result.translation_mode,
        automatic_metric_rows=automatic_metrics_result.automatic_metric_rows,
        judge_score_rows=judge_score_rows,
        factual_metric_rows=factual_audit["factual_metric_rows"],
        factual_consistency_status=factual_consistency_result["factual_consistency_status"],
        overall_assessment=llm_judge_result["overall_assessment"],
        corpus_gap_count=len(corpus_gap_rows),
    )

    resolve_final_validation_gate(
        final_validation_result=final_validation,
        fail_on_invalid_evaluation=evaluation_policy["fail_on_invalid_evaluation"],
        validation_report_path="evaluation_validation_report.json",
    )

    return {
        "ground_truth_plain_text": ground_truth_plain_text,
        "ground_truth_metadata": ground_truth_metadata,
        "ground_truth_source_path": ground_truth_source_path,
        "generated_language": generated_language,
        "ground_truth_language": ground_truth_language,
        "automatic_metrics_result": automatic_metrics_result,
        "factual_audit": factual_audit,
        "factual_consistency_result": factual_consistency_result,
        "judge_run": judge_run,
        "llm_judge_result": llm_judge_result,
        "judge_score_rows": judge_score_rows,
        "judge_errors": judge_errors,
        "final_selected_metrics": final_selected_metrics,
        "final_validation": final_validation,
        "corpus_gap_rows": corpus_gap_rows,
        "corpus_gap_markdown": corpus_gap_markdown,
        "evaluation_summary": evaluation_summary,
    }
