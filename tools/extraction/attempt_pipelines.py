"""
Pipelines de intento 1 y 2 del Agente 03 (extracción científica).

Extraídos literalmente de ``ExtractionAgent.execute()`` -- copia EXACTA
del código previo, sin cambiar ninguna condición, orden, mensaje ni
comentario. Solo se envolvió en funciones con parámetros explícitos en
vez de closures sobre variables locales de ``execute()``, y se
sustituyeron las referencias a ``self.dependencies``/``self._quality_row``/
``self._collect_output_artifacts`` por los parámetros equivalentes
(``dependencies``, ``quality_row_fn``, ``collect_artifacts_fn``), que el
caller (``ExtractionAgent.execute()``) sigue pasando exactamente iguales
a como los usaba antes.

Cada función devuelve o bien un ``AgentResult`` (cuando el intento 1
termina en una transición ``RETRY`` anticipada, igual que antes) o un
``dict`` con el estado actualizado que ``execute()`` debe seguir usando.
"""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import pandas as pd

from src.contracts.agent_result import (
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.tools.extraction.card_extraction import (
    generate_repaired_card_for_source,
)
from src.tools.extraction.card_validation import (
    QUALITY_COLUMNS,
    SUMMARY_COLUMNS,
    build_summary_row,
    is_bad_card,
)
from src.tools.extraction.retrieval import (
    build_context_from_chunks,
)
from src.tools.extraction.stage_artifacts import (
    save_dataframe_even_if_empty,
)
from src.tools.extraction.revision_strategy import (
    REVISION_PLAN_COLUMNS,
    build_revision_plan,
    plan_by_source,
)
from src.tools.extraction.review_exclusion import (
    apply_review_exclusion_policy,
)
from src.tools.extraction.corpus_eligibility import (
    apply_pre_eligibility_policy,
)


def run_first_attempt_pipeline(
    *,
    dependencies: Any,
    quality_row_fn: Any,
    collect_artifacts_fn: Any,
    card_json_parser: Any,
    retrieve: Any,
    df_chunks_clean: pd.DataFrame,
    metrics: dict[str, Any],
    paths: Mapping[str, Any],
    policy: Mapping[str, Any],
    quarantine_audit_rows: list[dict[str, Any]],
    retrieval_policy: Mapping[str, Any],
    review_exclusion_audit_rows: list[dict[str, Any]],
    signature_policy: Mapping[str, Any],
    started_at: str,
    validation_calls: int,
    warnings: list[Any],
    llm_calls: int,
    retrieval_rounds: int,
) -> "AgentResult | dict[str, Any]":
    """Copia EXACTA del bloque ``if agent_input.is_first_attempt():`` de
    ``ExtractionAgent.execute()`` -- ver docstring del módulo."""

    source_filenames = sorted(
        str(value)
        for value in df_chunks_clean["source_filename"].unique()
    )
    # Ejecuta la extracción inicial de fichas científicas para todos los papers
    # y registra las fichas, trazas de recuperación, errores y llamadas realizadas.
    extraction_created_at = dependencies.now_factory()
    initial_result = dependencies.run_initial(
        source_filenames,
        retrieve=retrieve,
        build_context=build_context_from_chunks,
        prompt_builder=dependencies.extraction_prompt_builder,
        experiment_profile=signature_policy["experiment_profile"],
        llm=dependencies.main_llm,
        json_parser=card_json_parser,
        message_factory=dependencies.message_factory,
        max_chunks_per_paper=int(
            retrieval_policy["max_chunks_per_paper"]
        ),
        max_context_chars=int(
            retrieval_policy["max_context_chars"]
        ),
        created_at=extraction_created_at,
    )
    cards = list(initial_result["cards"])
    retrieval_trace_rows = list(
        initial_result["retrieval_trace_rows"]
    )
    extraction_errors = list(initial_result["extraction_errors"])
    initial_calls = int(initial_result["llm_calls"])
    llm_calls += initial_calls
    retrieval_rounds += len(source_filenames)

    # Repara títulos faltantes o incorrectos después de la extracción inicial
    # y actualiza las fichas, los errores y el número de llamadas al LLM.
    title_result_initial = dependencies.run_title_repair(
        cards,
        df_chunks_clean=df_chunks_clean,
        title_repair_first_chunks=int(
            policy["title_repair_first_chunks"]
        ),
        repair_llm=dependencies.repair_llm,
        json_parser=dependencies.json_parser,
        should_rebuild_extraction=True,
        message_factory=dependencies.message_factory,
        created_at=dependencies.now_factory(),
        extraction_errors=extraction_errors,
    )
    cards = title_result_initial["cards"]
    extraction_errors = title_result_initial["extraction_errors"]
    title_calls_initial = int(title_result_initial["llm_calls"])
    llm_calls += title_calls_initial

    # Exclusión determinista y auditable de reviews
    exclusion_policy = dict(
        signature_policy.get("extraction_policy", {})
    )
                    
    # Aplica la política de exclusión de papers de revisión,
    # actualiza las fichas y registra en la auditoría cuáles fueron excluidos.
    exclusion_result_initial = apply_review_exclusion_policy(
        cards,
        exclude_reviews=bool(exclusion_policy.get("exclude_reviews", True)),
        created_at=dependencies.now_factory(),
    )
    cards = exclusion_result_initial["cards"]
    review_exclusion_audit_rows.extend(
        exclusion_result_initial["audit_rows"]
    )

    # Aplica una preclasificación documental para marcar papers que deben
    # incluirse, excluirse o enviarse a cuarentena antes del control de calidad científico.
    pre_eligibility_result = apply_pre_eligibility_policy(
        cards, created_at=dependencies.now_factory(),
    )
    cards = pre_eligibility_result["cards"]
    quarantine_audit_rows.extend(
        pre_eligibility_result["quarantine_audit_rows"]
    )

    # Construye el plan de revisión y los reportes de las fichas extraídas,
    # guarda los resultados de la etapa.
    revision_rows = build_revision_plan(
        cards,
        extraction_errors,
        retrieval_trace_rows,
    )
    summary_rows = [build_summary_row(card) for card in cards]
    quality_rows = [quality_row_fn(card) for card in cards]
    dependencies.save_jsonl(cards, paths["CARDS_JSONL_PATH"])
    dependencies.save_dataframe(
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS),
        paths["CARDS_SUMMARY_CSV_PATH"],
    )
    dependencies.save_dataframe(
        pd.DataFrame(quality_rows, columns=QUALITY_COLUMNS),
        paths["CARDS_QUALITY_CSV_PATH"],
    )
    dependencies.save_dataframe(
        pd.DataFrame(revision_rows, columns=REVISION_PLAN_COLUMNS),
        paths["CARDS_REVISION_PLAN_CSV_PATH"],
    )
    save_dataframe_even_if_empty(
        extraction_errors,
        paths["CARDS_ERRORS_CSV_PATH"],
        policy["error_columns"],
    )
    save_dataframe_even_if_empty(
        retrieval_trace_rows,
        paths["RETRIEVAL_TRACE_CSV_PATH"],
        policy["trace_columns"],
    )
    metrics["scientific"].update({
        "num_cards": len(cards),
        "num_trace_rows": len(retrieval_trace_rows),
        "num_extraction_errors": len(extraction_errors),
        "num_titles_selected_for_repair": len(
            title_result_initial["missing_title_cards"]
        ),
        "num_title_llm_calls": title_calls_initial,
        "num_revision_plan_rows": len(revision_rows),
    })

    # Si existen fichas que requieren revisión, identifica la causa principal,
    # clasifica el problema y solicita un único reintento de la etapa.
    if revision_rows:
        reason_codes = tuple(dict.fromkeys(
            str(row["primary_reason_code"])
            for row in revision_rows
        ))
        if any(
            code in {
                "INVALID_LLM_OUTPUT",
                "INVALID_CARD_SCHEMA",
                "MISSING_OR_INVALID_TITLE",
                "MISSING_CRITICAL_FIELDS",
            }
            for code in reason_codes
        ):
            quality_status = QualityStatus.NEEDS_REVISION
        else:
            quality_status = QualityStatus.NEEDS_MORE_EVIDENCE
        completed_at = dependencies.now_factory()
        return AgentResult(
            execution_status=ExecutionStatus.COMPLETED,
            quality_status=quality_status,
            decision=DecisionInfo(
                code=quality_status.value,
                rationale=(
                    "El intento 1 generó un plan de revisión por ficha; "
                    "se solicita una única transacción con attempt_number=2."
                ),
            ),
            quality_metrics=metrics,
            warnings=tuple(warnings),
            failure_reason_codes=reason_codes,
            requested_transition=RequestedTransition(
                action=TransitionAction.RETRY,
                target_stage=None,
                reason_code=quality_status.value,
                requires_human_confirmation=False,
            ),
            output_artifacts=collect_artifacts_fn(paths),
            tool_usage=ToolUsage(
                retrieval_rounds=retrieval_rounds,
                llm_calls=llm_calls,
                validation_calls=validation_calls,
            ),
            attempt_number=1,
            started_at=started_at,
            completed_at=completed_at,
            error=None,
        )

    # Prepara el segundo intento: reinicia los contadores de reparación y,
    # si corresponde, recupera las fichas, errores y trazas generados en el intento 1.
    bad_after_repair = []
    repair_calls = 0

    return {
        "cards": cards,
        "extraction_errors": extraction_errors,
        "retrieval_trace_rows": retrieval_trace_rows,
        "llm_calls": llm_calls,
        "retrieval_rounds": retrieval_rounds,
        "initial_calls": initial_calls,
        "bad_after_repair": bad_after_repair,
        "repair_calls": repair_calls,
        "metrics": metrics,
        "quarantine_audit_rows": quarantine_audit_rows,
        "review_exclusion_audit_rows": review_exclusion_audit_rows,
    }


def run_second_attempt_pipeline(
    *,
    dependencies: Any,
    card_json_parser: Any,
    retrieve: Any,
    df_chunks_clean: pd.DataFrame,
    metrics: dict[str, Any],
    paths: Mapping[str, Any],
    policy: Mapping[str, Any],
    previous_cards_for_attempt2: list[dict[str, Any]] | None,
    previous_errors_for_attempt2: list[dict[str, Any]],
    previous_trace_for_attempt2: list[dict[str, Any]],
    quarantine_audit_rows: list[dict[str, Any]],
    retrieval_policy: Mapping[str, Any],
    signature_policy: Mapping[str, Any],
    llm_calls: int,
    retrieval_rounds: int,
) -> dict[str, Any]:
    """Copia EXACTA del bloque ``else:`` (intento 2) de
    ``ExtractionAgent.execute()`` -- ver docstring del módulo."""

    if previous_cards_for_attempt2 is None:
        raise ValueError(
            "El intento 2 no recibió fichas preliminares del intento 1."
        )
    cards = [dict(card) for card in previous_cards_for_attempt2]
    extraction_errors = list(previous_errors_for_attempt2)
    retrieval_trace_rows = list(previous_trace_for_attempt2)

    # Reaplica la preclasificación documental al inicio del intento 2
    # para asegurar que ninguna ficha no elegible bloquee la revisión científica.
    pre_eligibility_result_attempt2 = apply_pre_eligibility_policy(
        cards, created_at=dependencies.now_factory(),
    )
    cards = pre_eligibility_result_attempt2["cards"]
    quarantine_audit_rows.extend(
        pre_eligibility_result_attempt2["quarantine_audit_rows"]
    )

    # Construye el plan de revisión del intento 2, organiza las correcciones
    # por paper y guarda el plan actualizado para continuar con la reparación.
    revision_rows = build_revision_plan(
        cards,
        extraction_errors,
        retrieval_trace_rows,
    )
    revisions = plan_by_source(revision_rows)
    dependencies.save_dataframe(
        pd.DataFrame(revision_rows, columns=REVISION_PLAN_COLUMNS),
        paths["CARDS_REVISION_PLAN_CSV_PATH"],
    )

    # Title-only repairs preserve every scientific field.
    title_sources = {
        source
        for source, row in revisions.items()
        if row["recommended_strategy"] == "REPAIR_TITLE_ONLY"
    }
    title_cards = [
        card for card in cards
        if str(card.get("source_filename", "")) in title_sources
    ]
    if title_cards:
        title_result_attempt2 = dependencies.run_title_repair(
            title_cards,
            df_chunks_clean=df_chunks_clean,
            title_repair_first_chunks=int(
                policy["title_repair_first_chunks"]
            ),
            repair_llm=dependencies.repair_llm,
            json_parser=dependencies.json_parser,
            should_rebuild_extraction=True,
            message_factory=dependencies.message_factory,
            created_at=dependencies.now_factory(),
            extraction_errors=extraction_errors,
        )
        extraction_errors = title_result_attempt2["extraction_errors"]
        title_calls = int(title_result_attempt2["llm_calls"])
        llm_calls += title_calls
    else:
        title_calls = 0

    # Ejecuta las reparaciones dirigidas del intento 2 según el problema de cada paper,
    # amplía la evidencia cuando hace falta y actualiza fichas, trazas, errores y métricas.
    repaired_sources: set[str] = set()
    repair_calls = 0
    by_source = {
        str(card.get("source_filename", "")): index
        for index, card in enumerate(cards)
    }
    for source, row in revisions.items():
        strategy = str(row["recommended_strategy"])
        if strategy == "REPAIR_TITLE_ONLY":
            continue
        max_chunks = int(
            retrieval_policy["max_chunks_per_paper"]
        )
        max_chars = int(retrieval_policy["max_context_chars"])
        if strategy in {
            "REPAIR_SCHEMA_EXPANDED_EVIDENCE",
            "EXPAND_EVIDENCE",
        }:
            max_chunks = int(
                retrieval_policy["repair_max_chunks_per_paper"]
            )
            max_chars = int(
                retrieval_policy["repair_max_context_chars"]
            )
        try:
            repaired_card, _raw, trace_rows = (
                generate_repaired_card_for_source(
                    source,
                    retrieve=retrieve,
                    build_context=build_context_from_chunks,
                    prompt_builder=dependencies.extraction_prompt_builder,
                    experiment_profile=signature_policy[
                        "experiment_profile"
                    ],
                    repair_llm=dependencies.repair_llm,
                    json_parser=card_json_parser,
                    message_factory=dependencies.message_factory,
                    repair_max_chunks_per_paper=max_chunks,
                    repair_max_context_chars=max_chars,
                )
            )
            repair_calls += 1
            retrieval_rounds += 1
            retrieval_trace_rows.extend(trace_rows)
            cards[by_source[source]] = repaired_card
            repaired_sources.add(source)
        except Exception as error:
            repair_calls += 1
            extraction_errors.append({
                "source_filename": source,
                "stage": "directed_attempt_2",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "created_at": dependencies.now_factory(),
            })
    llm_calls += repair_calls
    extraction_errors = [
        row for row in extraction_errors
        if not (
            str(row.get("source_filename", "")) in repaired_sources
            and str(row.get("stage", "")) == "initial_extraction"
        )
    ]
    # Verifica qué fichas siguen siendo inválidas después de la reparación,
    # guarda los resultados finales y actualiza las métricas del intento 2.
    bad_after_repair = [
        str(card.get("source_filename", ""))
        for card in cards
        if is_bad_card(card)
    ]
    dependencies.save_jsonl(cards, paths["CARDS_JSONL_PATH"])
    save_dataframe_even_if_empty(
        extraction_errors,
        paths["CARDS_ERRORS_CSV_PATH"],
        policy["error_columns"],
    )
    save_dataframe_even_if_empty(
        retrieval_trace_rows,
        paths["RETRIEVAL_TRACE_CSV_PATH"],
        policy["trace_columns"],
    )
    metrics["scientific"].update({
        "num_cards": len(cards),
        "num_trace_rows": len(retrieval_trace_rows),
        "num_extraction_errors": len(extraction_errors),
        "num_bad_cards_after_repair": len(bad_after_repair),
        "num_title_llm_calls": title_calls,
        "num_directed_repair_calls": repair_calls,
        "num_revision_plan_rows": len(revision_rows),
    })

    return {
        "cards": cards,
        "extraction_errors": extraction_errors,
        "retrieval_trace_rows": retrieval_trace_rows,
        "llm_calls": llm_calls,
        "retrieval_rounds": retrieval_rounds,
        "bad_after_repair": bad_after_repair,
        "repair_calls": repair_calls,
        "metrics": metrics,
        "quarantine_audit_rows": quarantine_audit_rows,
        "title_calls": title_calls,
    }
