"""
Fases finales de ``ExtractionAgent.execute()`` (Agente 03): manejo de
errores, title-repair general + reclasificación de relevancia, y
exclusión final + elegibilidad del corpus + construcción de la KB.

Cada función es copia EXACTA del bloque correspondiente en
``execute()`` -- ver ``attempt_pipelines.py`` para la misma convención
(parámetros explícitos en vez de ``self``/``self.dependencies``, sin
cambiar ninguna condición, mensaje ni comentario).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping

from src.contracts.agent_input import AgentInput
from src.contracts.agent_result import (
    AgentResult,
    AgentWarning,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
    WarningSeverity,
)
from src.tools.extraction.relevance_classification import (
    classify_card_relevance,
    determine_relevance_reclassification,
)
from src.tools.extraction.review_exclusion import (
    apply_review_exclusion_policy,
    is_review_excluded,
)
from src.tools.extraction.corpus_eligibility import (
    apply_corpus_eligibility_policy,
    is_corpus_quarantined,
)
from src.tools.extraction.stage_artifacts import (
    TRACKED_STAGE_OUTPUT_KEYS,
    any_stage_outputs_exist,
    backup_stage_outputs,
    save_dataframe_even_if_empty,
)

EXPECTED_STAGE_NAME = "03_agente_extraccion_kb"


def handle_execution_error(
    error: Exception,
    *,
    agent_input: AgentInput,
    dependencies: Any,
    resolve_paths_fn: Any,
    collect_artifacts_fn: Any,
    technical_failure_reason_codes_fn: Any,
    sanitize_error_message_fn: Any,
    metrics: dict[str, Any],
    warnings: list[Any],
    llm_calls: int,
    retrieval_rounds: int,
    validation_calls: int,
    started_at: str,
) -> AgentResult:
    """Copia EXACTA del bloque ``except Exception as error:`` de
    ``ExtractionAgent.execute()``."""

    completed_at = (
        dependencies.now_factory()
    )
    try:
        policy = dict(
            agent_input.policy
        )
        paths = resolve_paths_fn(
            policy
        )
        output_artifacts = (
            collect_artifacts_fn(
                paths
            )
        )
    except Exception:
        output_artifacts = {}

    failure_reason_codes = technical_failure_reason_codes_fn(error)
    safe_message = sanitize_error_message_fn(error)
    warnings.append(
        AgentWarning(
            code=failure_reason_codes[0],
            severity=(
                WarningSeverity.ERROR
            ),
            blocking=True,
            message=safe_message,
        )
    )
    metrics["technical"][
        "artifact_count"
    ] = len(output_artifacts)

    # Si la etapa falla, devuelve un resultado de error al orquestador,
    # registra el problema y solicita detener el Agente 03.
    return AgentResult(
        execution_status=(
            ExecutionStatus.FAILED
        ),
        quality_status=(
            QualityStatus.REJECTED
        ),
        decision=DecisionInfo(
            code="EXTRACTION_FAILED",
            rationale=(
                "La etapa 03 no completó "
                "su coordinación."
            ),
        ),
        quality_metrics=metrics,
        warnings=tuple(warnings),
        failure_reason_codes=failure_reason_codes,
        requested_transition=(
            RequestedTransition(
                action=(
                    TransitionAction.HALT_STAGE
                ),
                target_stage=None,
                reason_code=(
                    "EXTRACTION_FAILED"
                ),
                requires_human_confirmation=(
                    False
                ),
            )
        ),
        output_artifacts=(
            output_artifacts
        ),
        tool_usage=ToolUsage(
            retrieval_rounds=(
                retrieval_rounds
            ),
            llm_calls=llm_calls,
            validation_calls=(
                validation_calls
            ),
        ),
        attempt_number=(
            agent_input.attempt_number
        ),
        started_at=started_at,
        completed_at=completed_at,
        error={
            "type": (
                type(error).__name__
            ),
            "message": safe_message,
            "stage": (
                EXPECTED_STAGE_NAME
            ),
        },
    )


def run_title_repair_and_relevance_reclassification(
    *,
    dependencies: Any,
    validate_classification_dependencies_fn: Any,
    agent_input: AgentInput,
    df_chunks_clean: Any,
    metrics: dict[str, Any],
    paths: Mapping[str, Any],
    policy: Mapping[str, Any],
    should_rebuild: bool,
    signature_policy: Mapping[str, Any],
    cards: list[dict[str, Any]],
    extraction_errors: list[dict[str, Any]],
    llm_calls: int,
    backup_dir_created: Any = None,
) -> dict[str, Any]:
    """Copia EXACTA del bloque de title-repair general + reclasificación
    de relevancia de ``ExtractionAgent.execute()``.

    ``backup_dir_created`` viene de fuera (inicializado antes del fork de
    intento 1/2, y posiblemente ya asignado ahí) -- este bloque solo lo
    SOBRESCRIBE si ``reclassify_relevance and not should_rebuild``; si esa
    condición es falsa, el valor de entrada debe conservarse intacto."""

    title_selected = 0
    title_calls = 0
    cards_need_classification = False

    # Revisa y repara nuevamente los títulos de las fichas, actualiza los errores
    # y registra cuántas fichas necesitaron reparación y cuántas llamadas al LLM se usaron.
    title_result = (
        dependencies.run_title_repair(
            cards,
            df_chunks_clean=df_chunks_clean,
            title_repair_first_chunks=int(
                policy[
                    "title_repair_first_chunks"
                ]
            ),
            repair_llm=dependencies.repair_llm,
            json_parser=dependencies.json_parser,
            should_rebuild_extraction=should_rebuild,
            message_factory=(
                dependencies.message_factory
            ),
            created_at=(
                dependencies.now_factory()
            ),
            extraction_errors=(
                extraction_errors
            ),
        )
    )
    cards = title_result["cards"]
    extraction_errors = (
        title_result[
            "extraction_errors"
        ]
    )
    title_selected = len(
        title_result[
            "missing_title_cards"
        ]
    )
    title_calls = int(
        title_result["llm_calls"]
    )
    llm_calls += title_calls
    metrics["scientific"].update({
        "num_cards": int(
            len(cards)
        ),
        "num_extraction_errors": int(
            len(extraction_errors)
        ),
        "num_titles_selected_for_repair": int(
            title_selected
        ),
        "num_title_llm_calls": int(
            title_calls
        ),
    })

    # Guarda las fichas si hubo reparaciones de título y determina
    # si deben volver a clasificarse por relevancia.
    if title_result["repair_titles"]:
        dependencies.save_jsonl(
            cards,
            paths["CARDS_JSONL_PATH"],
        )
        save_dataframe_even_if_empty(
            extraction_errors,
            paths[
                "CARDS_ERRORS_CSV_PATH"
            ],
            policy["error_columns"],
        )

    (
        cards_need_classification,
        reclassify_relevance,
    ) = determine_relevance_reclassification(
        cards,
        should_rebuild_extraction=(
            should_rebuild
        ),
    )
    metrics["scientific"][
        "cards_need_classification"
    ] = bool(
        cards_need_classification
    )

    # Si las fichas necesitan reclasificarse y no se está reconstruyendo toda la etapa,
    # crea un respaldo de los resultados existentes antes de modificarlos.           
    if (
        reclassify_relevance
        and not should_rebuild
    ):
        tracked_outputs = [
            paths[key]
            for key in TRACKED_STAGE_OUTPUT_KEYS
        ]
        if any_stage_outputs_exist(
            tracked_outputs
        ):
            backup_dir_created = backup_stage_outputs(
                outputs_dir=paths["OUTPUTS_DIR"],
                dir_extraction=paths[
                    "DIR_EXTRACTION"
                ],
                dir_kb=paths["DIR_KB"],
                experiment_id=agent_input.experiment_id,
                reason=(
                    "auto_reclassify_missing_fields"
                ),
            )
            metrics["technical"][
                "backup_created"
            ] = True

    # Inicializa los contadores de clasificación y determina si la base de conocimiento
    # debe recrearse; si hay que reclasificar relevancia, valida sus dependencias.
    classification_calls = 0
    kb_should_recreate = bool(
        should_rebuild
    )
    if reclassify_relevance:
        validate_classification_dependencies_fn()

        # Define una función que clasifica la relevancia de una ficha científica
        # usando el perfil del experimento, el prompt configurado y el LLM principal.
        def classify(
            card: dict[str, Any],
        ) -> Any:
            return classify_card_relevance(
                card,
                experiment_profile=signature_policy[
                    "experiment_profile"
                ],
                prompt_builder=(
                    dependencies.relevance_prompt_builder
                ),
                llm=dependencies.main_llm,
                json_parser=(
                    dependencies.json_parser
                ),
                message_factory=(
                    dependencies.message_factory
                ),
            )

        relevance_result = (
            dependencies.run_relevance(
                [c for c in cards if not is_review_excluded(c) and not is_corpus_quarantined(c)],
                should_rebuild_extraction=(
                    should_rebuild
                ),
                classify=classify,
                created_at=(
                    dependencies.now_factory()
                ),
            )
        )
                
        # Conserva intactas las fichas ya excluidas o en cuarentena, integra la
        # clasificación de relevancia del resto y actualiza errores, métricas y la KB.
        classified_by_source = {
            str(c.get("source_filename", "")): c
            for c in relevance_result["cards"]
        }
        cards = [
            card if (is_review_excluded(card) or is_corpus_quarantined(card))
            else classified_by_source.get(str(card.get("source_filename", "")), card)
            for card in cards
        ]
        extraction_errors.extend(
            relevance_result["errors"]
        )
        classification_calls = int(
            relevance_result[
                "classification_calls"
            ]
        )
        llm_calls += classification_calls
        metrics["scientific"].update({
            "num_cards": int(
                len(cards)
            ),
            "num_extraction_errors": int(
                len(extraction_errors)
            ),
            "num_classification_calls": int(
                classification_calls
            ),
        })
        kb_should_recreate = bool(
            relevance_result[
                "kb_should_recreate"
            ]
            or should_rebuild
        )
        dependencies.save_jsonl(
            cards,
            paths["CARDS_JSONL_PATH"],
        )
        save_dataframe_even_if_empty(
            extraction_errors,
            paths[
                "CARDS_ERRORS_CSV_PATH"
            ],
            policy["error_columns"],
        )


    return {
        "cards": cards,
        "extraction_errors": extraction_errors,
        "llm_calls": llm_calls,
        "metrics": metrics,
        "backup_dir_created": backup_dir_created,
        "kb_should_recreate": kb_should_recreate,
        "title_selected": title_selected,
        "title_calls": title_calls,
        "classification_calls": classification_calls,
        "cards_need_classification": cards_need_classification,
    }


def run_final_eligibility_and_kb(
    *,
    dependencies: Any,
    paths: Mapping[str, Any],
    signature_policy: Mapping[str, Any],
    cards: list[dict[str, Any]],
    kb_should_recreate: bool,
    quarantine_audit_rows: list[dict[str, Any]],
    review_exclusion_audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Copia EXACTA del bloque de exclusión final + elegibilidad del
    corpus + construcción/reuso de la KB de ``ExtractionAgent.execute()``."""

    # Aplica por última vez la exclusión determinista de papers de revisión,
    # guarda las fichas finales y marca que la KB debe recrearse si hubo exclusiones.
    exclusion_policy_final = dict(
        signature_policy.get("extraction_policy", {})
    )
    exclusion_result_final = apply_review_exclusion_policy(
        cards,
        exclude_reviews=bool(exclusion_policy_final.get("exclude_reviews", True)),
        created_at=dependencies.now_factory(),
    )
    cards = exclusion_result_final["cards"]
    review_exclusion_audit_rows.extend(
        exclusion_result_final["audit_rows"]
    )
    if exclusion_result_final["num_excluded"]:
        kb_should_recreate = True
        dependencies.save_jsonl(
            cards,
            paths["CARDS_JSONL_PATH"],
        )

    # Aplica la clasificación final de elegibilidad del corpus y asigna a cada
    # ficha un estado canónico: INCLUDE, EXCLUDE o QUARANTINE.
    eligibility_result = apply_corpus_eligibility_policy(
        cards, created_at=dependencies.now_factory(),
    )
    cards = eligibility_result["cards"]
    corpus_eligibility_counts = eligibility_result["counts"]
    quarantine_audit_rows.extend(
        eligibility_result["quarantine_audit_rows"]
    )
    if eligibility_result["quarantine_audit_rows"]:
        kb_should_recreate = True
                
    # Guarda siempre las fichas con su estado final de elegibilidad
    # y reutiliza la KB existente solo si está completa y no necesita reconstruirse.
    dependencies.save_jsonl(
        cards,
        paths["CARDS_JSONL_PATH"],
    )
    kb_csv_exists = Path(
        paths["KB_CSV_PATH"]
    ).exists()
    kb_jsonl_exists = Path(
        paths["KB_JSONL_PATH"]
    ).exists()
    existing_kb_dataframe = None
    if (
        kb_csv_exists
        and kb_jsonl_exists
        and not kb_should_recreate
    ):
        existing_kb_dataframe = (
            dependencies.load_dataframe(
                paths["KB_CSV_PATH"]
            )
        )
                
    # Ejecuta la construcción o reutilización de la Knowledge Base y,
    # si fue creada nuevamente, guarda sus versiones CSV y JSONL.
    (
        kb_status,
        df_kb,
        kb_rows,
    ) = dependencies.execute_kb(
        cards,
        kb_csv_exists=kb_csv_exists,
        kb_jsonl_exists=kb_jsonl_exists,
        kb_should_recreate=(
            kb_should_recreate
        ),
        existing_csv_dataframe=(
            existing_kb_dataframe
        ),
    )
    if kb_status == "created":
        dependencies.save_dataframe(
            df_kb,
            paths["KB_CSV_PATH"],
        )
        dependencies.save_jsonl(
            kb_rows or [],
            paths["KB_JSONL_PATH"],
        )


    return {
        "cards": cards,
        "kb_should_recreate": kb_should_recreate,
        "corpus_eligibility_counts": corpus_eligibility_counts,
        "quarantine_audit_rows": quarantine_audit_rows,
        "review_exclusion_audit_rows": review_exclusion_audit_rows,
        "kb_status": kb_status,
        "df_kb": df_kb,
        "kb_rows": kb_rows,
    }
