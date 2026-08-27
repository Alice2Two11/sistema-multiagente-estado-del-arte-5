# ============================================================
# 06 - AGENTE REDACTOR DEL ESTADO DEL ARTE
# Recupera evidencia, redacta las secciones y valida el borrador
# antes de enviarlo al agente verificador.
# ============================================================

#canonical_sentences_v2 organiza el borrador a nivel de oraciones 
#para que luego sea más fácil verificar, rastrear y corregir afirmaciones 
#concretas.

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.contracts.agent_input import ArtifactReference
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
from src.state.fingerprints import sha256_file
from src.tools.draft_writing.artifacts import (
    NAMES,
    write_draft_artifacts,
    write_partial_validation,
)
from src.tools.draft_writing.input_validation import validate_draft_dependencies
from src.tools.draft_writing.prompting import (
    assign_section_budgets,
    build_source_free_organizational_section,
)
from src.tools.draft_writing.retrieval import (
    build_section_query,
    retrieve_section_evidence,
)
from src.tools.draft_writing.validation import (
    build_draft_reports,
    validate_draft_global,
    section_allows_no_sources,
)
from src.tools.draft_writing.length_repair import attempt_directed_length_repair


# Define la estrategia de recuperación (cómo busca el agente 06 la evidencia que va a usar para redactar cada sección)
# y el formato canónico con el que se representa el borrador (cómo se representa internamente el borrador para que todas las partes del sistema lo entiendan)
LEGACY_RETRIEVAL_STRATEGY = "legacy_chroma_then_csv_restricted"
CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT = "canonical_sentences_v2"

# Define códigos de validación para controlar la longitud del borrador
# y evitar completar el mínimo con contenido sin respaldo suficiente.
TOTAL_WORD_COUNT_BELOW_MINIMUM = "TOTAL_WORD_COUNT_BELOW_MINIMUM"
TOTAL_WORD_COUNT_ABOVE_MAXIMUM = "TOTAL_WORD_COUNT_ABOVE_MAXIMUM"
INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH = "INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH"

# Registra las versiones del comportamiento histórico del agente 06
# para mantener trazabilidad y reproducibilidad de las corridas.
LEGACY_VERSIONS = {
    "stage_version": "06_AGENTIC_V16_BEHAVIOR_PRESERVING",
    "rag_version": "legacy_chroma_then_csv_restricted_v1",
    "validation_version": "legacy_notebook06_validation_v2_hard_word_range_configured_min_gate_soft_failure_length_repair",
    "normalization_version": "sentence_claim_exact_match_preserve_unmatched_v1_immediate_numeric_salvage_v2_discourse_connector_feedback_v3",
}

class DraftWritingAgent:
    def __init__(self, runtime):
        self.runtime = runtime

    # limpia la lista de papers de una sección y devuelve
    # las fuentes válidas y únicas que el agente 06 puede usar.
    @staticmethod
    def _section_sources(section: Mapping[str, Any]) -> list[str]:
        sources: list[str] = []
        for paper in section.get("papers_to_use") or []:
            if not isinstance(paper, Mapping):
                continue
            source = str(paper.get("source_filename", "")).strip()
            if source and source not in sources:
                sources.append(source)
        return sources

    # Obtiene todos los pares válidos paper + chunk disponibles en la memoria documental.
    @staticmethod
    def _valid_source_chunk_pairs(chunks: pd.DataFrame) -> set[tuple[str, str]]:
        if chunks.empty or not {"source_filename", "chunk_id"}.issubset(chunks.columns):
            return set()
        return {
            (str(row["source_filename"]).strip(), str(row["chunk_id"]).strip())
            for _, row in chunks.iterrows()
            if str(row["source_filename"]).strip() and str(row["chunk_id"]).strip()
        }

    # prepara los datos numéricos y de datasets que el redactor puede usar en una sección, 
    # pero únicamente de los papers asignados a esa sección y sin traer demasiados registros.
    def _quant_context(
        self,
        section: Mapping[str, Any],
        bundle: Mapping[str, Any],
        limit: int,
    ) -> dict[str, list[dict[str, Any]]]:
        sources = set(self._section_sources(section))
        quantitative = bundle["quantitative"]
        dataset_summary = bundle["dataset_summary"]
        quantitative_rows = (
            quantitative[
                quantitative["source_filename"].astype(str).isin(sources)
            ].head(limit).to_dict("records")
            if not quantitative.empty and "source_filename" in quantitative.columns
            else []
        )
        dataset_rows = (
            dataset_summary[
                dataset_summary["source_filename"].astype(str).isin(sources)
            ].head(limit).to_dict("records")
            if not dataset_summary.empty and "source_filename" in dataset_summary.columns
            else []
        )
        return {
            "quantitative_results": quantitative_rows,
            "dataset_technique_summary": dataset_rows,
        }

    # Lee qué estrategia de búsqueda de evidencia está configurada
    # y comprueba que sea la única estrategia permitida por el agente 06.
    @staticmethod
    def _strategy(policy: Mapping[str, Any]) -> str:
        strategy = policy.get("retrieval_strategy")
        if strategy != LEGACY_RETRIEVAL_STRATEGY:
            raise ValueError(
                f"UNSUPPORTED_DRAFT_RETRIEVAL_STRATEGY:{strategy!r}"
            )
        return strategy

    # Devuelve las versiones de recuperación, validación,
    # normalización y comportamiento general usadas por el agente 06.
    @staticmethod
    def _effective_versions(
        policy: Mapping[str, Any], strategy: str
    ) -> dict[str, str]:
        del policy, strategy
        return dict(LEGACY_VERSIONS)

    # Calcula cuántas palabras debe tener cada sección
    # según el total de palabras definido para el borrador.
    @staticmethod
    def _section_budgets(
        sections: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any],
        strategy: str,
    ) -> dict[str, dict[str, Any]]:
        del strategy
        target_total_words = int(policy["target_total_words"])
        return assign_section_budgets(sections, target_total_words)

    # Prepara la búsqueda de evidencia para una sección
    # usando los fragmentos disponibles de los papers.
    def _retrieve_section_evidence(
        self,
        section: Mapping[str, Any], #la sección que se va a redactar
        bundle: Mapping[str, Any], #contiene los datos disponibles del proyecto
        policy: Mapping[str, Any], #trae las reglas de recuperación configuradas
        strategy: str, #indica qué estrategia de búsqueda se debe usar
        quantitative_context: Mapping[str, Any], #parte de la información cuantitativa del 03B que corresponde específicamente a la sección que el 06 está redactando
    ) -> list[dict[str, Any]]:
        chunks = bundle["chunks"]

        # Define los límites de evidencia que se recuperarán para la sección
        # y ejecuta la búsqueda sobre Chroma usando los chunks disponibles.
        del strategy, quantitative_context
        top_k = int(policy["top_k_evidence_per_section"])
        max_chars = int(policy["max_evidence_chars"])
        return retrieve_section_evidence(
            section,
            self.runtime.collection,
            chunks,
            top_k,
            max_chars,
        )

    # devuelve el resultado oficial del agente indicando que esa sección falló la validación, 
    # junto con toda la información necesaria para saber qué pasó y por qué.
    @staticmethod
    def _build_v2_section_validation_failed_result(
        *,
        agent_input,
        sid: str,
        out: Path,
        raw_dir: Path,
        attempt_logs: dict[str, list[dict[str, Any]]],
        retrieval_rounds: int,
        llm_calls: int,
        validation_calls: int,
        last_errors: list[str],
        validation_version: Any,
        start: str,
    ) -> AgentResult:
        

        # Guarda el reporte del fallo, registra los artefactos generados
        # y decide si Stage06 debe reintentarse o detenerse.
        section_attempts = len(attempt_logs.get(sid) or [])
        partial_validation = {
            "stage": "06_agente_redactor",
            "experiment_id": agent_input.experiment_id,
            "validation_version": validation_version,
            "validation_ok": False,
            "failed_section": sid,
            "section_attempts": section_attempts,
            "contract": "canonical_sentences_v2",
            "last_attempt_errors": list(last_errors),
            "generation_attempts": attempt_logs,
            "current_raw_attempt_directory": str(raw_dir),
            "published_draft": False,
        }
        report_path = write_partial_validation(out, partial_validation)
        artifacts = {
            "draft_validation_report.json": ArtifactReference(
                str(report_path), sha256_file(report_path)
            ),
            "raw_section_outputs": ArtifactReference(
                str(out / "raw_section_outputs"), "DIRECTORY"
            ),
        }
        action = (
            TransitionAction.RETRY
            if agent_input.is_first_attempt()
            else TransitionAction.HALT_STAGE
        )
        return AgentResult(
            execution_status=ExecutionStatus.COMPLETED,
            quality_status=QualityStatus.NEEDS_REVISION,
            decision=DecisionInfo(
                code="SECTION_VALIDATION_FAILED",
                rationale=(
                    f"La sección {sid} (canonical_sentences_v2) agotó sus "
                    "reintentos; se preservaron salidas y validaciones por "
                    "intento."
                ),
            ),
            quality_metrics={
                "scientific": {},
                "technical": {
                    "validation_ok": False,
                    "reused": False,
                    "failed_section": sid,
                    "section_attempts": section_attempts,
                    "contract": "canonical_sentences_v2",
                },
            },
            warnings=(
                AgentWarning(
                    code="SECTION_VALIDATION_FAILED",
                    severity=WarningSeverity.ERROR,
                    blocking=True,
                    message=(
                        f"La sección {sid} (canonical_sentences_v2) no "
                        "superó la validación tras agotar sus intentos."
                    ),
                ),
            ),
            failure_reason_codes=("SECTION_VALIDATION_FAILED",),
            requested_transition=RequestedTransition(
                action=action,
                target_stage=None,
                reason_code="NEEDS_REVISION",
                requires_human_confirmation=False,
            ),
            output_artifacts=artifacts,
            tool_usage=ToolUsage(
                retrieval_rounds=retrieval_rounds,
                llm_calls=llm_calls,
                validation_calls=validation_calls,
            ),
            attempt_number=agent_input.attempt_number,
            started_at=start,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    # Inicia la ejecución del agente 06, crea los contadores
    # y prepara la carpeta donde se guardarán los intentos de redacción.
    def execute(self, agent_input):
        start = datetime.now(timezone.utc).isoformat()
        llm_calls = 0
        retrieval_rounds = 0
        validation_calls = 0
        out = Path(agent_input.agent_context.output_directory)
        raw_dir = out / "raw_section_outputs" / f"agent_attempt_{agent_input.attempt_number:02d}"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Valida que estén disponibles las entradas necesarias, carga la configuración
        # del agente 06 y prepara los archivos requeridos para reutilizar una corrida previa.
        try:
            bundle = validate_draft_dependencies(agent_input)
            policy = dict(agent_input.policy)
            strategy = self._strategy(policy)
            versions = self._effective_versions(policy, strategy)
            policy.update(versions)
            manifest_path = out / "draft_generation_manifest.json"
            reuse = False
            required_reuse = (
                "state_of_art_draft.json",
                "state_of_art_draft.md",
                "draft_sections.csv",
                "draft_rag_evidence.csv",
                "draft_quality_check.csv",
                "draft_length_check.csv",
                "draft_claim_evidence.csv",
                "numeric_hallucination_check.csv",
                "draft_validation_report.json",
                "draft_generation_manifest.json",
            )
            # Comprueba si existe una corrida anterior válida y completa
            # para reutilizarla sin volver a generar el borrador.
            if manifest_path.exists() and not policy["force_rebuild"]:
                try:
                    old = json.loads(manifest_path.read_text())
                    report = json.loads((out / "draft_validation_report.json").read_text())
                    reuse = (
                        old.get("fingerprint") == policy.get("current_fingerprint")
                        and report.get("validation_ok") is True
                        and all((out / name).exists() for name in required_reuse)
                    )
                except Exception:
                    reuse = False

            # Si existe una corrida anterior válida, reutiliza sus archivos
            # y avanza directamente al agente verificador sin volver a generar el borrador.
            if reuse:
                artifacts = {
                    name: ArtifactReference(str(out / name), sha256_file(out / name))
                    for name in NAMES
                    if (out / name).exists()
                }
                artifacts["raw_section_outputs"] = ArtifactReference(
                    str(out / "raw_section_outputs"), "DIRECTORY"
                )
                return AgentResult(
                    execution_status=ExecutionStatus.COMPLETED,
                    quality_status=QualityStatus.APPROVED,
                    decision=DecisionInfo(
                        code="DRAFT_REUSED",
                        rationale="Borrador válido reutilizado con fingerprint vigente.",
                    ),
                    quality_metrics={
                        "scientific": {},
                        "technical": {"validation_ok": True, "reused": True},
                    },
                    warnings=(),
                    failure_reason_codes=(),
                    requested_transition=RequestedTransition(
                    action=TransitionAction.ADVANCE,
                    target_stage="07_agente_verificador",
                    reason_code="APPROVED",
                    requires_human_confirmation=False,
                ),
                    output_artifacts=artifacts,
                    tool_usage=ToolUsage(
                        retrieval_rounds=0, llm_calls=0, validation_calls=0
                    ),
                    attempt_number=agent_input.attempt_number,
                    started_at=start,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

            # Obtiene las secciones del esquema, verifica que sean válidas
            # y calcula cuántas palabras le corresponden a cada una.
            sections = bundle["outline"].get("sections") or []
            if not isinstance(sections, list) or not sections:
                raise ValueError("INVALID_OUTLINE_SCHEMA")
            policy["outline_sections"] = sections
            policy["section_budgets"] = self._section_budgets(
                sections, policy, strategy
            )

            # Comprueba que el borrador use el formato canónico esperado.
            # Si viene otro formato o falta la configuración, detiene la ejecución.
            contract = policy.get("draft_representation_contract")
            if contract != CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
                raise ValueError(
                    f"UNKNOWN_DRAFT_REPRESENTATION_CONTRACT:{contract!r}"
                )

            generated: list[dict[str, Any]] = []
            all_evidence: list[dict[str, Any]] = []
            attempt_logs: dict[str, list[dict[str, Any]]] = {}

            # Recorre cada sección, busca la evidencia que le corresponde
            # y la guarda para usarla después al redactar.
            for section in sections:
                sid = str(section.get("section_id", "")).strip()
                section_query = build_section_query(section)
                quant_context = self._quant_context(
                    section,
                    bundle,
                    int(policy["max_quantitative_rows_per_section"]),
                )
                evidence = self._retrieve_section_evidence(
                    section,
                    bundle,
                    policy,
                    strategy,
                    quant_context,
                )
                if section.get("papers_to_use"):
                    retrieval_rounds += 1
                all_evidence.extend({"section_id": sid, **row} for row in evidence)

                # Si no hay evidencia, revisa mediante section_allows_no_sources() -> introducción, conclusiones
                # si esa sección está autorizada para redactarse sin fuentes.
                if not evidence:
                    if not section_allows_no_sources(section):
                        raise ValueError(f"MISSING_SECTION_EVIDENCE:{sid}")
                    generated_section = build_source_free_organizational_section(
                        section, policy.get("output_language", "español")
                    )
                    attempt_logs[sid] = [
                        {
                            "attempt": 0,
                            "mode": "deterministic_source_free_organizational_section",
                            "validation": generated_section["section_validation"],
                        }
                    ]
                    generated.append(generated_section)
                    continue

                if contract == CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
                    from src.tools.draft_writing.canonical_sentences import (
                        generate_section_canonical_v2,
                    )

                    # Genera la sección usando el formato canónico V2,
                    # la evidencia recuperada y el contexto cuantitativo disponible.
                    generated_section = generate_section_canonical_v2(
                        section=section,
                        evidence=evidence,
                        quant_context=quant_context,
                        previous_errors=[],
                        policy=policy,
                        runtime=self.runtime,
                        raw_dir=raw_dir,
                        sid=sid,
                        runtime_invoke_sequence_base=llm_calls,
                    )
                   
                    v2_execution = generated_section.pop("_v2_execution")
                    llm_calls += v2_execution["llm_calls"]
                    validation_calls += v2_execution["validation_calls"]
                    attempt_logs[sid] = v2_execution["attempt_logs"]

                    # Si la generación V2 de la sección falló,
                    # construye y devuelve el resultado formal de error de Stage06.
                    if v2_execution["failed"]:
                        return self._build_v2_section_validation_failed_result(
                            agent_input=agent_input,
                            sid=sid,
                            out=out,
                            raw_dir=raw_dir,
                            attempt_logs=attempt_logs,
                            retrieval_rounds=retrieval_rounds,
                            llm_calls=llm_calls,
                            validation_calls=validation_calls,
                            last_errors=v2_execution["last_errors"],
                            validation_version=policy.get("validation_version"),
                            start=start,
                        )

                    generated.append(generated_section)
                    continue

            # Organiza la evidencia por sección y valida el borrador completo.
            evidence_map: dict[str, list[dict[str, Any]]] = {}
            for row in all_evidence:
                evidence_map.setdefault(row["section_id"], []).append(
                    {key: value for key, value in row.items() if key != "section_id"}
                )
            _, quality_rows, section_rows, claim_rows, numeric_rows = (
                build_draft_reports(generated, sections, evidence_map, policy) #genera reportes sobre calidad, secciones, claims y datos numéricos
            )
            validation = validate_draft_global( #revisa si el borrador completo cumple las reglas de 
                generated, sections, evidence_map, policy
            )
            validation.update( # revisa si el borrador cumple las reglas de longitud, evidencia, citas y estructura.
                {
                    "stage": "06_agente_redactor",
                    "experiment_id": agent_input.experiment_id,
                    "validation_version": policy.get("validation_version"),
                    "generation_attempts": attempt_logs,
                    "current_raw_attempt_directory": str(raw_dir),
                }
            )
            validation_calls += 1

            # Comprueba si el único problema del borrador es que no cumple
            # con la longitud requerida.
            length_repair_attempted = False
            length_repair_successful = False
            length_only_failure = ( #¿todo está bien excepto la cantidad de palabras?
                not validation["validation_ok"]
                and validation.get("word_count_compliant", True) is False
                and validation.get("all_section_validations_ok", False)
                and validation.get("invalid_citation_count", 1) == 0
                and not validation.get("sections_without_valid_citations", True)
                and not validation.get("sections_with_low_citation_density", True)
                and not validation.get("sections_with_claim_support_errors", True)
                and not validation.get("sections_with_quantitative_support_errors", True)
                and validation.get("numeric_failure_count", 1) == 0
            )
            # Si el único problema es la longitud y el borrador usa el formato canónico,
            # intenta ajustar el texto y vuelve a validarlo.
            if length_only_failure and contract == CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
                length_repair_attempted = True
                repaired_generated, repair_meta = attempt_directed_length_repair( #intenta ajustar la longitud con attempt_directed_length_repair(...)
                    generated, sections, evidence_map, policy, self.runtime,
                )
                if repair_meta["attempted"]:
                    _, quality_rows, section_rows, claim_rows, numeric_rows = (
                        build_draft_reports(repaired_generated, sections, evidence_map, policy)
                    )
                    repaired_validation = validate_draft_global( #Si realmente hizo una reparación, vuelve a generar los reportes y vuelve a validar todo el borrador.
                        repaired_generated, sections, evidence_map, policy
                    )
                    if repaired_validation["validation_ok"]:
                        generated = repaired_generated
                        validation = repaired_validation
                        length_repair_successful = True
                    else:
                        validation = repaired_validation
            validation["length_repair_attempted"] = length_repair_attempted
            validation["length_repair_successful"] = length_repair_successful

            # Si el borrador no pasa la validación, guarda el reporte parcial
            # y prepara los artefactos necesarios para registrar el fallo.
            if not validation["validation_ok"]:
                path = write_partial_validation(out, validation)
                artifacts = {
                    "draft_validation_report.json": ArtifactReference(
                        str(path), sha256_file(path)
                    ),
                    "raw_section_outputs": ArtifactReference(
                        str(out / "raw_section_outputs"), "DIRECTORY"
                    ),
                }
                is_final_attempt = agent_input.attempt_number != 1

                # Si el único problema del borrador es la longitud,
                # determina si quedó demasiado corto o demasiado largo.
                if (
                    validation.get("word_count_compliant", True) is False
                    and validation.get("all_section_validations_ok", False)
                    and validation.get("invalid_citation_count", 1) == 0
                    and not validation.get("sections_without_valid_citations", True)
                    and not validation.get("sections_with_low_citation_density", True)
                    and not validation.get("sections_with_claim_support_errors", True)
                    and not validation.get("sections_with_quantitative_support_errors", True)
                    and validation.get("numeric_failure_count", 1) == 0
                ):
                    if validation.get("word_deficit", 0) > 0:
                        length_reason_code = (
                            INSUFFICIENT_SUPPORTED_CONTENT_FOR_MIN_LENGTH
                            if length_repair_attempted
                            else TOTAL_WORD_COUNT_BELOW_MINIMUM
                        )
                    else:
                        length_reason_code = TOTAL_WORD_COUNT_ABOVE_MAXIMUM

                    # Si ya es el último intento y el único problema sigue siendo la longitud,
                    # detiene Stage06 y manda el borrador a revisión manual.
                    if is_final_attempt:
                        return AgentResult(
                            execution_status=ExecutionStatus.COMPLETED,
                            quality_status=QualityStatus.APPROVED_PENDING_MANUAL_REVIEW,
                            decision=DecisionInfo(
                                code="DRAFT_LENGTH_OUT_OF_RANGE_MANUAL_REVIEW",
                                rationale=(
                                    "El borrador quedó fuera del rango de longitud contractual "
                                    "tras agotar los intentos; requiere revisión manual antes de "
                                    "continuar. No se publican salidas finales desde Agent06."
                                ),
                            ),
                            quality_metrics={ #guarda cuántas palabras debía tener, cuántas tiene y si se intentó reparar.
                                "scientific": {
                                    "configured_min_total_words": validation["configured_min_total_words"],
                                    "configured_max_total_words": validation["configured_max_total_words"],
                                    "target_total_words": validation["target_total_words"],
                                    "actual_total_words": validation["actual_total_words"],
                                    "effective_min_total_words": validation["effective_min_total_words"],
                                    "word_deficit": validation.get("word_deficit"),
                                    "word_excess": validation["word_excess"],
                                    "word_count_compliant": validation.get("word_count_compliant"),
                                },
                                "technical": {
                                    "validation_ok": False, "reused": False,
                                    "length_repair_attempted": length_repair_attempted,
                                    "length_repair_successful": length_repair_successful,
                                },
                            },
                            warnings=( #genera una advertencia bloqueante con el motivo exacto.
                                AgentWarning(
                                    code=length_reason_code,
                                    severity=WarningSeverity.WARNING,
                                    blocking=True,
                                    message=(
                                        f"Longitud fuera de rango tras agotar intentos: "
                                        f"actual_total_words={validation['actual_total_words']}, "
                                        f"configured_min_total_words={validation['configured_min_total_words']}, "
                                        f"configured_max_total_words={validation['configured_max_total_words']}."
                                    ),
                                ),
                            ),
                            failure_reason_codes=(length_reason_code,),
                            requested_transition=RequestedTransition(
                                action=TransitionAction.HALT_STAGE,
                                target_stage=None,
                                reason_code=length_reason_code,
                                requires_human_confirmation=True,
                            ),
                            output_artifacts=artifacts,
                            tool_usage=ToolUsage(
                                retrieval_rounds=retrieval_rounds,
                                llm_calls=llm_calls,
                                validation_calls=validation_calls,
                            ),
                            attempt_number=agent_input.attempt_number,
                            started_at=start,
                            completed_at=datetime.now(timezone.utc).isoformat(),
                        )

                    # Si la validación global falla pero todavía se puede reintentar,
                    # devuelve el resultado indicando que el borrador necesita revisión.
                    return AgentResult(
                        execution_status=ExecutionStatus.COMPLETED,
                        quality_status=QualityStatus.NEEDS_REVISION,
                        decision=DecisionInfo(
                            code="DRAFT_VALIDATION_FAILED",
                            rationale=(
                                "El borrador no superó la validación global; "
                                "no se publicaron salidas finales."
                            ),
                        ),
                        quality_metrics={
                            "scientific": {
                                "configured_min_total_words": validation["configured_min_total_words"],
                                "configured_max_total_words": validation["configured_max_total_words"],
                                "target_total_words": validation["target_total_words"],
                                "actual_total_words": validation["actual_total_words"],
                                "effective_min_total_words": validation["effective_min_total_words"],
                                "word_deficit": validation.get("word_deficit"),
                                "word_excess": validation["word_excess"],
                                "word_count_compliant": validation.get("word_count_compliant"),
                            },
                            "technical": {
                                "validation_ok": False, "reused": False,
                                "length_repair_attempted": length_repair_attempted,
                                "length_repair_successful": length_repair_successful,
                            },
                        },
                        warnings=(
                            AgentWarning(
                                code=length_reason_code,
                                severity=WarningSeverity.ERROR,
                                blocking=True,
                                message="La validación global de longitud fue negativa.",
                            ),
                        ),
                        failure_reason_codes=(length_reason_code,),
                        requested_transition=RequestedTransition(
                            action=TransitionAction.RETRY,
                            target_stage=None,
                            reason_code=length_reason_code,
                            requires_human_confirmation=False,
                        ),
                        output_artifacts=artifacts,
                        tool_usage=ToolUsage(
                            retrieval_rounds=retrieval_rounds,
                            llm_calls=llm_calls,
                            validation_calls=validation_calls,
                        ),
                        attempt_number=agent_input.attempt_number,
                        started_at=start,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )

                # Decide si Stage06 debe reintentarse o detenerse según
                # si todavía queda otro intento disponible.
                action = (
                    TransitionAction.RETRY
                    if agent_input.is_first_attempt()
                    else TransitionAction.HALT_STAGE
                )
                return AgentResult(
                    execution_status=ExecutionStatus.COMPLETED,
                    quality_status=QualityStatus.NEEDS_REVISION,
                    decision=DecisionInfo(
                        code="DRAFT_VALIDATION_FAILED",
                        rationale=(
                            "El borrador no superó la validación global; "
                            "no se publicaron salidas finales."
                        ),
                    ),
                    quality_metrics={
                        "scientific": {},
                        "technical": {"validation_ok": False, "reused": False},
                    },
                    warnings=(
                        AgentWarning(
                            code="INVALID_DRAFT",
                            severity=WarningSeverity.ERROR,
                            blocking=True,
                            message="La validación global fue negativa.",
                        ),
                    ),
                    failure_reason_codes=("INVALID_DRAFT",),
                    requested_transition=RequestedTransition(
                        action=action,
                        target_stage=None,
                        reason_code="NEEDS_REVISION",
                        requires_human_confirmation=False,
                    ),
                    output_artifacts=artifacts,
                    tool_usage=ToolUsage(
                        retrieval_rounds=retrieval_rounds,
                        llm_calls=llm_calls,
                        validation_calls=validation_calls,
                    ),
                    attempt_number=agent_input.attempt_number,
                    started_at=start,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

            # Construye el borrador final de Stage06 con sus secciones
            # y guarda un resumen de cómo fue generado.
            draft = {
                "title": bundle["outline"].get(
                    "title", "Borrador del estado del arte"
                ),
                "topic": bundle["outline"].get("topic", ""),
                "status": "draft_validated_for_verification",
                "sections": generated,
                "generation_summary": {
                    "experiment_id": agent_input.experiment_id,
                    "section_count": len(generated),
                    "ground_truth_used": False,
                    "open_search_used": False,
                    "citation_format": "[source_filename | chunk_id]",
                    "retrieval_strategy": strategy,
                    **versions,
                },
            }
            manifest_versions = { # Guarda las versiones de los componentes usados para generar el borrador.
                "stage": versions["stage_version"],
                "prompt": policy.get("prompt_version"),
                "rag": versions["rag_version"],
                "validation": versions["validation_version"],
                "normalization": versions["normalization_version"],
            }
            manifest = { # Guarda los datos principales de la ejecución y las reglas de seguridad usadas.
                "stage": agent_input.stage_name,
                "experiment_id": agent_input.experiment_id,
                "run_id": agent_input.run_id,
                "attempt_number": agent_input.attempt_number,
                "fingerprint": policy.get("current_fingerprint"),
                "retrieval_strategy": strategy,
                "validation_ok": True,
                "safety_policy": {
                    "uses_ground_truth": False,
                    "uses_external_knowledge": False,
                    "open_search_used": False,
                },
                "counts": {
                    "sections": len(generated),
                    "llm_calls": llm_calls,
                    "retrieval_rounds": retrieval_rounds,
                },
                "versions": manifest_versions,
                "current_raw_attempt_directory": str(raw_dir),
            }
            # Guarda en el manifest el formato del borrador cuando usa el contrato canónico
            # y luego escribe todos los archivos finales generados por Stage06.
            if contract == CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
                manifest["draft_representation_contract"] = contract
            artifacts = write_draft_artifacts(
                out,
                draft,
                all_evidence,
                validation,
                bundle["quantitative"],
                bundle["dataset_summary"],
                manifest,
                quality_rows,
                section_rows,
                claim_rows,
                numeric_rows,
            )
            # Devuelve el resultado final cuando el borrador pasó la validación
            # y autoriza avanzar al agente verificador.
            return AgentResult(
                execution_status=ExecutionStatus.COMPLETED,
                quality_status=QualityStatus.APPROVED,
                decision=DecisionInfo(
                    code="DRAFT_APPROVED",
                    rationale=(
                        "Borrador generado por secciones y validado con "
                        "evidencia restringida."
                    ),
                ),
                quality_metrics={ #guarda las métricas de longitud y si hubo reparación
                    "scientific": { 
                        "configured_min_total_words": validation.get("configured_min_total_words"),
                        "configured_max_total_words": validation.get("configured_max_total_words"),
                        "target_total_words": validation.get("target_total_words"),
                        "actual_total_words": validation.get("actual_total_words"),
                        "effective_min_total_words": validation.get("effective_min_total_words"),
                        "word_deficit": validation.get("word_deficit"),
                        "word_excess": validation.get("word_excess"),
                        "word_count_compliant": validation.get("word_count_compliant"),
                    },
                    "technical": {
                        "validation_ok": True, "reused": False,
                        "length_repair_attempted": validation.get("length_repair_attempted", False),
                        "length_repair_successful": validation.get("length_repair_successful", False),
                    },
                },
                warnings=(),
                failure_reason_codes=(),
                requested_transition=RequestedTransition(
                    action=TransitionAction.ADVANCE,
                    target_stage="07_agente_verificador", #el siguiente paso es Stage07
                    reason_code="APPROVED",
                    requires_human_confirmation=False,
                ),
                output_artifacts=artifacts,
                tool_usage=ToolUsage(
                    retrieval_rounds=retrieval_rounds,
                    llm_calls=llm_calls,
                    validation_calls=validation_calls,
                ),
                attempt_number=agent_input.attempt_number,
                started_at=start,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        # Si ocurre un error, identifica si corresponde a un problema conocido
        # y asigna un código específico; si no, usa un error general.
        except Exception as exc:
            message = str(exc)
            known = (
                "DRAFT_INPUT_NOT_FOUND",
                "OUTLINE_NOT_APPROVED",
                "OUTLINE_MANIFEST_MISMATCH",
                "GROUND_TRUTH_POLICY_VIOLATION",
                "INVALID_DRAFT_KB_SCHEMA",
                "INVALID_CHUNKS_SCHEMA",
                "INVALID_QUANTITATIVE_CONTEXT",
                "THEMATIC_NOT_APPROVED",
                "OUTLINE_MANIFEST_NOT_APPROVED",
                "THEMATIC_MANIFEST_NOT_APPROVED",
                "OUTLINE_SOURCES_NOT_VALIDATED",
                "OUTLINE_TITLES_NOT_VALIDATED",
                "CHROMA_COLLECTION_MISMATCH",
                "CHROMA_EMBEDDING_MODEL_MISMATCH",
                "UNSAFE_CHROMA_INDEX",
                "DUPLICATE_KB_SOURCE",
                "DUPLICATE_CHUNK_ID",
                "UNSAFE_CHUNKS",
                "CHROMA_CHUNK_COUNT_MISMATCH",
                "INVALID_OUTLINE_SECTION_IDS",
                "INVALID_OUTLINE_MAPPING_SCHEMA",
                "OUTLINE_MAPPING_INCONSISTENT",
                "QUANTITATIVE_MANIFEST_MISMATCH",
                "INVALID_OUTLINE_SCHEMA",
                "MISSING_SECTION_EVIDENCE",
                "SECTION_VALIDATION_FAILED",
                "INVALID_LLM_OUTPUT",
                "CREDENTIAL_NOT_FOUND",
                "ATOMIC_WRITE_FAILED",
                "UNSUPPORTED_DRAFT_RETRIEVAL_STRATEGY",
            )
            code = next((item for item in known if item in message), "RUNTIME_DEPENDENCY_FAILED")
            # Si ocurre un fallo definitivo en Stage06, devuelve un resultado de error,
            # registra la causa y detiene la etapa.
            return AgentResult(
                execution_status=ExecutionStatus.FAILED,
                quality_status=QualityStatus.REJECTED,
                decision=DecisionInfo(
                    code="DRAFT_WRITING_FAILED",
                    rationale="Falló la ejecución del Agente Redactor.",
                ),
                quality_metrics={"scientific": {}, "technical": {}},
                warnings=(
                    AgentWarning(
                        code=code,
                        severity=WarningSeverity.ERROR,
                        blocking=True,
                        message=message,
                    ),
                ),
                failure_reason_codes=(code,),
                requested_transition=RequestedTransition(
                    action=TransitionAction.HALT_STAGE,
                    target_stage=None,
                    reason_code=code,
                    requires_human_confirmation=False,
                ),
                output_artifacts={},
                tool_usage=ToolUsage(
                    retrieval_rounds=retrieval_rounds,
                    llm_calls=llm_calls,
                    validation_calls=validation_calls,
                ),
                attempt_number=agent_input.attempt_number,
                started_at=start,
                completed_at=datetime.now(timezone.utc).isoformat(),
                error={
                    "type": type(exc).__name__,
                    "message": message,
                    "stage": agent_input.stage_name,
                },
            )
