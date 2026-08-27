"""
Coordinador del Agente 03 de extracción científica.

Organiza la ejecución de los módulos internos encargados de extraer y
estructurar información científica de los papers. Sus dependencias se
inyectan desde el runtime, por lo que el agente no crea directamente
conexiones con el LLM, Chroma ni la persistencia transaccional.

El agente produce un AgentResult con los resultados de la extracción y
puede solicitar una transición mediante requested_transition, pero la
decisión final sobre el flujo del pipeline corresponde al orquestador.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import pandas as pd

# Importa las estructuras que estandarizan la entrada del agente,
# las referencias a artefactos y el modo en que debe ejecutarse.
from src.contracts.agent_input import (
    AgentInput,
    ArtifactReference,
    ExecutionMode,
)
# Importa las estructuras que estandarizan el resultado del agente,
# incluyendo estados, advertencias, decisiones, transiciones y uso de herramientas.
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
# Importa funciones para guardar archivos de forma segura,
# evitando que queden incompletos si ocurre un error durante la escritura.
from src.io.atomic_write import (
    atomic_write_jsonl,
    atomic_write_text,
)
from src.state.fingerprints import sha256_file
# Importa las funciones que realizan la extracción inicial de fichas
# científicas y reparan aquellas que quedaron incompletas o inválidas.
from src.tools.extraction.card_extraction import (
    generate_repaired_card_for_source,
    run_bad_card_repair,
    run_initial_extraction,
)
# Importa las reglas y funciones usadas para validar las fichas científicas,
# detectar fichas problemáticas y construir sus resúmenes e indicadores de calidad.
from src.tools.extraction.card_validation import (
    CARD_REQUIRED_FIELDS,
    QUALITY_COLUMNS,
    SUMMARY_COLUMNS,
    build_quality_row,
    build_summary_row,
    is_bad_card,
)
# Importa la función que revisa que los chunks estén bien formados
# y tengan la información necesaria antes de usarlos en la extracción.
from src.tools.extraction.chunk_validation import (
    validate_chunks_dataframe,
)
# Importa la función que construye y ejecuta la rama encargada
# de generar la base de conocimiento estructurada de la extracción.
from src.tools.extraction.knowledge_base import (
    execute_knowledge_base_branch,
)
# Importa las funciones que clasifican la relevancia de las fichas
# científicas y determinan cuándo deben volver a evaluarse.
from src.tools.extraction.relevance_classification import (
    classify_card_relevance,
    determine_relevance_reclassification,
    run_relevance_classification,
)
# Importa las funciones que recuperan los chunks relevantes de cada paper
# y construyen con ellos el contexto que utilizará el agente de extracción.
from src.tools.extraction.retrieval import (
    build_context_from_chunks,
    retrieve_chunks_for_paper,
)
# Importa funciones para guardar, revisar y controlar los archivos generados
# por la etapa de extracción, incluyendo respaldos, manifiestos y reconstrucción.
from src.tools.extraction.stage_artifacts import (
    FINAL_OUTPUT_KEYS,
    TRACKED_STAGE_OUTPUT_KEYS,
    any_stage_outputs_exist,
    audit_final_consistency,
    backup_stage_outputs,
    build_current_extraction_signature,
    build_extraction_manifest,
    decide_extraction_rebuild,
    load_json_file,
    report_output_status,
    required_stage_outputs_exist,
    reset_stage_outputs,
    restore_validation_report,
    save_dataframe_even_if_empty,
    save_json_file,
    stable_hash_dict,
)
# Importa la función que revisa y corrige títulos incorrectos
# o faltantes en las fichas científicas extraídas.
from src.tools.extraction.title_repair import (
    run_title_repair,
)
# Importa funciones que revisan qué información importante falta en cada ficha
# y construyen un plan para corregir o completar únicamente esos campos.
from src.tools.extraction.revision_strategy import (
    REVISION_PLAN_COLUMNS,
    build_revision_plan,
    normalize_card_payload,
    plan_by_source,
    missing_critical_fields,
)
# Importa funciones que identifican y excluyen papers de revisión
# que no deben formar parte de la extracción científica principal.
from src.tools.extraction.review_exclusion import (
    EXCLUSION_AUDIT_COLUMNS,
    apply_review_exclusion_policy,
    is_review_excluded,
)
# Importa funciones que determinan qué papers pueden formar parte del corpus,
# cuáles deben excluirse y cuáles deben quedar en cuarentena para revisión.
from src.tools.extraction.corpus_eligibility import (
    QUARANTINE_AUDIT_COLUMNS,
    apply_corpus_eligibility_policy,
    apply_pre_eligibility_policy,
    is_corpus_include,
    is_corpus_quarantined,
)


EXPECTED_STAGE_NAME = "03_agente_extraccion_kb"

_REQUIRED_DEPENDENCIES = (
    "chunks_clean",
    "chroma_manifest",
)

# Define las rutas de archivos y carpetas que deben existir
# para que la etapa de extracción pueda ejecutarse correctamente.
_REQUIRED_PATH_KEYS = (
    "OUTPUTS_DIR",
    "DIR_EXTRACTION",
    "DIR_KB",
    "CARDS_JSONL_PATH",
    "CARDS_SUMMARY_CSV_PATH",
    "CARDS_ERRORS_CSV_PATH",
    "CARDS_QUALITY_CSV_PATH",
    "CARDS_REVISION_PLAN_CSV_PATH",
    "CARDS_REVIEW_EXCLUSION_AUDIT_CSV_PATH",
    "CARDS_QUARANTINE_AUDIT_CSV_PATH",
    "RETRIEVAL_TRACE_CSV_PATH",
    "EXTRACTION_MANIFEST_PATH",
    "CHUNKS_VALIDATION_REPORT_PATH",
    "KB_CSV_PATH",
    "KB_JSONL_PATH",
)

# Define los datos de configuración que se comparan entre ejecuciones
# para detectar si hubo cambios que obliguen a rehacer la extracción.
_REQUIRED_SIGNATURE_KEYS = (
    "experiment_dir",
    "chroma_collection_name",
    "openai_model",
    "embedding_model_name",
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
)

# Define los parámetros que controlan cómo se recuperan los chunks
# y cuánto contexto puede utilizarse durante la extracción y reparación.
_REQUIRED_RETRIEVAL_KEYS = (
    "queries",
    "profile",
    "profile_config",
    "max_chunks_per_paper",
    "max_context_chars",
    "repair_max_chunks_per_paper",
    "repair_max_context_chars",
)

class _DefaultMessage:
    def __init__(self, content: Any):
        self.content = content

# Guarda archivos de forma segura, evitando que queden incompletos
# si ocurre un error durante la escritura.
class AtomicRawWriter:
    """Writer adapter expected by ``run_bad_card_repair``."""

    def write(
        self,
        path: str | Path,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> Any:
        return atomic_write_text(
            path,
            content,
            encoding=encoding,
        )


# Lee un archivo JSONL línea por línea, convierte cada línea JSON
# en un objeto de Python y devuelve todos los registros en una lista.
def _load_jsonl(path: str | Path) -> list[Any]:
    source = Path(path)
    records = []

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    return records


def _save_jsonl(
    records: Sequence[Any],
    path: str | Path,
) -> Any:
    return atomic_write_jsonl(
        path,
        records,
        ensure_ascii=False,
        sort_keys=False,
    )


def _save_dataframe(
    dataframe: pd.DataFrame,
    path: str | Path,
) -> Any:
    return atomic_write_text(
        path,
        dataframe.to_csv(index=False),
        encoding="utf-8",
    )


def _default_now() -> str:
    return datetime.now().isoformat()


# Define las dependencias que necesita el Agente 03 para funcionar,
# incluyendo LLMs, prompts, lectura/escritura de archivos, validaciones,
# reparación de fichas, clasificación de relevancia y construcción de la KB.
@dataclass(slots=True)
class ExtractionAgentDependencies:
    main_llm: Any = None
    repair_llm: Any = None
    extraction_prompt_builder: Callable[..., str] | None = None
    relevance_prompt_builder: Callable[..., str] | None = None
    json_parser: Callable[[Any], Any] = json.loads
    message_factory: Callable[..., Any] = _DefaultMessage
    load_collection: Callable[[AgentInput], Any] | None = None

    load_dataframe: Callable[[str | Path], pd.DataFrame] = pd.read_csv
    load_json: Callable[[str | Path], Any] = load_json_file
    load_jsonl: Callable[[str | Path], list[Any]] = _load_jsonl
    save_json: Callable[[Any, str | Path], Any] = save_json_file
    save_jsonl: Callable[[Sequence[Any], str | Path], Any] = _save_jsonl
    save_dataframe: Callable[[pd.DataFrame, str | Path], Any] = _save_dataframe
    hash_file: Callable[[str | Path], str] = sha256_file
    raw_writer: Any = field(default_factory=AtomicRawWriter)
    now_factory: Callable[[], str] = _default_now
    print_fn: Callable[..., Any] = print

    validate_chunks: Callable[..., Any] = validate_chunks_dataframe
    run_initial: Callable[..., Mapping[str, Any]] = run_initial_extraction
    run_bad_repair: Callable[..., Mapping[str, Any]] = run_bad_card_repair
    run_title_repair: Callable[..., Mapping[str, Any]] = run_title_repair
    run_relevance: Callable[..., Mapping[str, Any]] = (
        run_relevance_classification
    )
    execute_kb: Callable[..., Any] = execute_knowledge_base_branch
    audit: Callable[..., Any] = audit_final_consistency


class ExtractionAgent:
    """Coordinate the approved extraction modules as AgentInput → AgentResult."""

    def __init__(
        self,
        dependencies: ExtractionAgentDependencies,
    ) -> None:
        self.dependencies = dependencies

    def execute(
        self,
        agent_input: AgentInput,
    ) -> AgentResult:
        started_at = self.dependencies.now_factory()
        warnings: list[AgentWarning] = []
        # Inicializa las métricas técnicas de la ejecución,
        # como reutilización, reconstrucción, respaldo y número de artefactos.
        metrics: dict[str, Any] = {
            "technical": {
                "reused_outputs": False,
                "rebuild_executed": False,
                "backup_created": False,
                "artifact_count": 0,
            },
        # Inicializa las métricas científicas de la extracción,
        # incluyendo fichas, KB, errores, reparaciones y llamadas al LLM.
            "scientific": {
                "num_cards": 0,
                "num_kb_rows": 0,
                "num_trace_rows": 0,
                "num_extraction_errors": 0,
                "num_validation_errors": 0,
                "num_bad_cards_after_repair": 0,
                "num_titles_selected_for_repair": 0,
                "num_title_llm_calls": 0,
                "num_classification_calls": 0,
            },
        }
        output_artifacts: dict[str, ArtifactReference] = {}
        llm_calls = 0
        retrieval_rounds = 0
        validation_calls = 0

        try:
            self._validate_agent_input(agent_input)

            policy = dict(agent_input.policy)
            paths = self._resolve_paths(policy)
            
            # Valida la entrada del Agente 03 y comprueba que las políticas
            # de firma y recuperación contengan todos los parámetros requeridos.
            signature_policy = self._require_mapping(
                policy,
                "signature",
            )
            retrieval_policy = self._require_mapping(
                policy,
                "retrieval",
            )
            self._validate_mapping_keys(
                signature_policy,
                _REQUIRED_SIGNATURE_KEYS,
                "policy.signature",
            )
            self._validate_mapping_keys(
                retrieval_policy,
                _REQUIRED_RETRIEVAL_KEYS,
                "policy.retrieval",
            )

            # Obtiene las referencias a los chunks limpios y al manifiesto de Chroma,
            # y valida que ambas dependencias estén disponibles y sean correctas.
            chunks_reference = agent_input.dependencies[
                "chunks_clean"
            ]
            chroma_reference = agent_input.dependencies[
                "chroma_manifest"
            ]
            self._validate_dependency(
                "chunks_clean",
                chunks_reference,
            )
            self._validate_dependency(
                "chroma_manifest",
                chroma_reference,
            )

            # Obtiene los recursos disponibles en tiempo de ejecución y carga los
            # chunks limpios desde memoria o, si no están disponibles, desde el archivo.
            # Luego verifica que el resultado sea un DataFrame válido.
            runtime_resources = dict(
                agent_input.agent_context.runtime_resources
            )
            input_dataframe = runtime_resources.get(
                "df_chunks_clean"
            )
            if input_dataframe is None:
                input_dataframe = self.dependencies.load_dataframe(
                    chunks_reference.path
                )
            if not isinstance(input_dataframe, pd.DataFrame):
                raise TypeError(
                    "df_chunks_clean debe ser un pandas.DataFrame."
                )

            # Construye la firma actual de la extracción usando la configuración,
            # los modelos, los prompts, los esquemas y los artefactos de entrada.
            current_signature = (
                build_current_extraction_signature(
                    experiment_id=agent_input.experiment_id,
                    experiment_dir=signature_policy[
                        "experiment_dir"
                    ],
                    chunks_clean_path=chunks_reference.path,
                    chroma_manifest_path=chroma_reference.path,
                    chroma_collection_name=signature_policy[
                        "chroma_collection_name"
                    ],
                    openai_model=signature_policy[
                        "openai_model"
                    ],
                    embedding_model_name=signature_policy[
                        "embedding_model_name"
                    ],
                    experiment_profile=signature_policy[
                        "experiment_profile"
                    ],
                    topic_profile=signature_policy[
                        "topic_profile"
                    ],
                    generation_profile=signature_policy[
                        "generation_profile"
                    ],
                    rag_policy=signature_policy[
                        "rag_policy"
                    ],
                    extraction_policy=signature_policy[
                        "extraction_policy"
                    ],
                    card_required_fields=signature_policy[
                        "card_required_fields"
                    ],
                    card_list_fields=signature_policy[
                        "card_list_fields"
                    ],
                    classification_fields=signature_policy[
                        "classification_fields"
                    ],
                    extraction_prompt_version=signature_policy[
                        "extraction_prompt_version"
                    ],
                    relevance_prompt_version=signature_policy[
                        "relevance_prompt_version"
                    ],
                    kb_schema_version=signature_policy[
                        "kb_schema_version"
                    ],
                    rag_clean_validation_version=signature_policy[
                        "rag_clean_validation_version"
                    ],
                )
            )
            current_fingerprint = stable_hash_dict(
                current_signature
            )

            previous_manifest = self.dependencies.load_json(
                paths["EXTRACTION_MANIFEST_PATH"]
            )
            # Define los archivos principales que deben existir para considerar
            # completa y reutilizable una ejecución previa del Agente 03.
            required_outputs = [
                paths[key]
                for key in (
                    "CARDS_JSONL_PATH",
                    "CARDS_SUMMARY_CSV_PATH",
                    "CARDS_ERRORS_CSV_PATH",
                    "CARDS_QUALITY_CSV_PATH",
                    "RETRIEVAL_TRACE_CSV_PATH",
                    "KB_CSV_PATH",
                    "KB_JSONL_PATH",
                )
            ]
            stage_outputs_exist = (
                required_stage_outputs_exist(
                    required_outputs
                )
            )
            rebuild_decision = decide_extraction_rebuild(
                force_rebuild_extraction=policy.get(
                    "force_rebuild_extraction",
                    False,
                ),
                previous_manifest=previous_manifest,
                stage_outputs_exist=stage_outputs_exist,
                current_fingerprint=current_fingerprint,
                auto_rebuild_extraction=policy.get(
                    "auto_rebuild_extraction",
                    True,
                ),
            )
            extraction_status = rebuild_decision[
                "extraction_status"
            ]
            should_rebuild = rebuild_decision[
                "should_rebuild_extraction"
            ]

            # Prepara los datos del intento anterior cuando el Agente 03 está en su
            # intento final, recuperando fichas, errores y trazas para continuar la reparación.
            backup_dir_created: Path | None = None
            previous_cards_for_attempt2: list[dict[str, Any]] | None = None
            previous_errors_for_attempt2: list[dict[str, Any]] = []
            previous_trace_for_attempt2: list[dict[str, Any]] = []
            if agent_input.is_final_attempt():
                previous_cards_path = paths["CARDS_JSONL_PATH"]
                previous_errors_path = paths["CARDS_ERRORS_CSV_PATH"]
                previous_trace_path = paths["RETRIEVAL_TRACE_CSV_PATH"]
                if agent_input.previous_attempt is not None:
                    previous_artifacts = agent_input.previous_attempt.previous_artifacts
                    if "CARDS_JSONL_PATH" in previous_artifacts:
                        previous_cards_path = Path(
                            previous_artifacts["CARDS_JSONL_PATH"].path
                        )
                    if "CARDS_ERRORS_CSV_PATH" in previous_artifacts:
                        previous_errors_path = Path(
                            previous_artifacts["CARDS_ERRORS_CSV_PATH"].path
                        )
                    if "RETRIEVAL_TRACE_CSV_PATH" in previous_artifacts:
                        previous_trace_path = Path(
                            previous_artifacts["RETRIEVAL_TRACE_CSV_PATH"].path
                        )
                if not Path(previous_cards_path).is_file():
                    raise FileNotFoundError(
                        "El intento 2 requiere scientific_cards.jsonl del intento 1."
                    )
                previous_cards_for_attempt2 = self.dependencies.load_jsonl(
                    previous_cards_path
                )
                if Path(previous_errors_path).is_file():
                    previous_errors_for_attempt2 = self.dependencies.load_dataframe(
                        previous_errors_path
                    ).to_dict(orient="records")
                if Path(previous_trace_path).is_file():
                    previous_trace_for_attempt2 = self.dependencies.load_dataframe(
                        previous_trace_path
                    ).to_dict(orient="records")
            # Si la extracción debe reconstruirse, respalda los resultados anteriores,
            # limpia las salidas de la etapa y prepara una nueva ejecución desde cero.
            if should_rebuild:
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
                        reason=extraction_status,
                    )
                reset_stage_outputs(
                    paths["DIR_EXTRACTION"],
                    paths["DIR_KB"],
                )
                if agent_input.is_first_attempt():
                    revision_path = paths["CARDS_REVISION_PLAN_CSV_PATH"]
                    if revision_path.exists():
                        revision_path.unlink()

            # Registra si se reutilizaron o reconstruyeron resultados y si se creó
            # un respaldo; luego valida los chunks limpios antes de continuar.
            metrics["technical"][
                "reused_outputs"
            ] = not bool(should_rebuild)
            metrics["technical"][
                "rebuild_executed"
            ] = bool(should_rebuild)
            metrics["technical"][
                "backup_created"
            ] = backup_dir_created is not None

            validation_created_at = (
                self.dependencies.now_factory()
            )
            (
                df_chunks_clean,
                validation_report,
            ) = self.dependencies.validate_chunks(
                input_dataframe,
                experiment_id=agent_input.experiment_id,
                chunks_file=chunks_reference.path,
                created_at=validation_created_at,
            )
            validation_calls += 1

            restore_validation_report(
                validation_report,
                paths[
                    "CHUNKS_VALIDATION_REPORT_PATH"
                ],
                save_json_file_fn=self.dependencies.save_json,
                print_fn=self.dependencies.print_fn,
            )

            # Obtiene los errores encontrados en la validación, registra cuántos hay
            # y detiene la extracción si los chunks no son seguros para continuar. 
            validation_errors = list(
                validation_report.get(
                    "errors",
                    [],
                )
            )
            metrics["scientific"][
                "num_validation_errors"
            ] = len(validation_errors)
            if validation_errors:
                raise ValueError(
                    "chunks_clean_for_rag.csv no es seguro "
                    "para la extracción científica."
                )

            # Carga el manifiesto de Chroma y verifica que corresponda al experimento,
            # a la colección configurada y a los chunks limpios usados por el Agente 03.
            chroma_manifest = self.dependencies.load_json(
                chroma_reference.path
            )
            self._validate_chroma_manifest(
                chroma_manifest=chroma_manifest,
                experiment_id=agent_input.experiment_id,
                collection_name=signature_policy[
                    "chroma_collection_name"
                ],
                chunks_clean_path=chunks_reference.path,
            )

            # Inicializa las estructuras que almacenarán los resultados de la extracción,
            # las auditorías, los conteos de elegibilidad y las llamadas realizadas.
            extraction_errors: list[dict[str, Any]]
            retrieval_trace_rows: list[dict[str, Any]]
            cards: list[dict[str, Any]]
            review_exclusion_audit_rows: list[dict[str, Any]] = []
            quarantine_audit_rows: list[dict[str, Any]] = []
            corpus_eligibility_counts: dict[str, int] = {
                "include": 0, "exclude": 0, "quarantine": 0,
            }
            initial_calls = 0
            repair_calls = 0
            bad_after_repair: list[str] = []

            # Si la etapa debe reconstruirse, obtiene la colección Chroma necesaria
            # y verifica que estén disponibles las dependencias para generar la extracción.
            if should_rebuild:
                collection = runtime_resources.get(
                    "collection"
                )
                if collection is None:
                    if self.dependencies.load_collection is None:
                        raise ValueError(
                            "No se recibió una colección ni un loader de colección."
                        )
                    collection = self.dependencies.load_collection(agent_input)

                self._validate_generation_dependencies()

                # Define una función de recuperación que busca los chunks más relevantes
                # de un paper usando Chroma y la configuración RAG de la extracción.
                def retrieve(*, source_filename: str, max_chunks: int) -> Any:
                    return retrieve_chunks_for_paper(
                        source_filename,
                        max_chunks,
                        df_chunks_clean=df_chunks_clean,
                        collection=collection,
                        retrieval_queries=retrieval_policy["queries"],
                        retrieval_profile=retrieval_policy["profile"],
                        retrieval_profile_config=retrieval_policy["profile_config"],
                    )

                def card_json_parser(raw: Any) -> dict[str, Any]:
                    return normalize_card_payload(
                        self.dependencies.json_parser(raw)
                    )

                if agent_input.is_first_attempt():
                    source_filenames = sorted(
                        str(value)
                        for value in df_chunks_clean["source_filename"].unique()
                    )
                    # Ejecuta la extracción inicial de fichas científicas para todos los papers
                    # y registra las fichas, trazas de recuperación, errores y llamadas realizadas.
                    extraction_created_at = self.dependencies.now_factory()
                    initial_result = self.dependencies.run_initial(
                        source_filenames,
                        retrieve=retrieve,
                        build_context=build_context_from_chunks,
                        prompt_builder=self.dependencies.extraction_prompt_builder,
                        experiment_profile=signature_policy["experiment_profile"],
                        llm=self.dependencies.main_llm,
                        json_parser=card_json_parser,
                        message_factory=self.dependencies.message_factory,
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
                    title_result_initial = self.dependencies.run_title_repair(
                        cards,
                        df_chunks_clean=df_chunks_clean,
                        title_repair_first_chunks=int(
                            policy["title_repair_first_chunks"]
                        ),
                        repair_llm=self.dependencies.repair_llm,
                        json_parser=self.dependencies.json_parser,
                        should_rebuild_extraction=True,
                        message_factory=self.dependencies.message_factory,
                        created_at=self.dependencies.now_factory(),
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
                        created_at=self.dependencies.now_factory(),
                    )
                    cards = exclusion_result_initial["cards"]
                    review_exclusion_audit_rows.extend(
                        exclusion_result_initial["audit_rows"]
                    )

                    # Aplica una preclasificación documental para marcar papers que deben
                    # incluirse, excluirse o enviarse a cuarentena antes del control de calidad científico.
                    pre_eligibility_result = apply_pre_eligibility_policy(
                        cards, created_at=self.dependencies.now_factory(),
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
                    quality_rows = [self._quality_row(card) for card in cards]
                    self.dependencies.save_jsonl(cards, paths["CARDS_JSONL_PATH"])
                    self.dependencies.save_dataframe(
                        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS),
                        paths["CARDS_SUMMARY_CSV_PATH"],
                    )
                    self.dependencies.save_dataframe(
                        pd.DataFrame(quality_rows, columns=QUALITY_COLUMNS),
                        paths["CARDS_QUALITY_CSV_PATH"],
                    )
                    self.dependencies.save_dataframe(
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
                        completed_at = self.dependencies.now_factory()
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
                            output_artifacts=self._collect_output_artifacts(paths),
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
                else:
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
                        cards, created_at=self.dependencies.now_factory(),
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
                    self.dependencies.save_dataframe(
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
                        title_result_attempt2 = self.dependencies.run_title_repair(
                            title_cards,
                            df_chunks_clean=df_chunks_clean,
                            title_repair_first_chunks=int(
                                policy["title_repair_first_chunks"]
                            ),
                            repair_llm=self.dependencies.repair_llm,
                            json_parser=self.dependencies.json_parser,
                            should_rebuild_extraction=True,
                            message_factory=self.dependencies.message_factory,
                            created_at=self.dependencies.now_factory(),
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
                                    prompt_builder=self.dependencies.extraction_prompt_builder,
                                    experiment_profile=signature_policy[
                                        "experiment_profile"
                                    ],
                                    repair_llm=self.dependencies.repair_llm,
                                    json_parser=card_json_parser,
                                    message_factory=self.dependencies.message_factory,
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
                                "created_at": self.dependencies.now_factory(),
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
                    self.dependencies.save_jsonl(cards, paths["CARDS_JSONL_PATH"])
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
           
            # Si el agente NO necesita volver a hacer la extracción desde cero, reutiliza las fichas, trazas y errores
            # ya guardados en disco y reinicia los indicadores para las validaciones siguientes.
            else:
                cards = self.dependencies.load_jsonl(
                    paths["CARDS_JSONL_PATH"]
                )
                trace_dataframe = (
                    self.dependencies.load_dataframe(
                        paths[
                            "RETRIEVAL_TRACE_CSV_PATH"
                        ]
                    )
                )
                errors_dataframe = (
                    self.dependencies.load_dataframe(
                        paths[
                            "CARDS_ERRORS_CSV_PATH"
                        ]
                    )
                )
                retrieval_trace_rows = (
                    trace_dataframe.to_dict(
                        orient="records"
                    )
                )
                extraction_errors = (
                    errors_dataframe.to_dict(
                        orient="records"
                    )
                )

            title_selected = 0
            title_calls = 0
            cards_need_classification = False

            # Revisa y repara nuevamente los títulos de las fichas, actualiza los errores
            # y registra cuántas fichas necesitaron reparación y cuántas llamadas al LLM se usaron.
            title_result = (
                self.dependencies.run_title_repair(
                    cards,
                    df_chunks_clean=df_chunks_clean,
                    title_repair_first_chunks=int(
                        policy[
                            "title_repair_first_chunks"
                        ]
                    ),
                    repair_llm=self.dependencies.repair_llm,
                    json_parser=self.dependencies.json_parser,
                    should_rebuild_extraction=should_rebuild,
                    message_factory=(
                        self.dependencies.message_factory
                    ),
                    created_at=(
                        self.dependencies.now_factory()
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
                self.dependencies.save_jsonl(
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
                self._validate_classification_dependencies()

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
                            self.dependencies.relevance_prompt_builder
                        ),
                        llm=self.dependencies.main_llm,
                        json_parser=(
                            self.dependencies.json_parser
                        ),
                        message_factory=(
                            self.dependencies.message_factory
                        ),
                    )

                relevance_result = (
                    self.dependencies.run_relevance(
                        [c for c in cards if not is_review_excluded(c) and not is_corpus_quarantined(c)],
                        should_rebuild_extraction=(
                            should_rebuild
                        ),
                        classify=classify,
                        created_at=(
                            self.dependencies.now_factory()
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
                self.dependencies.save_jsonl(
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

            # Aplica por última vez la exclusión determinista de papers de revisión,
            # guarda las fichas finales y marca que la KB debe recrearse si hubo exclusiones.
            exclusion_policy_final = dict(
                signature_policy.get("extraction_policy", {})
            )
            exclusion_result_final = apply_review_exclusion_policy(
                cards,
                exclude_reviews=bool(exclusion_policy_final.get("exclude_reviews", True)),
                created_at=self.dependencies.now_factory(),
            )
            cards = exclusion_result_final["cards"]
            review_exclusion_audit_rows.extend(
                exclusion_result_final["audit_rows"]
            )
            if exclusion_result_final["num_excluded"]:
                kb_should_recreate = True
                self.dependencies.save_jsonl(
                    cards,
                    paths["CARDS_JSONL_PATH"],
                )

            # Aplica la clasificación final de elegibilidad del corpus y asigna a cada
            # ficha un estado canónico: INCLUDE, EXCLUDE o QUARANTINE.
            eligibility_result = apply_corpus_eligibility_policy(
                cards, created_at=self.dependencies.now_factory(),
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
            self.dependencies.save_jsonl(
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
                    self.dependencies.load_dataframe(
                        paths["KB_CSV_PATH"]
                    )
                )
                
            # Ejecuta la construcción o reutilización de la Knowledge Base y,
            # si fue creada nuevamente, guarda sus versiones CSV y JSONL.
            (
                kb_status,
                df_kb,
                kb_rows,
            ) = self.dependencies.execute_kb(
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
                self.dependencies.save_dataframe(
                    df_kb,
                    paths["KB_CSV_PATH"],
                )
                self.dependencies.save_jsonl(
                    kb_rows or [],
                    paths["KB_JSONL_PATH"],
                )

            # Actualiza las métricas científicas finales de la etapa de extracción
            # con los conteos de fichas, KB, errores, reparaciones y clasificaciones.
            metrics["scientific"].update({
                "num_cards": int(
                    len(cards)
                ),
                "num_kb_rows": int(
                    len(df_kb)
                ),
                "num_trace_rows": int(
                    len(retrieval_trace_rows)
                ),
                "num_extraction_errors": int(
                    len(extraction_errors)
                ),
                "num_bad_cards_after_repair": int(
                    len(bad_after_repair)
                ),
                "num_titles_selected_for_repair": int(
                    title_selected
                ),
                "num_title_llm_calls": int(
                    title_calls
                ),
                "num_classification_calls": int(
                    classification_calls
                ),
                "cards_need_classification": bool(
                    cards_need_classification
                ),
                "kb_status": kb_status,
            })

            # Construye y guarda los reportes finales de resumen y calidad
            # usando las fichas definitivas después de todas las correcciones y filtros.
            summary_rows = [
                build_summary_row(card) for card in cards
            ]
            quality_rows = [
                self._quality_row(card) for card in cards
            ]
            self.dependencies.save_dataframe(
                pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS),
                paths["CARDS_SUMMARY_CSV_PATH"],
            )
            self.dependencies.save_dataframe(
                pd.DataFrame(quality_rows, columns=QUALITY_COLUMNS),
                paths["CARDS_QUALITY_CSV_PATH"],
            )

            # Reconstruye y guarda el plan de revisión usando las fichas finales,
            # para que el archivo refleje exactamente el estado definitivo del corpus.
            revision_rows_final = build_revision_plan(
                cards, extraction_errors, retrieval_trace_rows,
            )
            self.dependencies.save_dataframe(
                pd.DataFrame(revision_rows_final, columns=REVISION_PLAN_COLUMNS),
                paths["CARDS_REVISION_PLAN_CSV_PATH"],
            )

            # Elimina registros duplicados de la auditoría de exclusión y conserva
            # únicamente la evaluación más reciente de cada paper antes de guardarla.
            deduplicated_audit_rows: dict[str, dict[str, Any]] = {}
            for row in review_exclusion_audit_rows:
                deduplicated_audit_rows[str(row.get("source_filename", ""))] = row
            save_dataframe_even_if_empty(
                list(deduplicated_audit_rows.values()),
                paths["CARDS_REVIEW_EXCLUSION_AUDIT_CSV_PATH"],
                EXCLUSION_AUDIT_COLUMNS,
            )

            # Elimina registros duplicados de cuarentena y guarda una sola entrada final
            # por paper para mantener una auditoría clara de los documentos no utilizables.
            deduplicated_quarantine_rows: dict[str, dict[str, Any]] = {}
            for row in quarantine_audit_rows:
                deduplicated_quarantine_rows[str(row.get("source_filename", ""))] = row
            save_dataframe_even_if_empty(
                list(deduplicated_quarantine_rows.values()),
                paths["CARDS_QUARANTINE_AUDIT_CSV_PATH"],
                QUARANTINE_AUDIT_COLUMNS,
            )

            # Calcula las métricas finales del corpus: cuántos papers se procesaron,
            # incluyeron, excluyeron o quedaron en cuarentena, y por qué motivo.
            papers_processed = len(cards)
            papers_excluded_by_review_policy = sum(
                1 for card in cards if is_review_excluded(card)
            )
            papers_excluded_total = sum(
                1 for card in cards
                if card.get("include_in_state_of_art") is False
            )
            papers_included = papers_processed - papers_excluded_total
            metrics["scientific"].update({
                "papers_processed": int(papers_processed),
                "papers_included": int(papers_included),
                "papers_excluded": int(papers_excluded_total),
                "papers_excluded_by_review_policy": int(
                    papers_excluded_by_review_policy
                ),
                "papers_excluded_other_reasons": int(
                    papers_excluded_total - papers_excluded_by_review_policy
                ),
                "papers_excluded_uncertain_review_classification": int(
                    sum(
                        1 for row in review_exclusion_audit_rows
                        if str(row.get("action")) == "UNCERTAIN"
                    )
                ),
                "papers_corpus_include": int(
                    corpus_eligibility_counts.get("include", 0)
                ),
                "papers_corpus_exclude": int(
                    corpus_eligibility_counts.get("exclude", 0)
                ),
                "papers_corpus_quarantine": int(
                    corpus_eligibility_counts.get("quarantine", 0)
                ),
            })

            # Verifica que exista un número mínimo de papers elegibles en el corpus;
            # si no se cumple, detiene la etapa porque no hay corpus suficiente para continuar.
            corpus_eligibility_policy = dict(
                exclusion_policy_final.get("corpus_eligibility_policy", {})
            )
            min_include_corpus_size = int(
                corpus_eligibility_policy.get("min_include_corpus_size", 1)
            )
            papers_corpus_include = corpus_eligibility_counts.get("include", 0)
            if papers_corpus_include < max(1, min_include_corpus_size):
                artifacts = self._collect_output_artifacts(paths)
                return AgentResult(
                    execution_status=ExecutionStatus.COMPLETED,
                    quality_status=QualityStatus.REJECTED,
                    decision=DecisionInfo(
                        code="CORPUS_ELIGIBILITY_INSUFFICIENT",
                        rationale=(
                            f"Solo {papers_corpus_include} documento(s) elegible(s) "
                            f"(INCLUDE) para el corpus -- mínimo configurado: "
                            f"{max(1, min_include_corpus_size)}. Condición sistémica: "
                            "Stage03 no tiene corpus utilizable, nunca por un "
                            "documento individual."
                        ),
                    ),
                    quality_metrics=metrics,
                    warnings=(),
                    failure_reason_codes=("CORPUS_ELIGIBILITY_INSUFFICIENT",),
                    requested_transition=RequestedTransition(
                        action=TransitionAction.HALT_STAGE,
                        target_stage=None,
                        reason_code="CORPUS_ELIGIBILITY_INSUFFICIENT",
                        requires_human_confirmation=True,
                    ),
                    output_artifacts=artifacts,
                    tool_usage=ToolUsage(
                        retrieval_rounds=retrieval_rounds,
                        llm_calls=llm_calls,
                        validation_calls=validation_calls,
                    ),
                    attempt_number=agent_input.attempt_number,
                    started_at=started_at,
                    completed_at=self.dependencies.now_factory(),
                )

            # Verifica que existan los archivos principales de fichas, errores y trazas;
            # si alguno falta, lo crea antes de generar el manifiesto y la auditoría final.
            if not Path(
                paths["CARDS_JSONL_PATH"]
            ).exists():
                self.dependencies.save_jsonl(
                    cards,
                    paths["CARDS_JSONL_PATH"],
                )
            if not Path(
                paths["CARDS_ERRORS_CSV_PATH"]
            ).exists():
                save_dataframe_even_if_empty(
                    extraction_errors,
                    paths["CARDS_ERRORS_CSV_PATH"],
                    policy["error_columns"],
                )
            if not Path(
                paths["RETRIEVAL_TRACE_CSV_PATH"]
            ).exists():
                save_dataframe_even_if_empty(
                    retrieval_trace_rows,
                    paths[
                        "RETRIEVAL_TRACE_CSV_PATH"
                    ],
                    policy["trace_columns"],
                )

            # Construye y guarda el manifiesto final de la extracción con la configuración,
            # estado, métricas, firmas, políticas, validaciones y artefactos usados.
            trace_dataframe = pd.DataFrame(
                retrieval_trace_rows,
                columns=policy["trace_columns"],
            )
            errors_dataframe = pd.DataFrame(
                extraction_errors,
                columns=policy["error_columns"],
            )
            manifest_created_at = (
                self.dependencies.now_factory()
            )
            manifest = build_extraction_manifest(
                cards=cards,
                df_kb=df_kb,
                df_chunks_clean=df_chunks_clean,
                trace_dataframe=trace_dataframe,
                errors_dataframe=errors_dataframe,
                experiment_id=agent_input.experiment_id,
                created_at=manifest_created_at,
                current_fingerprint=current_fingerprint,
                current_extraction_signature=(
                    current_signature
                ),
                extraction_status=extraction_status,
                auto_rebuild_extraction=policy.get(
                    "auto_rebuild_extraction",
                    True,
                ),
                force_rebuild_extraction=policy.get(
                    "force_rebuild_extraction",
                    False,
                ),
                should_rebuild_extraction=(
                    should_rebuild
                ),
                reclassify_relevance=(
                    reclassify_relevance
                ),
                kb_should_recreate=(
                    kb_should_recreate
                ),
                backup_dir_created=(
                    backup_dir_created
                ),
                chroma_collection_name=signature_policy[
                    "chroma_collection_name"
                ],
                embedding_model_name=signature_policy[
                    "embedding_model_name"
                ],
                extraction_retrieval_profile=retrieval_policy[
                    "profile"
                ],
                extraction_retrieval_profile_config=retrieval_policy[
                    "profile_config"
                ],
                extraction_retrieval_queries=retrieval_policy[
                    "queries"
                ],
                retrieval_trace_csv_path=paths[
                    "RETRIEVAL_TRACE_CSV_PATH"
                ],
                validation_report=validation_report,
                paths=paths,
            )
            self.dependencies.save_json(
                manifest,
                paths[
                    "EXTRACTION_MANIFEST_PATH"
                ],
            )

            # Restaura y guarda el reporte de validación de los chunks
            # para conservar el resultado de la validación realizada al inicio.
            restore_validation_report(
                validation_report,
                paths[
                    "CHUNKS_VALIDATION_REPORT_PATH"
                ],
                save_json_file_fn=(
                    self.dependencies.save_json
                ),
                print_fn=(
                    self.dependencies.print_fn
                ),
            )

            # Revisa que todos los artefactos finales existan y ejecuta una auditoría
            # de consistencia entre chunks, fichas, trazas y manifiesto de extracción.
            outputs = [
                paths[key]
                for key in FINAL_OUTPUT_KEYS
            ]
            report_output_status(
                outputs,
                print_fn=self.dependencies.print_fn,
            )
            self.dependencies.audit(
                chunks_validation_report_path=paths[
                    "CHUNKS_VALIDATION_REPORT_PATH"
                ],
                extraction_manifest_path=paths[
                    "EXTRACTION_MANIFEST_PATH"
                ],
                df_chunks_clean=df_chunks_clean,
                cards=cards,
                retrieval_trace_csv_path=paths[
                    "RETRIEVAL_TRACE_CSV_PATH"
                ],
                load_json_file_fn=(
                    self.dependencies.load_json
                ),
                read_csv_fn=(
                    self.dependencies.load_dataframe
                ),
                print_fn=(
                    self.dependencies.print_fn
                ),
            )
            validation_calls += 1

            # Convierte los errores de extracción en advertencias y añade una alerta
            # adicional si todavía quedan fichas inválidas después de la reparación.
            for error_row in extraction_errors:
                warnings.append(
                    self._warning_from_error_row(
                        error_row
                    )
                )
            if bad_after_repair:
                warnings.append(
                    AgentWarning(
                        code="BAD_CARDS_REMAIN",
                        severity=(
                            WarningSeverity.WARNING
                        ),
                        blocking=False,
                        message=(
                            "Persisten fichas inválidas: "
                            + ", ".join(
                                bad_after_repair
                            )
                        ),
                    )
                )

            # Junta los archivos finales, mide qué tan completas quedaron las fichas
            # y revisa si todavía existen problemas o si hace falta revisión humana.
            output_artifacts = self._collect_output_artifacts(
                paths
            )
            metrics["technical"][
                "artifact_count"
            ] = len(output_artifacts)

            completed_at = (
                self.dependencies.now_factory()
            )
            has_warnings = bool(warnings)
            coverage = self._critical_field_coverage(quality_rows)
            metrics["scientific"][
                "critical_field_coverage"
            ] = coverage
            extraction_policy = dict(
                signature_policy.get("extraction_policy", {})
            )
            reason_codes, can_manual = self._compute_reason_codes_and_manual_review_eligibility(
                extraction_policy=extraction_policy,
                coverage=coverage,
                cards=cards,
                bad_after_repair=bad_after_repair,
                extraction_errors=extraction_errors,
            ) 

            # Decide el estado final de calidad de la extracción
            # y qué debe hacer el pipeline después.
            quality_status, transition = self._decide_final_quality_status_and_transition(
                reason_codes=reason_codes,
                can_manual=can_manual,
                has_warnings=has_warnings,
                agent_input=agent_input,
            )  

            # Devuelve al orquestador el resultado final de la etapa,
            # con su calidad, métricas, advertencias, archivos generados y la acción siguiente.
            return AgentResult(
                execution_status=ExecutionStatus.COMPLETED,
                quality_status=quality_status,
                decision=DecisionInfo(
                    code=(
                        "REBUILT_OUTPUTS"
                        if should_rebuild
                        else "REUSED_CURRENT_OUTPUTS"
                    ),
                    rationale=(
                        "La etapa 03 completó la evaluación contractual v1.6."
                    ),
                ),
                quality_metrics=metrics,
                warnings=tuple(warnings),
                failure_reason_codes=reason_codes,
                requested_transition=transition,
                output_artifacts=output_artifacts,
                tool_usage=ToolUsage(
                    retrieval_rounds=retrieval_rounds,
                    llm_calls=llm_calls,
                    validation_calls=validation_calls,
                ),
                attempt_number=agent_input.attempt_number,
                started_at=started_at,
                completed_at=completed_at,
                error=None,
            )

        # Si ocurre un error inesperado, registra la hora de finalización
        # e intenta recuperar los artefactos que ya se alcanzaron a generar.
        except Exception as error:
            completed_at = (
                self.dependencies.now_factory()
            )
            try:
                policy = dict(
                    agent_input.policy
                )
                paths = self._resolve_paths(
                    policy
                )
                output_artifacts = (
                    self._collect_output_artifacts(
                        paths
                    )
                )
            except Exception:
                output_artifacts = {}

            failure_reason_codes = self._technical_failure_reason_codes(error)
            safe_message = self._sanitize_error_message(error)
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

    # Construye la fila de calidad de una ficha, identifica qué campos críticos
    # faltan y registra si el paper fue excluido por la política de reviews.
    @staticmethod
    def _quality_row(card: Mapping[str, Any]) -> dict[str, Any]:
        row = build_quality_row(dict(card))
        missing = missing_critical_fields(card)
        row["missing_fields"] = missing
        row["num_missing_fields"] = len(missing)
        row["_excluded_by_policy_rule"] = is_review_excluded(card)
        return row

    # Calcula qué tan completas quedaron las fichas incluidas,
    # sin contar las reviews que ya fueron excluidas.
    @staticmethod
    def _critical_field_coverage(
        quality_rows: Sequence[Mapping[str, Any]],
    ) -> float:
        included_rows = [
            row for row in quality_rows if not row.get("_excluded_by_policy_rule")
        ]
        if not included_rows:
            return 0.0
        required = max(len(CARD_REQUIRED_FIELDS), 1)
        covered = [
            max(0.0, 1.0 - (float(row.get("num_missing_fields", 0)) / required))
            for row in included_rows
        ]
        return sum(covered) / len(covered)

    # Revisa la cobertura y los problemas de las fichas para decidir
    # qué códigos de error quedan y si el caso puede pasar a revisión manual.
    @staticmethod
    def _compute_reason_codes_and_manual_review_eligibility(
        *,
        extraction_policy: Mapping[str, Any],
        coverage: float,
        cards: Sequence[Mapping[str, Any]],
        bad_after_repair: Sequence[str],
        extraction_errors: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[str, ...], bool]:
        """Extracción mecánica (Bloque D, D2) del cálculo de
        ``reason_codes``/``can_manual`` -- copia EXACTA de la lógica
        que antes vivía inline en execute(), justo antes de la
        decisión final (D1, ``_decide_final_quality_status_and_
        transition``). Ningún cambio de condición, threshold, orden
        ni reason_code. ``thresholds``/``approval_threshold``/
        ``minimum_usable``/``manual_policy``/``allowed_manual_codes``
        son puramente locales a este cálculo -- no se usan en
        ningún otro punto de execute() (verificado antes de mover).
        """

        thresholds = dict(
            extraction_policy.get("thresholds", {})
        )
        approval_threshold = float(
            dict(thresholds.get("approval", {})).get(
                "critical_field_coverage", 0.92
            )
        )
        minimum_usable = float(
            dict(
                thresholds.get(
                    "minimum_usable_quality", {}
                )
            ).get("critical_field_coverage", 0.80)
        )
        reason_codes = ExtractionAgent._scientific_reason_codes(
            cards=cards,
            bad_sources=bad_after_repair,
            extraction_errors=extraction_errors,
        )
        if coverage < approval_threshold and (
            "MISSING_CRITICAL_FIELDS" not in reason_codes
        ):
            reason_codes = tuple(
                dict.fromkeys(
                    (*reason_codes, "MISSING_CRITICAL_FIELDS")
                )
            )

        manual_policy = dict(
            extraction_policy.get("manual_review_policy", {})
        )
        allowed_manual_codes = set(
            manual_policy.get("allowed_reason_codes", ())
        )
        can_manual = (
            bool(manual_policy.get("allowed", False))
            and coverage >= minimum_usable
            and bool(set(reason_codes) & allowed_manual_codes)
        )

        return reason_codes, can_manual

    # Decide cómo termina la etapa según los problemas encontrados:
    # reintenta, pide revisión humana, rechaza o avanza.
    @staticmethod
    def _decide_final_quality_status_and_transition(
        *,
        reason_codes: tuple[str, ...],
        can_manual: bool,
        has_warnings: bool,
        agent_input: AgentInput,
    ) -> tuple[QualityStatus, RequestedTransition]:
        """Extracción mecánica (Bloque D, D1) de la decisión final
        RETRY/HALT/APPROVED -- copia EXACTA de la lógica que antes
        vivía inline en execute(), sin ningún cambio de condición,
        orden, reason_code ni mensaje. ``has_warnings`` se agrega
        como cuarto parámetro (no estaba en la especificación
        original de entrada: reason_codes/can_manual/agent_input)
        porque la rama sin reason_codes lo necesita para decidir
        entre APPROVED_WITH_WARNINGS y APPROVED -- omitirlo habría
        sido un cambio de comportamiento, no una extracción mecánica.
        """

        if reason_codes:
            if agent_input.is_first_attempt():
                quality_status = QualityStatus.NEEDS_REVISION
                transition = RequestedTransition(
                    action=TransitionAction.RETRY,
                    target_stage=None,
                    reason_code="NEEDS_REVISION",
                    requires_human_confirmation=False,
                )
            elif can_manual:
                quality_status = (
                    QualityStatus.APPROVED_PENDING_MANUAL_REVIEW
                )
                transition = RequestedTransition(
                    action=TransitionAction.HALT_STAGE,
                    target_stage=None,
                    reason_code=(
                        "APPROVED_PENDING_MANUAL_REVIEW"
                    ),
                    requires_human_confirmation=True,
                )
            else:
                quality_status = QualityStatus.REJECTED
                transition = RequestedTransition(
                    action=TransitionAction.HALT_STAGE,
                    target_stage=None,
                    reason_code="REJECTED",
                    requires_human_confirmation=False,
                )
        else:
            quality_status = (
                QualityStatus.APPROVED_WITH_WARNINGS
                if has_warnings
                else QualityStatus.APPROVED
            )
            transition = RequestedTransition(
                action=TransitionAction.ADVANCE,
                target_stage=None,
                reason_code="EXTRACTION_COMPLETED",
                requires_human_confirmation=False,
            )

        return quality_status, transition

    # Detecta los problemas de calidad que todavía tienen las fichas
    # y devuelve una lista de esos problemas sin repetirlos.
    @staticmethod
    def _scientific_reason_codes(
        *,
        cards: Sequence[Mapping[str, Any]],
        bad_sources: Sequence[str],
        extraction_errors: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        revision_rows = build_revision_plan(
            cards,
            extraction_errors,
            (),
        )
        codes = [
            str(row["primary_reason_code"])
            for row in revision_rows
        ]
        return tuple(dict.fromkeys(codes))

    # Clasifica un error técnico según su causa:
    # dependencia faltante, dependencia incorrecta o error general de ejecución.
    @staticmethod
    def _technical_failure_reason_codes(error: Exception) -> tuple[str, ...]:
        text = str(error).casefold()
        if isinstance(error, FileNotFoundError) or "no existe" in text or "not found" in text:
            return ("DEPENDENCY_NOT_FOUND",)
        if any(token in text for token in (
            "no coincide", "mismatch", "desaline", "sha-256", "hash",
            "collection.count", "colección", "collection",
        )):
            return ("DEPENDENCY_MISMATCH",)
        return ("EXECUTION_ERROR",)


    @staticmethod
    # Evita mostrar información sensible, como claves de API,
    # dentro de los mensajes de error.
    def _sanitize_error_message(error: Exception) -> str:
        text = str(error)
        for marker in ("sk-", "OPENAI_API_KEY", "openai_api_key.key", "openai_api_key.enc"):
            if marker in text:
                return "Error técnico sanitizado durante la etapa 03."
        return text
    # Verifica que la entrada recibida por el Agente 03 sea válida,
    # corresponda a esta etapa y tenga las dependencias necesarias.
    def _validate_agent_input(
        self,
        agent_input: AgentInput,
    ) -> None:
        if not isinstance(
            agent_input,
            AgentInput,
        ):
            raise TypeError(
                "agent_input debe ser AgentInput."
            )
        if (
            agent_input.stage_name
            != EXPECTED_STAGE_NAME
        ):
            raise ValueError(
                "AgentInput.stage_name debe ser "
                f"'{EXPECTED_STAGE_NAME}'."
            )
        if agent_input.attempt_number > 2:
            raise ValueError(
                "Agent 03 admite como máximo attempt_number=2."
            )
        if (
            agent_input.mode
            is not ExecutionMode.FULL_RUN
        ):
            raise ValueError(
                "extraction_agent solo admite "
                "ExecutionMode.FULL_RUN en este "
                "entregable."
            )

        missing = [
            name
            for name in _REQUIRED_DEPENDENCIES
            if name not in agent_input.dependencies
        ]
        if missing:
            raise ValueError(
                "Faltan dependencias obligatorias: "
                + ", ".join(missing)
            )
    # Lee y valida las rutas de archivos configuradas
    # para la etapa de extracción.
    def _resolve_paths(
        self,
        policy: Mapping[str, Any],
    ) -> dict[str, Path]:
        raw_paths = self._require_mapping(
            policy,
            "paths",
        )
        self._validate_mapping_keys(
            raw_paths,
            _REQUIRED_PATH_KEYS,
            "policy.paths",
        )
        paths = {
            key: Path(raw_paths[key])
            for key in _REQUIRED_PATH_KEYS
        }
        for key, path in paths.items():
            if not str(path).strip():
                raise ValueError(
                    f"policy.paths.{key} no "
                    "puede estar vacío."
                )
        return paths

    # Comprueba que una dependencia exista
    # y que su archivo no haya cambiado.
    def _validate_dependency(
        self,
        name: str,
        reference: ArtifactReference,
    ) -> None:
        path = Path(reference.path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe la dependencia "
                f"'{name}': {path}"
            )
        actual_hash = (
            self.dependencies.hash_file(
                path
            )
        )
        if actual_hash != reference.hash:
            raise ValueError(
                f"Hash inválido para dependencia "
                f"'{name}'."
            )

    # Verifica que Chroma corresponda al experimento,
    # colección y archivo de chunks correctos.
    def _validate_chroma_manifest(
        self,
        *,
        chroma_manifest: Any,
        experiment_id: str,
        collection_name: str,
        chunks_clean_path: str,
    ) -> None:
        if not isinstance(
            chroma_manifest,
            Mapping,
        ):
            raise TypeError(
                "El manifiesto Chroma debe "
                "ser un mapping."
            )
        if (
            chroma_manifest.get(
                "experiment_id"
            )
            != experiment_id
        ):
            raise ValueError(
                "El manifiesto Chroma pertenece "
                "a otro experimento."
            )
        if (
            chroma_manifest.get(
                "collection_name"
            )
            != collection_name
        ):
            raise ValueError(
                "La colección indicada por la "
                "política no coincide con el "
                "manifiesto Chroma."
            )
        if (
            chroma_manifest.get(
                "chunks_source_file"
            )
            != str(chunks_clean_path)
        ):
            raise ValueError(
                "Chroma fue construido con un "
                "archivo de chunks distinto al "
                "usado por el agente 03."
            )

    # Comprueba que estén disponibles el LLM y el prompt
    # necesarios para generar o reconstruir la extracción.
    def _validate_generation_dependencies(
        self,
    ) -> None:
        if self.dependencies.main_llm is None:
            raise ValueError(
                "main_llm es obligatorio para "
                "reconstruir la extracción."
            )
        if self.dependencies.repair_llm is None:
            raise ValueError(
                "repair_llm es obligatorio para "
                "reconstruir la extracción."
            )
        if (
            self.dependencies.extraction_prompt_builder
            is None
        ):
            raise ValueError(
                "extraction_prompt_builder es "
                "obligatorio para reconstruir."
            )

    # Comprueba que estén disponibles el LLM y el prompt
    # necesarios para clasificar la relevancia de las fichas.
    def _validate_classification_dependencies(
        self,
    ) -> None:
        if self.dependencies.main_llm is None:
            raise ValueError(
                "main_llm es obligatorio para "
                "clasificar relevancia."
            )
        if (
            self.dependencies.relevance_prompt_builder
            is None
        ):
            raise ValueError(
                "relevance_prompt_builder es "
                "obligatorio para clasificar."
            )

    # Recopila los archivos finales generados por la etapa
    # y guarda una referencia y hash de cada uno.
    def _collect_output_artifacts(
        self,
        paths: Mapping[str, Path],
    ) -> dict[str, ArtifactReference]:
        artifacts = {}
        artifact_keys = list(FINAL_OUTPUT_KEYS)
        if "CARDS_REVISION_PLAN_CSV_PATH" not in artifact_keys:
            artifact_keys.append("CARDS_REVISION_PLAN_CSV_PATH")
        for key in artifact_keys:
            if key not in paths:
                continue
            path = Path(paths[key])
            if path.is_file():
                artifacts[key] = (
                    ArtifactReference(
                        path=str(path),
                        hash=(
                            self.dependencies.hash_file(
                                path
                            )
                        ),
                    )
                )
        return artifacts

    # Busca un dato dentro de la configuración
    # y verifica que sea un diccionario.
    @staticmethod
    def _require_mapping(
        mapping: Mapping[str, Any],
        key: str,
    ) -> Mapping[str, Any]:
        value = mapping.get(key)
        if not isinstance(
            value,
            Mapping,
        ):
            raise TypeError(
                f"{key} debe ser un mapping."
            )
        return value
        
    # Verifica que la configuración tenga todos los datos obligatorios.
    # Si falta alguno, genera un error indicando cuál falta.
    @staticmethod
    def _validate_mapping_keys(
        mapping: Mapping[str, Any],
        required: Sequence[str],
        label: str,
    ) -> None:
        missing = [
            key
            for key in required
            if key not in mapping
        ]
        if missing:
            raise ValueError(
                f"Faltan claves en {label}: "
                + ", ".join(missing)
            )
    
    # Convierte un error de extracción en una advertencia
    # para informar el problema sin detener el pipeline.
    @staticmethod
    def _warning_from_error_row(
        error_row: Mapping[str, Any],
    ) -> AgentWarning:
        stage = str(
            error_row.get(
                "stage",
                "unknown",
            )
        )
        source = str(
            error_row.get(
                "source_filename",
                "",
            )
        )
        message = str(
            error_row.get(
                "error_message",
                "",
            )
        )
        return AgentWarning(
            code=(
                "PARTIAL_"
                + stage.upper()
            ),
            severity=(
                WarningSeverity.WARNING
            ),
            blocking=False,
            message=(
                f"{source}: {message}"
                if source
                else message
            ),
        )
