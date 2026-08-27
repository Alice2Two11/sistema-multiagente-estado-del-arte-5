"""Real-service adapter for the approved stage-03 ExtractionAgent.

This module resolves configuration, prompts, parser, message type, model
clients, embeddings, Chroma collection, DataFrame loading, and physical
writers. It contains no scientific extraction rules and does not coordinate
the pipeline transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import pandas as pd

from src.agents.extraction_agent import (
    ExtractionAgentDependencies,
)
from src.io.credentials import resolve_openai_api_key
from src.contracts.agent_input import (
    AgentContext,
    AgentInput,
    ArtifactReference,
    ExecutionMode,
)
from src.state.fingerprints import sha256_file
from src.tools.extraction.card_validation import (
    CARD_LIST_FIELDS,
    CARD_REQUIRED_FIELDS,
)
from src.tools.extraction.chunk_validation import (
    RAG_CLEAN_VALIDATION_VERSION,
)
from src.tools.extraction.relevance_classification import (
    RELEVANCE_CLASSIFICATION_RESPONSE_FIELDS,
)


RELEVANCE_PROMPT_VERSION = (
    "v4_topic_profile_relevance_domain_agnostic"
)

ERROR_COLUMNS = [
    "source_filename",
    "stage",
    "error_type",
    "error_message",
    "created_at",
]

TRACE_COLUMNS = [
    "source_filename",
    "retrieval_query",
    "chunk_id",
    "chunk_index",
    "score",
    "retrieval_profile",
    "retrieval_mode",
]


@dataclass(frozen=True, slots=True)
class ExtractionRuntimeConfiguration:
    project_dir: Path
    experiment_id: str
    run_id: str
    experiment_dir: Path
    outputs_dir: Path
    chunks_clean_path: Path
    chroma_dir: Path
    chroma_manifest_path: Path
    chroma_collection_name: str
    openai_model: str
    embedding_model_name: str
    experiment_profile: Mapping[str, Any]
    topic_profile: Mapping[str, Any]
    generation_profile: Mapping[str, Any]
    rag_policy: Mapping[str, Any]
    extraction_policy: Mapping[str, Any]
    retrieval_config: Mapping[str, Any]
    extraction_prompt_version: str
    relevance_prompt_version: str
    kb_schema_version: str
    rag_clean_validation_version: str
    code_root: Path | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "project_dir",
            "experiment_dir",
            "outputs_dir",
            "chunks_clean_path",
            "chroma_dir",
            "chroma_manifest_path",
        ):
            object.__setattr__(
                self,
                field_name,
                Path(getattr(self, field_name)),
            )

        resolved_code_root = (
            Path(self.code_root)
            if self.code_root is not None
            else Path(__file__).resolve().parents[2]
        ).resolve()
        object.__setattr__(
            self,
            "code_root",
            resolved_code_root,
        )

        for field_name in (
            "experiment_id",
            "run_id",
            "chroma_collection_name",
            "openai_model",
            "embedding_model_name",
            "extraction_prompt_version",
            "relevance_prompt_version",
            "kb_schema_version",
            "rag_clean_validation_version",
        ):
            value = str(
                getattr(self, field_name)
            ).strip()
            if not value:
                raise ValueError(
                    f"{field_name} no puede estar vacío."
                )
            object.__setattr__(
                self,
                field_name,
                value,
            )

        for field_name in (
            "experiment_profile",
            "topic_profile",
            "generation_profile",
            "rag_policy",
            "extraction_policy",
            "retrieval_config",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"{field_name} debe ser un mapping."
                )
            object.__setattr__(
                self,
                field_name,
                dict(value),
            )

    @property
    def scientific_extraction_dir(self) -> Path:
        return (
            self.outputs_dir
            / "01_scientific_extraction"
        )

    @property
    def knowledge_base_dir(self) -> Path:
        return (
            self.outputs_dir
            / "02_scientific_knowledge_base"
        )

    def agent_paths(self) -> dict[str, str]:
        extraction_dir = (
            self.scientific_extraction_dir
        )
        kb_dir = self.knowledge_base_dir

        return {
            "OUTPUTS_DIR": str(
                self.outputs_dir
            ),
            "DIR_EXTRACTION": str(
                extraction_dir
            ),
            "DIR_KB": str(kb_dir),
            "CARDS_JSONL_PATH": str(
                extraction_dir
                / "scientific_cards.jsonl"
            ),
            "CARDS_SUMMARY_CSV_PATH": str(
                extraction_dir
                / "scientific_cards_summary.csv"
            ),
            "CARDS_ERRORS_CSV_PATH": str(
                extraction_dir
                / "scientific_cards_errors.csv"
            ),
            "CARDS_QUALITY_CSV_PATH": str(
                extraction_dir
                / "scientific_cards_quality_check.csv"
            ),
            "CARDS_REVISION_PLAN_CSV_PATH": str(
                extraction_dir
                / "scientific_cards_revision_plan.csv"
            ),
            "CARDS_REVIEW_EXCLUSION_AUDIT_CSV_PATH": str(
                extraction_dir
                / "scientific_cards_review_exclusion_audit.csv"
            ),
            "CARDS_QUARANTINE_AUDIT_CSV_PATH": str(
                extraction_dir
                / "scientific_cards_quarantine_audit.csv"
            ),
            "RETRIEVAL_TRACE_CSV_PATH": str(
                extraction_dir
                / "extraction_retrieval_trace.csv"
            ),
            "EXTRACTION_MANIFEST_PATH": str(
                extraction_dir
                / "scientific_extraction_manifest.json"
            ),
            "CHUNKS_VALIDATION_REPORT_PATH": str(
                extraction_dir
                / "chunks_clean_validation_report.json"
            ),
            "KB_CSV_PATH": str(
                kb_dir
                / "scientific_knowledge_base.csv"
            ),
            "KB_JSONL_PATH": str(
                kb_dir
                / "scientific_knowledge_base.jsonl"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_root": str(
                self.code_root
            ),
            "project_dir": str(
                self.project_dir
            ),
            "experiment_id": (
                self.experiment_id
            ),
            "run_id": self.run_id,
            "experiment_dir": str(
                self.experiment_dir
            ),
            "outputs_dir": str(
                self.outputs_dir
            ),
            "chunks_clean_path": str(
                self.chunks_clean_path
            ),
            "chroma_dir": str(
                self.chroma_dir
            ),
            "chroma_manifest_path": str(
                self.chroma_manifest_path
            ),
            "chroma_collection_name": (
                self.chroma_collection_name
            ),
            "openai_model": (
                self.openai_model
            ),
            "embedding_model_name": (
                self.embedding_model_name
            ),
            "experiment_profile": dict(
                self.experiment_profile
            ),
            "topic_profile": dict(
                self.topic_profile
            ),
            "generation_profile": dict(
                self.generation_profile
            ),
            "rag_policy": dict(
                self.rag_policy
            ),
            "extraction_policy": dict(
                self.extraction_policy
            ),
            "retrieval_config": dict(
                self.retrieval_config
            ),
            "extraction_prompt_version": (
                self.extraction_prompt_version
            ),
            "relevance_prompt_version": (
                self.relevance_prompt_version
            ),
            "kb_schema_version": (
                self.kb_schema_version
            ),
            "rag_clean_validation_version": (
                self.rag_clean_validation_version
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "ExtractionRuntimeConfiguration":
        return cls(
            code_root=Path(
                data.get(
                    "code_root",
                    data["project_dir"],
                )
            ),
            project_dir=Path(
                data["project_dir"]
            ),
            experiment_id=str(
                data["experiment_id"]
            ),
            run_id=str(data["run_id"]),
            experiment_dir=Path(
                data["experiment_dir"]
            ),
            outputs_dir=Path(
                data["outputs_dir"]
            ),
            chunks_clean_path=Path(
                data["chunks_clean_path"]
            ),
            chroma_dir=Path(
                data["chroma_dir"]
            ),
            chroma_manifest_path=Path(
                data["chroma_manifest_path"]
            ),
            chroma_collection_name=str(
                data[
                    "chroma_collection_name"
                ]
            ),
            openai_model=str(
                data["openai_model"]
            ),
            embedding_model_name=str(
                data[
                    "embedding_model_name"
                ]
            ),
            experiment_profile=dict(
                data["experiment_profile"]
            ),
            topic_profile=dict(
                data["topic_profile"]
            ),
            generation_profile=dict(
                data["generation_profile"]
            ),
            rag_policy=dict(
                data["rag_policy"]
            ),
            extraction_policy=dict(
                data["extraction_policy"]
            ),
            retrieval_config=dict(
                data["retrieval_config"]
            ),
            extraction_prompt_version=str(
                data[
                    "extraction_prompt_version"
                ]
            ),
            relevance_prompt_version=str(
                data[
                    "relevance_prompt_version"
                ]
            ),
            kb_schema_version=str(
                data["kb_schema_version"]
            ),
            rag_clean_validation_version=str(
                data[
                    "rag_clean_validation_version"
                ]
            ),
        )


@dataclass(frozen=True, slots=True)
class ProjectRuntimeComponents:
    extraction_prompt_builder: Callable[
        ..., str
    ]
    relevance_prompt_builder: Callable[
        ..., str
    ]
    json_parser: Callable[[Any], Any]
    human_message_factory: Callable[
        ..., Any
    ]


@dataclass(frozen=True, slots=True)
class ExtractionRuntime:
    configuration: (
        ExtractionRuntimeConfiguration
    )
    dependencies: (
        ExtractionAgentDependencies
    )
    dataframe: pd.DataFrame
    collection: Any
    main_llm: Any
    repair_llm: Any
    embedding_function: Any


def _load_module_from_path(
    module_name: str,
    path: str | Path,
) -> Any:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"No existe el módulo requerido: {source}"
        )

    unique_name = (
        "_extraction_runtime_"
        + module_name
        + "_"
        + str(abs(hash(source.resolve())))
    )
    spec = importlib.util.spec_from_file_location(
        unique_name,
        source,
    )
    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"No se pudo cargar: {source}"
        )

    module = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(module)
    return module


def resolve_human_message_factory() -> Callable[
    ..., Any
]:
    try:
        from langchain_core.messages import (
            HumanMessage,
        )
    except ImportError as error:
        raise RuntimeError(
            "Falta langchain-core para "
            "resolver HumanMessage."
        ) from error

    return HumanMessage


def resolve_project_runtime_components(
    project_dir: str | Path,
    *,
    human_message_factory: Callable[
        ..., Any
    ] | None = None,
) -> ProjectRuntimeComponents:
    source_dir = (
        Path(project_dir).resolve()
        / "src"
    )
    prompts_path = (
        source_dir / "prompts.py"
    )
    llm_utils_path = (
        source_dir / "llm_utils.py"
    )

    if not prompts_path.is_file():
        raise FileNotFoundError(
            "No existe PROJECT_DIR/src/prompts.py. "
            "Ejecuta nuevamente 00_setup_config."
        )
    if not llm_utils_path.is_file():
        raise FileNotFoundError(
            "No existe PROJECT_DIR/src/llm_utils.py. "
            "Ejecuta nuevamente 00_setup_config."
        )

    prompts_module = _load_module_from_path(
        "project_prompts",
        prompts_path,
    )
    llm_utils_module = _load_module_from_path(
        "project_llm_utils_runtime",
        llm_utils_path,
    )

    extraction_builder = getattr(
        prompts_module,
        "build_scientific_extraction_prompt",
        None,
    )
    relevance_builder = getattr(
        prompts_module,
        "build_relevance_classification_prompt",
        None,
    )
    json_parser = getattr(
        llm_utils_module,
        "parse_json_safely",
        None,
    )

    if not callable(
        extraction_builder
    ):
        raise AttributeError(
            "PROJECT_DIR/src/prompts.py no "
            "expone build_scientific_extraction_prompt."
        )
    if not callable(
        relevance_builder
    ):
        raise AttributeError(
            "PROJECT_DIR/src/prompts.py no "
            "expone build_relevance_classification_prompt."
        )
    if not callable(json_parser):
        raise AttributeError(
            "PROJECT_DIR/src/llm_utils.py no "
            "expone parse_json_safely."
        )

    return ProjectRuntimeComponents(
        extraction_prompt_builder=(
            extraction_builder
        ),
        relevance_prompt_builder=(
            relevance_builder
        ),
        json_parser=json_parser,
        human_message_factory=(
            human_message_factory
            or resolve_human_message_factory()
        ),
    )

def assert_generation_safe_configuration(
    configuration: (
        ExtractionRuntimeConfiguration
    ),
) -> None:
    rag_policy = dict(
        configuration.rag_policy
    )
    nested = rag_policy.get(
        "ground_truth_policy"
    )
    if isinstance(nested, Mapping):
        for key in (
            "use_ground_truth_for_generation",
            "use_ground_truth_for_rag",
            "use_ground_truth_for_verification",
            "use_ground_truth_for_evaluation",
        ):
            rag_policy.setdefault(
                key,
                nested.get(key),
            )

    for key in (
        "use_ground_truth_for_generation",
        "use_ground_truth_for_rag",
        "use_ground_truth_for_verification",
    ):
        if rag_policy.get(key) is not False:
            raise ValueError(
                f"{key} debe ser False "
                "durante la generación."
            )

    if rag_policy.get(
        "ground_truth_usage"
    ) != "evaluation_only":
        raise ValueError(
            "ground_truth_usage debe ser "
            "'evaluation_only'."
        )

    def reject_payload(
        value: Any,
        location: str,
    ) -> None:
        if isinstance(value, Mapping):
            for key, nested_value in (
                value.items()
            ):
                normalized = str(
                    key
                ).casefold()
                if (
                    "ground_truth"
                    in normalized
                    and any(
                        marker in normalized
                        for marker in (
                            "path",
                            "dir",
                            "file",
                            "document",
                            "content",
                            "text",
                            "corpus",
                        )
                    )
                ):
                    raise ValueError(
                        f"{location} no debe "
                        "exponer rutas o contenido "
                        "Ground Truth."
                    )
                reject_payload(
                    nested_value,
                    f"{location}.{key}",
                )
        elif isinstance(
            value,
            (list, tuple),
        ):
            for index, nested_value in enumerate(
                value
            ):
                reject_payload(
                    nested_value,
                    f"{location}[{index}]",
                )

    reject_payload(
        configuration.to_dict(),
        "configuration",
    )


def load_runtime_configuration(
    project_dir: str | Path,
    *,
    code_root: str | Path | None = None,
) -> ExtractionRuntimeConfiguration:
    """Load real notebook-00 configuration and validate notebook-01/02 outputs."""

    data_root = Path(
        project_dir
    ).resolve()
    resolved_code_root = Path(
        code_root
        if code_root is not None
        else Path(__file__).resolve().parents[2]
    ).resolve()

    active_path = (
        data_root
        / "active_experiment.json"
    )
    adapter_path = (
        resolved_code_root
        / "src/adapters/extraction_runtime.py"
    )

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"No existe PROJECT_DIR: {data_root}"
        )
    if not active_path.is_file():
        raise FileNotFoundError(
            "active_experiment.json inexistente: "
            f"{active_path}"
        )
    if not adapter_path.is_file():
        raise FileNotFoundError(
            "No existe el adaptador dentro "
            f"de CODE_ROOT: {adapter_path}"
        )

    os.environ[
        "THESIS_CODE_ROOT"
    ] = str(resolved_code_root)
    os.environ[
        "THESIS_PROJECT_DIR"
    ] = str(data_root)

    code_text = str(
        resolved_code_root
    )
    while code_text in sys.path:
        sys.path.remove(code_text)
    sys.path.insert(0, code_text)
    importlib.invalidate_caches()

    for module_name in (
        "src.config.generation_policy_config",
        "src.config.common_config",
    ):
        sys.modules.pop(
            module_name,
            None,
        )

    common_config = importlib.import_module(
        "src.config.common_config"
    )
    generation_config = importlib.import_module(
        "src.config.generation_policy_config"
    )

    if (
        Path(
            common_config.PROJECT_DIR
        ).resolve()
        != data_root
    ):
        raise ValueError(
            "PROJECT_DIR no coincide con "
            "src.config.common_config.PROJECT_DIR."
        )
    if (
        Path(
            common_config.CODE_ROOT
        ).resolve()
        != resolved_code_root
    ):
        raise ValueError(
            "CODE_ROOT no coincide con "
            "src.config.common_config.CODE_ROOT."
        )

    active = (
        common_config
        .get_active_experiment_config()
    )
    extraction_policy = (
        generation_config
        .get_extraction_policy()
    )
    extraction_paths = (
        generation_config
        .get_extraction_paths()
    )
    retrieval_config = dict(
        generation_config.RETRIEVAL_CONFIG
    )

    experiment_dir = Path(
        common_config.EXPERIMENT_DIR
    )
    chunks_path = Path(
        extraction_paths[
            "chunks_clean_path"
        ]
    )
    chroma_dir = Path(
        extraction_paths[
            "chroma_dir"
        ]
    )
    manifest_path = Path(
        extraction_paths[
            "chroma_manifest_path"
        ]
    )
    outputs_dir = Path(
        common_config.OUTPUTS_DIR
    )

    if not experiment_dir.is_dir():
        raise FileNotFoundError(
            "experiment_dir inexistente: "
            f"{experiment_dir}"
        )
    if not chunks_path.is_file():
        raise FileNotFoundError(
            "chunks_clean_for_rag.csv inexistente: "
            f"{chunks_path}. Ejecuta el notebook 01."
        )
    if not chroma_dir.is_dir():
        raise FileNotFoundError(
            "Directorio real de Chroma inexistente: "
            f"{chroma_dir}. Ejecuta el notebook 02."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "chroma_index_manifest.json inexistente: "
            f"{manifest_path}. Ejecuta el notebook 02."
        )
    if not outputs_dir.is_dir():
        raise FileNotFoundError(
            "05_outputs inexistente: "
            f"{outputs_dir}. Ejecuta el notebook 00."
        )

    try:
        chroma_manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "chroma_index_manifest.json "
            "contiene JSON inválido."
        ) from error

    if not isinstance(
        chroma_manifest,
        Mapping,
    ):
        raise TypeError(
            "chroma_index_manifest.json "
            "debe contener un objeto."
        )

    expected_manifest = {
        "experiment_id": (
            common_config.EXPERIMENT_ID
        ),
        "collection_name": (
            common_config
            .CHROMA_COLLECTION_NAME
        ),
        "chunks_source_file": str(
            chunks_path
        ),
    }
    for key, expected_value in (
        expected_manifest.items()
    ):
        if (
            str(
                chroma_manifest.get(
                    key,
                    ""
                )
            )
            != str(expected_value)
        ):
            raise ValueError(
                "Configuración inconsistente: "
                f"{key} del manifiesto Chroma "
                "no coincide con el experimento activo."
            )

    if (
        "chroma_dir"
        in chroma_manifest
        and Path(
            str(
                chroma_manifest[
                    "chroma_dir"
                ]
            )
        ).resolve()
        != chroma_dir.resolve()
    ):
        raise ValueError(
            "Configuración inconsistente: "
            "chroma_dir del manifiesto no coincide."
        )

    topic_profile = dict(
        active.get(
            "topic_profile",
            {},
        )
    )
    project_generation_path = (
        data_root / "src" / "generation_config.py"
    )
    if not project_generation_path.is_file():
        raise FileNotFoundError(
            "No existe PROJECT_DIR/src/generation_config.py. "
            "Ejecuta nuevamente 00_setup_config."
        )
    import types
    previous_project_config = sys.modules.get("config")
    project_config_proxy = types.ModuleType("config")
    project_config_proxy.GENERATION_PROFILE = dict(
        active.get("generation_profile", {})
    )
    project_config_proxy.RAG_POLICY = dict(
        active.get("rag_policy", {})
    )
    sys.modules["config"] = project_config_proxy
    importlib.invalidate_caches()
    try:
        project_generation_module = _load_module_from_path(
            "project_generation_config",
            project_generation_path,
        )
    finally:
        if previous_project_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous_project_config
        importlib.invalidate_caches()
    get_generation_profile = getattr(
        project_generation_module,
        "get_generation_profile",
        None,
    )
    if not callable(get_generation_profile):
        raise AttributeError(
            "PROJECT_DIR/src/generation_config.py no expone "
            "get_generation_profile()."
        )
    generation_profile = dict(
        get_generation_profile()
    )
    rag_policy = dict(
        active["rag_policy"]
    )
    experiment_profile = dict(
        topic_profile
    )
    experiment_profile.setdefault(
        "experiment_id",
        common_config.EXPERIMENT_ID,
    )
    experiment_profile.setdefault(
        "output_language",
        generation_profile.get(
            "output_language_label",
            generation_profile.get(
                "output_language",
                "español académico",
            ),
        ),
    )

    configuration = (
        ExtractionRuntimeConfiguration(
            code_root=resolved_code_root,
            project_dir=data_root,
            experiment_id=(
                common_config.EXPERIMENT_ID
            ),
            run_id=common_config.RUN_ID,
            experiment_dir=(
                experiment_dir
            ),
            outputs_dir=outputs_dir,
            chunks_clean_path=chunks_path,
            chroma_dir=chroma_dir,
            chroma_manifest_path=(
                manifest_path
            ),
            chroma_collection_name=(
                common_config
                .CHROMA_COLLECTION_NAME
            ),
            openai_model=(
                common_config.OPENAI_MODEL
            ),
            embedding_model_name=(
                common_config
                .EMBEDDING_MODEL_NAME
            ),
            experiment_profile=(
                experiment_profile
            ),
            topic_profile=topic_profile,
            generation_profile=(
                generation_profile
            ),
            rag_policy=rag_policy,
            extraction_policy=(
                extraction_policy
            ),
            retrieval_config=(
                retrieval_config
            ),
            extraction_prompt_version=(
                generation_config
                .EXTRACTION_PROMPT_VERSION
            ),
            relevance_prompt_version=(
                RELEVANCE_PROMPT_VERSION
            ),
            kb_schema_version=(
                generation_config
                .KB_SCHEMA_VERSION
            ),
            rag_clean_validation_version=(
                RAG_CLEAN_VALIDATION_VERSION
            ),
        )
    )
    assert_generation_safe_configuration(
        configuration
    )
    return configuration

def build_chat_clients(
    configuration: (
        ExtractionRuntimeConfiguration
    ),
    *,
    api_key: str,
    chat_model_factory: Callable[
        ..., Any
    ] | None = None,
) -> tuple[Any, Any]:
    if not isinstance(api_key, str) or (
        not api_key.strip()
    ):
        raise ValueError(
            "api_key no puede estar vacío."
        )

    factory = chat_model_factory

    if factory is None:
        try:
            from langchain_openai import (
                ChatOpenAI,
            )
        except ImportError as error:
            raise RuntimeError(
                "Falta langchain-openai."
            ) from error
        factory = ChatOpenAI

    extraction_policy = (
        configuration.extraction_policy
    )

    json_model_kwargs = {
        "response_format": {
            "type": "json_object",
        }
    }

    main_llm = factory(
        model=configuration.openai_model,
        temperature=float(
            extraction_policy[
                "temperature"
            ]
        ),
        api_key=api_key,
        model_kwargs=json_model_kwargs,
    )

    repair_llm = factory(
        model=configuration.openai_model,
        temperature=float(
            extraction_policy[
                "repair_temperature"
            ]
        ),
        api_key=api_key,
        model_kwargs=json_model_kwargs,
    )

    return main_llm, repair_llm


def build_embedding_function(
    model_name: str,
    *,
    embedding_factory: Callable[
        ..., Any
    ] | None = None,
) -> Any:
    factory = embedding_factory

    if factory is None:
        try:
            from chromadb.utils import (
                embedding_functions,
            )
        except ImportError as error:
            raise RuntimeError(
                "Falta chromadb."
            ) from error
        factory = (
            embedding_functions
            .SentenceTransformerEmbeddingFunction
        )

    return factory(
        model_name=model_name
    )


def open_chroma_collection(
    *,
    chroma_dir: str | Path,
    collection_name: str,
    embedding_function: Any,
    chroma_client_factory: Callable[
        ..., Any
    ] | None = None,
) -> Any:
    resolved_chroma_dir = Path(
        chroma_dir
    ).resolve()
    if not resolved_chroma_dir.is_dir():
        raise FileNotFoundError(
            "Directorio real de Chroma inexistente: "
            f"{resolved_chroma_dir}"
        )

    factory = chroma_client_factory

    if factory is None:
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "Falta chromadb."
            ) from error
        factory = chromadb.PersistentClient

    client = factory(
        path=str(
            resolved_chroma_dir
        )
    )
    try:
        return client.get_collection(
            name=collection_name,
            embedding_function=(
                embedding_function
            ),
        )
    except Exception as error:
        raise RuntimeError(
            "Colección Chroma inexistente o "
            "inaccesible: "
            f"{collection_name!r} en "
            f"{resolved_chroma_dir}. Ejecuta "
            "nuevamente el notebook 02."
        ) from error


def load_chunks_dataframe(
    path: str | Path,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"No existe: {source}"
        )
    return pd.read_csv(source)


def build_extraction_agent_dependencies(
    *,
    main_llm: Any,
    repair_llm: Any,
    extraction_prompt_builder: (
        Callable[..., str] | None
    ),
    relevance_prompt_builder: (
        Callable[..., str] | None
    ),
    json_parser: Callable[
        [Any], Any
    ] = json.loads,
    message_factory: Callable[
        ..., Any
    ] | None = None,
    dataframe_loader: Callable[
        [str | Path], pd.DataFrame
    ] = load_chunks_dataframe,
    collection: Any = None,
) -> ExtractionAgentDependencies:
    resolved_message_factory = (
        message_factory
        if message_factory is not None
        else (
            resolve_human_message_factory()
            if (
                main_llm is not None
                or repair_llm is not None
            )
            else None
        )
    )

    load_collection = (
        (lambda _agent_input: collection)
        if collection is not None
        else None
    )

    return ExtractionAgentDependencies(
        main_llm=main_llm,
        repair_llm=repair_llm,
        extraction_prompt_builder=(
            extraction_prompt_builder
        ),
        relevance_prompt_builder=(
            relevance_prompt_builder
        ),
        json_parser=json_parser,
        message_factory=(
            resolved_message_factory
            if resolved_message_factory
            is not None
            else (
                lambda content: {
                    "content": content
                }
            )
        ),
        load_collection=load_collection,
        load_dataframe=dataframe_loader,
    )


def build_extraction_runtime(
    configuration: (
        ExtractionRuntimeConfiguration
    ),
    *,
    api_key: str,
    components: (
        ProjectRuntimeComponents | None
    ) = None,
    chat_model_factory: Callable[
        ..., Any
    ] | None = None,
    embedding_factory: Callable[
        ..., Any
    ] | None = None,
    chroma_client_factory: Callable[
        ..., Any
    ] | None = None,
) -> ExtractionRuntime:
    assert_generation_safe_configuration(
        configuration
    )

    resolved_components = (
        components
        or resolve_project_runtime_components(
            configuration.project_dir
        )
    )
    dataframe = load_chunks_dataframe(
        configuration.chunks_clean_path
    )
    main_llm, repair_llm = (
        build_chat_clients(
            configuration,
            api_key=api_key,
            chat_model_factory=(
                chat_model_factory
            ),
        )
    )
    embedding_function = (
        build_embedding_function(
            configuration
            .embedding_model_name,
            embedding_factory=(
                embedding_factory
            ),
        )
    )
    collection = open_chroma_collection(
        chroma_dir=(
            configuration.chroma_dir
        ),
        collection_name=(
            configuration
            .chroma_collection_name
        ),
        embedding_function=(
            embedding_function
        ),
        chroma_client_factory=(
            chroma_client_factory
        ),
    )

    count = getattr(
        collection,
        "count",
        None,
    )
    if callable(count) and (
        count() != len(dataframe)
    ):
        raise ValueError(
            "La colección Chroma y "
            "chunks_clean_for_rag.csv no "
            "contienen la misma cantidad "
            "de fragmentos."
        )

    dependencies = (
        build_extraction_agent_dependencies(
            main_llm=main_llm,
            repair_llm=repair_llm,
            extraction_prompt_builder=(
                resolved_components
                .extraction_prompt_builder
            ),
            relevance_prompt_builder=(
                resolved_components
                .relevance_prompt_builder
            ),
            json_parser=(
                resolved_components
                .json_parser
            ),
            message_factory=(
                resolved_components
                .human_message_factory
            ),
            dataframe_loader=(
                load_chunks_dataframe
            ),
            collection=collection,
        )
    )

    return ExtractionRuntime(
        configuration=configuration,
        dependencies=dependencies,
        dataframe=dataframe,
        collection=collection,
        main_llm=main_llm,
        repair_llm=repair_llm,
        embedding_function=(
            embedding_function
        ),
    )


def build_agent_input(
    configuration: (
        ExtractionRuntimeConfiguration
    ),
    *,
    attempt_number: int = 1,
    runtime_resources: (
        Mapping[str, Any] | None
    ) = None,
    previous_attempt: Any = None,
) -> AgentInput:
    assert_generation_safe_configuration(
        configuration
    )

    if not configuration.chunks_clean_path.is_file():
        raise FileNotFoundError(
            "No existe chunks_clean_for_rag.csv: "
            f"{configuration.chunks_clean_path}"
        )
    if not configuration.chroma_manifest_path.is_file():
        raise FileNotFoundError(
            "No existe chroma_index_manifest.json: "
            f"{configuration.chroma_manifest_path}"
        )

    policy = configuration.extraction_policy
    retrieval = (
        configuration.retrieval_config
    )

    agent_policy = {
        "paths": (
            configuration.agent_paths()
        ),
        "signature": {
            "experiment_dir": str(
                configuration.experiment_dir
            ),
            "chroma_collection_name": (
                configuration
                .chroma_collection_name
            ),
            "openai_model": (
                configuration.openai_model
            ),
            "embedding_model_name": (
                configuration
                .embedding_model_name
            ),
            "experiment_profile": dict(
                configuration
                .experiment_profile
            ),
            "topic_profile": dict(
                configuration.topic_profile
            ),
            "generation_profile": dict(
                configuration
                .generation_profile
            ),
            "rag_policy": dict(
                configuration.rag_policy
            ),
            "extraction_policy": dict(
                policy
            ),
            "card_required_fields": list(
                CARD_REQUIRED_FIELDS
            ),
            "card_list_fields": list(
                CARD_LIST_FIELDS
            ),
            "classification_fields": list(
                RELEVANCE_CLASSIFICATION_RESPONSE_FIELDS
            ),
            "extraction_prompt_version": (
                configuration
                .extraction_prompt_version
            ),
            "relevance_prompt_version": (
                configuration
                .relevance_prompt_version
            ),
            "kb_schema_version": (
                configuration
                .kb_schema_version
            ),
            "rag_clean_validation_version": (
                configuration
                .rag_clean_validation_version
            ),
        },
        "retrieval": {
            "queries": list(
                policy[
                    "retrieval_queries"
                ]
            ),
            "profile": str(
                policy[
                    "retrieval_profile"
                ]
            ),
            "profile_config": dict(
                retrieval
            ),
            "max_chunks_per_paper": int(
                policy[
                    "max_chunks_per_paper"
                ]
            ),
            "max_context_chars": int(
                policy[
                    "max_context_chars"
                ]
            ),
            "repair_max_chunks_per_paper": int(
                policy[
                    "repair_max_chunks_per_paper"
                ]
            ),
            "repair_max_context_chars": int(
                policy[
                    "repair_max_context_chars"
                ]
            ),
        },
        "auto_rebuild_extraction": bool(
            policy["auto_rebuild"]
        ),
        "force_rebuild_extraction": bool(
            policy["force_rebuild"]
        ),
        "title_repair_first_chunks": int(
            policy[
                "title_repair_first_chunks"
            ]
        ),
        "error_columns": list(
            ERROR_COLUMNS
        ),
        "trace_columns": list(
            TRACE_COLUMNS
        ),
        "next_stage": (
            "04_analisis_tematico"
        ),
    }

    return AgentInput(
        experiment_id=(
            configuration.experiment_id
        ),
        run_id=configuration.run_id,
        stage_name=(
            "03_agente_extraccion_kb"
        ),
        attempt_number=attempt_number,
        mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(
            allowed_tools=(
                "openai_chat",
                "chroma_retrieval",
            ),
            output_directory=str(
                configuration.outputs_dir
            ),
            runtime_resources=dict(
                runtime_resources or {}
            ),
        ),
        dependencies={
            "chunks_clean": (
                ArtifactReference(
                    path=str(
                        configuration
                        .chunks_clean_path
                    ),
                    hash=sha256_file(
                        configuration
                        .chunks_clean_path
                    ),
                )
            ),
            "chroma_manifest": (
                ArtifactReference(
                    path=str(
                        configuration
                        .chroma_manifest_path
                    ),
                    hash=sha256_file(
                        configuration
                        .chroma_manifest_path
                    ),
                )
            ),
        },
        policy=agent_policy,
        previous_attempt=previous_attempt,
    )
