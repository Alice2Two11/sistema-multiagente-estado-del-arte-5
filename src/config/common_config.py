"""Common configuration shared by migrated agents.

Code is imported from ``CODE_ROOT`` while active experiment data remains under
``PROJECT_DIR``. The public configuration view preserves technical RAG fields
but removes Ground Truth paths, files, documents, and content.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


COMMON_CONFIG_SCHEMA_VERSION = "1.2"
PIPELINE_STATE_SCHEMA_VERSION = "1.0"
AGENT_CONTRACT_SCHEMA_VERSION = "1.0"

_MODULE_CODE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(
    os.environ.get(
        "THESIS_CODE_ROOT",
        _MODULE_CODE_ROOT,
    )
).resolve()
PROJECT_DIR = Path(
    os.environ.get(
        "THESIS_PROJECT_DIR",
        CODE_ROOT,
    )
).resolve()
SRC_DIR = CODE_ROOT / "src"
ACTIVE_EXPERIMENT_PATH = (
    PROJECT_DIR / "active_experiment.json"
)
PROJECT_CONFIG_PATH = PROJECT_DIR / "src" / "config.py"
PROJECT_GENERATION_CONFIG_PATH = (
    PROJECT_DIR / "src" / "generation_config.py"
)
PROJECT_RAG_POLICY_PATH = PROJECT_DIR / "src" / "rag_policy.py"


def get_project_module_paths() -> dict[str, Path]:
    """Return existing project-owned configuration modules without replacing them."""
    return {
        "config": PROJECT_CONFIG_PATH,
        "generation_config": PROJECT_GENERATION_CONFIG_PATH,
        "rag_policy": PROJECT_RAG_POLICY_PATH,
    }

_REQUIRED_FALSE_FLAGS = (
    "use_ground_truth_for_generation",
    "use_ground_truth_for_rag",
    "use_ground_truth_for_verification",
)
_FORBIDDEN_GT_PAYLOAD_MARKERS = (
    "path",
    "dir",
    "directory",
    "file",
    "document",
    "documents",
    "content",
    "text",
    "corpus",
    "reference",
)


def _require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string"
        )
    normalized = value.strip()
    if not normalized:
        raise ValueError(
            f"{field_name} must not be empty"
        )
    return normalized


def _require_bool(
    value: Any,
    field_name: str,
) -> bool:
    if not isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be bool"
        )
    return value


def _require_positive_int(
    value: Any,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{field_name} must be an integer"
        )
    if value < 1:
        raise ValueError(
            f"{field_name} must be greater than or equal to 1"
        )
    return value


def _require_number(
    value: Any,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise TypeError(
            f"{field_name} must be numeric"
        )
    normalized = float(value)
    if not minimum <= normalized <= maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return normalized


def _validate_string_list(
    value: Any,
    field_name: str,
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(
            f"{field_name} must be a list"
        )
    return [
        _require_non_empty_string(
            item,
            f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    ]


def _is_forbidden_ground_truth_key(
    key: Any,
) -> bool:
    normalized = str(key).casefold()
    if "ground_truth" not in normalized:
        return False
    if normalized in {
        "ground_truth_usage",
        "ground_truth_policy",
        "use_ground_truth_for_generation",
        "use_ground_truth_for_rag",
        "use_ground_truth_for_verification",
        "use_ground_truth_for_evaluation",
        "ground_truth_indexed",
    }:
        return False
    return any(
        marker in normalized
        for marker in _FORBIDDEN_GT_PAYLOAD_MARKERS
    )


def _safe_public_copy(
    value: Any,
) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, nested in value.items():
            if _is_forbidden_ground_truth_key(
                key
            ):
                continue
            safe[str(key)] = (
                _safe_public_copy(nested)
            )
        return safe
    if isinstance(value, list):
        return [
            _safe_public_copy(item)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _safe_public_copy(item)
            for item in value
        ]
    return deepcopy(value)


def _validate_retrieval_profiles(
    value: Any,
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "rag_policy.retrieval_profiles must be a mapping"
        )
    profiles: dict[str, dict[str, int]] = {}
    for raw_name, raw_profile in value.items():
        name = _require_non_empty_string(
            raw_name,
            "rag_policy.retrieval_profiles key",
        )
        if not isinstance(
            raw_profile,
            Mapping,
        ):
            raise TypeError(
                f"rag_policy.retrieval_profiles.{name} must be a mapping"
            )
        required = {
            "top_k",
            "fetch_k",
            "max_per_source",
        }
        missing = sorted(
            required - set(raw_profile)
        )
        if missing:
            raise ValueError(
                f"retrieval profile {name!r} is missing keys: {missing}"
            )
        profile = {
            "top_k": _require_positive_int(
                raw_profile["top_k"],
                f"rag_policy.retrieval_profiles.{name}.top_k",
            ),
            "fetch_k": _require_positive_int(
                raw_profile["fetch_k"],
                f"rag_policy.retrieval_profiles.{name}.fetch_k",
            ),
            "max_per_source": _require_positive_int(
                raw_profile[
                    "max_per_source"
                ],
                f"rag_policy.retrieval_profiles.{name}.max_per_source",
            ),
        }
        if (
            profile["fetch_k"]
            < profile["top_k"]
        ):
            raise ValueError(
                f"retrieval profile {name!r} fetch_k must be "
                "greater than or equal to top_k"
            )
        profiles[name] = profile
    if not profiles:
        raise ValueError(
            "rag_policy.retrieval_profiles must not be empty"
        )
    return profiles


def _validate_safe_rag_policy(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "rag_policy must be a mapping"
        )

    policy = _safe_public_copy(value)

    nested_policy = policy.get(
        "ground_truth_policy"
    )
    if isinstance(
        nested_policy,
        Mapping,
    ):
        for key in (
            *_REQUIRED_FALSE_FLAGS,
            "use_ground_truth_for_evaluation",
        ):
            if (
                key not in policy
                and key in nested_policy
            ):
                policy[key] = nested_policy[
                    key
                ]

    for flag in _REQUIRED_FALSE_FLAGS:
        if flag not in policy:
            raise ValueError(
                f"rag_policy is missing required key: {flag}"
            )
        if policy[flag] is not False:
            raise ValueError(
                f"{flag} must be False"
            )

    usage = _require_non_empty_string(
        policy.get(
            "ground_truth_usage",
            "evaluation_only",
        ),
        "rag_policy.ground_truth_usage",
    )
    if usage != "evaluation_only":
        raise ValueError(
            "rag_policy.ground_truth_usage must be 'evaluation_only'"
        )
    policy[
        "ground_truth_usage"
    ] = usage

    if (
        "use_ground_truth_for_evaluation"
        in policy
    ):
        policy[
            "use_ground_truth_for_evaluation"
        ] = _require_bool(
            policy[
                "use_ground_truth_for_evaluation"
            ],
            "rag_policy.use_ground_truth_for_evaluation",
        )

    if (
        "exclude_review_sections_from_reference_papers"
        in policy
    ):
        policy[
            "exclude_review_sections_from_reference_papers"
        ] = _require_bool(
            policy[
                "exclude_review_sections_from_reference_papers"
            ],
            (
                "rag_policy."
                "exclude_review_sections_from_reference_papers"
            ),
        )

    if (
        "excluded_reference_section_types"
        in policy
    ):
        policy[
            "excluded_reference_section_types"
        ] = _validate_string_list(
            policy[
                "excluded_reference_section_types"
            ],
            (
                "rag_policy."
                "excluded_reference_section_types"
            ),
        )

    if "retrieval_profiles" in policy:
        policy[
            "retrieval_profiles"
        ] = _validate_retrieval_profiles(
            policy["retrieval_profiles"]
        )

    if "indexing" in policy:
        indexing = policy["indexing"]
        if not isinstance(
            indexing,
            Mapping,
        ):
            raise TypeError(
                "rag_policy.indexing must be a mapping"
            )
        indexing = dict(indexing)
        if "batch_size" in indexing:
            indexing[
                "batch_size"
            ] = _require_positive_int(
                indexing["batch_size"],
                "rag_policy.indexing.batch_size",
            )
        policy["indexing"] = indexing

    if "generation" in policy:
        generation = policy[
            "generation"
        ]
        if not isinstance(
            generation,
            Mapping,
        ):
            raise TypeError(
                "rag_policy.generation must be a mapping"
            )
        generation = dict(generation)
        if "temperature" in generation:
            generation[
                "temperature"
            ] = _require_number(
                generation["temperature"],
                "rag_policy.generation.temperature",
                minimum=0.0,
                maximum=2.0,
            )
        if (
            "answer_max_words"
            in generation
        ):
            generation[
                "answer_max_words"
            ] = _require_positive_int(
                generation[
                    "answer_max_words"
                ],
                "rag_policy.generation.answer_max_words",
            )
        policy["generation"] = generation

    return policy


def load_active_experiment(
    path: str | Path = (
        ACTIVE_EXPERIMENT_PATH
    ),
) -> dict[str, Any]:
    if (
        isinstance(path, bool)
        or not isinstance(
            path,
            (str, Path),
        )
    ):
        raise TypeError(
            "path must be a string or pathlib.Path"
        )

    active_path = Path(path)
    if not active_path.is_file():
        raise FileNotFoundError(
            "active_experiment.json inexistente: "
            f"{active_path}"
        )

    try:
        data = json.loads(
            active_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "active_experiment.json contiene JSON inválido: "
            f"{active_path}"
        ) from error

    if not isinstance(data, dict):
        raise TypeError(
            "active_experiment.json debe contener un objeto"
        )

    required = {
        "active_experiment_id",
        "experiment_dir",
        "openai_model",
        "embedding_model",
        "chroma_collection_name",
        "rag_policy",
    }
    missing = sorted(
        required - set(data)
    )
    if missing:
        raise ValueError(
            "active_experiment.json está incompleto; "
            f"faltan claves: {missing}"
        )

    experiment_id = (
        _require_non_empty_string(
            data[
                "active_experiment_id"
            ],
            "active_experiment_id",
        )
    )
    project_dir = (
        active_path.parent.resolve()
    )
    experiment_dir = Path(
        _require_non_empty_string(
            data["experiment_dir"],
            "experiment_dir",
        )
    )
    if not experiment_dir.is_absolute():
        experiment_dir = (
            project_dir
            / experiment_dir
        )
    experiment_dir = (
        experiment_dir.resolve()
    )
    expected = (
        project_dir / experiment_id
    ).resolve()
    if experiment_dir != expected:
        raise ValueError(
            "experiment_dir no coincide con "
            "active_experiment_id y PROJECT_DIR"
        )

    if "project_dir" in data:
        configured_project = Path(
            _require_non_empty_string(
                data["project_dir"],
                "project_dir",
            )
        ).resolve()
        if (
            configured_project
            != project_dir
        ):
            raise ValueError(
                "project_dir de active_experiment.json "
                "no coincide con su ubicación real"
            )

    safe = _safe_public_copy(data)
    safe.update({
        "active_experiment_id": (
            experiment_id
        ),
        "run_id": _require_non_empty_string(
            data.get(
                "run_id",
                experiment_id,
            ),
            "run_id",
        ),
        "experiment_dir": str(
            experiment_dir
        ),
        "project_dir": str(
            project_dir
        ),
        "openai_model": (
            _require_non_empty_string(
                data["openai_model"],
                "openai_model",
            )
        ),
        "embedding_model": (
            _require_non_empty_string(
                data[
                    "embedding_model"
                ],
                "embedding_model",
            )
        ),
        "chroma_collection_name": (
            _require_non_empty_string(
                data[
                    "chroma_collection_name"
                ],
                "chroma_collection_name",
            )
        ),
        "rag_policy": (
            _validate_safe_rag_policy(
                data["rag_policy"]
            )
        ),
    })

    for mapping_key in (
        "generation_profile",
        "topic_profile",
        "extraction_policy",
    ):
        raw = safe.get(
            mapping_key,
            {},
        )
        if not isinstance(
            raw,
            Mapping,
        ):
            raise TypeError(
                f"{mapping_key} must be a mapping"
            )
        safe[mapping_key] = (
            deepcopy(dict(raw))
        )

    return safe


def require_active_experiment(
    path: str | Path = (
        ACTIVE_EXPERIMENT_PATH
    ),
) -> dict[str, Any]:
    return load_active_experiment(
        path
    )


def get_active_experiment_config() -> dict[
    str, Any
]:
    return deepcopy(
        _ACTIVE_EXPERIMENT
    )


def build_experiment_paths(
    project_dir: str | Path,
    experiment_id: str,
) -> dict[str, Path]:
    normalized_project_dir = Path(
        project_dir
    ).resolve()
    normalized_experiment_id = (
        _require_non_empty_string(
            experiment_id,
            "experiment_id",
        )
    )
    experiment_dir = (
        normalized_project_dir
        / normalized_experiment_id
    )
    outputs_dir = (
        experiment_dir / "05_outputs"
    )
    return {
        "PROJECT_DIR": (
            normalized_project_dir
        ),
        "SRC_DIR": CODE_ROOT / "src",
        "ACTIVE_EXPERIMENT_PATH": (
            normalized_project_dir
            / "active_experiment.json"
        ),
        "EXPERIMENT_DIR": (
            experiment_dir
        ),
        "INPUT_PDFS_DIR": (
            experiment_dir
            / "01_input_references_pdfs"
        ),
        "EXTRACTED_TEXTS_DIR": (
            experiment_dir
            / "02_extracted_texts"
        ),
        "CHUNKS_DIR": (
            experiment_dir
            / "03_chunks"
        ),
        "CHROMA_DIR": (
            experiment_dir
            / "04_chroma_index"
        ),
        "OUTPUTS_DIR": outputs_dir,
        "ORCHESTRATOR_DIR": (
            outputs_dir
            / "00_orchestrator_planner"
        ),
    }


_ACTIVE_EXPERIMENT = (
    require_active_experiment()
)
EXPERIMENT_ID = (
    _ACTIVE_EXPERIMENT[
        "active_experiment_id"
    ]
)
RUN_ID = _ACTIVE_EXPERIMENT[
    "run_id"
]
_PATHS = build_experiment_paths(
    PROJECT_DIR,
    EXPERIMENT_ID,
)
EXPERIMENT_DIR = _PATHS[
    "EXPERIMENT_DIR"
]
INPUT_PDFS_DIR = _PATHS[
    "INPUT_PDFS_DIR"
]
EXTRACTED_TEXTS_DIR = _PATHS[
    "EXTRACTED_TEXTS_DIR"
]
CHUNKS_DIR = _PATHS[
    "CHUNKS_DIR"
]
CHROMA_DIR = _PATHS[
    "CHROMA_DIR"
]
OUTPUTS_DIR = _PATHS[
    "OUTPUTS_DIR"
]
ORCHESTRATOR_DIR = _PATHS[
    "ORCHESTRATOR_DIR"
]

if (
    EXPERIMENT_DIR.resolve()
    != Path(
        _ACTIVE_EXPERIMENT[
            "experiment_dir"
        ]
    ).resolve()
):
    raise ValueError(
        "active experiment path is inconsistent "
        "with PROJECT_DIR"
    )

OPENAI_MODEL = _ACTIVE_EXPERIMENT[
    "openai_model"
]
EMBEDDING_MODEL_NAME = (
    _ACTIVE_EXPERIMENT[
        "embedding_model"
    ]
)
CHROMA_COLLECTION_NAME = (
    _ACTIVE_EXPERIMENT[
        "chroma_collection_name"
    ]
)
RAG_POLICY = MappingProxyType(
    deepcopy(
        _ACTIVE_EXPERIMENT[
            "rag_policy"
        ]
    )
)


def common_config_snapshot() -> dict[
    str, Any
]:
    return {
        "schema_versions": {
            "common_config": (
                COMMON_CONFIG_SCHEMA_VERSION
            ),
            "pipeline_state": (
                PIPELINE_STATE_SCHEMA_VERSION
            ),
            "agent_contract": (
                AGENT_CONTRACT_SCHEMA_VERSION
            ),
        },
        "code_root": str(CODE_ROOT),
        "project_dir": str(
            PROJECT_DIR
        ),
        "src_dir": str(SRC_DIR),
        "active_experiment_path": str(
            ACTIVE_EXPERIMENT_PATH
        ),
        "project_config_path": str(PROJECT_CONFIG_PATH),
        "project_generation_config_path": str(PROJECT_GENERATION_CONFIG_PATH),
        "project_rag_policy_path": str(PROJECT_RAG_POLICY_PATH),
        "experiment_id": (
            EXPERIMENT_ID
        ),
        "run_id": RUN_ID,
        "experiment_dir": str(
            EXPERIMENT_DIR
        ),
        "input_pdfs_dir": str(
            INPUT_PDFS_DIR
        ),
        "extracted_texts_dir": str(
            EXTRACTED_TEXTS_DIR
        ),
        "chunks_dir": str(
            CHUNKS_DIR
        ),
        "chroma_dir": str(
            CHROMA_DIR
        ),
        "outputs_dir": str(
            OUTPUTS_DIR
        ),
        "orchestrator_dir": str(
            ORCHESTRATOR_DIR
        ),
        "openai_model": (
            OPENAI_MODEL
        ),
        "embedding_model_name": (
            EMBEDDING_MODEL_NAME
        ),
        "chroma_collection_name": (
            CHROMA_COLLECTION_NAME
        ),
        "rag_policy": deepcopy(
            dict(RAG_POLICY)
        ),
    }
