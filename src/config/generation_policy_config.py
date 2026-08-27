"""Policies used exclusively by the migrated Agent 03.

This module does not replace or modify PROJECT_DIR/src/generation_config.py.
It contains no policies for stages 04-08.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .common_config import (
    CHROMA_DIR,
    CHUNKS_DIR,
    EXPERIMENT_DIR,
    OUTPUTS_DIR,
    get_active_experiment_config,
)


EXTRACTION_PROMPT_VERSION = (
    "v4_chroma_per_paper_domain_agnostic"
)
KB_SCHEMA_VERSION = (
    "v3_canonical_domain_agnostic"
)
EXTRACTION_AGENT_VERSION = (
    "v16_agent03_policy"
)

SCIENTIFIC_EXTRACTION_DIR = (
    OUTPUTS_DIR
    / "01_scientific_extraction"
)
KNOWLEDGE_BASE_DIR = (
    OUTPUTS_DIR
    / "02_scientific_knowledge_base"
)
CHUNKS_CLEAN_PATH = (
    CHUNKS_DIR
    / "chunks_clean_for_rag.csv"
)
CHROMA_MANIFEST_PATH = (
    CHROMA_DIR
    / "chroma_index_manifest.json"
)

SCIENTIFIC_CARDS_JSONL_PATH = (
    SCIENTIFIC_EXTRACTION_DIR
    / "scientific_cards.jsonl"
)
SCIENTIFIC_CARDS_SUMMARY_CSV_PATH = (
    SCIENTIFIC_EXTRACTION_DIR
    / "scientific_cards_summary.csv"
)
SCIENTIFIC_CARDS_ERRORS_CSV_PATH = (
    SCIENTIFIC_EXTRACTION_DIR
    / "scientific_cards_errors.csv"
)
SCIENTIFIC_CARDS_QUALITY_CSV_PATH = (
    SCIENTIFIC_EXTRACTION_DIR
    / "scientific_cards_quality_check.csv"
)
SCIENTIFIC_CARDS_REVISION_PLAN_CSV_PATH = (
    SCIENTIFIC_EXTRACTION_DIR
    / "scientific_cards_revision_plan.csv"
)
EXTRACTION_RETRIEVAL_TRACE_CSV_PATH = (
    SCIENTIFIC_EXTRACTION_DIR
    / "extraction_retrieval_trace.csv"
)
SCIENTIFIC_EXTRACTION_MANIFEST_PATH = (
    SCIENTIFIC_EXTRACTION_DIR
    / "scientific_extraction_manifest.json"
)
CHUNKS_VALIDATION_REPORT_PATH = (
    SCIENTIFIC_EXTRACTION_DIR
    / "chunks_clean_validation_report.json"
)
KB_CSV_PATH = (
    KNOWLEDGE_BASE_DIR
    / "scientific_knowledge_base.csv"
)
KB_JSONL_PATH = (
    KNOWLEDGE_BASE_DIR
    / "scientific_knowledge_base.jsonl"
)

_DEFAULT_EXTRACTION_POLICY = {
    "max_chunks_per_paper": 10,
    "max_context_chars": 18000,
    "repair_max_chunks_per_paper": 18,
    "repair_max_context_chars": 26000,
    "temperature": 0.1,
    "repair_temperature": 0.0,
    "retrieval_profile": "strict",
    "retrieval_queries": [
        (
            "research problem objective "
            "scientific contribution"
        ),
        (
            "methodology methods models "
            "algorithms experimental design"
        ),
        (
            "dataset data sources input variables "
            "study population case study"
        ),
        (
            "evaluation metrics experimental "
            "results comparative performance"
        ),
        (
            "main findings conclusions limitations "
            "research gaps future work"
        ),
    ],
    "title_repair_first_chunks": 3,
    "auto_rebuild": True,
    "force_rebuild": False,
    "max_attempts": 2,
    "max_retrieval_rounds": 2,
    # Política metodológica CANÓNICA: todo experimento nuevo nace con
    # los reviews/surveys completos excluidos del corpus de
    # generación -- esto ya NO depende de que una celda de notebook
    # escriba active_experiment.json manualmente (ver Corrida_03_a_08:
    # esa celda ahora solo hace un preflight fail-closed, nunca
    # escribe la policy). Sigue siendo configurable explícitamente:
    # un experimento que declare "exclude_reviews": false en su
    # active_experiment.json lo sobrescribe conscientemente, y
    # extraction_agent.py/classify_review_exclusion() lo respetan sin
    # cambios -- lo único que cambió es este DEFAULT, nunca la
    # semántica del clasificador.
    "exclude_reviews": True,
    # Corpus Eligibility Gate (ver corpus_eligibility.py): el gate en
    # sí (INCLUDE/EXCLUDE/QUARANTINE) es parte OBLIGATORIA de Stage03
    # -- siempre corre, no existe ("ni debe existir) un interruptor
    # "enabled" que lo desactive, porque un documento individual no
    # útil o no validable nunca debe poder saltarse la clasificación
    # y bloquear todo el corpus por accidente. Lo único configurable
    # es el umbral mínimo de corpus elegible antes de HALT global.
    "corpus_eligibility_policy": {"min_include_corpus_size": 1},
    "thresholds": {
        "approval": {"critical_field_coverage": 0.92},
        "minimum_usable_quality": {"critical_field_coverage": 0.80},
    },
    "manual_review_policy": {
        "allowed": True,
        "allowed_reason_codes": [
            "INSUFFICIENT_EVIDENCE",
            "MISSING_CRITICAL_FIELDS",
            "INVALID_LLM_OUTPUT",
            "INVALID_CARD_SCHEMA",
            "MISSING_OR_INVALID_TITLE",
        ],
        "minimum_usable_quality": {"critical_field_coverage": 0.80},
        "resume_requires_confirmation": True,
    },
}

_DEFAULT_RETRIEVAL_PROFILES = {
    "default": {
        "top_k": 8,
        "fetch_k": 35,
        "max_per_source": 2,
    },
    "compact": {
        "top_k": 6,
        "fetch_k": 35,
        "max_per_source": 2,
    },
    "strict": {
        "top_k": 10,
        "fetch_k": 40,
        "max_per_source": 2,
    },
    "testing": {
        "top_k": 5,
        "fetch_k": 30,
        "max_per_source": 2,
    },
}

_REQUIRED_POLICY_KEYS = {
    "max_chunks_per_paper",
    "max_context_chars",
    "repair_max_chunks_per_paper",
    "repair_max_context_chars",
    "temperature",
    "repair_temperature",
    "retrieval_profile",
    "retrieval_queries",
    "title_repair_first_chunks",
    "auto_rebuild",
    "force_rebuild",
}
_ALLOWED_POLICY_KEYS = set(_DEFAULT_EXTRACTION_POLICY)


def _mapping(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping"
        )
    return deepcopy(dict(value))


def _string(
    value: Any,
    name: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string"
        )
    normalized = value.strip()
    if not normalized:
        raise ValueError(
            f"{name} must not be empty"
        )
    return normalized


def _positive_int(
    value: Any,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{name} must be an integer"
        )
    if value < 1:
        raise ValueError(
            f"{name} must be greater than or equal to 1"
        )
    return value


def _number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise TypeError(
            f"{name} must be numeric"
        )
    normalized = float(value)
    if not 0.0 <= normalized <= 2.0:
        raise ValueError(
            f"{name} must be between 0.0 and 2.0"
        )
    return normalized


def _boolean(
    value: Any,
    name: str,
) -> bool:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be bool"
        )
    return value


def _validate_profiles(
    value: Any,
) -> dict[str, dict[str, int]]:
    raw = _mapping(
        value,
        "rag_policy.retrieval_profiles",
    )
    validated = {}
    for name, profile_value in raw.items():
        profile_name = _string(
            name,
            "retrieval profile name",
        )
        profile = _mapping(
            profile_value,
            f"retrieval_profiles.{profile_name}",
        )
        required = {
            "top_k",
            "fetch_k",
            "max_per_source",
        }
        missing = sorted(
            required - set(profile)
        )
        if missing:
            raise ValueError(
                f"retrieval profile {profile_name!r} "
                f"is missing keys: {missing}"
            )
        normalized = {
            key: _positive_int(
                profile[key],
                (
                    f"retrieval_profiles."
                    f"{profile_name}.{key}"
                ),
            )
            for key in required
        }
        if (
            normalized["fetch_k"]
            < normalized["top_k"]
        ):
            raise ValueError(
                "retrieval fetch_k must be "
                "greater than or equal to top_k"
            )
        validated[
            profile_name
        ] = normalized
    return validated


def _validate_policy(
    value: Any,
    profiles: Mapping[str, Any],
) -> dict[str, Any]:
    supplied_policy = _mapping(
        value,
        "extraction_policy",
    )
    # CONFIG-A (Stage 03): los 11 campos de _REQUIRED_POLICY_KEYS son
    # responsabilidad de 00 (coinciden exactamente con
    # FIXED_EXTRACTION_POLICY del notebook) -- deben venir de
    # active_experiment.json["extraction_policy"], nunca rellenarse en
    # silencio desde _DEFAULT_EXTRACTION_POLICY. El chequeo se hace
    # ANTES de aplicar el default, para que realmente pueda fallar.
    # Los campos fuera de este conjunto (exclude_reviews,
    # corpus_eligibility_policy, max_attempts, max_retrieval_rounds)
    # son contrato interno de Stage 03 y SÍ siguen tomando su valor de
    # _DEFAULT_EXTRACTION_POLICY si 00 no los proporciona -- 00 nunca
    # tuvo la responsabilidad de configurarlos.
    missing_from_00 = sorted(
        _REQUIRED_POLICY_KEYS
        - set(supplied_policy)
    )
    if missing_from_00:
        raise ValueError(
            "extraction_policy: faltan campos obligatorios que "
            "00_setup_config.ipynb debe proporcionar (sin default "
            f"interno): {missing_from_00}"
        )
    policy = deepcopy(_DEFAULT_EXTRACTION_POLICY)
    policy.update(supplied_policy)
    missing = sorted(
        _REQUIRED_POLICY_KEYS
        - set(policy)
    )
    if missing:
        raise ValueError(
            "extraction_policy is missing keys: "
            f"{missing}"
        )
    unexpected = sorted(
        set(supplied_policy)
        - _ALLOWED_POLICY_KEYS
    )
    if unexpected:
        raise ValueError(
            "extraction_policy contains unsupported keys: "
            f"{unexpected}"
        )

    for key in (
        "max_chunks_per_paper",
        "max_context_chars",
        "repair_max_chunks_per_paper",
        "repair_max_context_chars",
        "title_repair_first_chunks",
    ):
        policy[key] = _positive_int(
            policy[key],
            f"extraction_policy.{key}",
        )

    if (
        policy[
            "repair_max_chunks_per_paper"
        ]
        < policy[
            "max_chunks_per_paper"
        ]
    ):
        raise ValueError(
            "repair_max_chunks_per_paper must be "
            "greater than or equal to "
            "max_chunks_per_paper"
        )
    if (
        policy[
            "repair_max_context_chars"
        ]
        < policy[
            "max_context_chars"
        ]
    ):
        raise ValueError(
            "repair_max_context_chars must be "
            "greater than or equal to "
            "max_context_chars"
        )

    policy["temperature"] = (
        _number(
            policy["temperature"],
            "extraction_policy.temperature",
        )
    )
    policy["repair_temperature"] = (
        _number(
            policy[
                "repair_temperature"
            ],
            (
                "extraction_policy."
                "repair_temperature"
            ),
        )
    )
    profile_name = _string(
        policy["retrieval_profile"],
        (
            "extraction_policy."
            "retrieval_profile"
        ),
    )
    if profile_name not in profiles:
        raise ValueError(
            "retrieval_profile no existe en "
            "rag_policy.retrieval_profiles: "
            f"{profile_name}"
        )
    policy[
        "retrieval_profile"
    ] = profile_name

    queries = policy[
        "retrieval_queries"
    ]
    if not isinstance(
        queries,
        list,
    ):
        raise TypeError(
            "extraction_policy.retrieval_queries "
            "must be a list"
        )
    policy[
        "retrieval_queries"
    ] = tuple(
        _string(
            item,
            (
                "extraction_policy."
                f"retrieval_queries[{index}]"
            ),
        )
        for index, item in enumerate(
            queries
        )
    )
    if not policy[
        "retrieval_queries"
    ]:
        raise ValueError(
            "extraction_policy.retrieval_queries "
            "must not be empty"
        )

    policy["auto_rebuild"] = (
        _boolean(
            policy["auto_rebuild"],
            "extraction_policy.auto_rebuild",
        )
    )
    policy["force_rebuild"] = (
        _boolean(
            policy["force_rebuild"],
            "extraction_policy.force_rebuild",
        )
    )
    policy["exclude_reviews"] = (
        _boolean(
            policy.get("exclude_reviews", True),
            "extraction_policy.exclude_reviews",
        )
    )
    corpus_eligibility_policy = dict(
        policy.get("corpus_eligibility_policy") or {}
    )
    corpus_eligibility_policy["min_include_corpus_size"] = (
        _positive_int(
            corpus_eligibility_policy.get("min_include_corpus_size", 1),
            "extraction_policy.corpus_eligibility_policy.min_include_corpus_size",
        )
    )
    policy["corpus_eligibility_policy"] = corpus_eligibility_policy
    policy["max_attempts"] = (
        _positive_int(
            policy.get(
                "max_attempts",
                2,
            ),
            "extraction_policy.max_attempts",
        )
    )
    policy[
        "max_retrieval_rounds"
    ] = _positive_int(
        policy.get(
            "max_retrieval_rounds",
            2,
        ),
        (
            "extraction_policy."
            "max_retrieval_rounds"
        ),
    )
    policy.setdefault(
        "thresholds",
        {
            "approval": {
                "critical_field_coverage": (
                    0.92
                )
            }
        },
    )
    policy.setdefault(
        "manual_review_policy",
        {
            "allowed": True,
            "allowed_reason_codes": [
                "AMBIGUOUS_PAPER_RELEVANCE",
                "CRITICAL_FIELDS_NOT_REPORTED",
            ],
            "minimum_usable_quality": {
                "critical_field_coverage": (
                    0.80
                )
            },
            "resume_requires_confirmation": (
                True
            ),
        },
    )
    return policy


def get_extraction_policy() -> dict[
    str, Any
]:
    policy = deepcopy(
        dict(EXTRACTION_POLICY)
    )
    policy[
        "retrieval_queries"
    ] = list(
        EXTRACTION_POLICY[
            "retrieval_queries"
        ]
    )
    return policy


def get_extraction_paths() -> dict[
    str, Path
]:
    return {
        "experiment_dir": (
            EXPERIMENT_DIR
        ),
        "chunks_dir": CHUNKS_DIR,
        "chroma_dir": CHROMA_DIR,
        "scientific_extraction_dir": (
            SCIENTIFIC_EXTRACTION_DIR
        ),
        "knowledge_base_dir": (
            KNOWLEDGE_BASE_DIR
        ),
        "chunks_clean_path": (
            CHUNKS_CLEAN_PATH
        ),
        "chroma_manifest_path": (
            CHROMA_MANIFEST_PATH
        ),
        "scientific_cards_jsonl_path": (
            SCIENTIFIC_CARDS_JSONL_PATH
        ),
        "scientific_cards_summary_csv_path": (
            SCIENTIFIC_CARDS_SUMMARY_CSV_PATH
        ),
        "scientific_cards_errors_csv_path": (
            SCIENTIFIC_CARDS_ERRORS_CSV_PATH
        ),
        "scientific_cards_quality_csv_path": (
            SCIENTIFIC_CARDS_QUALITY_CSV_PATH
        ),
        "scientific_cards_revision_plan_csv_path": (
            SCIENTIFIC_CARDS_REVISION_PLAN_CSV_PATH
        ),
        "retrieval_trace_csv_path": (
            EXTRACTION_RETRIEVAL_TRACE_CSV_PATH
        ),
        "extraction_manifest_path": (
            SCIENTIFIC_EXTRACTION_MANIFEST_PATH
        ),
        "chunks_validation_report_path": (
            CHUNKS_VALIDATION_REPORT_PATH
        ),
        "kb_csv_path": KB_CSV_PATH,
        "kb_jsonl_path": KB_JSONL_PATH,
    }


def get_extraction_fingerprint_config() -> dict[
    str, Any
]:
    return {
        "agent_version": (
            EXTRACTION_AGENT_VERSION
        ),
        "prompt_version": (
            EXTRACTION_PROMPT_VERSION
        ),
        "kb_schema_version": (
            KB_SCHEMA_VERSION
        ),
        "policy": (
            get_extraction_policy()
        ),
        "retrieval_config": dict(
            RETRIEVAL_CONFIG
        ),
    }


def generation_config_snapshot() -> dict[
    str, Any
]:
    return {
        "versions": {
            "agent": (
                EXTRACTION_AGENT_VERSION
            ),
            "prompt": (
                EXTRACTION_PROMPT_VERSION
            ),
            "kb_schema": (
                KB_SCHEMA_VERSION
            ),
        },
        "paths": {
            key: str(value)
            for key, value
            in get_extraction_paths().items()
        },
        "extraction_policy": (
            get_extraction_policy()
        ),
        "retrieval_config": dict(
            RETRIEVAL_CONFIG
        ),
    }


_ACTIVE = get_active_experiment_config()
_RAW_PROFILES = (
    _ACTIVE["rag_policy"].get(
        "retrieval_profiles",
        _DEFAULT_RETRIEVAL_PROFILES,
    )
)
_VALIDATED_PROFILES = (
    _validate_profiles(
        _RAW_PROFILES
    )
)
_RAW_POLICY = _ACTIVE.get(
    "extraction_policy",
    _DEFAULT_EXTRACTION_POLICY,
)
_VALIDATED_POLICY = (
    _validate_policy(
        _RAW_POLICY,
        _VALIDATED_PROFILES,
    )
)
_PROFILE_NAME = (
    _VALIDATED_POLICY[
        "retrieval_profile"
    ]
)
_VALIDATED_RETRIEVAL = {
    "profile_name": (
        _PROFILE_NAME
    ),
    **deepcopy(
        _VALIDATED_PROFILES[
            _PROFILE_NAME
        ]
    ),
}

EXTRACTION_POLICY = (
    MappingProxyType(
        deepcopy(
            _VALIDATED_POLICY
        )
    )
)
RETRIEVAL_CONFIG = (
    MappingProxyType(
        deepcopy(
            _VALIDATED_RETRIEVAL
        )
    )
)
