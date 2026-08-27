"""Configuración determinista del futuro Agente 07.

Fases 1R–2: contrato estricto de entrada, identidad documental, clasificación
bilingüe de claims y política de evaluación de evidencia heredada. No contiene
runtime, prompts, retrieval independiente ni llamadas a modelos.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

AGENT06_STAGE_NAME = "06_agente_redactor"
AGENT07_STAGE_NAME = "07_agente_verificador_trazabilidad"
OFFICIAL_DRAFT_DIRECTORY_NAME = "05_draft"
FORBIDDEN_DRAFT_DIRECTORY_NAMES = ("05_draft_v17_candidate",)

REQUIRED_AGENT06_ARTIFACTS = (
    "state_of_art_draft.json",
    "draft_claim_evidence.csv",
    "draft_sections.csv",
    "draft_generation_manifest.json",
    "draft_validation_report.json",
)

CLAIM_TYPES = (
    "SUBSTANTIVE_FACTUAL",
    "QUANTITATIVE",
    "COMPARATIVE",
    "METHODOLOGICAL",
    "ATTRIBUTION",
    "INTERPRETIVE",
    "ORGANIZATIONAL",
    "TRANSITIONAL",
)

VERIFICATION_INTENSITIES = ("NONE", "LIGHT", "STANDARD", "STRICT")

RESOLUTION_STATUSES = (
    "RESOLVED",
    "MISSING_REFERENCE",
    "UNAUTHORIZED_SOURCE",
    "AMBIGUOUS_REFERENCE",
    "TEXT_MISMATCH",
    "INHERITED_EVIDENCE_EMPTY",
    "DUPLICATE_EXACT_PAIR",
)

RESOLUTION_ISSUE_CODES = (
    "DUPLICATE_EXACT_PAIR",
    "AMBIGUOUS_REFERENCE",
    "MISSING_REFERENCE",
    "UNAUTHORIZED_SOURCE",
    "TEXT_MISMATCH",
)

TEXT_MATCH_STATUSES = (
    "NOT_APPLICABLE",
    "EXACT_MATCH",
    "NORMALIZED_MATCH",
    "TEXT_MISMATCH",
)

RETRIEVAL_REASON_CODES = (
    "NO_INHERITED_EVIDENCE",
    "MISSING_REFERENCE",
    "UNAUTHORIZED_SOURCE",
    "AMBIGUOUS_REFERENCE",
    "TEXT_MISMATCH",
    "DUPLICATE_WITHIN_CLAIM",
    "QUANTITATIVE_COVERAGE_INCOMPLETE",
    "COMPARATIVE_ENTITY_COVERAGE_INCOMPLETE",
    "ATTRIBUTION_SOURCE_MISSING",
    "RESOLVABLE_EVIDENCE_BELOW_MINIMUM",
    "QUANTITATIVE_TOKEN_EXTRACTION_EMPTY",
)

DEFAULT_CLAIM_VERIFICATION_INTENSITY = {
    "SUBSTANTIVE_FACTUAL": "STANDARD",
    "QUANTITATIVE": "STRICT",
    "COMPARATIVE": "STRICT",
    "METHODOLOGICAL": "STANDARD",
    "ATTRIBUTION": "STRICT",
    "INTERPRETIVE": "STANDARD",
    "ORGANIZATIONAL": "LIGHT",
    "TRANSITIONAL": "NONE",
}

DEFAULT_MINIMUM_RESOLVED_EVIDENCE = {
    "SUBSTANTIVE_FACTUAL": 1,
    "QUANTITATIVE": 1,
    "COMPARATIVE": 1,
    "METHODOLOGICAL": 1,
    "ATTRIBUTION": 1,
    "INTERPRETIVE": 1,
    "ORGANIZATIONAL": 0,
    "TRANSITIONAL": 0,
}

DEFAULT_VERIFICATION_INPUT_POLICY: dict[str, Any] = {
    "stage_version": "07_INHERITED_EVIDENCE_V2R",
    "agent06_stage_name": AGENT06_STAGE_NAME,
    "agent07_stage_name": AGENT07_STAGE_NAME,
    "official_draft_directory_name": OFFICIAL_DRAFT_DIRECTORY_NAME,
    "forbidden_draft_directory_names": FORBIDDEN_DRAFT_DIRECTORY_NAMES,
    "required_agent06_artifacts": REQUIRED_AGENT06_ARTIFACTS,
    "accepted_agent06_quality_statuses": (
        "APPROVED",
        "APPROVED_WITH_WARNINGS",
        "APPROVED_AFTER_MANUAL_REVIEW",
    ),
    "required_agent06_decision_code": "DRAFT_APPROVED",
    "required_agent06_transition_action": "ADVANCE",
    "claim_id_prefix": "V07",
    "claim_id_hash_length": 12,
    "claim_verification_intensity": DEFAULT_CLAIM_VERIFICATION_INTENSITY,
    "minimum_resolved_evidence_by_claim_type": DEFAULT_MINIMUM_RESOLVED_EVIDENCE,
    "document_identity_fields": ("source_filename", "chunk_id"),
    "allowed_evidence_flag_field": "allowed_for_section",
    "evidence_text_field": "evidence_text",
    "chunk_text_field": "text",
    "require_text_match": True,
    "record_cross_claim_reuse": True,
    "allow_missing_allowed_for_section": False,
    # Compatibilidad solo por opt-in. El contrato estricto es el default.
    "allow_legacy_incomplete_committed_result": False,
    "allow_artifact_basename_fallback": True,
}


def _nonempty_string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:expected_nonempty_string")
    return value.strip()


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:expected_sequence")
    normalized = tuple(_nonempty_string(item, f"{key}[]") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:duplicates")
    return normalized


def _strict_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:expected_boolean")
    return value


def _nonnegative_int_mapping(value: Any, key: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:expected_mapping")
    normalized: dict[str, int] = {}
    for claim_type in CLAIM_TYPES:
        if claim_type not in value:
            raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:missing_{claim_type}")
        item = value[claim_type]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:{claim_type}:expected_nonnegative_integer")
        normalized[claim_type] = item
    return normalized


def validate_verification_input_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        raise ValueError("VERIFICATION_POLICY_INVALID:policy:expected_mapping")
    value = dict(policy)
    for key in (
        "stage_version",
        "agent06_stage_name",
        "agent07_stage_name",
        "official_draft_directory_name",
        "required_agent06_decision_code",
        "required_agent06_transition_action",
        "claim_id_prefix",
        "allowed_evidence_flag_field",
        "evidence_text_field",
        "chunk_text_field",
    ):
        value[key] = _nonempty_string(value.get(key), key)

    for key in (
        "forbidden_draft_directory_names",
        "required_agent06_artifacts",
        "accepted_agent06_quality_statuses",
        "document_identity_fields",
    ):
        value[key] = _string_tuple(value.get(key), key)

    for key in (
        "allow_legacy_incomplete_committed_result",
        "allow_artifact_basename_fallback",
        "require_text_match",
        "record_cross_claim_reuse",
        "allow_missing_allowed_for_section",
    ):
        value[key] = _strict_bool(value.get(key), key)

    hash_length = value.get("claim_id_hash_length")
    if isinstance(hash_length, bool) or not isinstance(hash_length, int):
        raise ValueError("VERIFICATION_POLICY_INVALID:claim_id_hash_length:expected_integer")
    if hash_length < 8 or hash_length > 64:
        raise ValueError("VERIFICATION_POLICY_INVALID:claim_id_hash_length:must_be_between_8_and_64")

    intensities = value.get("claim_verification_intensity")
    if not isinstance(intensities, Mapping):
        raise ValueError("VERIFICATION_POLICY_INVALID:claim_verification_intensity:expected_mapping")
    normalized_intensities: dict[str, str] = {}
    for claim_type in CLAIM_TYPES:
        if claim_type not in intensities:
            raise ValueError(f"VERIFICATION_POLICY_INVALID:claim_verification_intensity:missing_{claim_type}")
        intensity = _nonempty_string(intensities[claim_type], claim_type)
        if intensity not in VERIFICATION_INTENSITIES:
            raise ValueError(f"VERIFICATION_POLICY_INVALID:claim_verification_intensity:unsupported_{intensity}")
        normalized_intensities[claim_type] = intensity
    value["claim_verification_intensity"] = normalized_intensities
    value["minimum_resolved_evidence_by_claim_type"] = _nonnegative_int_mapping(
        value.get("minimum_resolved_evidence_by_claim_type"),
        "minimum_resolved_evidence_by_claim_type",
    )

    if tuple(value["document_identity_fields"]) != ("source_filename", "chunk_id"):
        raise ValueError("VERIFICATION_POLICY_INVALID:document_identity_fields:must_be_source_filename_chunk_id")
    return value


def get_verification_input_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_VERIFICATION_INPUT_POLICY)
    policy["claim_verification_intensity"] = dict(DEFAULT_CLAIM_VERIFICATION_INTENSITY)
    policy["minimum_resolved_evidence_by_claim_type"] = dict(DEFAULT_MINIMUM_RESOLVED_EVIDENCE)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("VERIFICATION_POLICY_INVALID:overrides:expected_mapping")
        policy.update(dict(overrides))
    return validate_verification_input_policy(policy)

# Phase 3: retrieval independiente inyectable y determinista por claim.
RETRIEVAL_MODES = (
    "SECTION_SCOPED",
    "CORPUS_WIDE_CONTRADICTION",
    "CORPUS_WIDE_TRANSVERSAL",
)

RETRIEVAL_TECHNICAL_STATUSES = (
    "NOT_ATTEMPTED",
    "COMPLETED",
    "NO_NEW_EVIDENCE",
    "RETRIEVER_UNAVAILABLE",
    "INDEX_SCHEMA_INVALID",
    "AUTHORIZED_SOURCE_SET_EMPTY",
    "CSV_RANKING_UNAVAILABLE",
    "BUDGET_EXHAUSTED",
)

CONTRADICTION_SIGNAL_CODES = (
    "VALUE_VARIATION",
    "UNIT_VARIATION",
    "NEGATION_SIGNAL",
    "ATTRIBUTION_VARIATION",
    "POLARITY_SIGNAL",
)

DEFAULT_RETRIEVAL_ROUNDS_BY_INTENSITY = {
    "NONE": 0,
    "LIGHT": 0,
    "STANDARD": 1,
    "STRICT": 2,
}

DEFAULT_VERIFICATION_INPUT_POLICY.update({
    "independent_retrieval_enabled": True,
    "retrieval_strategy": "hybrid_injected_rrf_by_claim",
    "retrieval_rounds_by_intensity": DEFAULT_RETRIEVAL_ROUNDS_BY_INTENSITY,
    "light_retrieval_reason_codes": ("ATTRIBUTION_SOURCE_MISSING",),
    "max_queries_per_claim": 3,
    "top_k_per_retriever": 8,
    "max_total_candidates_per_claim": 16,
    "max_final_evidence_per_claim": 8,
    "max_candidates_per_source": 4,
    "max_evidence_chars_per_claim": 12000,
    "rrf_k": 60,
    "allow_corpus_wide_retrieval": False,
    "allow_corpus_wide_contradiction": False,
    "allow_corpus_wide_transversal": False,
    "stop_when_structural_coverage_satisfied": True,
    "stop_when_no_new_pairs": True,
    "authorized_source_legacy_fallback_enabled": False,
})


def _positive_int(value: Any, key: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:expected_integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:must_be_at_least_{minimum}")
    return value


def _retrieval_round_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("VERIFICATION_POLICY_INVALID:retrieval_rounds_by_intensity:expected_mapping")
    result: dict[str, int] = {}
    for intensity in VERIFICATION_INTENSITIES:
        if intensity not in value:
            raise ValueError(f"VERIFICATION_POLICY_INVALID:retrieval_rounds_by_intensity:missing_{intensity}")
        result[intensity] = _positive_int(value[intensity], f"retrieval_rounds_by_intensity:{intensity}", allow_zero=True)
    if result["NONE"] != 0 or result["LIGHT"] != 0 or result["STANDARD"] > 1 or result["STRICT"] > 2:
        raise ValueError("VERIFICATION_POLICY_INVALID:retrieval_rounds_by_intensity:exceeds_conservative_limits")
    return result


# Wrap the prior validator to extend it without changing the accepted Phase 1R/2R API.
_validate_verification_input_policy_phase2r = validate_verification_input_policy


def validate_verification_input_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_verification_input_policy_phase2r(policy)
    for key in (
        "independent_retrieval_enabled",
        "allow_corpus_wide_retrieval",
        "allow_corpus_wide_contradiction",
        "allow_corpus_wide_transversal",
        "stop_when_structural_coverage_satisfied",
        "stop_when_no_new_pairs",
        "authorized_source_legacy_fallback_enabled",
    ):
        value[key] = _strict_bool(value.get(key), key)
    value["retrieval_strategy"] = _nonempty_string(value.get("retrieval_strategy"), "retrieval_strategy")
    value["retrieval_rounds_by_intensity"] = _retrieval_round_mapping(value.get("retrieval_rounds_by_intensity"))
    value["light_retrieval_reason_codes"] = _string_tuple(value.get("light_retrieval_reason_codes"), "light_retrieval_reason_codes")
    for key in (
        "max_queries_per_claim",
        "top_k_per_retriever",
        "max_total_candidates_per_claim",
        "max_final_evidence_per_claim",
        "max_candidates_per_source",
        "max_evidence_chars_per_claim",
        "rrf_k",
    ):
        value[key] = _positive_int(value.get(key), key)
    return value


def get_verification_input_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_VERIFICATION_INPUT_POLICY)
    policy["claim_verification_intensity"] = dict(DEFAULT_CLAIM_VERIFICATION_INTENSITY)
    policy["minimum_resolved_evidence_by_claim_type"] = dict(DEFAULT_MINIMUM_RESOLVED_EVIDENCE)
    policy["retrieval_rounds_by_intensity"] = dict(DEFAULT_RETRIEVAL_ROUNDS_BY_INTENSITY)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("VERIFICATION_POLICY_INVALID:overrides:expected_mapping")
        policy.update(dict(overrides))
    return validate_verification_input_policy(policy)

# Phase 4: núcleo científico por claim con LLM/retrieval inyectables.
SCIENTIFIC_VERDICTS = (
    "NOT_APPLICABLE",
    "NOT_EVALUATED",
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
    "NOT_VERIFIABLE",
)

SUPPORT_LEVELS = ("NONE", "WEAK", "PARTIAL", "STRONG")

SUPPORT_LEVEL_BY_VERDICT = {
    "NOT_APPLICABLE": "NONE",
    "NOT_EVALUATED": "NONE",
    "SUPPORTED": "STRONG",
    "PARTIALLY_SUPPORTED": "PARTIAL",
    "CONTRADICTED": "NONE",
    "INSUFFICIENT_EVIDENCE": "NONE",
    "NOT_VERIFIABLE": "NONE",
}

SCIENTIFIC_JUDGMENT_STATUSES = ("NOT_REQUIRED", "PENDING", "COMPLETED", "BLOCKED")
CLAIM_EXECUTION_STATUSES = ("COMPLETED", "FAILED")
CLAIM_TECHNICAL_STATUSES = (
    "OK",
    "RETRIEVAL_BLOCKED",
    "LLM_VALIDATION_ATTEMPTS_EXHAUSTED",
    "INVALID_INPUT",
    "LLM_UNAVAILABLE",
    "LLM_INVOCATION_FAILED",
    "ADDITIONAL_RETRIEVER_UNAVAILABLE",
    "ADDITIONAL_RETRIEVAL_BUDGET_EXHAUSTED",
    "ADDITIONAL_RETRIEVAL_FAILED",
)

# Historical terminal serializers may report execution completion while the
# scientific judgment remained unresolved. Only statuses explicitly listed here
# may use that compatibility path; all other non-OK statuses must retain their
# normal BLOCKED/invalid semantics.
CLAIM_COMPLETED_UNRESOLVED_TECHNICAL_STATUSES = (
    "LLM_VALIDATION_ATTEMPTS_EXHAUSTED",
)

CONTRADICTION_TYPES = (
    "NONE",
    "CLAIM_EVIDENCE_CONFLICT",
    "CROSS_SOURCE_DISAGREEMENT",
    "INTERNAL_TEXT_INCONSISTENCY",
)
NUMERIC_ASSESSMENTS = ("NOT_APPLICABLE", "SUPPORTED", "UNSUPPORTED", "CONTEXT_MISMATCH")
ATTRIBUTION_ASSESSMENTS = ("NOT_APPLICABLE", "CORRECT", "INCORRECT", "INSUFFICIENT_EVIDENCE")
EXTRAPOLATION_ASSESSMENTS = ("NOT_APPLICABLE", "WITHIN_EVIDENCE_SCOPE", "BEYOND_EVIDENCE_SCOPE", "UNCLEAR")
HALLUCINATION_RISKS = ("LOW", "MEDIUM", "HIGH")
CORRECTION_ELIGIBILITIES = (
    "AUTO_CORRECTION_ELIGIBLE",
    "POTENTIALLY_AUTO_CORRECTABLE",
    "MANUAL_REVIEW_REQUIRED",
    "NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE",
    "NO_CORRECTION_NEEDED",
)
DETERMINISTIC_ISSUE_CODES = (
    "INVALID_CITATION",
    "UNSUPPORTED_NUMERIC_VALUE",
    "DOCUMENT_IDENTITY_INVALID",
    "UNAUTHORIZED_SOURCE",
    "TEXT_INTEGRITY_INVALID",
    "RETRIEVAL_TECHNICAL_BLOCKER",
)
SEMANTIC_ISSUE_CODES = (
    "ATTRIBUTION_ERROR",
    "UNSUPPORTED_EXTRAPOLATION",
    "UNSUPPORTED_NUMERIC_VALUE",
    "NUMERIC_CONTEXT_MISMATCH",
    "CLAIM_EVIDENCE_CONFLICT",
    "CROSS_SOURCE_DISAGREEMENT",
    "INTERNAL_TEXT_INCONSISTENCY",
    "PARTIAL_SUPPORT",
    "INSUFFICIENT_EVIDENCE",
)


SEMANTIC_REASON_CODES = (
    "SCOPE_LIMITED", "PARTIAL_EVIDENCE", "NO_COVERAGE", "WRONG_ATTRIBUTION",
    "EXTRAPOLATION_BEYOND_SCOPE", "CLAIM_EVIDENCE_CONFLICT",
    "CROSS_SOURCE_DISAGREEMENT", "INTERNAL_TEXT_INCONSISTENCY",
    "EVIDENCE_INSUFFICIENT", "CONTEXT_MISMATCH",
)
ADDITIONAL_RETRIEVAL_REASON_CODES = (
    "MISSING_QUANTITATIVE_PAIR", "MISSING_COMPARATIVE_ENTITY",
    "MISSING_ATTRIBUTION", "INSUFFICIENT_AUTHORIZED_SUPPORT",
    "CONTRAST_REQUIRED",
)
TECHNICAL_ISSUE_CODES = (
    "LLM_UNAVAILABLE", "LLM_INVOCATION_FAILED",
    "ADDITIONAL_RETRIEVER_UNAVAILABLE", "ADDITIONAL_RETRIEVAL_BUDGET_EXHAUSTED",
    "ADDITIONAL_RETRIEVAL_FAILED", "LLM_VALIDATION_ATTEMPTS_EXHAUSTED",
    "INVALID_INPUT", "RETRIEVAL_TECHNICAL_BLOCKER",
)

DEFAULT_VERIFICATION_INPUT_POLICY.update({
    "verification_system_prompt_version": "AGENT07_VERIFY_SYSTEM_V1",
    "verification_user_prompt_version": "AGENT07_VERIFY_USER_V1",
    "max_llm_attempts_per_claim": 3,
    "max_format_repair_attempts": 2,
    "max_additional_retrieval_requests": 1,
    "max_total_evidence_chars": 12000,
    "max_llm_evidence_chunks_per_claim": 8,
    "max_contrast_evidence_chunks": 3,
    "max_llm_evidence_per_source": 4,
    "reject_unknown_llm_fields": True,
})

_validate_verification_input_policy_phase3s = validate_verification_input_policy


def validate_verification_input_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_verification_input_policy_phase3s(policy)
    for key in (
        "verification_system_prompt_version",
        "verification_user_prompt_version",
    ):
        value[key] = _nonempty_string(value.get(key), key)
    for key in (
        "max_llm_attempts_per_claim",
        "max_format_repair_attempts",
        "max_additional_retrieval_requests",
        "max_total_evidence_chars",
        "max_llm_evidence_chunks_per_claim",
        "max_contrast_evidence_chunks",
        "max_llm_evidence_per_source",
    ):
        value[key] = _positive_int(value.get(key), key, allow_zero=key in {
            "max_format_repair_attempts",
            "max_additional_retrieval_requests",
            "max_contrast_evidence_chunks",
        })
    value["reject_unknown_llm_fields"] = _strict_bool(
        value.get("reject_unknown_llm_fields"), "reject_unknown_llm_fields"
    )
    return value


def get_verification_input_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_VERIFICATION_INPUT_POLICY)
    policy["claim_verification_intensity"] = dict(DEFAULT_CLAIM_VERIFICATION_INTENSITY)
    policy["minimum_resolved_evidence_by_claim_type"] = dict(DEFAULT_MINIMUM_RESOLVED_EVIDENCE)
    policy["retrieval_rounds_by_intensity"] = dict(DEFAULT_RETRIEVAL_ROUNDS_BY_INTENSITY)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("VERIFICATION_POLICY_INVALID:overrides:expected_mapping")
        policy.update(dict(overrides))
    return validate_verification_input_policy(policy)


# Phase 5: recomendaciones y propuestas localizadas; nunca aplica cambios.
CORRECTION_DECISIONS = (
    "NO_CORRECTION", "PROPOSE_CHANGE", "DEFER_TO_MANUAL_REVIEW", "NOT_CORRECTABLE",
)
CORRECTION_ACTION_TYPES = (
    "REMOVE_UNSUPPORTED_FRAGMENT", "REPLACE_NUMERIC_VALUE", "CORRECT_ATTRIBUTION",
    "NARROW_SCOPE", "ADD_QUALIFICATION", "REPLACE_CITATION", "SPLIT_CLAIM",
)
CORRECTION_PROPOSAL_STATUSES = (
    "NOT_PROPOSED", "PROPOSED", "ACCEPTED_FOR_REVERIFICATION", "REJECTED", "DEFERRED",
)
CORRECTION_CHANGE_SCOPES = ("TOKEN", "PHRASE", "CLAUSE", "SENTENCE", "MULTI_SENTENCE")
CORRECTION_SEMANTIC_CHANGE_LEVELS = ("NONE", "MINIMAL", "MODERATE", "SUBSTANTIAL")
ATTRIBUTION_RELATIONS = (
    "PROPOSED_BY", "DEVELOPED_BY", "INTRODUCED_IN", "REPORTED_BY", "EVALUATED_BY",
)
CORRECTION_LOCALIZATION_METHODS = (
    "CONTRACTUAL_SPAN", "EXACT_UNIQUE_MATCH", "NORMALIZED_UNIQUE_MATCH",
)
CORRECTION_COORDINATE_BASES = ("CLAIM_TEXT",)
CORRECTION_COORDINATE_SYSTEMS = ("PYTHON_CODEPOINT_OFFSETS",)
CORRECTION_REASON_CODES = (
    "LOCALIZED_NUMERIC_ERROR", "LOCALIZED_ATTRIBUTION_ERROR", "LOCALIZED_UNSUPPORTED_FRAGMENT",
    "LOCALIZED_EXTRAPOLATION", "INVALID_CITATION_WITH_VALID_REPLACEMENT",
    "AUTHORIZED_EVIDENCE_AVAILABLE", "SUPPORTED_NEW_QUALIFICATION",
    "AUTHORIZED_CORRECTION_EVIDENCE_UNAVAILABLE", "CORRECTION_LLM_UNAVAILABLE",
    "CLAIM_SPAN_IN_SECTION_REQUIRED", "CORRECTION_PROPOSAL_LIMIT_REACHED",
    # Causas terminales deterministas y semánticas para las decisiones
    # DEFER_TO_MANUAL_REVIEW / NO_CORRECTION / NOT_CORRECTABLE cuando el
    # motivo no es un fallo técnico (evidencia/LLM/límite) sino una
    # clasificación de política -- antes estas tres ramas producían
    # reason_codes=() vacío, sin causa auditable.
    "VERIFIER_REQUIRES_MANUAL_REVIEW", "NO_AUTOMATIC_CORRECTION_REQUIRED",
    "CLAIM_NOT_AUTOCORRECTABLE", "CORRECTION_RESPONSE_VALIDATION_EXHAUSTED",
)
CORRECTION_VALIDATION_ISSUE_CODES = (
    "TARGET_SPAN_NOT_FOUND", "AMBIGUOUS_TARGET_SPAN", "SPAN_COORDINATE_BASE_INVALID",
    "ORIGINAL_FINGERPRINT_MISMATCH", "TARGET_TEXT_MISMATCH", "UNKNOWN_EVIDENCE_ID",
    "UNAUTHORIZED_CORRECTION_EVIDENCE", "UNSUPPORTED_NEW_NUMERIC_VALUE",
    "NUMERIC_CONTEXT_MISMATCH", "UNAUTHORIZED_NEW_CITATION", "UNSUPPORTED_NEW_ATTRIBUTION",
    "ATTRIBUTION_RELATION_INVALID", "UNSUPPORTED_NEW_INFORMATION", "REMOVAL_ALTERS_SUPPORTED_MEANING",
    "OVERLAPPING_CORRECTIONS", "CORPUS_WIDE_DEPENDENCY", "CROSS_SOURCE_DISAGREEMENT",
    "REPLACEMENT_SCOPE_EXCEEDED", "PUNCTUATION_INTEGRITY_INVALID", "WHITESPACE_INTEGRITY_INVALID",
    "BRACKET_BALANCE_INVALID", "CITATION_SYNTAX_INVALID", "SPLIT_CLAIM_MANUAL_REVIEW_ONLY",
    "NEW_ENTITY_UNSUPPORTED", "NEW_TECHNICAL_TERM_UNSUPPORTED",
    "CLAIM_FINGERPRINT_REQUIRED", "SECTION_FINGERPRINT_REQUIRED", "SECTION_FINGERPRINT_MISMATCH",
    "CLAIM_SPAN_TEXT_MISMATCH", "FINGERPRINT_CONFLICT", "CORRECTION_LLM_INVOCATION_FAILED",
    "AUTOMATIC_PROPOSAL_REQUIRES_EVIDENCE", "REPLACEMENT_TEXT_REQUIRED", "NUMERIC_PAIRS_REQUIRED",
    "ATTRIBUTION_FIELDS_REQUIRED", "CITATION_REFS_REQUIRED", "QUALIFICATION_CONDITIONS_REQUIRED",
    "CORRECTION_RESPONSE_CLAIM_ID_MISMATCH", "CLAIM_SPAN_IN_SECTION_REQUIRED",
    "CORRECTION_RECOMMENDATION_CONTRADICTION", "CORRECTION_ACTION_FIELD_MATRIX_VIOLATION",
    "CITATION_TEXT_SPAN_REQUIRED", "CITATION_TEXT_SPAN_INVALID",
    "MALFORMED_PRIOR_CORRECTION_PROPOSAL_IGNORED", "STALE_PRIOR_CORRECTION_PROPOSAL",
    "PRIOR_CORRECTION_SECTION_MISMATCH", "CORRECTION_PROPOSAL_LIMIT_REACHED",
    "NARROW_SCOPE_CONDITIONS_REQUIRED", "CITATION_TEXT_REFERENCE_MISMATCH",
    "AUTHORIZED_CORRECTION_EVIDENCE_UNAVAILABLE", "CORRECTION_LLM_UNAVAILABLE",
)
DEFAULT_VERIFICATION_INPUT_POLICY.update({
    "correction_system_prompt_version": "AGENT07_CORRECTION_SYSTEM_V1",
    "correction_user_prompt_version": "AGENT07_CORRECTION_USER_V3_5T",
    "max_correction_llm_attempts": 3,
    "max_correction_format_repair_attempts": 2,
    "max_correction_proposals_per_claim": 2,
    "max_replacement_chars": 500,
    "max_target_span_chars": 1000,
    "allow_multi_sentence_automatic_correction": False,
    "allow_corpus_wide_for_automatic_correction": False,
    "require_authorized_evidence_for_correction": True,
    "require_independent_reverification": True,
    "require_exact_original_fingerprint": True,
    "allow_unit_conversion": False,
})
_validate_verification_input_policy_phase4u = validate_verification_input_policy

def validate_verification_input_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_verification_input_policy_phase4u(policy)
    for key in ("correction_system_prompt_version", "correction_user_prompt_version"):
        value[key] = _nonempty_string(value.get(key), key)
    for key in ("max_correction_llm_attempts", "max_correction_format_repair_attempts",
                "max_correction_proposals_per_claim", "max_replacement_chars", "max_target_span_chars"):
        value[key] = _positive_int(value.get(key), key, allow_zero=key == "max_correction_format_repair_attempts")
    for key in ("allow_multi_sentence_automatic_correction", "allow_corpus_wide_for_automatic_correction",
                "require_authorized_evidence_for_correction", "require_independent_reverification",
                "require_exact_original_fingerprint", "allow_unit_conversion"):
        value[key] = _strict_bool(value.get(key), key)
    return value

def get_verification_input_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_VERIFICATION_INPUT_POLICY)
    policy["claim_verification_intensity"] = dict(DEFAULT_CLAIM_VERIFICATION_INTENSITY)
    policy["minimum_resolved_evidence_by_claim_type"] = dict(DEFAULT_MINIMUM_RESOLVED_EVIDENCE)
    policy["retrieval_rounds_by_intensity"] = dict(DEFAULT_RETRIEVAL_ROUNDS_BY_INTENSITY)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("VERIFICATION_POLICY_INVALID:overrides:expected_mapping")
        policy.update(dict(overrides))
    return validate_verification_input_policy(policy)

# Phase 6.1: contratos y policy para reverificación virtual independiente previa a aplicación.
REVERIFICATION_PROCESS_NAME = "VIRTUAL_INDEPENDENT_PRE_APPLICATION_REVERIFICATION"
REVERIFICATION_EXECUTION_STATUSES = (
    "NOT_REQUESTED", "PENDING", "COMPLETED", "BLOCKED", "FAILED",
)
REVERIFICATION_SCIENTIFIC_OUTCOMES = (
    "NOT_EVALUATED", "SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE", "NOT_VERIFIABLE", "AMBIGUOUS",
)
REVERIFICATION_ACCEPTANCE_DECISIONS = (
    "ACCEPT_FOR_07C", "REJECT_PROPOSAL", "DEFER_TO_MANUAL_REVIEW",
)
REVERIFICATION_RISK_DELTAS = (
    "REDUCED", "UNCHANGED", "INCREASED", "NOT_COMPARABLE",
)
REVERIFICATION_BLOCK_CATEGORIES = (
    "CONTRACTUAL_INCOMPATIBILITY", "TEMPORARY_TECHNICAL_DEPENDENCY",
    "NEGATIVE_SCIENTIFIC_RESULT", "SCIENTIFIC_AMBIGUITY",
)
REVERIFICATION_REASON_CODES = (
    "VIRTUAL_REVERIFICATION_NOT_REQUESTED",
    "PROPOSAL_CONTRACT_INCOMPATIBLE",
    "TEMPORARY_TECHNICAL_DEPENDENCY",
    "SCIENTIFIC_OUTCOME_NEGATIVE",
    "SCIENTIFIC_AMBIGUITY",
    "TARGET_ISSUE_RESOLVED",
    "TARGET_ISSUE_REMAINS",
    "NEW_CRITICAL_ISSUE_INTRODUCED",
    "RISK_REDUCED",
    "RISK_UNCHANGED",
    "RISK_INCREASED",
    "RISK_NOT_COMPARABLE",
    "SUPPORTED_MEANING_NOT_PRESERVED",
    "INTENDED_SEMANTIC_CHANGE_INVALID",
    "UNINTENDED_SEMANTIC_CHANGE_DETECTED",
    "FROZEN_EVIDENCE_REQUIRED",
    "REVERIFICATION_RETRIEVAL_FORBIDDEN",
    "MULTIPLE_ACCEPTABLE_PROPOSALS_FOR_CLAIM",
)
REVERIFICATION_TECHNICAL_ISSUE_CODES = (
    "REVERIFICATION_DEPENDENCY_UNAVAILABLE",
    "REVERIFICATION_LLM_INVOCATION_FAILED",
    "REVERIFICATION_RESPONSE_INVALID",
    "REVERIFICATION_ATTEMPTS_EXHAUSTED",
)
REVERIFICATION_CRITICAL_NEW_ISSUE_CODES = (
    "UNSUPPORTED_NUMERIC_VALUE",
    "NUMERIC_CONTEXT_MISMATCH",
    "INVALID_CITATION",
    "ATTRIBUTION_ERROR",
    "UNSUPPORTED_EXTRAPOLATION",
    "CLAIM_EVIDENCE_CONFLICT",
    "UNAUTHORIZED_SOURCE",
    "DOCUMENT_IDENTITY_INVALID",
    "TEXT_INTEGRITY_INVALID",
)
REVERIFICATION_APPLICATION_ORDER_FIELDS = (
    "section_id", "claim_span_in_section.start", "target_span_in_claim.start", "correction_id",
)
REVERIFICATION_ALLOWED_PROPOSAL_STATUSES = ("ACCEPTED_FOR_REVERIFICATION",)
REVERIFICATION_RISK_POLICY_VERSION = "AGENT07_HALLUCINATION_RISK_V1"
REVERIFICATION_ACTION_TARGET_ISSUE_MATRIX = {
    "REPLACE_NUMERIC_VALUE": ("UNSUPPORTED_NUMERIC_VALUE", "NUMERIC_CONTEXT_MISMATCH"),
    "CORRECT_ATTRIBUTION": ("ATTRIBUTION_ERROR",),
    "REPLACE_CITATION": ("INVALID_CITATION", "UNAUTHORIZED_SOURCE"),
    "NARROW_SCOPE": ("UNSUPPORTED_EXTRAPOLATION", "CLAIM_EVIDENCE_CONFLICT", "PARTIAL_SUPPORT"),
    "ADD_QUALIFICATION": ("UNSUPPORTED_EXTRAPOLATION", "CLAIM_EVIDENCE_CONFLICT", "PARTIAL_SUPPORT"),
    "REMOVE_UNSUPPORTED_FRAGMENT": ("CLAIM_EVIDENCE_CONFLICT", "PARTIAL_SUPPORT", "INSUFFICIENT_EVIDENCE"),
}

DEFAULT_VERIFICATION_INPUT_POLICY.update({
    "reverification_process_name": REVERIFICATION_PROCESS_NAME,
    "reverification_risk_policy_version": REVERIFICATION_RISK_POLICY_VERSION,
    "reverification_retrieval_rounds": 0,
    "max_accepted_proposals_per_claim": 1,
    "require_frozen_reverification_evidence": True,
    "require_same_risk_policy_version": True,
    "require_virtual_proposed_claim_reconstruction": True,
    "reverification_allowed_proposal_statuses": REVERIFICATION_ALLOWED_PROPOSAL_STATUSES,
    "reverification_critical_new_issue_codes": REVERIFICATION_CRITICAL_NEW_ISSUE_CODES,
    "reverification_application_order_fields": REVERIFICATION_APPLICATION_ORDER_FIELDS,
    "reverification_action_target_issue_matrix": REVERIFICATION_ACTION_TARGET_ISSUE_MATRIX,
})

_validate_verification_input_policy_phase5t = validate_verification_input_policy


def validate_verification_input_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_verification_input_policy_phase5t(policy)
    value["reverification_process_name"] = _nonempty_string(
        value.get("reverification_process_name"), "reverification_process_name"
    )
    if value["reverification_process_name"] != REVERIFICATION_PROCESS_NAME:
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_process_name:unsupported")
    value["reverification_risk_policy_version"] = _nonempty_string(
        value.get("reverification_risk_policy_version"), "reverification_risk_policy_version"
    )
    rounds = value.get("reverification_retrieval_rounds")
    if type(rounds) is not int or rounds != 0:
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_retrieval_rounds:must_be_zero")
    value["max_accepted_proposals_per_claim"] = _positive_int(
        value.get("max_accepted_proposals_per_claim"), "max_accepted_proposals_per_claim"
    )
    for key in (
        "require_frozen_reverification_evidence",
        "require_same_risk_policy_version",
        "require_virtual_proposed_claim_reconstruction",
    ):
        value[key] = _strict_bool(value.get(key), key)
        if value[key] is not True:
            raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:must_be_true")
    value["reverification_allowed_proposal_statuses"] = _string_tuple(
        value.get("reverification_allowed_proposal_statuses"),
        "reverification_allowed_proposal_statuses",
    )
    invalid_statuses = sorted(
        set(value["reverification_allowed_proposal_statuses"]) - set(CORRECTION_PROPOSAL_STATUSES)
    )
    if invalid_statuses:
        raise ValueError(
            "VERIFICATION_POLICY_INVALID:reverification_allowed_proposal_statuses:unsupported_"
            + ",".join(invalid_statuses)
        )
    value["reverification_critical_new_issue_codes"] = _string_tuple(
        value.get("reverification_critical_new_issue_codes"),
        "reverification_critical_new_issue_codes",
    )
    value["reverification_application_order_fields"] = _string_tuple(
        value.get("reverification_application_order_fields"),
        "reverification_application_order_fields",
    )
    if tuple(value["reverification_application_order_fields"]) != REVERIFICATION_APPLICATION_ORDER_FIELDS:
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_application_order_fields:unsupported_order")
    matrix=value.get("reverification_action_target_issue_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_action_target_issue_matrix:expected_mapping")
    normalized_matrix={}
    for action, issues in REVERIFICATION_ACTION_TARGET_ISSUE_MATRIX.items():
        if action not in matrix:
            raise ValueError(f"VERIFICATION_POLICY_INVALID:reverification_action_target_issue_matrix:missing_{action}")
        normalized_matrix[action]=_string_tuple(matrix[action], f"reverification_action_target_issue_matrix.{action}")
    value["reverification_action_target_issue_matrix"]=normalized_matrix
    return value


def get_verification_input_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_VERIFICATION_INPUT_POLICY)
    policy["claim_verification_intensity"] = dict(DEFAULT_CLAIM_VERIFICATION_INTENSITY)
    policy["minimum_resolved_evidence_by_claim_type"] = dict(DEFAULT_MINIMUM_RESOLVED_EVIDENCE)
    policy["retrieval_rounds_by_intensity"] = dict(DEFAULT_RETRIEVAL_ROUNDS_BY_INTENSITY)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("VERIFICATION_POLICY_INVALID:overrides:expected_mapping")
        policy.update(dict(overrides))
    return validate_verification_input_policy(policy)

# Phase 6.3: reverificación virtual independiente con double inyectable.
REVERIFICATION_PROMPT_VERSION = "AGENT07_REVERIFICATION_USER_V1"
REVERIFICATION_SYSTEM_PROMPT_VERSION = "AGENT07_REVERIFICATION_SYSTEM_V1"
REVERIFICATION_OUTPUT_FIELDS = (
    "correction_id", "claim_id", "proposed_verdict", "support_level",
    "evidence_ids_used", "observed_issue_codes", "target_issues_resolved",
    "supported_meaning_preserved", "intended_semantic_change_valid",
    "unintended_semantic_change_absent", "scope_change_valid",
    "numeric_change_valid", "attribution_change_valid", "citation_change_valid",
    "manual_review_recommended", "reason_codes", "rationale", "confidence",
)
REVERIFICATION_CONFIDENCE_MIN = 0.0
REVERIFICATION_CONFIDENCE_MAX = 1.0
DEFAULT_VERIFICATION_INPUT_POLICY.update({
    "reverification_system_prompt_version": REVERIFICATION_SYSTEM_PROMPT_VERSION,
    "reverification_user_prompt_version": REVERIFICATION_PROMPT_VERSION,
    "max_reverification_llm_attempts": 3,
    "max_reverification_format_repair_attempts": 2,
    "reverification_confidence_min": REVERIFICATION_CONFIDENCE_MIN,
    "reverification_confidence_max": REVERIFICATION_CONFIDENCE_MAX,
})
_validate_verification_input_policy_phase62r = validate_verification_input_policy

def validate_verification_input_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_verification_input_policy_phase62r(policy)
    for key in ("reverification_system_prompt_version", "reverification_user_prompt_version"):
        value[key] = _nonempty_string(value.get(key), key)
    value["max_reverification_llm_attempts"] = _positive_int(
        value.get("max_reverification_llm_attempts"), "max_reverification_llm_attempts"
    )
    value["max_reverification_format_repair_attempts"] = _positive_int(
        value.get("max_reverification_format_repair_attempts"),
        "max_reverification_format_repair_attempts", allow_zero=True,
    )
    for key in ("reverification_confidence_min", "reverification_confidence_max"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:expected_number")
        value[key] = float(item)
    if not (0.0 <= value["reverification_confidence_min"] < value["reverification_confidence_max"] <= 1.0):
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_confidence_range")
    return value

def get_verification_input_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_VERIFICATION_INPUT_POLICY)
    policy["claim_verification_intensity"] = dict(DEFAULT_CLAIM_VERIFICATION_INTENSITY)
    policy["minimum_resolved_evidence_by_claim_type"] = dict(DEFAULT_MINIMUM_RESOLVED_EVIDENCE)
    policy["retrieval_rounds_by_intensity"] = dict(DEFAULT_RETRIEVAL_ROUNDS_BY_INTENSITY)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("VERIFICATION_POLICY_INVALID:overrides:expected_mapping")
        policy.update(dict(overrides))
    return validate_verification_input_policy(policy)


# Phase 6.3R: immutable reverification context and scientific output coherence.
REVERIFICATION_ASSESSMENTS = ("VALID", "INVALID", "NOT_APPLICABLE")

# Catálogo explícito de hallazgos atribuibles al claim propuesto o a la
# evidencia científica usada para evaluarlo. Se excluyen deliberadamente
# códigos de parsing, schema, LLM, intentos, infraestructura, retrieval,
# dependencias, fingerprints y contratos.
#
# DOCUMENT_IDENTITY_INVALID y UNAUTHORIZED_SOURCE se conservan porque describen
# una invalidez determinista de la evidencia usada para sostener el claim, no
# una avería técnica del runtime. En cambio RETRIEVAL_TECHNICAL_BLOCKER se
# excluye porque describe disponibilidad/ejecución del recuperador.
REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES = (
    "INVALID_CITATION",
    "UNSUPPORTED_NUMERIC_VALUE",
    "NUMERIC_CONTEXT_MISMATCH",
    "DOCUMENT_IDENTITY_INVALID",
    "UNAUTHORIZED_SOURCE",
    "TEXT_INTEGRITY_INVALID",
    "ATTRIBUTION_ERROR",
    "UNSUPPORTED_EXTRAPOLATION",
    "CLAIM_EVIDENCE_CONFLICT",
    "CROSS_SOURCE_DISAGREEMENT",
    "INTERNAL_TEXT_INCONSISTENCY",
    "PARTIAL_SUPPORT",
    "INSUFFICIENT_EVIDENCE",
)

# Matriz reutilizada del núcleo científico del Agente 07. La comprobación
# integrada contra verification_agent.py queda pendiente para el repositorio
# completo; esta candidata mínima no contiene ese módulo.
REVERIFICATION_SUPPORT_LEVEL_BY_VERDICT = dict(SUPPORT_LEVEL_BY_VERDICT)
REVERIFICATION_LLM_REASON_CODES = (
    "TARGET_ISSUE_APPEARS_RESOLVED",
    "TARGET_ISSUE_APPEARS_UNRESOLVED",
    "SUPPORTED_MEANING_PRESERVED",
    "SUPPORTED_MEANING_NOT_PRESERVED",
    "INTENDED_SEMANTIC_CHANGE_VALID",
    "INTENDED_SEMANTIC_CHANGE_INVALID",
    "UNINTENDED_SEMANTIC_CHANGE_ABSENT",
    "UNINTENDED_SEMANTIC_CHANGE_DETECTED",
    "MANUAL_REVIEW_RECOMMENDED",
    "EVIDENCE_INSUFFICIENT_FOR_REVERIFICATION",
)
REVERIFICATION_SCIENTIFIC_COHERENCE_REASON_CODES = (
    "REVERIFICATION_SUPPORTED_REQUIRES_EVIDENCE",
    "REVERIFICATION_SUPPORTED_REQUIRES_SUPPORT_LEVEL",
    "REVERIFICATION_RESOLVED_ISSUE_STILL_OBSERVED",
    "REVERIFICATION_NOT_EVALUATED_CANNOT_RESOLVE_TARGET",
    "REVERIFICATION_ACTION_ASSESSMENT_MISMATCH",
    "REVERIFICATION_VERDICT_SUPPORT_LEVEL_MISMATCH",
)
COMPARISON_GATE_ACTION_NOT_AVAILABLE = "NOT_AVAILABLE"

# Phase 6.4V: complete, disjoint classification of every reason code emitted by
# run_virtual_reverification_prechecks and its contractual/action validators.
# Parameterized contract errors are classified by the explicit family before the first colon;
# the original full code remains preserved in decision_trace.
PRECHECK_TEMPORARY_TECHNICAL_REASON_CODES = (
    "VIRTUAL_PROPOSED_CLAIM_BUILD_FAILED",
    "PROPOSAL_FINGERPRINT_RECOMPUTATION_FAILED",
    "REVERIFICATION_DEPENDENCY_UNAVAILABLE",
    "REVERIFICATION_POLICY_UNAVAILABLE",
)
PRECHECK_PERMANENT_CONTRACTUAL_REASON_CODES = (
    "REVERIFICATION_CONTRACT_INVALID",
    "REVERIFICATION_INPUT_NOT_MAPPING",
    "REVERIFICATION_INPUT_FIELDS_MISSING",
    "REVERIFICATION_PROCESS_NAME_INVALID",
    "REVERIFICATION_PROPOSAL_STATUS_NOT_ALLOWED",
    "REVERIFICATION_CORRECTION_APPLIED_REQUIRED",
    "REVERIFICATION_PHYSICAL_APPLICATION_FORBIDDEN",
    "REVERIFICATION_RETRIEVAL_FORBIDDEN",
    "REVERIFICATION_EVIDENCE_REQUIRED",
    "REVERIFICATION_EVIDENCE_NOT_FROZEN",
    "REVERIFICATION_EVIDENCE_NOT_AUTHORIZED",
    "REVERIFICATION_AUTHORIZED_EVIDENCE_ID_DUPLICATE",
    "TARGET_ISSUE_CODE_NOT_PRESENT",
    "BASE_CLAIM_FINGERPRINT_MISMATCH",
    "BASE_SECTION_FINGERPRINT_MISMATCH",
    "REVERIFICATION_APPLICATION_ORDER_KEY_MISMATCH",
    "REVERIFICATION_INPUT_CONTRACT_INVALID",
    "REVERIFICATION_POLICY_INVARIANT_VIOLATION",
    "SECTION_TEXT_REQUIRED",
    "ORIGINAL_CLAIM_FINGERPRINT_MISMATCH",
    "ORIGINAL_SECTION_FINGERPRINT_MISMATCH",
    "CLAIM_SPAN_TEXT_MISMATCH",
    "TARGET_SPAN_TEXT_MISMATCH",
    "PROPOSED_CLAIM_TEXT_FINGERPRINT_MISMATCH",
    "REVERIFICATION_PROPOSAL_NOT_CURRENT",
    "PROPOSAL_FINGERPRINT_MISMATCH",
    "REVERIFICATION_EVIDENCE_ORDER_MISMATCH",
    "REVERIFICATION_EVIDENCE_SET_MISMATCH",
    "REVERIFICATION_EVIDENCE_ID_DUPLICATE",
    "REVERIFICATION_EVIDENCE_ID_MISSING",
    "DOCUMENT_IDENTITY_INVALID",
    "REVERIFICATION_CORPUS_WIDE_EVIDENCE_FORBIDDEN",
    "REVERIFICATION_EVIDENCE_TEXT_MISSING",
)
PRECHECK_DETERMINISTIC_SCIENTIFIC_REJECTION_CODES = (
    "PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH",
    "TEXT_INTEGRITY_INVALID",
    "ACTION_TARGET_ISSUE_MISMATCH",
    "NUMERIC_PAIRS_REQUIRED",
    "UNSUPPORTED_NEW_NUMERIC_VALUE",
    "NUMERIC_CONTEXT_MISMATCH",
    "UNDECLARED_NEW_NUMERIC_VALUE",
    "ATTRIBUTION_FIELDS_REQUIRED",
    "UNSUPPORTED_NEW_ATTRIBUTION",
    "ATTRIBUTION_RELATION_NOT_SUPPORTED",
    "ATTRIBUTION_OBJECT_NOT_SUPPORTED",
    "CITATION_TEXT_SPAN_REQUIRED",
    "CITATION_TEXT_SPAN_STALE",
    "CITATION_TEXT_REFERENCE_MISMATCH",
    "NEW_CITATION_MARKER_MISSING",
    "NEW_CITATION_REFERENCE_MISMATCH",
    "NEW_CITATION_DOES_NOT_SUPPORT_PROPOSED_CLAIM",
    "NARROW_SCOPE_CONDITIONS_REQUIRED",
    "SCOPE_NOT_NARROWED",
    "SCOPE_EXPANSION_DETECTED",
    "ORIGINAL_SCOPE_CONDITION_REMOVED",
    "UNSUPPORTED_NEW_INFORMATION",
    "QUALIFICATION_CONDITIONS_REQUIRED",
    "QUALIFICATION_DIFFERENTIAL_INVALID",
    "QUALIFICATION_INCREASES_CERTAINTY",
    "SUPPORTED_MEANING_NOT_PRESERVED",
    "EMPTY_PROPOSED_CLAIM",
    "REMOVAL_TARGET_HALLUCINATION_MISMATCH",
    "REMOVAL_ALTERS_SUPPORTED_MEANING",
    "REVERIFICATION_ACTION_UNSUPPORTED",
)
PRECHECK_PARAMETERIZED_REASON_CODE_FAMILIES = (
    "REVERIFICATION_CONTRACT_INVALID",
    "REVERIFICATION_INPUT_FIELDS_MISSING",
    "TARGET_ISSUE_CODE_NOT_PRESENT",
    "REVERIFICATION_INPUT_CONTRACT_INVALID",
)
# Independent audit inventory of all closed codes that the Phase 6.2 precheck
# path can return directly or through its input-contract normalizer.
PRECHECK_RUNTIME_EMITTED_REASON_CODES = (
    "REVERIFICATION_INPUT_NOT_MAPPING", "REVERIFICATION_INPUT_FIELDS_MISSING",
    "REVERIFICATION_PROCESS_NAME_INVALID", "REVERIFICATION_PROPOSAL_STATUS_NOT_ALLOWED",
    "REVERIFICATION_CORRECTION_APPLIED_REQUIRED", "REVERIFICATION_PHYSICAL_APPLICATION_FORBIDDEN",
    "REVERIFICATION_RETRIEVAL_FORBIDDEN", "REVERIFICATION_EVIDENCE_REQUIRED",
    "REVERIFICATION_EVIDENCE_NOT_FROZEN", "REVERIFICATION_EVIDENCE_NOT_AUTHORIZED",
    "REVERIFICATION_AUTHORIZED_EVIDENCE_ID_DUPLICATE", "TARGET_ISSUE_CODE_NOT_PRESENT",
    "BASE_CLAIM_FINGERPRINT_MISMATCH", "BASE_SECTION_FINGERPRINT_MISMATCH",
    "REVERIFICATION_APPLICATION_ORDER_KEY_MISMATCH", "REVERIFICATION_INPUT_CONTRACT_INVALID",
    "REVERIFICATION_POLICY_INVARIANT_VIOLATION", "SECTION_TEXT_REQUIRED",
    "ORIGINAL_CLAIM_FINGERPRINT_MISMATCH", "ORIGINAL_SECTION_FINGERPRINT_MISMATCH",
    "CLAIM_SPAN_TEXT_MISMATCH", "TARGET_SPAN_TEXT_MISMATCH",
    "VIRTUAL_PROPOSED_CLAIM_BUILD_FAILED", "PROPOSED_CLAIM_RECONSTRUCTION_MISMATCH",
    "PROPOSED_CLAIM_TEXT_FINGERPRINT_MISMATCH", "REVERIFICATION_PROPOSAL_NOT_CURRENT",
    "PROPOSAL_FINGERPRINT_RECOMPUTATION_FAILED", "PROPOSAL_FINGERPRINT_MISMATCH",
    "REVERIFICATION_EVIDENCE_ORDER_MISMATCH", "REVERIFICATION_EVIDENCE_SET_MISMATCH",
    "REVERIFICATION_EVIDENCE_ID_DUPLICATE", "REVERIFICATION_EVIDENCE_ID_MISSING",
    "DOCUMENT_IDENTITY_INVALID", "REVERIFICATION_CORPUS_WIDE_EVIDENCE_FORBIDDEN",
    "REVERIFICATION_EVIDENCE_TEXT_MISSING", "TEXT_INTEGRITY_INVALID",
    "ACTION_TARGET_ISSUE_MISMATCH", "NUMERIC_PAIRS_REQUIRED",
    "UNSUPPORTED_NEW_NUMERIC_VALUE", "NUMERIC_CONTEXT_MISMATCH",
    "UNDECLARED_NEW_NUMERIC_VALUE", "ATTRIBUTION_FIELDS_REQUIRED",
    "UNSUPPORTED_NEW_ATTRIBUTION", "ATTRIBUTION_RELATION_NOT_SUPPORTED",
    "ATTRIBUTION_OBJECT_NOT_SUPPORTED", "CITATION_TEXT_SPAN_REQUIRED",
    "CITATION_TEXT_SPAN_STALE", "CITATION_TEXT_REFERENCE_MISMATCH",
    "NEW_CITATION_MARKER_MISSING", "NEW_CITATION_REFERENCE_MISMATCH",
    "NEW_CITATION_DOES_NOT_SUPPORT_PROPOSED_CLAIM", "NARROW_SCOPE_CONDITIONS_REQUIRED",
    "SCOPE_NOT_NARROWED", "SCOPE_EXPANSION_DETECTED",
    "ORIGINAL_SCOPE_CONDITION_REMOVED", "UNSUPPORTED_NEW_INFORMATION",
    "QUALIFICATION_CONDITIONS_REQUIRED", "QUALIFICATION_DIFFERENTIAL_INVALID",
    "QUALIFICATION_INCREASES_CERTAINTY", "SUPPORTED_MEANING_NOT_PRESERVED",
    "EMPTY_PROPOSED_CLAIM", "REMOVAL_TARGET_HALLUCINATION_MISMATCH",
    "REMOVAL_ALTERS_SUPPORTED_MEANING", "REVERIFICATION_ACTION_UNSUPPORTED",
)
PRECHECK_EMITTABLE_REASON_CODES = PRECHECK_RUNTIME_EMITTED_REASON_CODES
PRECHECK_GATE_REASON_CODES = PRECHECK_EMITTABLE_REASON_CODES
# Phase 6.2 does not emit LLM invocation failures. Those remain exclusively in
# REVERIFICATION_TECHNICAL_ISSUE_CODES for Phase 6.3.
PRECHECK_GATE_TECHNICAL_ISSUE_CODES = (
    "REVERIFICATION_DEPENDENCY_UNAVAILABLE",
)


REVERIFICATION_ACTION_ASSESSMENT_FIELD = {
    "REPLACE_NUMERIC_VALUE": "numeric_assessment",
    "CORRECT_ATTRIBUTION": "attribution_assessment",
    "REPLACE_CITATION": "citation_assessment",
    "NARROW_SCOPE": "scope_assessment",
    "ADD_QUALIFICATION": "scope_assessment",
    "REMOVE_UNSUPPORTED_FRAGMENT": "scope_assessment",
    "SPLIT_CLAIM": "scope_assessment",
}
REVERIFICATION_OUTPUT_FIELDS = (
    "correction_id", "claim_id", "proposed_verdict", "support_level",
    "evidence_ids_used", "observed_issue_codes", "target_issues_resolved",
    "supported_meaning_preserved", "intended_semantic_change_valid",
    "unintended_semantic_change_absent", "scope_assessment",
    "numeric_assessment", "attribution_assessment", "citation_assessment",
    "manual_review_recommended", "reason_codes", "rationale", "confidence",
)

DEFAULT_VERIFICATION_INPUT_POLICY.update({
    "reverification_observed_issue_codes": REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES,
    "reverification_llm_reason_codes": REVERIFICATION_LLM_REASON_CODES,
    "reverification_support_level_by_verdict": REVERIFICATION_SUPPORT_LEVEL_BY_VERDICT,
})

_validate_verification_input_policy_phase63 = validate_verification_input_policy

def validate_verification_input_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_verification_input_policy_phase63(policy)
    value["reverification_observed_issue_codes"] = _string_tuple(
        value.get("reverification_observed_issue_codes"),
        "reverification_observed_issue_codes",
    )
    value["reverification_llm_reason_codes"] = _string_tuple(
        value.get("reverification_llm_reason_codes"),
        "reverification_llm_reason_codes",
    )
    if set(value["reverification_observed_issue_codes"]) != set(REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES):
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_observed_issue_codes:must_match_contract")
    if set(value["reverification_llm_reason_codes"]) != set(REVERIFICATION_LLM_REASON_CODES):
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_llm_reason_codes:must_match_contract")
    matrix = value.get("reverification_support_level_by_verdict")
    if not isinstance(matrix, Mapping):
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_support_level_by_verdict:expected_mapping")
    normalized_matrix = {str(k): str(v) for k, v in matrix.items()}
    if normalized_matrix != REVERIFICATION_SUPPORT_LEVEL_BY_VERDICT:
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_support_level_by_verdict:must_match_contract")
    value["reverification_support_level_by_verdict"] = normalized_matrix
    return value


def get_verification_input_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_VERIFICATION_INPUT_POLICY)
    policy["claim_verification_intensity"] = dict(DEFAULT_CLAIM_VERIFICATION_INTENSITY)
    policy["minimum_resolved_evidence_by_claim_type"] = dict(DEFAULT_MINIMUM_RESOLVED_EVIDENCE)
    policy["retrieval_rounds_by_intensity"] = dict(DEFAULT_RETRIEVAL_ROUNDS_BY_INTENSITY)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("VERIFICATION_POLICY_INVALID:overrides:expected_mapping")
        policy.update(dict(overrides))
    return validate_verification_input_policy(policy)


# Phase 6.4: deterministic before/after comparison and provisional 07C decision.
REVERIFICATION_NONCRITICAL_REVIEW_ISSUE_CODES = (
    "CROSS_SOURCE_DISAGREEMENT",
    "INTERNAL_TEXT_INCONSISTENCY",
    "PARTIAL_SUPPORT",
    "INSUFFICIENT_EVIDENCE",
)
REVERIFICATION_COMPARISON_REASON_CODES = (
    "COMPARISON_CONTEXT_MISMATCH",
    "COMPARISON_REQUIRED_IDENTITY_FIELD_MISSING",
    "TARGET_ISSUE_NOT_RESOLVED",
    "REPORTED_RESOLUTION_MISMATCH",
    "CRITICAL_NEW_ISSUE_INTRODUCED",
    "NONCRITICAL_NEW_ISSUE_REQUIRES_REVIEW",
    "REMAINING_SCIENTIFIC_AMBIGUITY",
    "RISK_REDUCED",
    "RISK_UNCHANGED",
    "RISK_INCREASED",
    "RISK_NOT_COMPARABLE",
    "ACTION_ASSESSMENT_INVALID",
    "ACTION_ASSESSMENT_NOT_APPLICABLE",
    "SUPPORTED_MEANING_NOT_PRESERVED",
    "INTENDED_SEMANTIC_CHANGE_INVALID",
    "UNINTENDED_SEMANTIC_CHANGE_DETECTED",
    "MANUAL_REVIEW_RECOMMENDED",
    "COMPARISON_ACCEPTED_FOR_07C",
)
REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES = (
    "COMPARISON_INPUT_INVALID",
    "COMPARISON_PRECHECK_INVALID",
    "COMPARISON_GATE_ACTION_NOT_AVAILABLE",
    "COMPARISON_REVERIFICATION_RESULT_INVALID",
    "COMPARISON_RISK_POLICY_MISMATCH",
)
REVERIFICATION_ISSUE_ORDER = REVERIFICATION_OBSERVED_SCIENTIFIC_ISSUE_CODES

DEFAULT_VERIFICATION_INPUT_POLICY.update({
    "reverification_noncritical_review_issue_codes": REVERIFICATION_NONCRITICAL_REVIEW_ISSUE_CODES,
    "reverification_comparison_reason_codes": REVERIFICATION_COMPARISON_REASON_CODES,
    "reverification_comparison_technical_issue_codes": REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES,
    "reverification_issue_order": REVERIFICATION_ISSUE_ORDER,
})

_validate_verification_input_policy_phase63s = validate_verification_input_policy

def validate_verification_input_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_verification_input_policy_phase63s(policy)
    exact = {
        "reverification_noncritical_review_issue_codes": REVERIFICATION_NONCRITICAL_REVIEW_ISSUE_CODES,
        "reverification_comparison_reason_codes": REVERIFICATION_COMPARISON_REASON_CODES,
        "reverification_comparison_technical_issue_codes": REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES,
        "reverification_issue_order": REVERIFICATION_ISSUE_ORDER,
    }
    for key, expected in exact.items():
        normalized = _string_tuple(value.get(key), key)
        if tuple(normalized) != tuple(expected):
            raise ValueError(f"VERIFICATION_POLICY_INVALID:{key}:must_match_contract")
        value[key] = normalized
    if tuple(value["reverification_critical_new_issue_codes"]) != tuple(REVERIFICATION_CRITICAL_NEW_ISSUE_CODES):
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_critical_new_issue_codes:must_match_contract")
    return value

# Phase 6.4R: closed comparison vocabularies and projection-specific risk policy.
REVERIFICATION_COMPARISON_RISK_POLICY_VERSION = "AGENT07_COMPARISON_RISK_PROJECTION_V1"
REVERIFICATION_COMPARISON_REASON_CODES = (
    "COMPARISON_INPUT_INVALID",
    "COMPARISON_CORRECTION_ACTION_INVALID",
    "COMPARISON_PRECHECK_BLOCKED",
    "COMPARISON_PRECHECK_REJECTED",
    "COMPARISON_PRECHECK_INVALID",
    "COMPARISON_GATE_ACTION_NOT_AVAILABLE",
    "COMPARISON_REVERIFICATION_RESULT_INVALID",
    "COMPARISON_CONTEXT_MISMATCH",
    "COMPARISON_REQUIRED_IDENTITY_FIELD_MISSING",
    "COMPARISON_CONTEXT_SNAPSHOT_MISMATCH",
    "COMPARISON_RISK_POLICY_MISMATCH",
    "TARGET_ISSUE_NOT_RESOLVED",
    "REPORTED_RESOLUTION_MISMATCH",
    "CRITICAL_NEW_ISSUE_INTRODUCED",
    "NONCRITICAL_NEW_ISSUE_REQUIRES_REVIEW",
    "REMAINING_SCIENTIFIC_AMBIGUITY",
    "ACTION_ASSESSMENT_INVALID",
    "ACTION_ASSESSMENT_NOT_APPLICABLE",
    "SUPPORTED_MEANING_NOT_PRESERVED",
    "INTENDED_SEMANTIC_CHANGE_INVALID",
    "UNINTENDED_SEMANTIC_CHANGE_DETECTED",
    "MANUAL_REVIEW_RECOMMENDED",
    "RISK_REDUCED",
    "RISK_UNCHANGED",
    "RISK_INCREASED",
    "RISK_NOT_COMPARABLE",
    "COMPARISON_ACCEPTED_FOR_07C",
)
REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES = (
    "COMPARISON_DEPENDENCY_UNAVAILABLE",
    "COMPARISON_POLICY_UNAVAILABLE",
    "COMPARISON_RESULT_ABSENT",
)
REVERIFICATION_COMPARISON_CLASSIFICATION_MATRIX = {
    "CONTRACTUAL_OR_IDENTITY_INVALID": "REJECT_PROPOSAL",
    "SCIENTIFICALLY_INVALID": "REJECT_PROPOSAL",
    "TECHNICAL_DEPENDENCY_TEMPORARY": "DEFER_TO_MANUAL_REVIEW",
    "RESULT_ABSENT": "DEFER_TO_MANUAL_REVIEW",
    "POLICY_UNAVAILABLE": "DEFER_TO_MANUAL_REVIEW",
}
DEFAULT_VERIFICATION_INPUT_POLICY.update({
    "reverification_comparison_risk_policy_version": REVERIFICATION_COMPARISON_RISK_POLICY_VERSION,
    "reverification_comparison_reason_codes": REVERIFICATION_COMPARISON_REASON_CODES,
    "reverification_comparison_technical_issue_codes": REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES,
    "reverification_comparison_classification_matrix": REVERIFICATION_COMPARISON_CLASSIFICATION_MATRIX,
})

_validate_verification_input_policy_phase64 = validate_verification_input_policy

def validate_verification_input_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_verification_input_policy_phase64(policy)
    if value.get("reverification_comparison_risk_policy_version") != REVERIFICATION_COMPARISON_RISK_POLICY_VERSION:
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_comparison_risk_policy_version:unsupported")
    if tuple(value.get("reverification_comparison_reason_codes") or ()) != REVERIFICATION_COMPARISON_REASON_CODES:
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_comparison_reason_codes:unsupported")
    if tuple(value.get("reverification_comparison_technical_issue_codes") or ()) != REVERIFICATION_COMPARISON_TECHNICAL_ISSUE_CODES:
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_comparison_technical_issue_codes:unsupported")
    if dict(value.get("reverification_comparison_classification_matrix") or {}) != REVERIFICATION_COMPARISON_CLASSIFICATION_MATRIX:
        raise ValueError("VERIFICATION_POLICY_INVALID:reverification_comparison_classification_matrix:unsupported")
    return value

def get_verification_input_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(DEFAULT_VERIFICATION_INPUT_POLICY)
    policy["claim_verification_intensity"] = dict(DEFAULT_CLAIM_VERIFICATION_INTENSITY)
    policy["minimum_resolved_evidence_by_claim_type"] = dict(DEFAULT_MINIMUM_RESOLVED_EVIDENCE)
    policy["retrieval_rounds_by_intensity"] = dict(DEFAULT_RETRIEVAL_ROUNDS_BY_INTENSITY)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("VERIFICATION_POLICY_INVALID:overrides:expected_mapping")
        policy.update(dict(overrides))
    return validate_verification_input_policy(policy)

# Phase 6.5.1A: structural aggregation contracts only.
AGGREGATION_STATUSES = ("VALID", "PARTIAL", "INVALID")
METRICS_STATUSES = ("COMPUTED", "PARTIALLY_COMPUTED", "NOT_COMPUTED")
TRACE_STAGE_AVAILABILITIES = (
    "AVAILABLE", "NOT_PRODUCED", "NOT_APPLICABLE", "BLOCKED_UPSTREAM", "FAILED",
)
METRIC_COMPUTATION_STATUSES = ("COMPUTED", "NOT_COMPUTABLE")
NORMALIZED_BUNDLE_STATUSES = ("COMPUTED", "NOT_COMPUTABLE")
AGGREGATION_PARTIAL_REASON_CODES = (
    "PARTIAL_EXPECTED",
    "PARTIAL_UPSTREAM_BLOCKED",
    "PARTIAL_STAGE_FAILED",
    "PARTIAL_STAGE_NOT_PRODUCED",
    "PARTIAL_MANUAL_REVIEW_REQUIRED",
)
EVIDENCE_SUPPORT_STATUSES = ("SUPPORTED", "NOT_SUPPORTED", "NOT_EVALUATED")
AGGREGATION_SCIENTIFIC_ACTION_TYPES = (
    "REMOVE_UNSUPPORTED_FRAGMENT", "REPLACE_NUMERIC_VALUE", "CORRECT_ATTRIBUTION",
    "NARROW_SCOPE", "ADD_QUALIFICATION", "REPLACE_CITATION", "SPLIT_CLAIM",
)
AGGREGATION_GATE_ACTION_NOT_AVAILABLE = "NOT_AVAILABLE"

# Phase 6.5.1AR: closed structural vocabularies for provisional aggregation.
AGGREGATION_PRECHECK_STATUSES = ("PRECHECK_PASSED", "PRECHECK_BLOCKED", "PRECHECK_REJECTED")
AGGREGATION_GATE_CLASSIFICATIONS = (
    "TEMPORARY_TECHNICAL", "PERMANENT_CONTRACTUAL",
    "DETERMINISTIC_SCIENTIFIC_REJECTION", "UNKNOWN_REASON_CODE",
    "INVALID_GATE_CONTRACT",
)

# Phase 6.5.2: collection validation and deduplication contracts.
COLLECTION_VALIDATION_STATUSES = ("VALID", "INVALID")
AGGREGATION_COLLECTION_NAMES = (
    "claim_verification_records",
    "correction_proposals",
    "correction_precheck_results",
    "independent_reverification_results",
    "before_after_comparison_results",
)
AGGREGATION_COLLECTION_ISSUE_CODES = (
    "AGGREGATION_COLLECTION_ELEMENT_INVALID",
    "AGGREGATION_IDENTICAL_DUPLICATE_DEDUPLICATED",
    "AGGREGATION_CONFLICTING_DUPLICATE",
)

# Phase 6.5.3: referential integrity contracts.
AGGREGATION_DUPLICATE_TYPES = ("IDENTICAL", "CONFLICTING")
AGGREGATION_REFERENTIAL_VALIDATION_STATUSES = ("VALID", "PARTIAL", "INVALID")
AGGREGATION_REFERENTIAL_ISSUE_CODES = (
    "AGGREGATION_UNKNOWN_CLAIM_ID",
    "AGGREGATION_ORPHAN_PRECHECK_RESULT",
    "AGGREGATION_ORPHAN_REVERIFICATION_RESULT",
    "AGGREGATION_ORPHAN_COMPARISON_RESULT",
    "AGGREGATION_CORRECTION_ID_CLAIM_CONFLICT",
    "AGGREGATION_SECTION_ID_MISMATCH",
    "AGGREGATION_PROPOSAL_FINGERPRINT_MISMATCH",
    "AGGREGATION_PROPOSED_TEXT_FINGERPRINT_MISMATCH",
    "AGGREGATION_EVIDENCE_SNAPSHOT_FINGERPRINT_MISMATCH",
    "AGGREGATION_REVERIFICATION_CONTEXT_FINGERPRINT_MISMATCH",
    "AGGREGATION_CORRECTION_ACTION_MISMATCH",
    "AGGREGATION_UNAUTHORIZED_REVERIFICATION_EVIDENCE",
    "AGGREGATION_TARGET_ISSUE_WITHOUT_PROVENANCE",
)
AGGREGATION_REFERENTIAL_WARNING_CODES = (
    "AGGREGATION_CLAIM_WITHOUT_PROPOSAL",
    "AGGREGATION_PROPOSAL_NOT_REVERIFIED",
    "AGGREGATION_PRECHECK_TERMINAL_WITHOUT_REVERIFICATION",
    "AGGREGATION_REVERIFICATION_TERMINAL_WITHOUT_COMPARISON",
)

# Phase 6.5.4: complete referential chain and provisional rows.
AGGREGATION_COLLECTION_NAMES = (
    "claim_verification_records", "correction_proposals", "correction_reverification_inputs",
    "correction_precheck_results", "independent_reverification_results", "before_after_comparison_results",
)
AGGREGATION_REFERENTIAL_ISSUE_CODES = AGGREGATION_REFERENTIAL_ISSUE_CODES + (
    "AGGREGATION_ORPHAN_REVERIFICATION_INPUT",
    "AGGREGATION_PRECHECK_WITHOUT_REVERIFICATION_INPUT",
    "AGGREGATION_REVERIFICATION_POLICY_FINGERPRINT_MISMATCH",
    "AGGREGATION_AUTHORIZED_EVIDENCE_IDENTITY_MISMATCH",
    "AGGREGATION_AUTHORIZED_EVIDENCE_ORDER_MISMATCH",
    "AGGREGATION_AUTHORIZED_EVIDENCE_CONTENT_MISMATCH",
)
AGGREGATION_REFERENTIAL_WARNING_CODES = AGGREGATION_REFERENTIAL_WARNING_CODES + (
    "AGGREGATION_ACCEPTED_PROPOSAL_INPUT_NOT_PRODUCED",
)
AGGREGATION_ROW_BUILD_STATUSES = ("VALID", "PARTIAL", "INVALID")
AGGREGATION_ROW_ISSUE_CODES = (
    "AGGREGATION_ROW_SOURCE_CLAIM_TEXT_UNAVAILABLE",
    "AGGREGATION_ROW_CONTRACT_INVALID",
)
AGGREGATION_ROW_WARNING_CODES = (
    "AGGREGATION_ROW_STAGE_NOT_AVAILABLE",
    "AGGREGATION_ROW_EVIDENCE_SUPPORT_NOT_EVALUATED",
    "AGGREGATION_ROW_MANUAL_REVIEW_REQUIRED",
)
AGGREGATION_CLAIM_TYPES = tuple(dict.fromkeys(CLAIM_TYPES + ("FACTUAL",)))

# Phase 6.5.5 row closure and metrics aggregation.
AGGREGATION_ROW_ISSUE_CODES = tuple(dict.fromkeys(AGGREGATION_ROW_ISSUE_CODES + (
    "AGGREGATION_ROW_SOURCE_CLAIM_CONFLICT",
    "AGGREGATION_ROW_EVIDENCE_IDENTITY_UNAVAILABLE",
    "AGGREGATION_ROW_DUPLICATE_KEY",
    "AGGREGATION_ROW_ORDER_INVALID",
    "AGGREGATION_ROW_INTERNAL_REFERENCE_INVALID",
)))
AGGREGATION_ROW_WARNING_CODES = tuple(dict.fromkeys(AGGREGATION_ROW_WARNING_CODES + (
    "AGGREGATION_ROW_SOURCE_CLAIM_TEXT_UNAVAILABLE",
    "AGGREGATION_ROW_EVIDENCE_IDENTITY_UNAVAILABLE",
)))
AGGREGATION_METRIC_ISSUE_CODES = (
    "AGGREGATION_METRICS_INPUT_INVALID",
    "AGGREGATION_METRICS_COUNTER_INVALID",
)
AGGREGATION_METRIC_WARNING_CODES = (
    "AGGREGATION_RECOMMENDATIONS_NOT_COMPUTABLE",
)
AGGREGATION_METRICS_STATUSES = ("COMPUTED", "PARTIALLY_COMPUTED", "NOT_COMPUTED")

# Phase 6.5.6: canonical fingerprint contracts for the provisional bundle.
PROVISIONAL_BUNDLE_FINGERPRINT_ALGORITHM = "SHA-256"
PROVISIONAL_BUNDLE_FINGERPRINT_VERSION_V1 = "AGENT07_PROVISIONAL_BUNDLE_V1"
PROVISIONAL_BUNDLE_FINGERPRINT_VERSION_V2 = "AGENT07_PROVISIONAL_BUNDLE_V2"
PROVISIONAL_BUNDLE_FINGERPRINT_VERSION_V3 = "AGENT07_PROVISIONAL_BUNDLE_V3"
PROVISIONAL_BUNDLE_FINGERPRINT_VERSION = "AGENT07_PROVISIONAL_BUNDLE_V4"
PROVISIONAL_AUDIT_FINGERPRINT_VERSION = "AGENT07_PROVISIONAL_AUDIT_V1"
PROVISIONAL_COLLECTION_FINGERPRINT_VERSION = "AGENT07_PROVISIONAL_COLLECTION_V1"
PROVISIONAL_COLLECTION_FINGERPRINT_FIELDS = (
    "claim_verification_records",
    "correction_proposals",
    "correction_reverification_inputs",
    "correction_precheck_results",
    "independent_reverification_results",
    "before_after_comparison_results",
)
