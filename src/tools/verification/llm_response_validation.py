"""Validación de la respuesta LLM de verificación de claims (Agente 07).

Extraído mecánicamente de src.tools.verification.validation (Bloque C,
C3) -- sin cambios de comportamiento, lógica, firmas ni mensajes de
error respecto al código original. validation.py reexporta estos
símbolos para preservar el contrato de importación existente.

Replica localmente el subconjunto del bloque "Phase 4" (ver
validation.py) que este cluster realmente usa -- el bloque completo
permanece en validation.py porque también lo usan otros
clusters/bloques contaminados que no se movieron en este Bloque C.
Importa _REQUIRED_LLM_FIELDS desde evidence_selection.py (Bloque C2)
en vez de duplicarla -- sin riesgo de import circular, ya que
evidence_selection.py no depende de este módulo.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence as _Sequence

from src.config.verification_policy_config import (
    ADDITIONAL_RETRIEVAL_REASON_CODES,
    ATTRIBUTION_ASSESSMENTS,
    CONTRADICTION_TYPES,
    EXTRAPOLATION_ASSESSMENTS,
    HALLUCINATION_RISKS,
    NUMERIC_ASSESSMENTS,
    SCIENTIFIC_VERDICTS,
    SEMANTIC_REASON_CODES,
    SUPPORT_LEVELS,
    SUPPORT_LEVEL_BY_VERDICT,
)

from .evidence_selection import _REQUIRED_LLM_FIELDS


def _require_exact_type(result: Mapping[str, Any], key: str, expected: type) -> Any:
    value = result[key]
    if type(value) is not expected:
        raise ValueError(f"LLM_RESPONSE_FIELD_INVALID_TYPE:{key}:{expected.__name__}")
    return value


def _require_string_list(result: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = _require_exact_type(result, key, list)
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"LLM_RESPONSE_FIELD_INVALID_STRING_LIST:{key}")
    return tuple(dict.fromkeys(item.strip() for item in value))


def validate_llm_verification_response(
    response: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    eligible_evidence: _Sequence[Mapping[str, Any]],
    allowed_verdicts: _Sequence[str],
) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise ValueError("LLM_RESPONSE_NOT_MAPPING")
    policy = context["policy"]
    keys = set(response)
    missing = sorted(_REQUIRED_LLM_FIELDS - keys)
    if missing:
        raise ValueError(f"LLM_RESPONSE_FIELDS_MISSING:{','.join(missing)}")
    unknown = sorted(keys - _REQUIRED_LLM_FIELDS)
    if unknown and policy["reject_unknown_llm_fields"]:
        raise ValueError(f"LLM_RESPONSE_UNKNOWN_FIELDS:{','.join(unknown)}")
    result = dict(response)
    for key in ("claim_id", "verdict", "support_level", "rationale", "contradiction_type",
                "numeric_assessment", "attribution_assessment", "extrapolation_assessment", "confidence"):
        _require_exact_type(result, key, str)
    if not result["rationale"].strip():
        raise ValueError("LLM_RESPONSE_RATIONALE_EMPTY")
    used = _require_string_list(result, "evidence_ids_used")
    rejected = _require_string_list(result, "evidence_ids_rejected")
    contradiction_ids = _require_string_list(result, "contradiction_evidence_ids")
    reason_codes = _require_string_list(result, "reason_codes")
    for key in ("additional_retrieval_needed", "llm_correction_recommendation", "manual_review_required"):
        _require_exact_type(result, key, bool)
    if set(used) & set(rejected):
        raise ValueError("EVIDENCE_IDS_USED_REJECTED_OVERLAP")
    if not set(contradiction_ids).issubset(set(used)):
        raise ValueError("CONTRADICTION_EVIDENCE_MUST_BE_USED")
    if set(contradiction_ids) & set(rejected):
        raise ValueError("CONTRADICTION_EVIDENCE_CANNOT_BE_REJECTED")

    if result["claim_id"] != context["claim_id"]:
        raise ValueError("LLM_RESPONSE_CLAIM_ID_MISMATCH")
    verdict = result["verdict"].upper()
    if verdict not in SCIENTIFIC_VERDICTS or verdict not in set(allowed_verdicts):
        raise ValueError(f"LLM_RESPONSE_VERDICT_NOT_ALLOWED:{verdict}")
    support = result["support_level"].upper()
    expected_support = SUPPORT_LEVEL_BY_VERDICT[verdict]
    if support not in SUPPORT_LEVELS:
        raise ValueError(f"LLM_RESPONSE_UNKNOWN_SUPPORT_LEVEL:{support}")
    if support != expected_support:
        
        if verdict == "SUPPORTED":
            raise ValueError("SUPPORTED_REQUIRES_STRONG_SUPPORT")
        if verdict == "PARTIALLY_SUPPORTED":
            raise ValueError("PARTIALLY_SUPPORTED_REQUIRES_PARTIAL_SUPPORT")
        raise ValueError(f"VERDICT_SUPPORT_LEVEL_INCOMPATIBLE:{verdict}:{support}:expected_{expected_support}")

    evidence_map = {str(row["evidence_id"]): row for row in eligible_evidence}
    unknown_ids = sorted((set(used) | set(rejected) | set(contradiction_ids)) - set(evidence_map))
    if unknown_ids:
        raise ValueError(f"UNKNOWN_EVIDENCE_ID:{','.join(unknown_ids)}")
    if verdict in {"SUPPORTED", "PARTIALLY_SUPPORTED"} and not used:
        raise ValueError(f"{verdict}_REQUIRES_EVIDENCE")
    if verdict == "CONTRADICTED" and not used:
        raise ValueError("CONTRADICTED_REQUIRES_USED_EVIDENCE")
    if verdict == "PARTIALLY_SUPPORTED" and not reason_codes:
        raise ValueError("PARTIALLY_SUPPORTED_REQUIRES_REASON_CODE")
    if verdict in {"SUPPORTED", "PARTIALLY_SUPPORTED"}:
        authorized_support = [e for e in used if evidence_map[e].get("usage_allowed") == "SUPPORT" and bool(evidence_map[e].get("authorized_for_section"))]
        if not authorized_support:
            raise ValueError(f"{verdict}_REQUIRES_AUTHORIZED_SUPPORT_EVIDENCE")

    contradiction = result["contradiction_type"].upper()
    if contradiction not in CONTRADICTION_TYPES:
        raise ValueError(f"LLM_RESPONSE_UNKNOWN_CONTRADICTION_TYPE:{contradiction}")
    if contradiction == "INTERNAL_TEXT_INCONSISTENCY" and not context.get("related_claims"):
        raise ValueError("INTERNAL_TEXT_INCONSISTENCY_REQUIRES_RELATED_CLAIMS")
    if verdict == "CONTRADICTED" and contradiction != "CLAIM_EVIDENCE_CONFLICT":
        raise ValueError("CONTRADICTED_REQUIRES_CLAIM_EVIDENCE_CONFLICT")
    if verdict == "CONTRADICTED" and not contradiction_ids:
        raise ValueError("CONTRADICTED_REQUIRES_CONTRADICTION_EVIDENCE")

    numeric = result["numeric_assessment"].upper()
    attribution = result["attribution_assessment"].upper()
    extrapolation = result["extrapolation_assessment"].upper()
    if numeric not in NUMERIC_ASSESSMENTS:
        raise ValueError(f"LLM_RESPONSE_UNKNOWN_NUMERIC_ASSESSMENT:{numeric}")
    if attribution not in ATTRIBUTION_ASSESSMENTS:
        raise ValueError(f"LLM_RESPONSE_UNKNOWN_ATTRIBUTION_ASSESSMENT:{attribution}")
    if extrapolation not in EXTRAPOLATION_ASSESSMENTS:
        raise ValueError(f"LLM_RESPONSE_UNKNOWN_EXTRAPOLATION_ASSESSMENT:{extrapolation}")
    if verdict == "SUPPORTED":
        if numeric in {"UNSUPPORTED", "CONTEXT_MISMATCH"}:
            raise ValueError("SUPPORTED_INCOMPATIBLE_WITH_UNSUPPORTED_NUMERIC")
        if attribution == "INCORRECT":
            raise ValueError("SUPPORTED_INCOMPATIBLE_WITH_INCORRECT_ATTRIBUTION")
        if extrapolation == "BEYOND_EVIDENCE_SCOPE":
            raise ValueError("SUPPORTED_INCOMPATIBLE_WITH_UNSUPPORTED_EXTRAPOLATION")
        if contradiction == "CLAIM_EVIDENCE_CONFLICT":
            raise ValueError("SUPPORTED_INCOMPATIBLE_WITH_CLAIM_EVIDENCE_CONFLICT")

    semantic_reasons = set(SEMANTIC_REASON_CODES)
    retrieval_reasons = set(ADDITIONAL_RETRIEVAL_REASON_CODES)
    unknown_reasons = sorted(set(reason_codes) - semantic_reasons - retrieval_reasons)
    if unknown_reasons:
        raise ValueError(f"LLM_RESPONSE_UNKNOWN_REASON_CODE:{','.join(unknown_reasons)}")
    if result["additional_retrieval_needed"]:
        if not reason_codes:
            raise ValueError("ADDITIONAL_RETRIEVAL_REQUIRES_REASON_CODE")
        invalid = sorted(set(reason_codes) - retrieval_reasons)
        if invalid:
            raise ValueError(f"ADDITIONAL_RETRIEVAL_REASON_NOT_ALLOWED:{','.join(invalid)}")
        if int(context["attempt_context"].get("remaining_retrieval_requests", 0)) <= 0:
            raise ValueError("ADDITIONAL_RETRIEVAL_WITHOUT_BUDGET")
    else:
        invalid = sorted(set(reason_codes) & retrieval_reasons)
        if invalid:
            raise ValueError(f"RETRIEVAL_REASON_WITHOUT_REQUEST:{','.join(invalid)}")
    if verdict == "SUPPORTED" and set(reason_codes) & {"NO_COVERAGE", "EVIDENCE_INSUFFICIENT"}:
        raise ValueError("SUPPORTED_INCOMPATIBLE_WITH_INSUFFICIENT_REASON")

    if contradiction == "CROSS_SOURCE_DISAGREEMENT" and len({evidence_map[e]["source_filename"] for e in contradiction_ids}) < 2:
        raise ValueError("CROSS_SOURCE_DISAGREEMENT_REQUIRES_MULTIPLE_SOURCES")
    confidence = result["confidence"].upper()
    if confidence not in HALLUCINATION_RISKS:
        raise ValueError(f"LLM_RESPONSE_UNKNOWN_CONFIDENCE:{confidence}")
    result.update({
        "verdict": verdict, "support_level": support, "evidence_ids_used": used,
        "evidence_ids_rejected": rejected, "contradiction_type": contradiction,
        "contradiction_evidence_ids": contradiction_ids, "numeric_assessment": numeric,
        "attribution_assessment": attribution, "extrapolation_assessment": extrapolation,
        "confidence": confidence, "rationale": result["rationale"].strip(),
        "reason_codes": tuple(sorted(set(reason_codes))),
    })
    return result


def derive_semantic_issue_codes(validated: Mapping[str, Any]) -> tuple[str, ...]:
    issues: list[str] = []
    if validated["verdict"] == "PARTIALLY_SUPPORTED":
        issues.append("PARTIAL_SUPPORT")
    if validated["verdict"] == "INSUFFICIENT_EVIDENCE":
        issues.append("INSUFFICIENT_EVIDENCE")
    if validated["contradiction_type"] == "CLAIM_EVIDENCE_CONFLICT":
        issues.append("CLAIM_EVIDENCE_CONFLICT")
    elif validated["contradiction_type"] == "CROSS_SOURCE_DISAGREEMENT":
        issues.append("CROSS_SOURCE_DISAGREEMENT")
    elif validated["contradiction_type"] == "INTERNAL_TEXT_INCONSISTENCY":
        issues.append("INTERNAL_TEXT_INCONSISTENCY")
    if validated["numeric_assessment"] == "UNSUPPORTED":
        issues.append("UNSUPPORTED_NUMERIC_VALUE")
    elif validated["numeric_assessment"] == "CONTEXT_MISMATCH":
        issues.append("NUMERIC_CONTEXT_MISMATCH")
    if validated["attribution_assessment"] == "INCORRECT":
        issues.append("ATTRIBUTION_ERROR")
    if validated["extrapolation_assessment"] == "BEYOND_EVIDENCE_SCOPE":
        issues.append("UNSUPPORTED_EXTRAPOLATION")
    return tuple(sorted(set(issues)))


def compute_hallucination_risk(
    *,
    deterministic_issue_codes: _Sequence[str],
    semantic_issue_codes: _Sequence[str],
    validated_response: Mapping[str, Any] | None,
    eligible_evidence: _Sequence[Mapping[str, Any]],
    technical_status: str,
) -> str:
    deterministic = set(deterministic_issue_codes)
    semantic = set(semantic_issue_codes)
    high = {
        "INVALID_CITATION", "UNSUPPORTED_NUMERIC_VALUE", "DOCUMENT_IDENTITY_INVALID",
        "UNAUTHORIZED_SOURCE", "CLAIM_EVIDENCE_CONFLICT", "ATTRIBUTION_ERROR",
        "UNSUPPORTED_EXTRAPOLATION", "NUMERIC_CONTEXT_MISMATCH",
    }
    if deterministic & high or semantic & high:
        return "HIGH"
    if technical_status != "OK":
        return "MEDIUM"
    if validated_response is None:
        return "MEDIUM"
    if validated_response["verdict"] in {"PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "NOT_VERIFIABLE"}:
        return "MEDIUM"
    used = set(validated_response["evidence_ids_used"])
    evidence_map = {row["evidence_id"]: row for row in eligible_evidence}
    if used and all(not evidence_map[item]["authorized_for_section"] for item in used):
        return "HIGH"
    if semantic:
        return "MEDIUM"
    return "LOW"


def determine_final_correction_eligibility(
    *,
    verdict: str,
    deterministic_issue_codes: _Sequence[str],
    semantic_issue_codes: _Sequence[str],
    llm_recommendation: bool,
    manual_review_required: bool,
    eligible_evidence: _Sequence[Mapping[str, Any]],
    evidence_ids_used: _Sequence[str] = (),
    correction_localized: bool = False,
) -> str:
    issues = set(deterministic_issue_codes) | set(semantic_issue_codes)
    if verdict in {"SUPPORTED", "NOT_APPLICABLE"} and not issues:
        return "NO_CORRECTION_NEEDED"
    if manual_review_required or "CROSS_SOURCE_DISAGREEMENT" in issues or "INTERNAL_TEXT_INCONSISTENCY" in issues:
        return "MANUAL_REVIEW_REQUIRED"
    if verdict in {"INSUFFICIENT_EVIDENCE", "NOT_VERIFIABLE", "NOT_EVALUATED"}:
        return "NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE"
    evidence_map = {str(row.get("evidence_id")): row for row in eligible_evidence}
    used_rows = [evidence_map[item] for item in evidence_ids_used if item in evidence_map]
    authorized_used = [row for row in used_rows if row.get("usage_allowed") == "SUPPORT" and row.get("authorized_for_section")]
    localized_correctable = bool(issues & {"ATTRIBUTION_ERROR", "UNSUPPORTED_EXTRAPOLATION", "PARTIAL_SUPPORT", "UNSUPPORTED_NUMERIC_VALUE"})
    if llm_recommendation and authorized_used and localized_correctable and "CROSS_SOURCE_DISAGREEMENT" not in issues:
        return "AUTO_CORRECTION_ELIGIBLE" if correction_localized else "POTENTIALLY_AUTO_CORRECTABLE"
    return "MANUAL_REVIEW_REQUIRED"
