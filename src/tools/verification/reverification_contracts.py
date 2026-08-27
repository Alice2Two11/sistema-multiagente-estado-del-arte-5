"""Contratos de reverificación virtual independiente (Fase 6.1/6.2R, Agente 07).

Extraído mecánicamente de src.tools.verification.validation (Bloque C,
C6) -- sin cambios de comportamiento, lógica, firmas ni mensajes de
error respecto al código original. validation.py reexporta estos
símbolos para preservar el contrato de importación existente.

Replica localmente el import de las constantes REVERIFICATION_* (ver
validation.py) que este cluster necesita -- ese mismo import
permanece también en validation.py porque otras zonas del archivo
(bloques contaminados, no movidos en este Bloque C) también lo usan.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

from src.config.verification_policy_config import (
    HALLUCINATION_RISKS,
    REVERIFICATION_ACCEPTANCE_DECISIONS,
    REVERIFICATION_EXECUTION_STATUSES,
    REVERIFICATION_PROCESS_NAME,
    REVERIFICATION_REASON_CODES,
    REVERIFICATION_RISK_DELTAS,
    REVERIFICATION_RISK_POLICY_VERSION,
    REVERIFICATION_SCIENTIFIC_OUTCOMES,
    REVERIFICATION_TECHNICAL_ISSUE_CODES,
    get_verification_input_policy,
)


def _reverification_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}")
    return value.strip()


def _reverification_string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}")
    normalized = tuple(_reverification_nonempty_string(item, f"{field}[]") for item in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}:duplicates")
    return normalized


def validate_correction_reverification_input_contract(context: Mapping[str, Any]) -> dict[str, Any]:
    """Valida solo el contrato de Fase 6.1; no reconstruye ni reverifica."""
    if not isinstance(context, Mapping):
        raise ValueError("REVERIFICATION_INPUT_NOT_MAPPING")
    required = (
        "correction_id", "claim_id", "section_id", "original_claim_text",
        "proposed_claim_text", "source_verdict", "source_issue_codes",
        "target_issue_codes", "correction_action_type", "claim_span_in_section",
        "target_span_in_claim",
        "replacement_text", "evidence_ids", "authorized_evidence",
        "correction_validation_result", "proposal_fingerprint",
        "proposed_claim_text_fingerprint", "original_claim_fingerprint",
        "original_section_fingerprint", "base_claim_fingerprint",
        "base_section_fingerprint", "application_order_key",
        "attempt_context", "policy",
    )
    missing = [field for field in required if field not in context]
    if missing:
        raise ValueError("REVERIFICATION_INPUT_FIELDS_MISSING:" + ",".join(missing))
    value = dict(context)
    for field in (
        "correction_id", "claim_id", "section_id", "original_claim_text",
        "proposed_claim_text", "source_verdict", "correction_action_type",
        "proposal_fingerprint", "proposed_claim_text_fingerprint",
        "original_claim_fingerprint", "original_section_fingerprint",
        "base_claim_fingerprint", "base_section_fingerprint",
    ):
        value[field] = _reverification_nonempty_string(value[field], field)
    order = value["application_order_key"]
    if type(order) not in (list, tuple) or len(order) != 4:
        raise ValueError("REVERIFICATION_CONTRACT_INVALID:application_order_key")
    if not isinstance(order[0], str) or not order[0].strip() or type(order[1]) is not int or order[1] < 0 or type(order[2]) is not int or order[2] < 0 or not isinstance(order[3], str) or not order[3].strip():
        raise ValueError("REVERIFICATION_CONTRACT_INVALID:application_order_key")
    value["application_order_key"] = (order[0].strip(), order[1], order[2], order[3].strip())
    if not isinstance(value["replacement_text"], str):
        raise ValueError("REVERIFICATION_CONTRACT_INVALID:replacement_text")
    value["source_issue_codes"] = _reverification_string_tuple(
        value["source_issue_codes"], "source_issue_codes"
    )
    value["target_issue_codes"] = _reverification_string_tuple(
        value["target_issue_codes"], "target_issue_codes"
    )
    if not value["target_issue_codes"]:
        raise ValueError("REVERIFICATION_CONTRACT_INVALID:target_issue_codes:empty")
    missing_targets = sorted(set(value["target_issue_codes"]) - set(value["source_issue_codes"]))
    if missing_targets:
        raise ValueError("TARGET_ISSUE_CODE_NOT_PRESENT:" + ",".join(missing_targets))
    value["evidence_ids"] = _reverification_string_tuple(value["evidence_ids"], "evidence_ids")
    def _validate_contract_span(span_value: Any, field: str, expected_base: str, expected_fingerprint: str) -> dict[str, Any]:
        if not isinstance(span_value, Mapping):
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}")
        span = dict(span_value)
        required_span = ("coordinate_base", "coordinate_system", "base_text_fingerprint", "start", "end", "text")
        missing_span = [name for name in required_span if name not in span]
        if missing_span:
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}:missing_" + ",".join(missing_span))
        if span["coordinate_base"] != expected_base:
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}:coordinate_base")
        if span["coordinate_system"] != "PYTHON_CODEPOINT_OFFSETS":
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}:coordinate_system")
        if span["base_text_fingerprint"] != expected_fingerprint:
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}:base_text_fingerprint")
        if type(span["start"]) is not int or type(span["end"]) is not int or span["start"] < 0 or span["end"] <= span["start"]:
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}:range")
        if not isinstance(span["text"], str) or not span["text"]:
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}:text")
        return span

    value["claim_span_in_section"] = _validate_contract_span(
        value["claim_span_in_section"], "claim_span_in_section", "SECTION_TEXT", value["base_section_fingerprint"]
    )
    value["target_span_in_claim"] = _validate_contract_span(
        value["target_span_in_claim"], "target_span_in_claim", "CLAIM_TEXT", value["base_claim_fingerprint"]
    )
    for field in ("authorized_evidence",):
        if type(value[field]) not in (list, tuple) or any(not isinstance(item, Mapping) for item in value[field]):
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}")
        value[field] = tuple(dict(item) for item in value[field])
    authorized_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in value["authorized_evidence"]
        if str(item.get("evidence_id", "")).strip()
    }
    if not set(value["evidence_ids"]).issubset(authorized_ids):
        raise ValueError("REVERIFICATION_EVIDENCE_NOT_FROZEN")
    authorized_by_id = {str(item.get("evidence_id", "")).strip(): item for item in value["authorized_evidence"]}
    if any(authorized_by_id[evidence_id].get("authorized_for_section") is not True for evidence_id in value["evidence_ids"]):
        raise ValueError("REVERIFICATION_EVIDENCE_NOT_AUTHORIZED")
    for field in ("correction_validation_result", "attempt_context"):
        if not isinstance(value[field], Mapping):
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}")
        value[field] = dict(value[field])
    if value["base_claim_fingerprint"] != value["original_claim_fingerprint"]:
        raise ValueError("BASE_CLAIM_FINGERPRINT_MISMATCH")
    if value["base_section_fingerprint"] != value["original_section_fingerprint"]:
        raise ValueError("BASE_SECTION_FINGERPRINT_MISMATCH")
    expected_order = (
        value["section_id"],
        value["claim_span_in_section"]["start"],
        value["target_span_in_claim"]["start"],
        value["correction_id"],
    )
    if value["application_order_key"] != expected_order:
        raise ValueError("REVERIFICATION_APPLICATION_ORDER_KEY_MISMATCH")
    if not value["evidence_ids"] or not value["authorized_evidence"]:
        raise ValueError("REVERIFICATION_EVIDENCE_REQUIRED")
    authorized_all_ids = [str(item.get("evidence_id", "")).strip() for item in value["authorized_evidence"]]
    if any(not item for item in authorized_all_ids):
        raise ValueError("REVERIFICATION_CONTRACT_INVALID:authorized_evidence:evidence_id")
    if len(authorized_all_ids) != len(set(authorized_all_ids)):
        raise ValueError("REVERIFICATION_AUTHORIZED_EVIDENCE_ID_DUPLICATE")
    if "correction_applied" not in value["correction_validation_result"]:
        raise ValueError("REVERIFICATION_CORRECTION_APPLIED_REQUIRED")
    if value["correction_validation_result"]["correction_applied"] is not False:
        raise ValueError("REVERIFICATION_PHYSICAL_APPLICATION_FORBIDDEN")
    value["policy"] = get_verification_input_policy(value["policy"])
    proposal_status = value["correction_validation_result"].get("proposal_status")
    if proposal_status not in value["policy"]["reverification_allowed_proposal_statuses"]:
        raise ValueError("REVERIFICATION_PROPOSAL_STATUS_NOT_ALLOWED")
    if value["policy"]["reverification_process_name"] != REVERIFICATION_PROCESS_NAME:
        raise ValueError("REVERIFICATION_PROCESS_NAME_INVALID")
    if value["policy"]["reverification_retrieval_rounds"] != 0:
        raise ValueError("REVERIFICATION_RETRIEVAL_FORBIDDEN")
    return value


def validate_correction_reverification_result_contract(result: Mapping[str, Any]) -> dict[str, Any]:
    """Valida dimensiones independientes; no calcula aceptación ni issues."""
    if not isinstance(result, Mapping):
        raise ValueError("REVERIFICATION_RESULT_NOT_MAPPING")
    required = (
        "correction_id", "claim_id", "section_id", "reverification_execution_status",
        "scientific_outcome", "acceptance_decision", "original_verdict",
        "proposed_verdict", "original_issue_codes", "remaining_issue_codes",
        "resolved_issue_codes", "new_issue_codes", "evidence_used",
        "supported_meaning_preserved", "intended_semantic_change_valid",
        "unintended_semantic_change_absent", "scope_change_valid",
        "numeric_change_valid", "attribution_change_valid", "citation_change_valid",
        "hallucination_risk_before", "hallucination_risk_after", "hallucination_risk_delta",
        "risk_policy_version", "risk_before_recomputed", "risk_after_computed",
        "manual_review_required", "reason_codes", "technical_issue_codes",
        "tool_usage", "decision_trace", "raw_attempts", "result_contract_valid",
        "correction_applied",
    )
    missing = [field for field in required if field not in result]
    if missing:
        raise ValueError("REVERIFICATION_RESULT_FIELDS_MISSING:" + ",".join(missing))
    value = dict(result)
    for field in ("correction_id", "claim_id", "section_id", "original_verdict", "proposed_verdict", "risk_policy_version"):
        value[field] = _reverification_nonempty_string(value[field], field)
    if value["risk_policy_version"] != REVERIFICATION_RISK_POLICY_VERSION:
        raise ValueError("REVERIFICATION_RISK_POLICY_VERSION_MISMATCH")
    if value["reverification_execution_status"] not in REVERIFICATION_EXECUTION_STATUSES:
        raise ValueError("REVERIFICATION_EXECUTION_STATUS_UNKNOWN")
    if value["scientific_outcome"] not in REVERIFICATION_SCIENTIFIC_OUTCOMES:
        raise ValueError("REVERIFICATION_SCIENTIFIC_OUTCOME_UNKNOWN")
    if value["acceptance_decision"] not in REVERIFICATION_ACCEPTANCE_DECISIONS:
        raise ValueError("REVERIFICATION_ACCEPTANCE_DECISION_UNKNOWN")
    for field in (
        "original_issue_codes", "remaining_issue_codes", "resolved_issue_codes",
        "new_issue_codes", "evidence_used", "reason_codes", "technical_issue_codes",
    ):
        value[field] = _reverification_string_tuple(value[field], field)
    if not set(value["reason_codes"]).issubset(REVERIFICATION_REASON_CODES):
        raise ValueError("REVERIFICATION_REASON_CODE_UNKNOWN")
    if not set(value["technical_issue_codes"]).issubset(REVERIFICATION_TECHNICAL_ISSUE_CODES):
        raise ValueError("REVERIFICATION_TECHNICAL_ISSUE_CODE_UNKNOWN")
    for field in (
        "supported_meaning_preserved", "intended_semantic_change_valid",
        "unintended_semantic_change_absent", "scope_change_valid",
        "numeric_change_valid", "attribution_change_valid", "citation_change_valid",
        "risk_before_recomputed", "risk_after_computed", "manual_review_required",
        "result_contract_valid", "correction_applied",
    ):
        if type(value[field]) is not bool:
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}")
    if value["correction_applied"] is not False:
        raise ValueError("REVERIFICATION_PHYSICAL_APPLICATION_FORBIDDEN")
    for field in ("hallucination_risk_before", "hallucination_risk_after"):
        if value[field] not in HALLUCINATION_RISKS:
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}")
    if value["hallucination_risk_delta"] not in REVERIFICATION_RISK_DELTAS:
        raise ValueError("REVERIFICATION_RISK_DELTA_UNKNOWN")
    if value["result_contract_valid"] is not True:
        raise ValueError("REVERIFICATION_RESULT_CONTRACT_NOT_VALID")
    if value["acceptance_decision"] == "ACCEPT_FOR_07C":
        if value["reverification_execution_status"] != "COMPLETED":
            raise ValueError("REVERIFICATION_ACCEPTANCE_REQUIRES_COMPLETED_EXECUTION")
        if value["scientific_outcome"] == "NOT_EVALUATED":
            raise ValueError("REVERIFICATION_ACCEPTANCE_REQUIRES_EVALUATED_OUTCOME")
        if value["manual_review_required"] is True:
            raise ValueError("REVERIFICATION_ACCEPTANCE_INCOMPATIBLE_WITH_MANUAL_REVIEW")
        if value["risk_before_recomputed"] is not True or value["risk_after_computed"] is not True:
            raise ValueError("REVERIFICATION_ACCEPTANCE_REQUIRES_COMPARABLE_RISK")
        if value["hallucination_risk_delta"] in {"NOT_COMPARABLE", "INCREASED"}:
            raise ValueError("REVERIFICATION_ACCEPTANCE_RISK_INVALID")
    if value["reverification_execution_status"] == "BLOCKED" and value["acceptance_decision"] == "ACCEPT_FOR_07C":
        raise ValueError("REVERIFICATION_BLOCKED_CANNOT_ACCEPT")
    for field in ("tool_usage",):
        if not isinstance(value[field], Mapping):
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}")
        value[field] = dict(value[field])
    for field in ("decision_trace", "raw_attempts"):
        if type(value[field]) not in (list, tuple) or any(not isinstance(item, Mapping) for item in value[field]):
            raise ValueError(f"REVERIFICATION_CONTRACT_INVALID:{field}")
        value[field] = tuple(dict(item) for item in value[field])
    return value


def validate_reverification_block_matrix(
    *, category: str, execution_status: str, acceptance_decision: str
) -> None:
    expected = {
        "CONTRACTUAL_INCOMPATIBILITY": ("BLOCKED", "REJECT_PROPOSAL"),
        "TEMPORARY_TECHNICAL_DEPENDENCY": ("BLOCKED", "DEFER_TO_MANUAL_REVIEW"),
        "NEGATIVE_SCIENTIFIC_RESULT": ("COMPLETED", "REJECT_PROPOSAL"),
        "SCIENTIFIC_AMBIGUITY": ("COMPLETED", "DEFER_TO_MANUAL_REVIEW"),
    }
    if category not in expected:
        raise ValueError("REVERIFICATION_BLOCK_CATEGORY_UNKNOWN")
    if (execution_status, acceptance_decision) != expected[category]:
        raise ValueError("REVERIFICATION_BLOCK_MATRIX_VIOLATION")


def _phase62_brackets_balanced(text: str) -> bool:
    pairs={')':'(',']':'[','}':'{'}
    stack=[]
    for ch in text:
        if ch in '([{': stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop()!=pairs[ch]: return False
    return not stack


# Phase 6.2R: integridad de propuesta y semántica de prechecks, sin LLM.
PRECHECK_STATUSES = ("PRECHECK_PASSED", "PRECHECK_BLOCKED", "PRECHECK_REJECTED")


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
