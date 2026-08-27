"""Closed in-memory runtime contract for Agent 07.

Operational failures are represented by ``BlockedRuntimeAuditRecord`` and do
not fabricate partial scientific contracts. Scientific results are validated
recursively before becoming a terminal runtime result.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from typing import Any, Callable, Mapping, Protocol, Sequence

from src.agents.verification_agent import VerificationAgent
from src.adapters.agent06_verification_handoff import (Agent07RetrieverBinding, validate_agent07_experiment_compatibility, validate_productive_retriever_binding)
from src.adapters.claim_verification_context import (
    build_claim_verification_context_from_agent06_handoff,
    _normalize_evidence as _canonicalize_claim_evidence,
)
from src.tools.verification.corrections import propose_correction, fingerprint_text
from src.tools.verification.resolution import (
    resolve_multiple_correction_proposals,
    validate_provisional_multi_proposal_resolution_result,
)
from src.tools.verification.validation import (
    build_provisional_verification_traceability_bundle,
    compare_virtual_reverification_before_after,
    run_independent_virtual_reverification,
    run_virtual_reverification_prechecks,
    validate_provisional_verification_traceability_bundle_contract,
    validate_sha256_hex,
    validate_correction_reverification_input_contract,
    compute_frozen_evidence_snapshot_fingerprint,
    compute_reverification_policy_fingerprint,
    compute_reverification_context_fingerprint,
    validate_additional_retrieval_delta,
)
from src.config.agentic_retrieval_policy_config import (
    DEFAULT_GRADER_THRESHOLDS,
    DEFAULT_MINIMUM_VIABLE_THRESHOLDS,
    validate_effective_top_k_max,
    compute_effective_budget_for_verify_claim,
)
from src.config.verification_policy_config import get_verification_input_policy
from src.tools.verification.agentic_retrieval_grader import grade_evidence, is_minimum_viable_evidence
from src.tools.verification.agentic_retrieval_controller import (
    AgenticRetrievalObservation,
    run_agentic_retrieval_cycle,
)
from src.tools.verification.agentic_retrieval_action_executor import AgenticRetrievalActionExecutor
from src.tools.verification.agentic_retrieval_tracing import (
    build_traced_execute_action_fn,
    build_agentic_retrieval_trace,
)
from src.tools.verification.prompting import normalize_verification_llm_response

AGENT07_RUNTIME_METRICS_VERSION = "AGENT07_RUNTIME_METRICS_V5"

RUNTIME_STATUSES = ("COMPLETED", "PARTIAL", "BLOCKED")
RUNTIME_ERROR_CLASSIFICATIONS = ("CONTRACTUAL", "TECHNICAL", "DEPENDENCY", "GLOBAL")
CANDIDATE_ARTIFACT_STATUSES = ("READY_CANDIDATE", "PARTIAL_CANDIDATE", "BLOCKED_AUDIT_ONLY")
CANDIDATE_ARTIFACT_TYPES = (
    "PROVISIONAL_VERIFICATION_TRACEABILITY_BUNDLE",
    "MULTI_PROPOSAL_RESOLUTION_RESULT",
)
RUNTIME_ISSUE_CODES = (
    "AGENT07_RUNTIME_INPUT_CONTRACT_INVALID",
    "AGENT07_RUNTIME_DEPENDENCY_MISSING",
    "AGENT07_RUNTIME_STAGE_FAILURE",
    "AGENT07_RUNTIME_GLOBAL_BLOCK",
)


class CorrectionContextFactory(Protocol):
    def __call__(self, claim_context: Mapping[str, Any], verification_result: Mapping[str, Any], runtime_config: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ReverificationInputFactory(Protocol):
    def __call__(self, claim_context: Mapping[str, Any], verification_result: Mapping[str, Any], correction_proposal: Mapping[str, Any], runtime_config: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class VerificationRuntimeDependencies:
    verification_llm: Any = None
    retrieval_tool: Any = None
    correction_llm: Any = None
    reverification_llm: Any = None
    correction_context_factory: CorrectionContextFactory | None = None
    reverification_input_factory: ReverificationInputFactory | None = None
    verification_agent_factory: Callable[..., Any] = VerificationAgent
    proposal_runner: Callable[..., Any] = propose_correction
    precheck_runner: Callable[..., Any] = run_virtual_reverification_prechecks
    reverification_runner: Callable[..., Any] = run_independent_virtual_reverification
    comparison_runner: Callable[..., Any] = compare_virtual_reverification_before_after
    bundle_builder: Callable[..., Any] = build_provisional_verification_traceability_bundle
    resolution_runner: Callable[..., Any] = resolve_multiple_correction_proposals
    retriever_binding: Mapping[str, str] | None = None




PRODUCTIVE_DEPENDENCY_REQUIRED_CONFIG = (
    "verification_policy", "correction_policy", "reverification_policy",
    "verification_prompt_version", "correction_prompt_version",
    "reverification_prompt_version", "verification_budgets",
    "correction_budgets", "reverification_budgets",
)

def _require_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"AGENT07_PRODUCTIVE_CONFIG_REQUIRED:{name}")
    return value

def _require_text(config: Mapping[str, Any], name: str) -> str:
    value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AGENT07_PRODUCTIVE_CONFIG_REQUIRED:{name}")
    return value.strip()

def _productive_correction_context(
    claim_context: Mapping[str, Any],
    verification_result: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    original = claim_context.get("original_claim_text", claim_context.get("claim_text"))
    section_text = claim_context.get("section_text")
    if not isinstance(original, str) or not original:
        raise ValueError("AGENT07_PRODUCTIVE_CONTEXT_REQUIRED:original_claim_text")
    if not isinstance(section_text, str) or not section_text:
        raise ValueError("AGENT07_PRODUCTIVE_CONTEXT_REQUIRED:section_text")
    claim_fp = str(claim_context.get("claim_fingerprint") or fingerprint_text(original))
    section_fp = str(claim_context.get("section_fingerprint") or fingerprint_text(section_text))
    span = claim_context.get("claim_span_in_section")
    if not isinstance(span, Mapping):
        if section_text.count(original) != 1:
            raise ValueError("AGENT07_PRODUCTIVE_CONTEXT_REQUIRED:claim_span_in_section")
        start = section_text.index(original)
        span = {"coordinate_base":"SECTION_TEXT","coordinate_system":"PYTHON_CODEPOINT_OFFSETS",
                "base_text_fingerprint":section_fp,"start":start,"end":start+len(original),"text":original}
    return {
        "claim_id": str(claim_context["claim_id"]), "section_id": str(claim_context["section_id"]),
        "original_claim_text": original, "claim_fingerprint": claim_fp,
        "section_text": section_text, "section_fingerprint": section_fp,
        "claim_span_in_section": deepcopy(dict(span)),
        "final_correction_eligibility": verification_result["final_correction_eligibility"],
        "eligible_evidence": deepcopy(tuple(verification_result.get("eligible_evidence", ()))),
        "source_verdict": verification_result["scientific_verdict"],
        "source_issue_codes": tuple(sorted(set(verification_result.get("deterministic_issue_codes", ())) | set(verification_result.get("semantic_issue_codes", ())))),
        "policy": deepcopy(dict(runtime_config["correction_policy"])),
        "prompt_version": runtime_config["correction_prompt_version"],
        "prior_correction_proposals": tuple(deepcopy(runtime_config.get("prior_correction_proposals", ()))),
    }

def _productive_reverification_input(
    claim_context: Mapping[str, Any],
    verification_result: Mapping[str, Any],
    correction_proposal: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    if correction_proposal.get("accepted_for_reverification") is not True:
        raise ValueError("AGENT07_REVERIFICATION_INPUT_PROPOSAL_NOT_ACCEPTED")
    authorized = tuple(deepcopy(correction_proposal.get("authorized_evidence", ())))
    if not authorized:
        eligible = {str(x.get("evidence_id")): deepcopy(dict(x)) for x in verification_result.get("eligible_evidence", ()) if isinstance(x, Mapping)}
        authorized = tuple(eligible[eid] for eid in correction_proposal.get("evidence_ids", ()) if eid in eligible)
    target_issues = tuple(runtime_config.get("target_issue_codes_by_correction", {}).get(correction_proposal["correction_id"], ()))
    source_issues = tuple(sorted(set(verification_result.get("deterministic_issue_codes", ())) | set(verification_result.get("semantic_issue_codes", ()))))
    if not target_issues:
        target_issues = source_issues
    if not target_issues:
        raise ValueError("AGENT07_PRODUCTIVE_CONTEXT_REQUIRED:target_issue_codes")
    policy = deepcopy(dict(runtime_config["reverification_policy"]))
    evidence_snapshot = compute_frozen_evidence_snapshot_fingerprint(authorized)
    policy_fp = compute_reverification_policy_fingerprint(policy)
    proposed_fp = fingerprint_text(str(correction_proposal["proposed_claim_text"]))
    base = {
        "correction_id": correction_proposal["correction_id"], "claim_id": correction_proposal["claim_id"],
        "section_id": correction_proposal["section_id"], "original_claim_text": correction_proposal["original_text"],
        "proposed_claim_text": correction_proposal["proposed_claim_text"],
        "source_verdict": verification_result["scientific_verdict"],
        "source_issue_codes": source_issues, "target_issue_codes": target_issues,
        "correction_action_type": correction_proposal["action_type"],
        "claim_span_in_section": deepcopy(correction_proposal["claim_span_in_section"]),
        "target_span_in_claim": deepcopy(correction_proposal["target_span_in_claim"]),
        "replacement_text": correction_proposal["replacement_text"],
        "evidence_ids": tuple(correction_proposal.get("evidence_ids", ())),
        "authorized_evidence": authorized,
        "correction_validation_result": {"proposal_status": correction_proposal["proposal_status"], "correction_applied": False},
        "proposal_fingerprint": correction_proposal["proposal_fingerprint"],
        "proposed_claim_text_fingerprint": proposed_fp,
        "original_claim_fingerprint": correction_proposal["original_claim_fingerprint"],
        "original_section_fingerprint": correction_proposal["original_section_fingerprint"],
        "base_claim_fingerprint": correction_proposal["original_claim_fingerprint"],
        "base_section_fingerprint": correction_proposal["original_section_fingerprint"],
        "application_order_key": (correction_proposal["section_id"], correction_proposal["claim_span_in_section"]["start"], correction_proposal["target_span_in_claim"]["start"], correction_proposal["correction_id"]),
        "attempt_context": {"prompt_version": runtime_config["reverification_prompt_version"], "remaining_llm_attempts": int(runtime_config["reverification_budgets"].get("max_llm_attempts", 1))},
        "policy": policy,
        "frozen_evidence_snapshot_fingerprint": evidence_snapshot,
        "reverification_policy_fingerprint": policy_fp,
    }
    base["reverification_context_fingerprint"] = compute_reverification_context_fingerprint(base, evidence_snapshot_fingerprint=evidence_snapshot, policy_fingerprint=policy_fp)
    validated = validate_correction_reverification_input_contract(base)
    # Defecto productivo real, corregido aquí (no en el contrato formal de
    # ReverificationInput, que no incluye section_text -- por eso se agrega
    # DESPUÉS de validate_correction_reverification_input_contract, no
    # antes): run_virtual_reverification_prechecks (el precheck_runner por
    # defecto, wired como tal en esta misma clase) exige
    # context.get("section_text") directamente sobre el mismo dict que
    # recibe -- ver validation.py, run_virtual_reverification_prechecks,
    # comprobación "SECTION_TEXT_REQUIRED". Ningún productor se lo entregaba
    # antes de esta corrección; el texto de sección SÍ está disponible en
    # claim_context (el mismo contexto que ya usó VerificationAgent), así
    # que se propaga aquí sin inventar nada.
    return {**validated, "section_text": str(claim_context.get("section_text") or "")}

def build_agent07_runtime_dependencies(
    *, config: Mapping[str, Any], experiment_paths: Mapping[str, str],
    verification_llm: Any, correction_llm: Any, reverification_llm: Any,
    incremental_retriever: Any = None, active_experiment_config: Mapping[str, Any] | None = None,
    retriever_binding: Agent07RetrieverBinding | Mapping[str, Any] | None = None,
    chroma_manifest_path: str | None = None, chunks_manifest_path: str | None = None,
    committed_experiment_id: str | None = None,
) -> VerificationRuntimeDependencies:
    """Build production dependencies without importing tests or fixture helpers."""
    if not isinstance(config, Mapping):
        raise ValueError("AGENT07_PRODUCTIVE_CONFIG_INVALID")
    if active_experiment_config is not None:
        validate_agent07_experiment_compatibility(active_config=active_experiment_config, agent07_config=config, experiment_paths=experiment_paths)
    if incremental_retriever is not None:
        if retriever_binding is None or chroma_manifest_path is None or chunks_manifest_path is None or committed_experiment_id is None:
            raise ValueError("AGENT07_PRODUCTIVE_RETRIEVER_BINDING_REQUIRED")
        validated_binding = validate_productive_retriever_binding(binding=retriever_binding, active_config=active_experiment_config or config, chroma_manifest_path=chroma_manifest_path, chunks_manifest_path=chunks_manifest_path, committed_experiment_id=committed_experiment_id)
        for key, value in validated_binding.items():
            if not hasattr(incremental_retriever, key):
                raise ValueError(f"AGENT07_PRODUCTIVE_RETRIEVER_IDENTITY_MISSING:{key}")
            observed = getattr(incremental_retriever, key)
            if observed != value:
                raise ValueError(f"AGENT07_PRODUCTIVE_RETRIEVER_OBJECT_MISMATCH:{key}")
    for name in ("verification_policy", "correction_policy", "reverification_policy", "verification_budgets", "correction_budgets", "reverification_budgets"):
        _require_mapping(config, name)
    for name in ("verification_prompt_version", "correction_prompt_version", "reverification_prompt_version"):
        _require_text(config, name)
    if not isinstance(experiment_paths, Mapping) or not experiment_paths:
        raise ValueError("AGENT07_PRODUCTIVE_EXPERIMENT_PATHS_INVALID")
    if verification_llm is None: raise ValueError("AGENT07_PRODUCTIVE_DEPENDENCY_REQUIRED:verification_llm")
    if correction_llm is None: raise ValueError("AGENT07_PRODUCTIVE_DEPENDENCY_REQUIRED:correction_llm")
    if reverification_llm is None: raise ValueError("AGENT07_PRODUCTIVE_DEPENDENCY_REQUIRED:reverification_llm")
    return VerificationRuntimeDependencies(
        verification_llm=verification_llm, retrieval_tool=incremental_retriever,
        correction_llm=correction_llm, reverification_llm=reverification_llm,
        correction_context_factory=_productive_correction_context,
        reverification_input_factory=_productive_reverification_input,
        verification_agent_factory=VerificationAgent, proposal_runner=propose_correction,
        precheck_runner=run_virtual_reverification_prechecks,
        reverification_runner=run_independent_virtual_reverification,
        comparison_runner=compare_virtual_reverification_before_after,
        bundle_builder=build_provisional_verification_traceability_bundle,
        resolution_runner=resolve_multiple_correction_proposals,
        retriever_binding=dict(validated_binding) if incremental_retriever is not None else None,
    )


@dataclass(frozen=True, slots=True)
class Agent07RuntimeInput:
    committed_agent06_output: Mapping[str, Any]
    agent07_config: Mapping[str, Any]
    policy_versions: Mapping[str, str]
    schema_versions: Mapping[str, str]
    experiment_paths: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeErrorRecord:
    stage: str
    claim_id: str | None
    section_id: str | None
    error_code: str
    error_classification: str

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True, slots=True)
class BlockedRuntimeAuditRecord:
    stage: str
    claim_id: str | None
    section_id: str | None
    error_code: str
    error_classification: str
    runtime_audit_fingerprint: str

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateArtifactRecord:
    artifact_type: str
    artifact_status: str
    candidate_only: bool
    producer: str
    schema_version: str
    normalized_fingerprint: str | None
    audit_fingerprint: str

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True, slots=True)
class Agent07RuntimeResult:
    provisional_bundle: Mapping[str, Any] | None
    multi_proposal_resolution_result: Mapping[str, Any] | None
    candidate_artifact_inventory: tuple[Mapping[str, Any], ...]
    execution_metrics: Mapping[str, int]
    runtime_warnings: tuple[str, ...]
    runtime_issue_codes: tuple[str, ...]
    runtime_error_records: tuple[Mapping[str, Any], ...]
    blocked_runtime_audit_record: Mapping[str, Any] | None
    runtime_status: str
    execution_metrics_version: str = AGENT07_RUNTIME_METRICS_VERSION
    correction_applied: bool = False
    official_artifacts_created: bool = False
    evaluation_ready_emitted: bool = False
    result_contract_valid: bool = False

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict): return _plain(value.to_dict())
    if isinstance(value, Mapping): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return tuple(_plain(v) for v in value)
    return value


def _audit_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _nonempty_string_mapping(value: Any, code: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value: raise ValueError(code)
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(item, str) or not item.strip(): raise ValueError(code)
        result[key] = item
    return result


def validate_committed_agent06_output_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the explicit Agent 06 committed hand-off identity used by 07."""
    if not isinstance(value, Mapping): raise ValueError("AGENT07_RUNTIME_AGENT06_OUTPUT_INVALID")
    if value.get("commit_status") != "COMMITTED": raise ValueError("AGENT07_RUNTIME_AGENT06_OUTPUT_NOT_COMMITTED")
    required_identity = ("run_id", "artifact_identity", "schema_version", "source_draft_fingerprint")
    for name in required_identity:
        item = value.get(name)
        if not isinstance(item, str) or not item.strip(): raise ValueError(f"AGENT07_RUNTIME_AGENT06_{name.upper()}_INVALID")
    validate_sha256_hex(value["source_draft_fingerprint"], field="source_draft_fingerprint")
    contexts = value.get("claim_verification_contexts")
    if not isinstance(contexts, (tuple, list)): raise ValueError("AGENT07_RUNTIME_CLAIM_CONTEXTS_INVALID")
    seen: set[tuple[str, str]] = set(); normalized = []
    for context in contexts:
        if not isinstance(context, Mapping): raise ValueError("AGENT07_RUNTIME_CLAIM_CONTEXT_INVALID")
        claim_id, section_id = context.get("claim_id"), context.get("section_id")
        if not isinstance(claim_id, str) or not claim_id.strip() or not isinstance(section_id, str) or not section_id.strip():
            raise ValueError("AGENT07_RUNTIME_CLAIM_IDENTITY_INVALID")
        key = (section_id, claim_id)
        if key in seen: raise ValueError("AGENT07_RUNTIME_DUPLICATE_CLAIM_CONTEXT")
        seen.add(key); normalized.append(deepcopy(dict(context)))
    result = deepcopy(dict(value)); result["claim_verification_contexts"] = tuple(normalized)
    return result


def validate_agent07_runtime_input_contract(value: Agent07RuntimeInput | Mapping[str, Any]) -> dict[str, Any]:
    payload = asdict(value) if isinstance(value, Agent07RuntimeInput) else deepcopy(dict(value)) if isinstance(value, Mapping) else None
    if payload is None: raise ValueError("AGENT07_RUNTIME_INPUT_TYPE_INVALID")
    if set(payload) != {f.name for f in fields(Agent07RuntimeInput)}: raise ValueError("AGENT07_RUNTIME_INPUT_SCHEMA_INVALID")
    committed = validate_committed_agent06_output_contract(payload["committed_agent06_output"])
    if not isinstance(payload["agent07_config"], Mapping) or any(not isinstance(k, str) or not k.strip() for k in payload["agent07_config"]):
        raise ValueError("AGENT07_RUNTIME_CONFIG_MAPPING_INVALID")
    return {
        "committed_agent06_output": committed,
        "agent07_config": deepcopy(dict(payload["agent07_config"])),
        "policy_versions": _nonempty_string_mapping(payload["policy_versions"], "AGENT07_RUNTIME_POLICY_VERSIONS_INVALID"),
        "schema_versions": _nonempty_string_mapping(payload["schema_versions"], "AGENT07_RUNTIME_SCHEMA_VERSIONS_INVALID"),
        "experiment_paths": _nonempty_string_mapping(payload["experiment_paths"], "AGENT07_RUNTIME_EXPERIMENT_PATHS_INVALID"),
    }


def _resolution_to_runtime_status(status: str) -> str:
    if status not in {"COMPLETED", "PARTIAL", "BLOCKED"}: raise ValueError("AGENT07_RUNTIME_RESOLUTION_STATUS_INVALID")
    return status


def _validate_error_record(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(RuntimeErrorRecord)}: raise ValueError("AGENT07_RUNTIME_ERROR_RECORD_SCHEMA_INVALID")
    record = dict(value)
    if not isinstance(record["stage"], str) or not record["stage"]: raise ValueError("AGENT07_RUNTIME_ERROR_STAGE_INVALID")
    if record["error_classification"] not in RUNTIME_ERROR_CLASSIFICATIONS: raise ValueError("AGENT07_RUNTIME_ERROR_CLASSIFICATION_INVALID")
    if not isinstance(record["error_code"], str) or not record["error_code"]: raise ValueError("AGENT07_RUNTIME_ERROR_CODE_INVALID")
    return record


def _validate_blocked_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(BlockedRuntimeAuditRecord)}: raise ValueError("AGENT07_RUNTIME_BLOCKED_AUDIT_SCHEMA_INVALID")
    record = dict(value); validate_sha256_hex(record["runtime_audit_fingerprint"], field="runtime_audit_fingerprint")
    expected = _audit_hash({k: record[k] for k in ("stage", "claim_id", "section_id", "error_code", "error_classification")})
    if record["runtime_audit_fingerprint"] != expected: raise ValueError("AGENT07_RUNTIME_BLOCKED_AUDIT_FINGERPRINT_MISMATCH")
    return record


def _validate_inventory_record(value: Mapping[str, Any], bundle: Mapping[str, Any], resolution: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(CandidateArtifactRecord)}: raise ValueError("AGENT07_RUNTIME_ARTIFACT_RECORD_SCHEMA_INVALID")
    record = dict(value)
    if record["artifact_type"] not in CANDIDATE_ARTIFACT_TYPES or record["artifact_status"] not in CANDIDATE_ARTIFACT_STATUSES: raise ValueError("AGENT07_RUNTIME_ARTIFACT_ENUM_INVALID")
    if record["candidate_only"] is not True: raise ValueError("AGENT07_RUNTIME_ARTIFACT_NOT_CANDIDATE_ONLY")
    validate_sha256_hex(record["audit_fingerprint"], field="audit_fingerprint")
    if record["normalized_fingerprint"] is not None: validate_sha256_hex(record["normalized_fingerprint"], field="normalized_fingerprint")
    if record["artifact_type"] == CANDIDATE_ARTIFACT_TYPES[0]:
        invalid = bundle["aggregation_status"] == "INVALID"; expected_n = None if invalid else bundle["normalized_bundle_fingerprint"]; expected_a = bundle["aggregation_audit_fingerprint"]
        expected_s = "BLOCKED_AUDIT_ONLY" if invalid else ("PARTIAL_CANDIDATE" if bundle["aggregation_status"] == "PARTIAL" else "READY_CANDIDATE")
    else:
        blocked = resolution["resolution_status"] == "BLOCKED"; expected_n = None if blocked else resolution["multi_proposal_resolution_fingerprint"]; expected_a = resolution["multi_proposal_audit_fingerprint"]
        expected_s = "BLOCKED_AUDIT_ONLY" if blocked else ("PARTIAL_CANDIDATE" if resolution["resolution_status"] == "PARTIAL" else "READY_CANDIDATE")
    if (record["artifact_status"], record["normalized_fingerprint"], record["audit_fingerprint"]) != (expected_s, expected_n, expected_a): raise ValueError("AGENT07_RUNTIME_ARTIFACT_FINGERPRINT_MISMATCH")
    return record


def validate_agent07_runtime_result_contract(value: Agent07RuntimeResult | Mapping[str, Any], *, allow_unvalidated: bool = False) -> dict[str, Any]:
    payload = _plain(asdict(value)) if isinstance(value, Agent07RuntimeResult) else _plain(value)
    if not isinstance(payload, Mapping) or set(payload) != {f.name for f in fields(Agent07RuntimeResult)}: raise ValueError("AGENT07_RUNTIME_RESULT_SCHEMA_INVALID")
    payload = dict(payload)
    if payload["runtime_status"] not in RUNTIME_STATUSES: raise ValueError("AGENT07_RUNTIME_STATUS_MISMATCH")
    if any(payload[name] is not False for name in ("correction_applied", "official_artifacts_created", "evaluation_ready_emitted")): raise ValueError("AGENT07_RUNTIME_ISOLATION_INVARIANT_VIOLATION")
    if payload.get("execution_metrics_version") != AGENT07_RUNTIME_METRICS_VERSION: raise ValueError("AGENT07_RUNTIME_EXECUTION_METRICS_VERSION_INVALID")
    metrics = payload["execution_metrics"]
    metric_keys = {"claims_processed","independent_rag_claims","independent_rag_claims_with_results","independent_rag_claims_without_results","independent_rag_claim_records","evidence_candidate_validation_claims","correction_proposals","reverification_inputs","prechecks","reverifications","comparisons","additional_llm_calls","additional_retrieval_rounds","official_writes","physical_corrections"}
    if not isinstance(metrics, Mapping) or set(metrics) != metric_keys: raise ValueError("AGENT07_RUNTIME_EXECUTION_METRICS_INVALID")
    integer_metric_keys = metric_keys - {"independent_rag_claim_records"}
    if any(type(metrics[k]) is not int or metrics[k] < 0 for k in integer_metric_keys): raise ValueError("AGENT07_RUNTIME_EXECUTION_METRICS_INVALID")
    records = metrics["independent_rag_claim_records"]
    if type(records) not in (tuple, list): raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_RECORDS_INVALID")
    normalized_records=[]; identities=set(); with_results=0; without_results=0
    required_record_fields={"claim_id","section_id","retrieval_requested","retrieval_rounds","retrieval_status","retriever_binding_fingerprint","retrieved_candidate_ids","retrieved_candidate_records","verification_context_snapshot","agentic_retrieval"}
    # E2E-BUG-02: "agentic_retrieval" es OBLIGATORIO -- confirmado por
    # auditoría que _independent_retrieve_claim es el ÚNICO productor
    # real de estos records (solo se invoca cuando dependencies.
    # retrieval_tool/retriever_binding existen; si no existen, no se
    # agrega ningún record para ese claim -- nunca un record "legacy"
    # sin agentic_retrieval) y SIEMPRE incluye esta clave, tanto en el
    # caso SUFFICIENT inicial como en el ciclo completo. Ningún test/
    # fixture en el repo construye un record sin ella.
    required_agentic_retrieval_fields={"planner_invoked","outcome","final_observation","steps","initial_grade_result","original_max_additional_retrieval_requests","agentic_additional_retrievals_used","effective_budget_for_verify_claim","trace"}
    for row in records:
        if not isinstance(row, Mapping) or set(row)!=required_record_fields:
            # E2E-DIAG-03: instrumentación temporal, solo diagnóstico.
            row_set = set(row) if isinstance(row, Mapping) else set()
            print(
                "AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_INVALID:FIELD_SET",
                "row_fields=", sorted(row_set),
                "required_fields=", sorted(required_record_fields),
                "extra_in_row=", sorted(row_set - required_record_fields),
                "missing_from_row=", sorted(required_record_fields - row_set),
                "claim_id=", row.get("claim_id") if isinstance(row, Mapping) else None,
                "section_id=", row.get("section_id") if isinstance(row, Mapping) else None,
            )
            raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_INVALID:FIELD_SET")
        row=dict(row); ident=(row["section_id"],row["claim_id"])
        if not all(isinstance(row[k],str) and row[k] for k in ("claim_id","section_id")):
            # E2E-DIAG-03: instrumentación temporal, solo diagnóstico.
            print(
                "AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_INVALID:IDENTITY",
                "claim_id=", repr(row.get("claim_id")), type(row.get("claim_id")).__name__,
                "section_id=", repr(row.get("section_id")), type(row.get("section_id")).__name__,
            )
            raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_INVALID:IDENTITY")
        if ident in identities: raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_DUPLICATE")
        identities.add(ident)
        if type(row["retrieval_requested"]) is not int or row["retrieval_requested"] < 1 or type(row["retrieval_rounds"]) is not int or row["retrieval_rounds"] < 1: raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_NOT_RETRIEVED")
        if row["retrieval_status"] not in {"COMPLETED_WITH_RESULTS","COMPLETED_NO_RESULTS"}: raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_STATUS_INVALID")
        validate_sha256_hex(row["retriever_binding_fingerprint"], field="retriever_binding_fingerprint")
        if type(row["retrieved_candidate_ids"]) not in (tuple,list) or any(not isinstance(x,str) or not x for x in row["retrieved_candidate_ids"]):
            # E2E-DIAG-03: instrumentación temporal, solo diagnóstico.
            print(
                "AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_INVALID:CANDIDATE_IDS",
                "retrieved_candidate_ids=", repr(row["retrieved_candidate_ids"]),
                "element_types=", [type(x).__name__ for x in row["retrieved_candidate_ids"]] if type(row["retrieved_candidate_ids"]) in (tuple, list) else "N/A (not tuple/list)",
            )
            raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_RECORD_INVALID:CANDIDATE_IDS")
        candidate_records=row["retrieved_candidate_records"]
        if type(candidate_records) not in (tuple,list): raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_CANDIDATES_INVALID")
        normalized_candidates=[]
        for candidate in candidate_records:
            required={"evidence_id","source_filename","chunk_id","query_ids","text_fingerprint"}
            if not isinstance(candidate,Mapping) or set(candidate)!=required: raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_CANDIDATE_INVALID")
            candidate=dict(candidate)
            if any(not isinstance(candidate[k],str) or not candidate[k] for k in ("evidence_id","source_filename","chunk_id")): raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_CANDIDATE_INVALID")
            validate_sha256_hex(candidate["text_fingerprint"], field="text_fingerprint")
            if type(candidate["query_ids"]) not in (tuple,list) or row["claim_id"] not in tuple(str(x) for x in candidate["query_ids"]): raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RETRIEVAL_CLAIM_MISMATCH")
            candidate["query_ids"]=tuple(str(x) for x in candidate["query_ids"]); normalized_candidates.append(candidate)
        agentic = row["agentic_retrieval"]
        if not isinstance(agentic, Mapping) or set(agentic) != required_agentic_retrieval_fields: raise ValueError("AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID")
        agentic = dict(agentic)
        if not isinstance(agentic["planner_invoked"], bool): raise ValueError("AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID")
        if agentic["outcome"] is not None and not isinstance(agentic["outcome"], str): raise ValueError("AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID")
        if agentic["initial_grade_result"] not in ("SUFFICIENT", "INSUFFICIENT"): raise ValueError("AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID")
        if type(agentic["steps"]) not in (tuple, list): raise ValueError("AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID")
        if not isinstance(agentic["trace"], Mapping): raise ValueError("AGENT07_RUNTIME_AGENTIC_RETRIEVAL_RECORD_INVALID")
        try:
            expected_effective_budget = compute_effective_budget_for_verify_claim(
                original_max_additional_retrieval_requests=agentic["original_max_additional_retrieval_requests"],
                agentic_additional_retrievals_used=agentic["agentic_additional_retrievals_used"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"AGENT07_RUNTIME_AGENTIC_RETRIEVAL_BUDGET_INVALID: {exc}") from exc
        if type(agentic["effective_budget_for_verify_claim"]) is not int or isinstance(agentic["effective_budget_for_verify_claim"], bool) or agentic["effective_budget_for_verify_claim"] != expected_effective_budget:
            raise ValueError("AGENT07_RUNTIME_AGENTIC_RETRIEVAL_BUDGET_MISMATCH")
        by_id={}; by_pair={}
        for candidate in normalized_candidates:
            canonical=json.dumps(candidate,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
            eid=candidate["evidence_id"]; pair=(candidate["source_filename"],candidate["chunk_id"])
            if eid in by_id and by_id[eid] != canonical: raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_CANDIDATE_CONFLICT")
            if pair in by_pair and by_pair[pair] != canonical: raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_CANDIDATE_CONFLICT")
            by_id[eid]=canonical; by_pair[pair]=canonical
        normalized_candidates=[json.loads(by_id[k]) for k in sorted(by_id)]
        for candidate in normalized_candidates: candidate["query_ids"]=tuple(candidate["query_ids"])
        snapshot=row["verification_context_snapshot"]
        if not isinstance(snapshot,Mapping) or set(snapshot)!={"claim_id","section_id","eligible_evidence"}: raise ValueError("AGENT07_RUNTIME_VERIFICATION_CONTEXT_SNAPSHOT_INVALID")
        if (str(snapshot["claim_id"]),str(snapshot["section_id"])) != (row["claim_id"],row["section_id"]): raise ValueError("AGENT07_RUNTIME_VERIFICATION_CONTEXT_SNAPSHOT_IDENTITY_MISMATCH")
        if type(snapshot["eligible_evidence"]) not in (tuple,list): raise ValueError("AGENT07_RUNTIME_VERIFICATION_CONTEXT_SNAPSHOT_INVALID")
        normalized_snapshot=[]; snapshot_ids=set(); snapshot_pairs=set()
        snapshot_required={"evidence_id","source_filename","chunk_id","authorized_for_section","text_fingerprint"}
        for evidence in snapshot["eligible_evidence"]:
            if not isinstance(evidence,Mapping) or set(evidence)!=snapshot_required: raise ValueError("AGENT07_RUNTIME_VERIFICATION_CONTEXT_EVIDENCE_INVALID")
            evidence=dict(evidence)
            if any(not isinstance(evidence[k],str) or not evidence[k] for k in ("evidence_id","source_filename","chunk_id")) or type(evidence["authorized_for_section"]) is not bool: raise ValueError("AGENT07_RUNTIME_VERIFICATION_CONTEXT_EVIDENCE_INVALID")
            validate_sha256_hex(evidence["text_fingerprint"], field="text_fingerprint")
            if evidence["evidence_id"] in snapshot_ids or (evidence["source_filename"],evidence["chunk_id"]) in snapshot_pairs: raise ValueError("AGENT07_RUNTIME_VERIFICATION_CONTEXT_EVIDENCE_DUPLICATE")
            snapshot_ids.add(evidence["evidence_id"]); snapshot_pairs.add((evidence["source_filename"],evidence["chunk_id"])); normalized_snapshot.append(evidence)
        snapshot={"claim_id":row["claim_id"],"section_id":row["section_id"],"eligible_evidence":tuple(sorted(normalized_snapshot,key=lambda x:(x["evidence_id"],x["source_filename"],x["chunk_id"])))}
        # evidence_id is a claim-local alias and may change during canonicalization.
        # Provenance closure uses the stable physical passage identity plus
        # authorization and content fingerprint.
        snapshot_set={(e["source_filename"],e["chunk_id"],e["authorized_for_section"],e["text_fingerprint"]) for e in snapshot["eligible_evidence"]}
        for candidate in normalized_candidates:
            identity=(candidate["source_filename"],candidate["chunk_id"],True,candidate["text_fingerprint"])
            if identity not in snapshot_set: raise ValueError("AGENT07_RUNTIME_RETRIEVED_EVIDENCE_SNAPSHOT_MISMATCH")
        ids=tuple(row["retrieved_candidate_ids"])
        if tuple(sorted(ids)) != tuple(sorted(c["evidence_id"] for c in normalized_candidates)): raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_CANDIDATE_ID_MISMATCH")
        if row["retrieval_status"]=="COMPLETED_WITH_RESULTS":
            if not ids: raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_STATUS_CONTRADICTION")
            with_results += 1
        else:
            if ids or normalized_candidates: raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_STATUS_CONTRADICTION")
            without_results += 1
        row["retrieved_candidate_ids"]=ids; row["retrieved_candidate_records"]=tuple(normalized_candidates); row["verification_context_snapshot"]=snapshot; normalized_records.append(row)
    if metrics["independent_rag_claims"] != len(normalized_records): raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_UNPROVEN:COUNT_MISMATCH")
    if metrics["independent_rag_claims_with_results"] != with_results or metrics["independent_rag_claims_without_results"] != without_results or with_results + without_results != metrics["independent_rag_claims"]: raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RAG_RESULT_COUNTS_MISMATCH")
    metrics=dict(metrics); metrics["independent_rag_claim_records"]=tuple(normalized_records); payload["execution_metrics"]=metrics
    if any(metrics[k] != 0 for k in ("additional_llm_calls","additional_retrieval_rounds","official_writes","physical_corrections")): raise ValueError("AGENT07_RUNTIME_EXECUTION_ISOLATION_VIOLATION")
    if metrics["independent_rag_claims"] > metrics["claims_processed"] or metrics["evidence_candidate_validation_claims"] > metrics["claims_processed"]: raise ValueError("AGENT07_RUNTIME_CLAIM_EXECUTION_METRICS_INCOHERENT")
    errors = tuple(_validate_error_record(x) for x in payload["runtime_error_records"])
    bundle, resolution, blocked = payload["provisional_bundle"], payload["multi_proposal_resolution_result"], payload["blocked_runtime_audit_record"]
    if blocked is not None:
        _validate_blocked_audit(blocked)
        if payload["runtime_status"] != "BLOCKED" or bundle is not None or resolution is not None or tuple(payload["candidate_artifact_inventory"]): raise ValueError("AGENT07_RUNTIME_OPERATIONAL_BLOCK_SHAPE_INVALID")
    else:
        if not isinstance(bundle, Mapping) or not isinstance(resolution, Mapping): raise ValueError("AGENT07_RUNTIME_RESULT_PAYLOAD_INVALID")
        validated_bundle = validate_provisional_verification_traceability_bundle_contract(bundle)
        validated_resolution = validate_provisional_multi_proposal_resolution_result(resolution)
        snapshot_by_claim={(r["section_id"],r["claim_id"]):{(e["source_filename"],e["chunk_id"],e["authorized_for_section"],e["text_fingerprint"]) for e in r["verification_context_snapshot"]["eligible_evidence"]} for r in normalized_records}
        for evidence in validated_bundle.get("claim_evidence_traceability_rows",()):
            key=(str(evidence["section_id"]),str(evidence["claim_id"]))
            if key in snapshot_by_claim:
                identity=(str(evidence["source_filename"]),str(evidence["chunk_id"]),bool(evidence["authorized_for_section"]),str(evidence["text_fingerprint"]))
                if identity not in snapshot_by_claim[key]: raise ValueError("AGENT07_RUNTIME_TERMINAL_EVIDENCE_CONTEXT_MISMATCH")
        if validated_resolution["source_bundle_audit_fingerprint"] != validated_bundle["aggregation_audit_fingerprint"]: raise ValueError("AGENT07_RUNTIME_SOURCE_BUNDLE_AUDIT_MISMATCH")
        expected = _resolution_to_runtime_status(validated_resolution["resolution_status"])
        if payload["runtime_status"] != expected: raise ValueError("AGENT07_RUNTIME_STATUS_MISMATCH")
        inv = tuple(_validate_inventory_record(x, validated_bundle, validated_resolution) for x in payload["candidate_artifact_inventory"])
        if len(inv) != 2 or {x["artifact_type"] for x in inv} != set(CANDIDATE_ARTIFACT_TYPES): raise ValueError("AGENT07_RUNTIME_ARTIFACT_INVENTORY_INVALID")
        payload["candidate_artifact_inventory"] = inv
    if not isinstance(payload["runtime_issue_codes"], (tuple,list)) or any(x not in RUNTIME_ISSUE_CODES for x in payload["runtime_issue_codes"]): raise ValueError("AGENT07_RUNTIME_ISSUE_CODES_INVALID")
    if type(payload["result_contract_valid"]) is not bool or (not allow_unvalidated and payload["result_contract_valid"] is not True): raise ValueError("AGENT07_RUNTIME_RESULT_VALIDITY_NOT_DERIVED")
    payload["runtime_error_records"] = errors; payload["runtime_warnings"] = tuple(payload["runtime_warnings"]); payload["runtime_issue_codes"] = tuple(payload["runtime_issue_codes"])
    return payload


def create_agent07_runtime_result(**kwargs: Any) -> Agent07RuntimeResult:
    if "result_contract_valid" in kwargs: raise TypeError("result_contract_valid is derived")
    provisional = Agent07RuntimeResult(**kwargs, result_contract_valid=False)
    validated = validate_agent07_runtime_result_contract(provisional, allow_unvalidated=True)
    final = Agent07RuntimeResult(**{**validated, "result_contract_valid": True})
    validate_agent07_runtime_result_contract(final); return final


def _candidate_inventory(bundle: Mapping[str, Any], resolution: Mapping[str, Any], schema_versions: Mapping[str, str]) -> tuple[dict[str, Any], ...]:
    bi = bundle["aggregation_status"] == "INVALID"; rb = resolution["resolution_status"] == "BLOCKED"
    return (
        CandidateArtifactRecord(CANDIDATE_ARTIFACT_TYPES[0], "BLOCKED_AUDIT_ONLY" if bi else ("PARTIAL_CANDIDATE" if bundle["aggregation_status"] == "PARTIAL" else "READY_CANDIDATE"), True, "AGENT07_RUNTIME", str(schema_versions.get("provisional_bundle", "UNSPECIFIED")), None if bi else bundle["normalized_bundle_fingerprint"], bundle["aggregation_audit_fingerprint"]).to_dict(),
        CandidateArtifactRecord(CANDIDATE_ARTIFACT_TYPES[1], "BLOCKED_AUDIT_ONLY" if rb else ("PARTIAL_CANDIDATE" if resolution["resolution_status"] == "PARTIAL" else "READY_CANDIDATE"), True, "AGENT07_RUNTIME", str(schema_versions.get("multi_proposal_resolution", "UNSPECIFIED")), None if rb else resolution["multi_proposal_resolution_fingerprint"], resolution["multi_proposal_audit_fingerprint"]).to_dict(),
    )


def _base_metrics(**overrides: Any) -> dict[str, Any]:
    result = {"claims_processed":0,"independent_rag_claims":0,"independent_rag_claims_with_results":0,"independent_rag_claims_without_results":0,"independent_rag_claim_records":(),"evidence_candidate_validation_claims":0,"correction_proposals":0,"reverification_inputs":0,"prechecks":0,"reverifications":0,"comparisons":0,"additional_llm_calls":0,"additional_retrieval_rounds":0,"official_writes":0,"physical_corrections":0}; result.update(overrides); return result

def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False).encode("utf-8")).hexdigest()

def _build_agentic_planner_invoke_fn(verification_llm: Any) -> Callable[[str], str]:
    """Adapter mínimo: prompt string (Bloque 2, exclusivo) -> infraestructura
    LLM existente de Stage 07 -> string normalizado para el parser de
    Bloque 2 (exclusivo). No introduce prompt/parser de VerificationAgent,
    no crea cliente/credenciales/config nuevos, no agrega system prompt."""

    def invoke(prompt: str) -> str:
        messages = ({"role": "user", "content": prompt},)
        raw = verification_llm.invoke(messages)
        normalized = normalize_verification_llm_response(raw)
        return normalized if isinstance(normalized, str) else json.dumps(normalized)

    return invoke


def _run_agentic_retrieval_for_claim(
    *,
    claim_id: str,
    claim_text: str,
    initial_candidates: list[dict[str, Any]],
    allowed_source_filenames: frozenset[str] | set[str],
    retrieval_tool: Any,
    verification_llm: Any,
    original_max_additional_retrieval_requests: int,
    initial_top_k: int,
    fetch_k: int,
) -> dict[str, Any]:
    """Unidad de integración aislada (AGENTIC-RETRIEVAL-BLOQUE-5): decide,
    sobre el retrieval inicial YA ejecutado (``initial_candidates``, sin un
    segundo retrieval inicial), si hace falta el ciclo Agentic
    (controller de Bloque 2 + executor de Bloque 4), y produce los
    candidatos finales con el schema real completo -- nunca reconstruidos
    desde ``eligible_evidence``.

    No reimplementa el controller ni el executor -- solo los orquesta.

    Presupuesto COMPARTIDO con ``VerificationAgent.verify_claim`` (fix de
    esta ronda, contrato ya cerrado en Bloques 1/2 -- no son presupuestos
    independientes): ``original_max_additional_retrieval_requests`` es
    EXACTAMENTE ``max_additional_retrieval_requests`` de la política de
    verificación oficial validada (``get_verification_input_policy``),
    sin ningún default propio inventado. Retorna
    ``effective_budget_for_verify_claim`` (vía
    ``compute_effective_budget_for_verify_claim``, Bloque 1, nunca
    reimplementado) para que el caller lo inyecte en la política que
    ``verify_claim`` realmente usará para este claim.

    Retorna un dict estructurado, nunca descartado por el caller:
        {
            "final_candidates": list[dict],   # schema real completo
            "planner_invoked": bool,
            "outcome": str | None,            # None si SUFFICIENT inicial (ciclo no ejecutado)
            "final_observation": dict | None, # asdict-like, o None
            "steps": tuple[dict, ...],
            "initial_grade_result": str,
            "agentic_additional_retrievals_used": int,
            "effective_budget_for_verify_claim": int,
        }
    """
    effective_top_k_max = validate_effective_top_k_max(fetch_k)
    authorized = frozenset(allowed_source_filenames)

    initial_grade = grade_evidence(
        claim_text=claim_text, candidates=initial_candidates, thresholds=DEFAULT_GRADER_THRESHOLDS,
    )
    initial_minimum_viable = is_minimum_viable_evidence(
        candidates=initial_candidates, thresholds=DEFAULT_MINIMUM_VIABLE_THRESHOLDS, authorized_sources=authorized,
    )
    evidence_ids = tuple(f"{c['source_filename']}::{c['chunk_id']}" for c in initial_candidates)

    if initial_grade["grade_result"] == "SUFFICIENT":
        # SUFFICIENT inicial: el planner NUNCA se invoca, no se ejecuta
        # run_agentic_retrieval_cycle -- los candidatos iniciales son los
        # finales. Agentic no consumió nada del presupuesto compartido --
        # verify_claim conserva el budget original íntegro.
        agentic_result_sufficient = {
            "planner_invoked": False,
            "outcome": None,
            "final_observation": None,
            "steps": (),
            "initial_grade_result": "SUFFICIENT",
            "agentic_additional_retrievals_used": 0,
            "effective_budget_for_verify_claim": compute_effective_budget_for_verify_claim(
                original_max_additional_retrieval_requests=original_max_additional_retrieval_requests,
                agentic_additional_retrievals_used=0,
            ),
        }
        return {
            "final_candidates": list(initial_candidates),
            "planner_invoked": False,
            "outcome": None,
            "final_observation": None,
            "steps": (),
            "initial_grade_result": "SUFFICIENT",
            "agentic_additional_retrievals_used": 0,
            "effective_budget_for_verify_claim": agentic_result_sufficient["effective_budget_for_verify_claim"],
            "trace": build_agentic_retrieval_trace(
                claim_id=claim_id,
                initial_candidate_count=initial_grade["candidate_count"],
                initial_max_relevance_score=initial_grade["max_relevance_score"],
                initial_top_k=initial_top_k,
                raw_transition_attempts=[],
                agentic_result=agentic_result_sufficient,
            ),
        }

    initial_observation = AgenticRetrievalObservation(
        claim_id=claim_id,
        claim_text=claim_text,
        current_query=claim_text,
        retrieval_round=0,
        current_top_k=initial_top_k,
        effective_top_k_max=effective_top_k_max,
        remaining_retrieval_budget=original_max_additional_retrieval_requests,
        candidate_count=initial_grade["candidate_count"],
        evidence_ids=evidence_ids,
        max_relevance_score=initial_grade["max_relevance_score"],
        grade_result=initial_grade["grade_result"],
        reason_codes=initial_grade["reason_codes"],
        minimum_viable_evidence=initial_minimum_viable,
        query_rewrite_count=0,
    )

    executor = AgenticRetrievalActionExecutor(
        retriever=retrieval_tool,
        allowed_source_filenames=authorized,
        claim_id=claim_id,
        claim_text=claim_text,
        initial_candidates=initial_candidates,
    )
    invoke_planner_fn = _build_agentic_planner_invoke_fn(verification_llm)
    # Bloque 6: instrumentación puramente observacional -- envuelve el
    # executor sin alterar su comportamiento, capturando Observations
    # reales antes/después de cada acción.
    traced_execute_action_fn, raw_transition_attempts = build_traced_execute_action_fn(executor)

    result = run_agentic_retrieval_cycle(
        initial_observation=initial_observation,
        invoke_planner_fn=invoke_planner_fn,
        execute_action_fn=traced_execute_action_fn,
    )

    planner_invoked = (
        any(step.get("planner_invoked") for step in result.steps)
        or result.outcome == "AGENTIC_PLANNER_FAILED"  # el planner sí fue consultado, aunque su respuesta fallara
    )
    final_candidates = list(executor.current_candidates) if result.steps else list(initial_candidates)

    # agentic_additional_retrievals_used: derivado de la fuente canónica
    # (remaining_retrieval_budget antes/después), no contando strings/steps.
    if result.final_observation is not None:
        agentic_used = initial_observation.remaining_retrieval_budget - result.final_observation.remaining_retrieval_budget
    else:
        # AGENTIC_PLANNER_FAILED en el primer step sin transición ejecutada
        # todavía: no se consumió presupuesto real de retrieval.
        agentic_used = 0

    effective_budget_for_verify_claim = compute_effective_budget_for_verify_claim(
        original_max_additional_retrieval_requests=original_max_additional_retrieval_requests,
        agentic_additional_retrievals_used=agentic_used,
    )

    final_observation_dict = None
    if result.final_observation is not None:
        final_observation_dict = {
            "claim_id": result.final_observation.claim_id,
            "current_query": result.final_observation.current_query,
            "retrieval_round": result.final_observation.retrieval_round,
            "current_top_k": result.final_observation.current_top_k,
            "remaining_retrieval_budget": result.final_observation.remaining_retrieval_budget,
            "candidate_count": result.final_observation.candidate_count,
            "evidence_ids": result.final_observation.evidence_ids,
            "max_relevance_score": result.final_observation.max_relevance_score,
            "grade_result": result.final_observation.grade_result,
            "reason_codes": result.final_observation.reason_codes,
            "minimum_viable_evidence": result.final_observation.minimum_viable_evidence,
            "query_rewrite_count": result.final_observation.query_rewrite_count,
        }

    agentic_result_insufficient = {
        "planner_invoked": planner_invoked,
        "outcome": result.outcome,
        "final_observation": final_observation_dict,
        "steps": tuple(result.steps),
        "initial_grade_result": "INSUFFICIENT",
        "agentic_additional_retrievals_used": agentic_used,
        "effective_budget_for_verify_claim": effective_budget_for_verify_claim,
    }

    return {
        "final_candidates": final_candidates,
        "planner_invoked": planner_invoked,
        "outcome": result.outcome,
        "final_observation": final_observation_dict,
        "steps": tuple(result.steps),
        "initial_grade_result": "INSUFFICIENT",
        "agentic_additional_retrievals_used": agentic_used,
        "effective_budget_for_verify_claim": effective_budget_for_verify_claim,
        "trace": build_agentic_retrieval_trace(
            claim_id=claim_id,
            initial_candidate_count=initial_grade["candidate_count"],
            initial_max_relevance_score=initial_grade["max_relevance_score"],
            initial_top_k=initial_top_k,
            raw_transition_attempts=raw_transition_attempts,
            agentic_result=agentic_result_insufficient,
        ),
    }


def _independent_retrieve_claim(
    context: Mapping[str, Any],
    dependencies: VerificationRuntimeDependencies,
    *,
    agentic_retrieval_policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if dependencies.retrieval_tool is None or dependencies.retriever_binding is None:
        raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RETRIEVER_REQUIRED")
    binding = dict(dependencies.retriever_binding)
    required = {"experiment_id", "collection_name", "embedding_model", "chroma_manifest_fingerprint", "chunks_manifest_fingerprint"}
    if set(binding) != required:
        raise ValueError("AGENT07_RUNTIME_RETRIEVER_BINDING_INVALID")
    binding_fp = _canonical_hash(binding)
    claim_id = str(context["claim_id"])
    section_id = str(context["section_id"])
    claim_text = str(context.get("claim_text") or context.get("original_claim_text") or "").strip()
    inherited = tuple(deepcopy(context.get("eligible_evidence", ())))
    authorized_sources = tuple(context.get("authorized_source_filenames", ()))
    if not authorized_sources or any(not isinstance(x, str) or not x for x in authorized_sources) or len(set(authorized_sources)) != len(authorized_sources):
        raise ValueError("AGENT07_RUNTIME_AUTHORIZED_SOURCE_UNIVERSE_INVALID")
    authorized_source_set = set(authorized_sources)
    request = {
        "claim_id": claim_id,
        "section_id": section_id,
        "claim_context": deepcopy(context),
        "retrieval_reason_codes": ("INDEPENDENT_CLAIM_RETRIEVAL",),
        "remaining_budget": 1,
        "eligible_evidence": inherited,
        "allowed_source_filenames": tuple(sorted(authorized_source_set)),
        "retriever_binding": deepcopy(binding),
    }
    raw = dependencies.retrieval_tool.retrieve_more(deepcopy(request))
    delta = validate_additional_retrieval_delta(raw, strict=True)
    rounds = int(delta.get("rounds_executed", 0) or 0)
    if rounds < 1:
        raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RETRIEVAL_ROUND_MISSING")

    selected = tuple(delta.get("selected_candidates", ()) or ())

    # AGENTIC-RETRIEVAL-BLOQUE-5: el retrieval inicial ya ejecutado
    # (`selected`, schema real completo) se reutiliza tal cual como
    # initial_candidates -- NUNCA se ejecuta un segundo retrieval
    # inicial. Solo la PARTE recuperada adaptativamente (`selected` ->
    # `final_candidates`) puede cambiar; `inherited` (evidencia previa
    # legítima del handoff) se preserva intacta más abajo, sin tocar.
    # AGENTIC-RETRIEVAL-BLOQUE-5 (fix de presupuesto compartido): el
    # retrieval inicial ya ejecutado (`selected`, schema real completo)
    # se reutiliza tal cual como initial_candidates -- NUNCA se ejecuta
    # un segundo retrieval inicial. Solo la PARTE recuperada
    # adaptativamente (`selected` -> `final_candidates`) puede cambiar;
    # `inherited` (evidencia previa legítima del handoff) se preserva
    # intacta más abajo, sin tocar.
    #
    # Presupuesto: NO es independiente del de verify_claim -- es el
    # MISMO max_additional_retrieval_requests compartido (contrato ya
    # cerrado en Bloques 1/2). Se obtiene exclusivamente de la política
    # de verificación oficial validada (get_verification_input_policy),
    # sin ningún default propio -- fail-closed si la política es
    # inválida.
    verification_policy_validated = get_verification_input_policy(agentic_retrieval_policy)
    original_max_additional_retrieval_requests = verification_policy_validated["max_additional_retrieval_requests"]

    # Acceso fail-closed a los atributos del retriever contractual --
    # deben existir y ser válidos, sin coerción int(...) ni defaults
    # inventados (8/35).
    initial_top_k = dependencies.retrieval_tool.top_k
    fetch_k = dependencies.retrieval_tool.fetch_k

    agentic_result = _run_agentic_retrieval_for_claim(
        claim_id=claim_id,
        claim_text=claim_text,
        initial_candidates=[dict(c) for c in selected],
        allowed_source_filenames=authorized_source_set,
        retrieval_tool=dependencies.retrieval_tool,
        verification_llm=dependencies.verification_llm,
        original_max_additional_retrieval_requests=original_max_additional_retrieval_requests,
        initial_top_k=initial_top_k,
        fetch_k=fetch_k,
    )
    selected = tuple(agentic_result["final_candidates"])

    recovered_rows: list[dict[str, Any]] = []
    recovered_queries_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in selected:
        query_ids = tuple(str(x) for x in row.get("query_ids", ())) or (claim_id,)
        if claim_id not in query_ids:
            raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RETRIEVAL_CLAIM_MISMATCH")
        source = str(row.get("source_filename") or "").strip()
        chunk = str(row.get("chunk_id") or "").strip()
        if not source or not chunk:
            raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RETRIEVAL_IDENTITY_MISSING")
        if source not in authorized_source_set:
            raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RETRIEVAL_OUTLINE_VIOLATION")
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RETRIEVAL_TEXT_MISSING")
        evidence_id = str(row.get("evidence_id") or f"{source}::{chunk}").strip()
        if not evidence_id:
            raise ValueError("AGENT07_RUNTIME_INDEPENDENT_RETRIEVAL_IDENTITY_MISSING")
        pair = (source, chunk)
        recovered_queries_by_pair.setdefault(pair, set()).update(query_ids)
        recovered_rows.append({
            "evidence_id": evidence_id,
            "source_filename": source,
            "chunk_id": chunk,
            "text": text,
            "canonical_text": text,
            "authorized_for_section": True,
            "retrieval_origin": "AGENT07_INDEPENDENT_RAG",
            "usage_role": "SUPPORT",
            "query_ids": query_ids,
        })

    # The producer uses the same physical-evidence canonicalizer as the
    # handoff-to-core adapter.  Therefore a retrieval alias for an inherited
    # (source_filename, chunk_id) cannot survive as a second snapshot row.
    # Representation-only differences are normalized there; semantic text,
    # authorization, identity, or role conflicts remain blocking errors.
    try:
        canonical_all = _canonicalize_claim_evidence((*inherited, *recovered_rows))
    except ValueError as exc:
        code = str(exc).strip()
        if code.startswith("AGENT07_CONTEXT_ADAPTER_EVIDENCE_"):
            raise ValueError(
                "AGENT07_RUNTIME_INDEPENDENT_RAG_CANDIDATE_CONFLICT:" + code
            ) from exc
        raise
    recovered_pairs = set(recovered_queries_by_pair)
    canonical_by_pair = {
        (str(row["source_filename"]), str(row["chunk_id"])): deepcopy(dict(row))
        for row in canonical_all
    }

    # Preserve explicit latest-stage provenance for runtime consumers while
    # retaining the complete origin/alias sets produced by canonicalization.
    for pair in recovered_pairs:
        canonical_by_pair[pair]["retrieval_origin"] = "AGENT07_INDEPENDENT_RAG"
        canonical_by_pair[pair]["query_ids"] = tuple(sorted(recovered_queries_by_pair[pair]))

    canonical_evidence = tuple(
        canonical_by_pair[pair]
        for pair in sorted(canonical_by_pair, key=lambda p: (str(canonical_by_pair[p]["evidence_id"]), p[0], p[1]))
    )

    audit_candidates = []
    for pair in sorted(recovered_pairs, key=lambda p: (str(canonical_by_pair[p]["evidence_id"]), p[0], p[1])):
        evidence = canonical_by_pair[pair]
        canonical_text = str(evidence.get("canonical_text", evidence.get("text", ""))).strip()
        if not canonical_text:
            raise ValueError("AGENT07_RUNTIME_VERIFICATION_CONTEXT_EVIDENCE_TEXT_MISSING")
        audit_candidates.append({
            "evidence_id": str(evidence["evidence_id"]),
            "source_filename": pair[0],
            "chunk_id": pair[1],
            "query_ids": tuple(sorted(recovered_queries_by_pair[pair])),
            "text_fingerprint": fingerprint_text(canonical_text),
        })

    updated = deepcopy(dict(context))
    updated["eligible_evidence"] = canonical_evidence
    updated["agent07_independent_retrieval_executed"] = True
    updated["agent07_independent_retrieval_rounds"] = rounds
    updated["agent07_independent_retrieval_status"] = "COMPLETED_WITH_RESULTS" if audit_candidates else "COMPLETED_NO_RESULTS"

    snapshot_evidence = []
    for evidence in updated["eligible_evidence"]:
        text = str(evidence.get("canonical_text", evidence.get("text", ""))).strip()
        if not text:
            raise ValueError("AGENT07_RUNTIME_VERIFICATION_CONTEXT_EVIDENCE_TEXT_MISSING")
        snapshot_evidence.append({
            "evidence_id": str(evidence["evidence_id"]),
            "source_filename": str(evidence["source_filename"]),
            "chunk_id": str(evidence["chunk_id"]),
            "authorized_for_section": bool(evidence.get("authorized_for_section") is True),
            "text_fingerprint": fingerprint_text(text),
        })
    snapshot = {
        "claim_id": claim_id,
        "section_id": section_id,
        "eligible_evidence": tuple(sorted(snapshot_evidence, key=lambda x: (x["evidence_id"], x["source_filename"], x["chunk_id"]))),
    }
    candidate_ids = tuple(candidate["evidence_id"] for candidate in audit_candidates)
    record = {
        "claim_id": claim_id,
        "section_id": section_id,
        "retrieval_requested": 1,
        "retrieval_rounds": rounds,
        "retrieval_status": "COMPLETED_WITH_RESULTS" if candidate_ids else "COMPLETED_NO_RESULTS",
        "retriever_binding_fingerprint": binding_fp,
        "retrieved_candidate_ids": candidate_ids,
        "retrieved_candidate_records": tuple(audit_candidates),
        "verification_context_snapshot": snapshot,
        "agentic_retrieval": {
            "planner_invoked": agentic_result["planner_invoked"],
            "outcome": agentic_result["outcome"],
            "final_observation": agentic_result["final_observation"],
            "steps": agentic_result["steps"],
            "initial_grade_result": agentic_result["initial_grade_result"],
            "original_max_additional_retrieval_requests": original_max_additional_retrieval_requests,
            "agentic_additional_retrievals_used": agentic_result["agentic_additional_retrievals_used"],
            "effective_budget_for_verify_claim": agentic_result["effective_budget_for_verify_claim"],
            "trace": agentic_result["trace"],
        },
    }
    return updated, record



def _sanitized_stage_error_code(exc: Exception) -> str:
    """Preserve a safe contractual code, not only the Python exception class."""
    import re
    raw=str(exc).strip()
    match=re.match(r"^([A-Z][A-Z0-9_]*(?::[A-Z0-9_,.-]+)*)", raw)
    candidate=match.group(1) if match else ""
    token=candidate if "_" in candidate else type(exc).__name__
    return f"AGENT07_RUNTIME_STAGE_FAILURE:{token}"

def _blocked_runtime_result(*, stage: str, claim_id: str | None, section_id: str | None, error_code: str, classification: str, schema_versions: Mapping[str, str], metrics: Mapping[str, int] | None = None) -> Agent07RuntimeResult:
    core = {"stage":stage,"claim_id":claim_id,"section_id":section_id,"error_code":error_code,"error_classification":classification}
    audit = BlockedRuntimeAuditRecord(**core, runtime_audit_fingerprint=_audit_hash(core)).to_dict()
    error = RuntimeErrorRecord(**core).to_dict()
    return create_agent07_runtime_result(provisional_bundle=None, multi_proposal_resolution_result=None, candidate_artifact_inventory=(), execution_metrics=_base_metrics(**dict(metrics or {})), runtime_warnings=(), runtime_issue_codes=("AGENT07_RUNTIME_GLOBAL_BLOCK",), runtime_error_records=(error,), blocked_runtime_audit_record=audit, runtime_status="BLOCKED", correction_applied=False, official_artifacts_created=False, evaluation_ready_emitted=False)


def run_agent07_in_memory(runtime_input: Agent07RuntimeInput, *, dependencies: VerificationRuntimeDependencies) -> Agent07RuntimeResult:
    validated = validate_agent07_runtime_input_contract(runtime_input)
    if dependencies.correction_context_factory is None or dependencies.reverification_input_factory is None: raise ValueError("AGENT07_RUNTIME_CONTEXT_FACTORY_REQUIRED")
    contexts = deepcopy(validated["committed_agent06_output"]["claim_verification_contexts"]); config = deepcopy(validated["agent07_config"])
    vr=[]; proposals=[]; ri=[]; pre=[]; rev=[]; comp=[]; independent_rag_records=[]; verification_context_snapshots={}; stage="AGENT_INITIALIZATION"; claim_id=section_id=None
    try:
        agent = dependencies.verification_agent_factory(llm=dependencies.verification_llm, retrieval_tool=dependencies.retrieval_tool)
        for source in sorted(contexts, key=lambda x:(str(x["section_id"]),str(x["claim_id"]))):
            ctx=deepcopy(source); claim_id=str(ctx["claim_id"]); section_id=str(ctx["section_id"]); stage="INDEPENDENT_RAG"
            if dependencies.retrieval_tool is not None and dependencies.retriever_binding is not None:
                policy_for_agentic = config.get("verification_policy", config.get("policy", {}))
                ctx, rag_record = _independent_retrieve_claim(ctx, dependencies, agentic_retrieval_policy=policy_for_agentic); independent_rag_records.append(rag_record)
            stage="VERIFICATION"
            snapshot = next((deepcopy(r["verification_context_snapshot"]) for r in independent_rag_records if r["claim_id"]==claim_id and r["section_id"]==section_id), None)
            # A strict snapshot is mandatory for productive independent RAG records.
            # Legacy/in-memory fixtures without retrieval retain ID-only validation.
            verification_context_snapshots[(section_id,claim_id)] = snapshot
            if isinstance(agent, VerificationAgent):
                policy_overrides=dict(config.get("verification_policy", config.get("policy", {})))
                # AGENTIC-RETRIEVAL-BLOQUE-5 (fix de presupuesto compartido):
                # verify_claim NO recibe el budget original completo -- recibe
                # exactamente lo que quedó después de lo que consumió Agentic
                # Retrieval para este claim (compute_effective_budget_for_verify_claim,
                # Bloque 1, ya cerrado). Ejemplo: original=1, Agentic usa 1 -> verify_claim recibe 0.
                agentic_record_for_claim = next(
                    (r for r in independent_rag_records if r["claim_id"] == claim_id and r["section_id"] == section_id), None,
                )
                if agentic_record_for_claim is not None and "agentic_retrieval" in agentic_record_for_claim:
                    policy_overrides["max_additional_retrieval_requests"] = agentic_record_for_claim["agentic_retrieval"]["effective_budget_for_verify_claim"]
                core_ctx=build_claim_verification_context_from_agent06_handoff(ctx, verification_policy=policy_overrides, attempt_number=int(config.get("attempt_number",1)))
            else:
                core_ctx=deepcopy(ctx)
            verification=_plain(agent.verify_claim(deepcopy(core_ctx))); vr.append({"section_id":section_id,"claim_verification_result":deepcopy(verification)})
            stage="CORRECTION_PROPOSAL"; cctx=dependencies.correction_context_factory(deepcopy(ctx),deepcopy(verification),deepcopy(config)); proposal=_plain(dependencies.proposal_runner(deepcopy(cctx),llm=dependencies.correction_llm)); proposals.append(deepcopy(proposal))
            if proposal.get("accepted_for_reverification") is not True: continue
            stage="REVERIFICATION_INPUT"; inp=_plain(dependencies.reverification_input_factory(deepcopy(ctx),deepcopy(verification),deepcopy(proposal),deepcopy(config))); ri.append(deepcopy(inp))
            stage="PRECHECK"; pc=_plain(dependencies.precheck_runner(deepcopy(inp))); pre.append(deepcopy(pc))
            stage="REVERIFICATION"; rv=_plain(dependencies.reverification_runner(deepcopy(inp),deepcopy(pc),reverification_llm=dependencies.reverification_llm)); rev.append(deepcopy(rv))
            stage="COMPARISON"; cp=_plain(dependencies.comparison_runner(deepcopy(inp),deepcopy(pc),deepcopy(rv))); comp.append(deepcopy(cp))
        aggregation={"claim_verification_records":tuple(vr),"correction_proposals":tuple(proposals),"correction_reverification_inputs":tuple(ri),"correction_precheck_results":tuple(pre),"independent_reverification_results":tuple(rev),"before_after_comparison_results":tuple(comp),"policy_versions":deepcopy(validated["policy_versions"]),"schema_versions":deepcopy(validated["schema_versions"]),"additional_llm_calls":0,"additional_retrieval_rounds":0,"correction_applied":False,"official_artifacts_created":False}
        stage="BUNDLE_BUILD"; bundle=_plain(dependencies.bundle_builder(deepcopy(aggregation)))
        stage="MULTI_PROPOSAL_RESOLUTION"; resolution=_plain(dependencies.resolution_runner(deepcopy(bundle)))
        independent_rag_claims = len(independent_rag_records)
        independent_rag_claims_with_results = sum(1 for x in independent_rag_records if x["retrieval_status"]=="COMPLETED_WITH_RESULTS")
        independent_rag_claims_without_results = sum(1 for x in independent_rag_records if x["retrieval_status"]=="COMPLETED_NO_RESULTS")
        evidence_candidate_validation_claims = 0
        def evidence_identity(e):
            text=str(e.get("canonical_text",e.get("text",""))).strip()
            return (str(e.get("source_filename","")),str(e.get("chunk_id","")),bool(e.get("authorized_for_section") is True),fingerprint_text(text))
        for wrapper in vr:
            verification = wrapper["claim_verification_result"]
            key = (str(wrapper["section_id"]), str(verification["claim_id"]))
            snapshot = verification_context_snapshots[key]
            if snapshot is not None:
                snapshot_set={(e["source_filename"],e["chunk_id"],e["authorized_for_section"],e["text_fingerprint"]) for e in snapshot["eligible_evidence"]}
                terminal_eligible={evidence_identity(e) for e in verification.get("eligible_evidence", ())}
                terminal_used={evidence_identity(e) for e in verification.get("evidence_used", ())}
                terminal_rejected={evidence_identity(e) for e in verification.get("evidence_rejected", ())}
                valid_evidence = terminal_eligible.issubset(snapshot_set) and terminal_used.union(terminal_rejected).issubset(snapshot_set)
            else:
                source = next(x for x in contexts if str(x["section_id"])==key[0] and str(x["claim_id"])==key[1])
                source_ids={str(e.get("evidence_id","")) for e in source.get("eligible_evidence",())}
                eligible_ids={str(e.get("evidence_id","")) for e in verification.get("eligible_evidence",())}
                used_ids={str(e.get("evidence_id","")) for e in verification.get("evidence_used",())}
                rejected_ids={str(e.get("evidence_id","")) for e in verification.get("evidence_rejected",())}
                valid_evidence = eligible_ids.issubset(source_ids) and used_ids.union(rejected_ids).issubset(source_ids)
            if valid_evidence and verification.get("result_contract_valid") is True:
                evidence_candidate_validation_claims += 1
        # E2E-DIAG-03: instrumentación temporal -- evita que el audit
        # atribuya erróneamente el error a MULTI_PROPOSAL_RESOLUTION si
        # la excepción real ocurre dentro de create_agent07_runtime_result
        # (validación del contrato), no en resolution_runner.
        stage = "RUNTIME_RESULT_BUILD"
        return create_agent07_runtime_result(provisional_bundle=bundle,multi_proposal_resolution_result=resolution,candidate_artifact_inventory=_candidate_inventory(bundle,resolution,validated["schema_versions"]),execution_metrics=_base_metrics(claims_processed=len(vr),independent_rag_claims=independent_rag_claims,independent_rag_claims_with_results=independent_rag_claims_with_results,independent_rag_claims_without_results=independent_rag_claims_without_results,independent_rag_claim_records=tuple(independent_rag_records),evidence_candidate_validation_claims=evidence_candidate_validation_claims,correction_proposals=len(proposals),reverification_inputs=len(ri),prechecks=len(pre),reverifications=len(rev),comparisons=len(comp)),runtime_warnings=(),runtime_issue_codes=(),runtime_error_records=(),blocked_runtime_audit_record=None,runtime_status=_resolution_to_runtime_status(resolution["resolution_status"]),correction_applied=False,official_artifacts_created=False,evaluation_ready_emitted=False)
    except Exception as exc:
        # E2E-DIAG-02: instrumentación temporal, solo diagnóstico -- no
        # cambia lógica ni contratos, no convierte ni captura nada nuevo.
        print(
            "AGENT07_EXCEPTION_DEBUG",
            type(exc).__module__,
            type(exc).__qualname__,
            repr(exc),
            str(exc),
        )
        return _blocked_runtime_result(stage=stage,claim_id=claim_id,section_id=section_id,error_code=_sanitized_stage_error_code(exc),classification="DEPENDENCY" if stage in {"AGENT_INITIALIZATION","BUNDLE_BUILD","MULTI_PROPOSAL_RESOLUTION"} else "TECHNICAL",schema_versions=validated["schema_versions"],metrics=_base_metrics(claims_processed=len(vr),correction_proposals=len(proposals),reverification_inputs=len(ri),prechecks=len(pre),reverifications=len(rev),comparisons=len(comp)))
