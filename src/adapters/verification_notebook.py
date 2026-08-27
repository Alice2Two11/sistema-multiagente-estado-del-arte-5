"""Notebook and transactional adapter for Agent 07.

The adapter reuses the frozen PipelineState/StateStore transaction model. It
supports PREPARE, EXECUTE, COMMIT and RESUME without modifying the draft or
emitting EVALUATION_READY.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import shutil
from typing import Any, Callable, Mapping

from src.adapters.agent06_verification_handoff import build_agent07_input_from_committed_agent06
from src.adapters.verification_runtime import (
    Agent07RuntimeInput, Agent07RuntimeResult, VerificationRuntimeDependencies,
    run_agent07_in_memory, validate_agent07_runtime_input_contract,
    validate_agent07_runtime_result_contract, validate_committed_agent06_output_contract,
)
from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import (
    AgentResult, DecisionInfo, ExecutionStatus, QualityStatus,
    RequestedTransition, ToolUsage, TransitionAction,
)
from src.io.atomic_write import atomic_write_json, atomic_write_bytes
from src.state.fingerprints import build_stage_fingerprints, sha256_bytes, sha256_file
from src.state.state_store import ResumeResolution, StateStore

NOTEBOOK_PREPARATION_STATUSES = ("READY", "BLOCKED")
NOTEBOOK_EXECUTION_STATUSES = ("NOT_EXECUTED", "COMPLETED", "PARTIAL", "BLOCKED")
AGENT07_STAGE_NAME = "07_agente_verificador"
AGENT07_ARTIFACT_NAMES = (
    "provisional_verification_traceability_bundle.json",
    "multi_proposal_resolution_result.json",
    "agent07_runtime_report.json",
    "agent07_artifact_manifest.json",
)
# Artefacto CONDICIONAL: solo existe cuando la transición clasificada es
# RETURN. No forma parte de AGENT07_ARTIFACT_NAMES (los 4 son
# incondicionales para toda ejecución científica) -- se trata aparte en
# cada punto de validación que antes asumía un conjunto fijo.
AGENT07_CONDITIONAL_ARTIFACT_NAME = "writer_revision_request.json"




def resolve_committed_agent06_output(*, store: StateStore, experiment_root: str | Path, stage_name: str = "06_agente_redactor", agent07_config: Mapping[str, Any] | None = None, policy_versions: Mapping[str, str] | None = None, schema_versions: Mapping[str, str] | None = None, experiment_paths: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the Agent 07 hand-off from the exact committed Agent 06 result."""
    return build_agent07_input_from_committed_agent06(
        store=store, stage_name=stage_name, agent07_config=agent07_config or {},
        policy_versions=policy_versions or {}, schema_versions=schema_versions or {},
        experiment_paths=experiment_paths or {"experiment_root": str(experiment_root)},
        outline_paper_mapping_path=Path(experiment_root) / "outputs" / "04_outline" / "outline_paper_mapping.csv",
    )


@dataclass(frozen=True, slots=True)
class Agent07NotebookRequest:
    configuration_source: Any
    committed_agent06_source: Any
    experiment_paths: Mapping[str, str]
    policy_versions: Mapping[str, str]
    schema_versions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Agent07NotebookPreparationResult:
    runtime_input: Agent07RuntimeInput | None
    dependencies: VerificationRuntimeDependencies | None
    configuration_errors: tuple[str, ...]
    input_contract_errors: tuple[str, ...]
    dependency_errors: tuple[str, ...]
    execution_errors: tuple[str, ...]
    runtime_result: Agent07RuntimeResult | None
    preparation_status: str
    execution_status: str
    official_artifacts_created: bool = False
    correction_applied: bool = False
    evaluation_ready_emitted: bool = False
    result_contract_valid: bool = False

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True, slots=True)
class Agent07ManifestArtifact:
    artifact_name: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True, slots=True)
class Agent07ArtifactManifest:
    stage: str
    decision_id: str
    attempt_number: int
    execution_fingerprint: str
    schema_versions: Mapping[str, str]
    source_draft_fingerprint: str
    artifacts: tuple[Agent07ManifestArtifact, ...]
    correction_applied: bool
    evaluation_ready_emitted: bool

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedAgent07Execution:
    decision_id: str
    runtime_input: Agent07RuntimeInput
    input_fingerprints: Mapping[str, str]
    stage_fingerprints: Mapping[str, str]
    attempt_number: int = 1
    execution_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class ExecutedAgent07Execution:
    decision_id: str
    runtime_input: Agent07RuntimeInput
    runtime_result: Agent07RuntimeResult
    candidate_payloads: Mapping[str, bytes]
    staging_manifest_path: str
    agent_result: AgentResult
    persisted_result_path: str
    stage_fingerprints: Mapping[str, str]
    attempt_number: int = 1
    execution_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class Agent07ResumeResult:
    action: str
    committed_result: AgentResult | None
    executed: ExecutedAgent07Execution | None


RESUME_ACTIONS = (
    "COMMITTED", "EXECUTED_NOT_COMMITTED", "REEXECUTE", "NO_COMMIT",
    "FINGERPRINT_MISMATCH", "ARTIFACT_MISMATCH", "MANIFEST_INCOMPLETE",
)
SCIENTIFIC_ARTIFACT_NAMES = AGENT07_ARTIFACT_NAMES[:3]
MANIFEST_NAME = AGENT07_ARTIFACT_NAMES[3]
OPERATIONAL_AUDIT_NAME = "agent07_operational_audit.json"


def _safe_error_code(prefix: str, exc: Exception) -> str: return f"{prefix}:{type(exc).__name__}"


def validate_agent07_notebook_preparation_result_contract(value: Agent07NotebookPreparationResult | Mapping[str, Any], *, allow_unvalidated: bool=False) -> dict[str, Any]:
    payload = asdict(value) if isinstance(value, Agent07NotebookPreparationResult) else deepcopy(dict(value)) if isinstance(value, Mapping) else None
    if payload is None or set(payload) != {f.name for f in fields(Agent07NotebookPreparationResult)}: raise ValueError("AGENT07_NOTEBOOK_PREPARATION_SCHEMA_INVALID")
    if payload["preparation_status"] not in NOTEBOOK_PREPARATION_STATUSES or payload["execution_status"] not in NOTEBOOK_EXECUTION_STATUSES: raise ValueError("AGENT07_NOTEBOOK_STATUS_INVALID")
    error_fields = ("configuration_errors","input_contract_errors","dependency_errors","execution_errors")
    for name in error_fields:
        if not isinstance(payload[name], (tuple,list)) or any(not isinstance(x,str) or not x for x in payload[name]): raise ValueError("AGENT07_NOTEBOOK_ERRORS_INVALID")
    all_errors = sum((len(payload[name]) for name in error_fields), 0)
    if payload["preparation_status"] == "READY":
        if payload["runtime_input"] is None or payload["dependencies"] is None or all_errors: raise ValueError("AGENT07_NOTEBOOK_READY_COHERENCE_INVALID")
    elif all_errors == 0: raise ValueError("AGENT07_NOTEBOOK_BLOCKED_WITHOUT_CAUSE")
    if payload["runtime_result"] is not None:
        validate_agent07_runtime_result_contract(payload["runtime_result"])
        expected = payload["runtime_result"]["runtime_status"] if isinstance(payload["runtime_result"], Mapping) else payload["runtime_result"].runtime_status
        if payload["execution_status"] != expected: raise ValueError("AGENT07_NOTEBOOK_EXECUTION_STATUS_MISMATCH")
    elif payload["execution_status"] != "NOT_EXECUTED" and not payload["execution_errors"]: raise ValueError("AGENT07_NOTEBOOK_EXECUTION_RESULT_MISSING")
    if any(payload[name] is not False for name in ("official_artifacts_created","correction_applied","evaluation_ready_emitted")): raise ValueError("AGENT07_NOTEBOOK_ISOLATION_INVALID")
    if type(payload["result_contract_valid"]) is not bool or (not allow_unvalidated and payload["result_contract_valid"] is not True): raise ValueError("AGENT07_NOTEBOOK_VALIDITY_NOT_DERIVED")
    return payload


def create_agent07_notebook_preparation_result(**kwargs: Any) -> Agent07NotebookPreparationResult:
    if "result_contract_valid" in kwargs: raise TypeError("result_contract_valid is derived")
    provisional=Agent07NotebookPreparationResult(**kwargs,result_contract_valid=False)
    validate_agent07_notebook_preparation_result_contract(provisional,allow_unvalidated=True)
    final=Agent07NotebookPreparationResult(**kwargs,result_contract_valid=True)
    validate_agent07_notebook_preparation_result_contract(final); return final


def prepare_agent07_notebook_execution(request: Agent07NotebookRequest, *, configuration_loader: Callable[[Any], Mapping[str, Any]], committed_output_loader: Callable[[Any], Mapping[str, Any]], dependency_resolver: Callable[[Mapping[str, Any]], VerificationRuntimeDependencies]) -> Agent07NotebookPreparationResult:
    ce=[]; ie=[]; de=[]; config=None; committed=None; dependencies=None; runtime_input=None
    try: config=deepcopy(dict(configuration_loader(deepcopy(request.configuration_source))))
    except Exception as exc: ce.append(_safe_error_code("AGENT07_NOTEBOOK_CONFIGURATION_ERROR",exc))
    try: committed=deepcopy(dict(committed_output_loader(deepcopy(request.committed_agent06_source))))
    except Exception as exc: ie.append(_safe_error_code("AGENT07_NOTEBOOK_AGENT06_LOAD_ERROR",exc))
    if committed is not None:
        candidate=Agent07RuntimeInput(committed, config or {}, deepcopy(dict(request.policy_versions)), deepcopy(dict(request.schema_versions)), deepcopy(dict(request.experiment_paths)))
        try:
            validate_agent07_runtime_input_contract(candidate)
            if config is not None: runtime_input=candidate
        except Exception as exc: ie.append(_safe_error_code("AGENT07_NOTEBOOK_INPUT_CONTRACT_ERROR",exc))
    if config is not None:
        try:
            dependencies=dependency_resolver(deepcopy(config))
            if not isinstance(dependencies,VerificationRuntimeDependencies): raise TypeError
        except Exception as exc: de.append(_safe_error_code("AGENT07_NOTEBOOK_DEPENDENCY_ERROR",exc)); dependencies=None
    status="READY" if runtime_input is not None and dependencies is not None and not (ce or ie or de) else "BLOCKED"
    return create_agent07_notebook_preparation_result(runtime_input=runtime_input,dependencies=dependencies,configuration_errors=tuple(ce),input_contract_errors=tuple(ie),dependency_errors=tuple(de),execution_errors=(),runtime_result=None,preparation_status=status,execution_status="NOT_EXECUTED",official_artifacts_created=False,correction_applied=False,evaluation_ready_emitted=False)


def execute_agent07_notebook_in_memory(preparation: Agent07NotebookPreparationResult) -> Agent07NotebookPreparationResult:
    validate_agent07_notebook_preparation_result_contract(preparation)
    if preparation.preparation_status != "READY" or preparation.runtime_input is None or preparation.dependencies is None: return preparation
    try:
        result=run_agent07_in_memory(preparation.runtime_input,dependencies=preparation.dependencies); validate_agent07_runtime_result_contract(result)
        return create_agent07_notebook_preparation_result(runtime_input=preparation.runtime_input,dependencies=preparation.dependencies,configuration_errors=preparation.configuration_errors,input_contract_errors=preparation.input_contract_errors,dependency_errors=preparation.dependency_errors,execution_errors=(),runtime_result=result,preparation_status="READY",execution_status=result.runtime_status,official_artifacts_created=False,correction_applied=False,evaluation_ready_emitted=False)
    except Exception as exc:
        return create_agent07_notebook_preparation_result(runtime_input=preparation.runtime_input,dependencies=preparation.dependencies,configuration_errors=preparation.configuration_errors,input_contract_errors=preparation.input_contract_errors,dependency_errors=preparation.dependency_errors,execution_errors=(_safe_error_code("AGENT07_NOTEBOOK_RUNTIME_ERROR",exc),),runtime_result=None,preparation_status="BLOCKED",execution_status="NOT_EXECUTED",official_artifacts_created=False,correction_applied=False,evaluation_ready_emitted=False)


def _stage_fingerprints(runtime_input: Agent07RuntimeInput):
    valid=validate_agent07_runtime_input_contract(runtime_input)
    return build_stage_fingerprints(input_data=valid["committed_agent06_output"],config_data={"agent07_config":valid["agent07_config"],"policy_versions":valid["policy_versions"],"schema_versions":valid["schema_versions"]},dependencies_data={"source_draft_fingerprint":valid["committed_agent06_output"]["source_draft_fingerprint"],"artifact_identity":valid["committed_agent06_output"]["artifact_identity"]})


def _validate_sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value): raise ValueError(code)
    return value


def validate_prepared_agent07_execution_contract(value: PreparedAgent07Execution) -> PreparedAgent07Execution:
    if not isinstance(value, PreparedAgent07Execution): raise ValueError("AGENT07_PREPARED_SCHEMA_INVALID")
    if not value.decision_id or value.attempt_number < 1: raise ValueError("AGENT07_PREPARED_IDENTITY_INVALID")
    validate_agent07_runtime_input_contract(value.runtime_input)
    required={"input","config","dependencies","composite"}
    if set(value.stage_fingerprints)!=required: raise ValueError("AGENT07_PREPARED_FINGERPRINTS_INVALID")
    for fp in value.stage_fingerprints.values(): _validate_sha(fp,"AGENT07_PREPARED_FINGERPRINT_INVALID")
    if value.execution_fingerprint != value.stage_fingerprints["composite"]: raise ValueError("AGENT07_PREPARED_EXECUTION_FINGERPRINT_MISMATCH")
    return value


def prepare_agent07_execution(*, store: StateStore, runtime_input: Agent07RuntimeInput) -> PreparedAgent07Execution:
    valid=validate_agent07_runtime_input_contract(runtime_input); fp=_stage_fingerprints(runtime_input)
    state=store.load(); attempt=state.stages.get(AGENT07_STAGE_NAME).attempts_used+1 if AGENT07_STAGE_NAME in state.stages else 1
    prepared=store.prepare_execution(target_stage=AGENT07_STAGE_NAME,intended_action="EXECUTE_AGENT07_VERIFICATION",attempt_number=attempt)
    result=PreparedAgent07Execution(prepared.decision_id,runtime_input,{"source_draft_fingerprint":valid["committed_agent06_output"]["source_draft_fingerprint"]},fp.to_dict(),attempt,fp.composite)
    return validate_prepared_agent07_execution_contract(result)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")


def _extract_claims_for_transition(runtime_result: Agent07RuntimeResult) -> list[dict[str, Any]]:
    """Adaptador productivo real, no fixture: ``provisional_bundle`` puede
    traer ``claim_verification_records`` (forma cruda, ver
    ``run_agent07_in_memory``) O ``claim_traceability_rows`` (forma real de
    ``build_provisional_verification_traceability_bundle``) -- nunca ambas
    con el mismo significado. Delega en
    ``canonicalize_claims_for_transition`` (``src/adapters/
    verification_claim_canonicalization.py``), que detecta cuál de las dos
    formas productivas reales trae el bundle y las une en un único modelo
    canónico antes de clasificar. Ver ese módulo para el mapeo de campos
    completo y la política de ``usage_role`` investigada."""

    from src.adapters.verification_claim_canonicalization import canonicalize_claims_for_transition

    return canonicalize_claims_for_transition(runtime_result.provisional_bundle)


def _agent07_has_classifiable_bundle(runtime_result: Agent07RuntimeResult) -> bool:
    """Determina si el runtime de 07 produjo un bundle real sobre el que
    vale la pena intentar clasificación científica.

    ``runtime_status`` puede ser ``BLOCKED`` por dos motivos distintos
    (ver ``_expected_candidate_payload_names``, que ya documentaba esta
    distinción sin que el resto del código la respetara):

    1. Bloqueo OPERATIVO temprano: ni ``provisional_bundle`` ni
       ``multi_proposal_resolution_result`` existen -- no hay nada que
       clasificar. Esto sigue siendo un fallo técnico real.
    2. Bloqueo CIENTÍFICO: el runtime sí completó la verificación y la
       resolución de propuestas, y produjo un bundle real -- pero alguna
       etapa posterior (ej. agregación) marcó ``runtime_status="BLOCKED"``.
       En este caso el bundle SÍ es clasificable: contiene claims reales
       con veredictos y elegibilidad de corrección reales, exactamente
       igual que un resultado ``COMPLETED``/``PARTIAL``.

    Antes de esta corrección, el caso 2 se trataba igual que el caso 1
    (nunca se intentaba clasificar, y el commit se rechazaba siempre) --
    eso convertía un resultado científico real (potencialmente
    corregible vía RETURN, o cerrable vía HALT_STAGE informado) en un
    ``RuntimeError`` técnico sin clasificar. Esta función es el único
    punto de verdad que usan ``_classify_agent07_transition``,
    ``_build_agent07_result`` y ``_validate_execution_for_commit`` para
    mantener el mismo criterio en los tres lugares.
    """

    if runtime_result.runtime_status in {"COMPLETED", "PARTIAL"}:
        return True
    return (
        runtime_result.runtime_status == "BLOCKED"
        and runtime_result.provisional_bundle is not None
        and runtime_result.multi_proposal_resolution_result is not None
    )


def _classify_agent07_transition(
    runtime_result: Agent07RuntimeResult, *, rounds_used: int, max_rounds: int,
    declared_contract_version: str | None = None, migration_signal: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Paso 1 (clasificación): SOLO deriva la acción real
    (ADVANCE/RETURN/HALT_STAGE) vía ``classify_verification_transition``.
    No construye ``writer_revision_request`` ni ``AgentResult`` -- eso
    queda en los pasos 2 y 5 respectivamente, para poder conocer si habrá
    ``writer_revision_request.json`` ANTES de construir los payloads
    (paso 3)."""

    from src.tools.verification.writer_revision_cycle import classify_verification_transition

    completed = _agent07_has_classifiable_bundle(runtime_result)
    if completed and runtime_result.provisional_bundle is not None:
        claims = _extract_claims_for_transition(runtime_result)
        decision = classify_verification_transition(
            claims=claims,
            technical_status="COMPLETED",
            rounds_used=rounds_used,
            max_rounds=max_rounds,
            declared_contract_version=declared_contract_version,
            migration_signal=migration_signal,
        )
    else:
        claims = []
        decision = {
            "action": "HALT_STAGE",
            "reason_code": f"AGENT07_{runtime_result.runtime_status}",
            "correctable_claim_uids": (),
            "blocking_claim_uids": (),
            "claim_identity_contract_version": None,
            "claim_identity_contract_version_newly_inferred": False,
            "claim_identity_contract_migrated": False,
            "rationale": "Fallo técnico o artefactos incompletos en el runtime de 07.",
        }
    return claims, decision


def _build_optional_writer_revision_request(
    *,
    decision: Mapping[str, Any],
    claims: list[dict[str, Any]],
    experiment_id: str,
    cycle_id: str,
    rounds_used: int,
    source_draft_path: str,
    source_draft_fingerprint: str,
    verification_fingerprint: str,
) -> dict[str, Any] | None:
    """Paso 2 (opcional): construye ``writer_revision_request`` SOLO
    cuando ``decision["action"] == "RETURN"``. ``None`` en cualquier otro
    caso -- ese ``None`` es la señal que decide, aguas abajo, si
    ``writer_revision_request.json`` participa o no del conjunto de
    payloads/manifest/refs de esta ejecución."""

    if decision["action"] != "RETURN":
        return None

    from src.tools.verification.writer_revision_cycle import build_writer_revision_request

    return build_writer_revision_request(
        experiment_id=experiment_id,
        cycle_id=cycle_id,
        round_number=rounds_used + 1,
        source_draft_path=source_draft_path,
        source_draft_fingerprint=source_draft_fingerprint,
        verification_fingerprint=verification_fingerprint,
        claims=claims,
        correctable_claim_uids=decision["correctable_claim_uids"],
        claim_identity_contract_version=decision["claim_identity_contract_version"],
        transition_reason=decision["reason_code"],
    )


def _build_agent07_result(
    runtime_result: Agent07RuntimeResult,
    decision: Mapping[str, Any],
    refs: Mapping[str, ArtifactReference],
    *,
    attempt_number: int,
) -> AgentResult:
    """Paso 5 (final): construye el ``AgentResult`` a partir de la
    clasificación YA hecha (paso 1) y las referencias de artefactos YA
    construidas (paso 4 -- incluyen ``writer_revision_request.json``
    cuando corresponde, porque ``refs`` se arma después de saber si el
    paso 2 produjo un request)."""

    completed = _agent07_has_classifiable_bundle(runtime_result)
    action = decision["action"]
    now = datetime.now(timezone.utc).isoformat()

    action_map = {
        "ADVANCE": TransitionAction.ADVANCE,
        "RETURN": TransitionAction.RETURN,
        "HALT_STAGE": TransitionAction.HALT_STAGE,
    }
    target_map = {
        "ADVANCE": "08_evaluacion_experimental",
        "RETURN": "06_agente_redactor",
        "HALT_STAGE": None,
    }
    quality_map = {
        "ADVANCE": QualityStatus.APPROVED if runtime_result.runtime_status == "COMPLETED" else QualityStatus.APPROVED_WITH_WARNINGS,
        "RETURN": QualityStatus.NEEDS_REVISION,
        "HALT_STAGE": QualityStatus.NEEDS_REVISION,
    }

    return AgentResult(
        execution_status=ExecutionStatus.COMPLETED if completed else ExecutionStatus.FAILED,
        quality_status=quality_map[action],
        decision=DecisionInfo(
            code=decision["reason_code"],
            rationale=decision.get("rationale", "Resultado terminal validado del runtime del Agente 07."),
        ),
        quality_metrics=dict(runtime_result.execution_metrics),
        warnings=(),
        failure_reason_codes=tuple(runtime_result.runtime_issue_codes),
        requested_transition=RequestedTransition(
            action=action_map[action],
            target_stage=target_map[action],
            reason_code=decision["reason_code"],
            requires_human_confirmation=(action != "ADVANCE"),
        ),
        output_artifacts=refs,
        tool_usage=ToolUsage(retrieval_rounds=0, llm_calls=0, validation_calls=1),
        attempt_number=attempt_number,
        started_at=now,
        completed_at=now,
        error=None if completed else {"code": "AGENT07_RUNTIME_BLOCKED"},
    )


def _runtime_agent_result(
    runtime_result: Agent07RuntimeResult,
    refs: Mapping[str, ArtifactReference],
    *,
    attempt_number: int,
    experiment_id: str = "",
    cycle_id: str = "",
    source_draft_path: str = "",
    source_draft_fingerprint: str = "",
    verification_fingerprint: str = "",
    rounds_used: int = 0,
    max_rounds: int = 3,
) -> tuple[AgentResult, dict[str, Any] | None]:
    """Wrapper de COMPATIBILIDAD RETROACTIVA para llamadoras que ya
    invocaban esta función directamente con ``refs`` ya construido (p.ej.
    pruebas existentes que ejercitan la clasificación real sin atravesar
    ``execute_prepared_agent07`` completo). Compone, sin duplicar lógica,
    los pasos 1+2+5 (``_classify_agent07_transition`` +
    ``_build_optional_writer_revision_request`` + ``_build_agent07_result``).

    ``execute_prepared_agent07`` NO usa este wrapper: necesita conocer
    ``revision_request`` (paso 2) ANTES de construir los payloads (paso 3)
    para poder incluir condicionalmente
    ``writer_revision_request.json`` en el manifest, así que llama a los
    tres pasos por separado en el orden real. Mismo comportamiento externo
    que antes de la refactorización -- las pruebas que llaman a esta
    función directamente no necesitan cambiar."""

    claims, decision = _classify_agent07_transition(
        runtime_result, rounds_used=rounds_used, max_rounds=max_rounds
    )
    revision_request = _build_optional_writer_revision_request(
        decision=decision,
        claims=claims,
        experiment_id=experiment_id,
        cycle_id=cycle_id,
        rounds_used=rounds_used,
        source_draft_path=source_draft_path,
        source_draft_fingerprint=source_draft_fingerprint,
        verification_fingerprint=verification_fingerprint,
    )
    result = _build_agent07_result(runtime_result, decision, refs, attempt_number=attempt_number)
    return result, revision_request


def _manifest_for(
    prepared: PreparedAgent07Execution,
    payloads: Mapping[str, bytes],
    *,
    include_conditional: bool = False,
) -> Agent07ArtifactManifest:
    names = SCIENTIFIC_ARTIFACT_NAMES + ((AGENT07_CONDITIONAL_ARTIFACT_NAME,) if include_conditional else ())
    artifacts=tuple(Agent07ManifestArtifact(name,sha256_bytes(payloads[name]),len(payloads[name])) for name in names)
    return Agent07ArtifactManifest(AGENT07_STAGE_NAME,prepared.decision_id,prepared.attempt_number,prepared.execution_fingerprint,dict(prepared.runtime_input.schema_versions),prepared.runtime_input.committed_agent06_output["source_draft_fingerprint"],artifacts,False,False)


def validate_agent07_artifact_manifest_contract(value: Agent07ArtifactManifest | Mapping[str,Any], *, artifact_bytes: Mapping[str,bytes] | None=None) -> dict[str,Any]:
    p=asdict(value) if isinstance(value,Agent07ArtifactManifest) else deepcopy(dict(value)) if isinstance(value,Mapping) else None
    expected={f.name for f in fields(Agent07ArtifactManifest)}
    if p is None or set(p)!=expected: raise ValueError("AGENT07_MANIFEST_SCHEMA_INVALID")
    if p["stage"]!=AGENT07_STAGE_NAME or not p["decision_id"] or type(p["attempt_number"]) is not int or p["attempt_number"]<1: raise ValueError("AGENT07_MANIFEST_IDENTITY_INVALID")
    _validate_sha(p["execution_fingerprint"],"AGENT07_MANIFEST_EXECUTION_FINGERPRINT_INVALID"); _validate_sha(p["source_draft_fingerprint"],"AGENT07_MANIFEST_SOURCE_FINGERPRINT_INVALID")
    if not isinstance(p["schema_versions"],Mapping) or not p["schema_versions"] or any(not str(k).strip() or not str(v).strip() for k,v in p["schema_versions"].items()): raise ValueError("AGENT07_MANIFEST_SCHEMA_VERSIONS_INVALID")
    if p["correction_applied"] is not False or p["evaluation_ready_emitted"] is not False: raise ValueError("AGENT07_MANIFEST_ISOLATION_INVALID")
    arts=p["artifacts"]
    if not isinstance(arts,(list,tuple)): raise ValueError("AGENT07_MANIFEST_ARTIFACT_SET_INVALID")
    names={a.get("artifact_name") for a in arts}
    # Conjunto condicional (punto 1 del pedido): los 3 científicos son
    # SIEMPRE obligatorios; writer_revision_request.json es la ÚNICA
    # extensión permitida más allá de ese núcleo (solo cuando la
    # transición real fue RETURN) -- cualquier otro nombre extra, o la
    # ausencia de alguno de los 3 obligatorios, invalida el manifest.
    if not set(SCIENTIFIC_ARTIFACT_NAMES).issubset(names): raise ValueError("AGENT07_MANIFEST_ARTIFACT_SET_INVALID")
    extra=names-set(SCIENTIFIC_ARTIFACT_NAMES)
    if extra and extra!={AGENT07_CONDITIONAL_ARTIFACT_NAME}: raise ValueError("AGENT07_MANIFEST_ARTIFACT_SET_INVALID")
    if len(arts)!=len(names): raise ValueError("AGENT07_MANIFEST_ARTIFACT_DUPLICATE")
    for a in arts:
        if set(a)!={"artifact_name","sha256","size_bytes"}: raise ValueError("AGENT07_MANIFEST_ARTIFACT_SCHEMA_INVALID")
        _validate_sha(a["sha256"],"AGENT07_MANIFEST_ARTIFACT_HASH_INVALID")
        if type(a["size_bytes"]) is not int or a["size_bytes"]<0: raise ValueError("AGENT07_MANIFEST_ARTIFACT_SIZE_INVALID")
        if artifact_bytes is not None:
            data=artifact_bytes.get(a["artifact_name"])
            if data is None or sha256_bytes(data)!=a["sha256"] or len(data)!=a["size_bytes"]: raise ValueError("AGENT07_MANIFEST_ARTIFACT_CONTENT_MISMATCH")
    return p


def execute_prepared_agent07(*, store: StateStore, prepared: PreparedAgent07Execution, dependencies: VerificationRuntimeDependencies) -> ExecutedAgent07Execution:
    validate_prepared_agent07_execution_contract(prepared)
    from src.tools.verification.validation import aggregation_diagnostic_sink
    _staging_dir = Path(prepared.runtime_input.experiment_paths.get("agent07_staging_dir", prepared.runtime_input.experiment_paths["root"] + "/.agent07_staging")) / prepared.decision_id
    _diagnostic_path = _staging_dir / "aggregation_invalid_elements_debug.json"
    with aggregation_diagnostic_sink(_diagnostic_path):
        runtime_result=run_agent07_in_memory(prepared.runtime_input,dependencies=dependencies); validate_agent07_runtime_result_contract(runtime_result)
    output_dir=Path(prepared.runtime_input.experiment_paths.get("agent07_output_dir",prepared.runtime_input.experiment_paths["root"]+"/07_verification"))

    state_for_cycle = store.load()
    cycle = state_for_cycle.cycles.get("writer_verifier")
    rounds_used = cycle.rounds_used if cycle is not None else 0
    max_rounds = cycle.max_rounds if cycle is not None else 3
    declared_contract_version = cycle.claim_identity_contract_version if cycle is not None else None
    # source_draft_path: no viene directo en committed_agent06_output (solo
    # guarda experiment_id/source_draft_fingerprint) -- se reconstruye con
    # la misma convención real de draft_writing_runtime.py
    # (outputs/"05_draft"/"state_of_art_draft.json"), confirmada por lectura
    # directa de ese módulo. Documentado como reconstrucción, no un dato
    # propagado verbatim.
    experiment_root = prepared.runtime_input.experiment_paths.get("root", "")
    reconstructed_draft_path = (
        str(Path(experiment_root) / "05_outputs" / "05_draft" / "state_of_art_draft.json")
        if experiment_root
        else ""
    )
    experiment_id = prepared.runtime_input.committed_agent06_output.get("experiment_id", "")
    source_draft_fingerprint = prepared.runtime_input.committed_agent06_output["source_draft_fingerprint"]

    if runtime_result.runtime_status=="BLOCKED" and runtime_result.provisional_bundle is None:
        # Bloqueo operativo temprano: no hay bundle que clasificar, no hay
        # paso 1/2 real -- HALT_STAGE fijo, sin writer_revision_request.
        payloads={"agent07_runtime_report.json":_json_bytes(runtime_result.to_dict()),OPERATIONAL_AUDIT_NAME:_json_bytes(runtime_result.blocked_runtime_audit_record)}
        decision={"action":"HALT_STAGE","reason_code":f"AGENT07_{runtime_result.runtime_status}","correctable_claim_uids":(),"blocking_claim_uids":(),"claim_identity_contract_version":None,"claim_identity_contract_version_newly_inferred":False,"claim_identity_contract_migrated":False,"rationale":"Fallo técnico o artefactos incompletos en el runtime de 07."}
        revision_request=None
    else:
        # Señal EXPLÍCITA de 06 (nunca inferida de la mera presencia de
        # claim_uid): 06 declara en su propio output comprometido que
        # este round produce claims deliberadamente bajo STABLE_UID_V1
        # -- única fuente que autoriza una migración LEGACY->STABLE_UID_V1.
        migration_signal = bool(
            prepared.runtime_input.committed_agent06_output.get("claim_identity_migration_signal", False)
        )
        # Paso 1: clasificación (ADVANCE/RETURN/HALT_STAGE real).
        claims, decision = _classify_agent07_transition(
            runtime_result, rounds_used=rounds_used, max_rounds=max_rounds,
            declared_contract_version=declared_contract_version, migration_signal=migration_signal,
        )
        # Frontera real (primera vez que este ciclo ve claims, sin
        # contrato declarado todavía): el contrato recién inferido se
        # persiste AHORA, antes de que el orquestador procese la
        # transición -- así, si la decisión es RETURN, apply_return_
        # with_cycle (decision_engine.py) encuentra un CycleState YA
        # CORRECTO (con claim_identity_contract_version ya resuelto) en
        # vez de crear uno nuevo con un valor por defecto sin validar
        # contra los claims reales de esta ronda.
        if decision.get("claim_identity_contract_version_newly_inferred") and cycle is None:
            from src.state.pipeline_state import CycleState

            fresh_cycle = CycleState(
                max_rounds=max_rounds,
                claim_identity_contract_version=decision["claim_identity_contract_version"],
            )
            updated_cycles = dict(state_for_cycle.cycles)
            updated_cycles["writer_verifier"] = fresh_cycle
            from dataclasses import replace as _dc_replace
            from datetime import datetime as _dt, timezone as _tz

            state_for_cycle = _dc_replace(
                state_for_cycle, cycles=updated_cycles,
                identity=_dc_replace(state_for_cycle.identity, updated_at=_dt.now(_tz.utc).isoformat()),
            )
            store.save(state_for_cycle)
        elif decision.get("claim_identity_contract_migrated") and cycle is not None:
            # Migración real LEGACY -> STABLE_UID_V1 (una sola vez, con
            # señal explícita ya validada por classify_verification_
            # transition). Se persiste el registro completo pedido:
            # from/to/round/decision_id/migration_mode -- nunca se
            # reescriben rondas anteriores, nunca se reconstruyen UIDs
            # históricos.
            from dataclasses import replace as _dc_replace
            from datetime import datetime as _dt, timezone as _tz

            migrated_cycle = _dc_replace(
                cycle,
                claim_identity_contract_version="STABLE_UID_V1",
                claim_identity_migration_from="LEGACY",
                claim_identity_migration_to="STABLE_UID_V1",
                claim_identity_migration_round=rounds_used + 1,
                claim_identity_migration_decision_id=prepared.decision_id,
                claim_identity_migration_mode="EXPLICIT_SIGNAL_FROM_06",
            )
            updated_cycles = dict(state_for_cycle.cycles)
            updated_cycles["writer_verifier"] = migrated_cycle
            state_for_cycle = _dc_replace(
                state_for_cycle, cycles=updated_cycles,
                identity=_dc_replace(state_for_cycle.identity, updated_at=_dt.now(_tz.utc).isoformat()),
            )
            store.save(state_for_cycle)
        # Paso 2: writer_revision_request opcional -- se conoce ANTES de
        # construir los payloads, precisamente para poder incluirlo
        # condicionalmente en el manifest (punto 1 del pedido: nunca por
        # una ruta lateral fuera del contrato de staging/commit).
        revision_request = _build_optional_writer_revision_request(
            decision=decision, claims=claims, experiment_id=experiment_id, cycle_id=prepared.decision_id,
            rounds_used=rounds_used, source_draft_path=reconstructed_draft_path,
            source_draft_fingerprint=source_draft_fingerprint, verification_fingerprint=prepared.execution_fingerprint,
        )
        # Paso 3: payloads -- writer_revision_request.json entra en
        # candidate_payloads (y por tanto en staging/manifest/commit) SOLO
        # si el paso 2 produjo un request (acción RETURN).
        payloads={SCIENTIFIC_ARTIFACT_NAMES[0]:_json_bytes(runtime_result.provisional_bundle),SCIENTIFIC_ARTIFACT_NAMES[1]:_json_bytes(runtime_result.multi_proposal_resolution_result),SCIENTIFIC_ARTIFACT_NAMES[2]:_json_bytes(runtime_result.to_dict())}
        if revision_request is not None:
            payloads[AGENT07_CONDITIONAL_ARTIFACT_NAME]=_json_bytes(revision_request)
        manifest=_manifest_for(prepared,payloads,include_conditional=revision_request is not None)
        validate_agent07_artifact_manifest_contract(manifest,artifact_bytes=payloads)
        payloads[MANIFEST_NAME]=_json_bytes(manifest.to_dict())

    staging_dir=Path(prepared.runtime_input.experiment_paths.get("agent07_staging_dir",prepared.runtime_input.experiment_paths["root"]+"/.agent07_staging"))/prepared.decision_id
    staging_dir.mkdir(parents=True,exist_ok=True)
    for name,data in payloads.items(): atomic_write_bytes(staging_dir/name,data)
    staging_manifest=staging_dir/"staging_index.json"; atomic_write_json(staging_manifest,{"decision_id":prepared.decision_id,"attempt_number":prepared.attempt_number,"execution_fingerprint":prepared.execution_fingerprint,"payload_hashes":{k:sha256_bytes(v) for k,v in payloads.items()},"fingerprints":dict(prepared.stage_fingerprints)})
    # Paso 4: referencias de artefactos -- writer_revision_request.json
    # apunta al MISMO output_dir/nombre que los demás (ya no hay una
    # escritura lateral con atomic_write_json fuera del contrato).
    refs={name:ArtifactReference(str(output_dir/name),sha256_bytes(data)) for name,data in payloads.items()}

    # Paso 5: AgentResult final, con las refs YA completas (incluyen el
    # condicional cuando corresponde).
    result = _build_agent07_result(runtime_result, decision, refs, attempt_number=prepared.attempt_number)

    if revision_request is not None:
        # Persistencia por ronda (punto 1 del pedido original de la ronda
        # anterior): copia inmutable del lado de 07 en
        # writer_verifier_cycle/round_NN/ -- SEPARADA del manifest de
        # candidate_payloads, que ahora TAMBIÉN protege
        # writer_revision_request.json con el mismo staging/commit atómico
        # que el resto. Esta segunda copia (round bundle) sigue existiendo
        # porque es la que consulta complete_round_revision/
        # _resolve_draft_execution_mode -- no se elimina, solo deja de ser
        # la única protección de integridad del archivo.
        from src.tools.verification.cycle_round_persistence import create_round_awaiting_revision

        create_round_awaiting_revision(
            project_dir=str(Path(experiment_root).parent) if experiment_root else "",
            experiment_id=experiment_id,
            cycle_id=revision_request["cycle_id"],
            round_number=revision_request["round_number"],
            writer_revision_request=revision_request,
            artifacts={
                "input_draft_reference.json": {
                    "source_draft_path": reconstructed_draft_path,
                    "source_draft_fingerprint": revision_request["source_draft_fingerprint"],
                },
                "agent07_result.json": {
                    "decision_code": result.decision.code,
                    "rationale": result.decision.rationale,
                    "execution_status": result.execution_status.value,
                    "quality_status": result.quality_status.value if result.quality_status else None,
                },
                "writer_revision_request.json": revision_request,
                "transition.json": {
                    "action": result.requested_transition.action.value,
                    "target_stage": result.requested_transition.target_stage,
                    "reason_code": result.requested_transition.reason_code,
                },
                "fingerprints.json": {
                    "verification_fingerprint": prepared.execution_fingerprint,
                    "source_draft_fingerprint": revision_request["source_draft_fingerprint"],
                },
            },
        )
    persisted=store.persist_agent_result(prepared.decision_id,result)
    executed=ExecutedAgent07Execution(prepared.decision_id,prepared.runtime_input,runtime_result,payloads,str(staging_manifest),result,str(persisted),prepared.stage_fingerprints,prepared.attempt_number,prepared.execution_fingerprint)
    return validate_executed_agent07_execution_contract(executed)


def _expected_candidate_payload_names(runtime_result: Agent07RuntimeResult, *, has_revision_request: bool = False) -> set[str]:
    """Derive the staging payload contract from the terminal result shape.

    A BLOCKED result can be either an early operational block (audit-only) or
    a terminal scientific block that already produced a bundle and resolution.
    The latter remains a complete *candidate* set in staging, although it is
    not eligible for official COMMIT.

    ``has_revision_request``: cuando la ejecución clasificó RETURN,
    ``writer_revision_request.json`` se suma al conjunto científico
    esperado -- nunca aparece en el conjunto operativo-solo
    (``agent07_runtime_report.json``/``OPERATIONAL_AUDIT_NAME``), porque
    ese caso nunca llega a clasificar (no hay bundle que clasificar).
    """
    has_bundle = runtime_result.provisional_bundle is not None
    has_resolution = runtime_result.multi_proposal_resolution_result is not None
    conditional = {AGENT07_CONDITIONAL_ARTIFACT_NAME} if has_revision_request else set()
    if runtime_result.runtime_status in {"COMPLETED", "PARTIAL"}:
        return set(AGENT07_ARTIFACT_NAMES) | conditional
    if runtime_result.runtime_status == "BLOCKED" and not has_bundle and not has_resolution:
        return {"agent07_runtime_report.json", OPERATIONAL_AUDIT_NAME}
    if runtime_result.runtime_status == "BLOCKED" and has_bundle and has_resolution:
        return set(AGENT07_ARTIFACT_NAMES) | conditional
    raise ValueError("AGENT07_EXECUTED_RUNTIME_PAYLOAD_SHAPE_INVALID")


def validate_executed_agent07_execution_contract(value: ExecutedAgent07Execution) -> ExecutedAgent07Execution:
    if not isinstance(value,ExecutedAgent07Execution): raise ValueError("AGENT07_EXECUTED_SCHEMA_INVALID")
    if value.agent_result.attempt_number!=value.attempt_number or value.decision_id=="": raise ValueError("AGENT07_EXECUTED_IDENTITY_INVALID")
    validate_prepared_agent07_execution_contract(PreparedAgent07Execution(value.decision_id,value.runtime_input,{"source_draft_fingerprint":value.runtime_input.committed_agent06_output["source_draft_fingerprint"]},value.stage_fingerprints,value.attempt_number,value.execution_fingerprint))
    validate_agent07_runtime_result_contract(value.runtime_result)
    if not Path(value.staging_manifest_path).is_file() or not Path(value.persisted_result_path).is_file(): raise ValueError("AGENT07_EXECUTED_STAGING_INVALID")
    has_revision_request = AGENT07_CONDITIONAL_ARTIFACT_NAME in value.candidate_payloads
    expected = _expected_candidate_payload_names(value.runtime_result, has_revision_request=has_revision_request)
    if set(value.candidate_payloads)!=expected: raise ValueError("AGENT07_EXECUTED_PAYLOAD_SET_INVALID")
    for name,data in value.candidate_payloads.items():
        if value.agent_result.output_artifacts[name].hash!=sha256_bytes(data): raise ValueError("AGENT07_EXECUTED_PAYLOAD_HASH_MISMATCH")
    if MANIFEST_NAME in value.candidate_payloads:
        validate_agent07_artifact_manifest_contract(json.loads(value.candidate_payloads[MANIFEST_NAME]),artifact_bytes={k:v for k,v in value.candidate_payloads.items() if k!=MANIFEST_NAME})
    return value


def _validate_execution_for_commit(executed: ExecutedAgent07Execution) -> None:
    runtime_result = executed.runtime_result
    if runtime_result.runtime_status == "BLOCKED":
        validate_executed_agent07_execution_contract(executed)
        if not _agent07_has_classifiable_bundle(runtime_result):
            # Bloqueo OPERATIVO: no hay bundle real que respalde ninguna
            # decisión científica -- sigue siendo un fallo técnico real,
            # nunca committable como resultado oficial.
            raise RuntimeError("AGENT07_OPERATIONAL_BLOCK_NOT_SCIENTIFIC_COMMITTABLE")
        # Bloqueo CIENTÍFICO con bundle+resolución reales: la clasificación
        # ya corrió sobre datos reales (ver _classify_agent07_transition)
        # y produjo una decisión ADVANCE/RETURN/HALT_STAGE genuina -- es
        # tan committable como un runtime_status COMPLETED/PARTIAL. No se
        # rechaza aquí; continúa al mismo chequeo de conjunto de payloads
        # que cualquier otro resultado.
    names = set(executed.candidate_payloads)
    extra = names - set(AGENT07_ARTIFACT_NAMES)
    if not set(AGENT07_ARTIFACT_NAMES).issubset(names) or (extra and extra != {AGENT07_CONDITIONAL_ARTIFACT_NAME}):
        raise RuntimeError("AGENT07_COMMIT_MANIFEST_INCOMPLETE")
    validate_executed_agent07_execution_contract(executed)


def _published_dir(executed: ExecutedAgent07Execution) -> Path:
    return Path(executed.runtime_input.experiment_paths.get("agent07_output_dir",executed.runtime_input.experiment_paths["root"]+"/07_verification"))


def commit_executed_agent07(*, store: StateStore, executed: ExecutedAgent07Execution, fail_after_writes: int | None=None):
    _validate_execution_for_commit(executed)
    output_dir=_published_dir(executed); parent=output_dir.parent; parent.mkdir(parents=True,exist_ok=True)
    release=parent/f".{output_dir.name}.{executed.decision_id}.publish"; shutil.rmtree(release,ignore_errors=True); release.mkdir(parents=True)
    # Nombres a publicar (sin el manifest, que se maneja aparte): los 3
    # científicos siempre, más writer_revision_request.json SOLO si esta
    # ejecución lo incluyó en candidate_payloads (transición RETURN).
    scientific_names = SCIENTIFIC_ARTIFACT_NAMES + (
        (AGENT07_CONDITIONAL_ARTIFACT_NAME,) if AGENT07_CONDITIONAL_ARTIFACT_NAME in executed.candidate_payloads else ()
    )
    written=0
    try:
        # Manifest is created in the release directory but validated and published as the final marker.
        for name in scientific_names:
            atomic_write_bytes(release/name,executed.candidate_payloads[name]); written+=1
            if fail_after_writes is not None and written>=fail_after_writes: raise RuntimeError("AGENT07_COMMIT_INJECTED_WRITE_FAILURE")
        manifest_payload=json.loads(executed.candidate_payloads[MANIFEST_NAME]); validate_agent07_artifact_manifest_contract(manifest_payload,artifact_bytes={n:(release/n).read_bytes() for n in scientific_names})
        atomic_write_bytes(release/MANIFEST_NAME,executed.candidate_payloads[MANIFEST_NAME]); written+=1
        if fail_after_writes is not None and written>=fail_after_writes: raise RuntimeError("AGENT07_COMMIT_INJECTED_WRITE_FAILURE")
        validate_agent07_artifact_manifest_contract(json.loads((release/MANIFEST_NAME).read_text()),artifact_bytes={n:(release/n).read_bytes() for n in scientific_names})
        backup=parent/f".{output_dir.name}.{executed.decision_id}.backup"; shutil.rmtree(backup,ignore_errors=True)
        if output_dir.exists(): os.replace(output_dir,backup)
        try: os.replace(release,output_dir)
        except Exception:
            if backup.exists(): os.replace(backup,output_dir)
            raise
        shutil.rmtree(backup,ignore_errors=True)
        # references now point to definitive paths and are rechecked before the single state transition.
        for name,ref in executed.agent_result.output_artifacts.items():
            if not Path(ref.path).is_file() or sha256_file(ref.path)!=ref.hash: raise RuntimeError(f"AGENT07_COMMIT_FINGERPRINT_MISMATCH:{name}")
        validate_agent07_artifact_manifest_contract(json.loads((output_dir/MANIFEST_NAME).read_text()),artifact_bytes={n:(output_dir/n).read_bytes() for n in scientific_names})
        return store.commit_execution(decision_id=executed.decision_id,result=executed.agent_result,stage_name=AGENT07_STAGE_NAME,fingerprints=executed.stage_fingerprints,observations={"correction_applied":False,"evaluation_ready_emitted":False})
    finally:
        shutil.rmtree(release,ignore_errors=True)


def _load_executed_from_staging(*, store: StateStore, runtime_input: Agent07RuntimeInput, decision_id: str) -> ExecutedAgent07Execution | None:
    result=store.find_persisted_agent_result(decision_id)
    staging_dir=Path(runtime_input.experiment_paths.get("agent07_staging_dir",runtime_input.experiment_paths["root"]+"/.agent07_staging"))/decision_id; index=staging_dir/"staging_index.json"
    if result is None or not index.is_file(): return None
    meta=json.loads(index.read_text()); payloads={p.name:p.read_bytes() for p in staging_dir.iterdir() if p.is_file() and p.name!="staging_index.json"}
    for name,h in meta.get("payload_hashes",{}).items():
        if name not in payloads or sha256_bytes(payloads[name])!=h: return None
    runtime_payload=json.loads(payloads["agent07_runtime_report.json"]); runtime_result=Agent07RuntimeResult(**runtime_payload); validate_agent07_runtime_result_contract(runtime_result)
    value=ExecutedAgent07Execution(decision_id,runtime_input,runtime_result,payloads,str(index),result,str(store._agent_result_path(decision_id)),meta["fingerprints"],meta["attempt_number"],meta["execution_fingerprint"])
    return validate_executed_agent07_execution_contract(value)


def validate_agent07_resume_result_contract(value: Agent07ResumeResult) -> Agent07ResumeResult:
    if not isinstance(value,Agent07ResumeResult) or value.action not in RESUME_ACTIONS: raise ValueError("AGENT07_RESUME_ACTION_INVALID")
    if value.action=="COMMITTED" and value.committed_result is None: raise ValueError("AGENT07_RESUME_COMMITTED_RESULT_MISSING")
    if value.action=="EXECUTED_NOT_COMMITTED" and value.executed is None: raise ValueError("AGENT07_RESUME_EXECUTED_RESULT_MISSING")
    return value


def _resume(action: str, committed=None, executed=None): return validate_agent07_resume_result_contract(Agent07ResumeResult(action,committed,executed))


def resume_agent07_execution(*, store: StateStore, runtime_input: Agent07RuntimeInput, dependencies: VerificationRuntimeDependencies | None=None) -> Agent07ResumeResult:
    state=store.load(); fp=_stage_fingerprints(runtime_input); output_dir=Path(runtime_input.experiment_paths.get("agent07_output_dir",runtime_input.experiment_paths["root"]+"/07_verification"))
    if state.pending_execution is not None:
        executed=_load_executed_from_staging(store=store,runtime_input=runtime_input,decision_id=state.pending_execution.decision_id)
        if executed is None: return _resume("REEXECUTE")
        # Partial definitive files without the final valid manifest are not a commit. Remove them and reuse staging.
        if output_dir.exists():
            try:
                manifest=output_dir/MANIFEST_NAME
                on_disk_names=set(p.name for p in output_dir.iterdir() if p.is_file())
                # Válido con o sin el condicional: se excluye antes de
                # comparar contra el núcleo fijo, así funciona para ambos
                # casos sin necesitar saber de antemano si hubo RETURN.
                valid_manifest=manifest.is_file() and (on_disk_names-{AGENT07_CONDITIONAL_ARTIFACT_NAME})==set(AGENT07_ARTIFACT_NAMES)
                if valid_manifest:
                    resume_scientific_names = SCIENTIFIC_ARTIFACT_NAMES + (
                        (AGENT07_CONDITIONAL_ARTIFACT_NAME,) if AGENT07_CONDITIONAL_ARTIFACT_NAME in on_disk_names else ()
                    )
                    validate_agent07_artifact_manifest_contract(json.loads(manifest.read_text()),artifact_bytes={n:(output_dir/n).read_bytes() for n in resume_scientific_names})
                    # Crash recovery: publication finished but the single state transition did not.
                    committed = store.commit_execution(decision_id=executed.decision_id, result=executed.agent_result, stage_name=AGENT07_STAGE_NAME, fingerprints=executed.stage_fingerprints, observations={"correction_applied":False,"evaluation_ready_emitted":False})
                    return _resume("COMMITTED", committed=executed.agent_result)
                else: shutil.rmtree(output_dir)
            except Exception: shutil.rmtree(output_dir,ignore_errors=True)
        if executed.runtime_result.runtime_status == "BLOCKED":
            # A blocked runtime result is terminal for this attempt but is not a
            # reusable scientific execution. Record the failed attempt without
            # publishing its staging-only artifacts, clear pending_execution via
            # the existing StateStore COMMIT transition, and request a fresh
            # PREPARE with a new decision_id and incremented attempt number.
            failed_result = replace(executed.agent_result, output_artifacts={})
            store.commit_execution(
                decision_id=executed.decision_id,
                result=failed_result,
                stage_name=AGENT07_STAGE_NAME,
                fingerprints=executed.stage_fingerprints,
                observations={
                    "resume_disposition": "BLOCKED_ATTEMPT_REEXECUTE",
                    "scientific_result_reused": False,
                    "correction_applied": False,
                    "evaluation_ready_emitted": False,
                },
            )
            return _resume("REEXECUTE")
        return _resume("EXECUTED_NOT_COMMITTED",executed=executed)
    stage=state.stages.get(AGENT07_STAGE_NAME)
    if stage is None or stage.execution_status != ExecutionStatus.COMPLETED: return _resume("NO_COMMIT")
    if stage.fingerprints != fp: return _resume("FINGERPRINT_MISMATCH")
    if not output_dir.is_dir(): return _resume("ARTIFACT_MISMATCH")
    has_conditional_on_disk = (output_dir/AGENT07_CONDITIONAL_ARTIFACT_NAME).is_file()
    committed_scientific_names = SCIENTIFIC_ARTIFACT_NAMES + (
        (AGENT07_CONDITIONAL_ARTIFACT_NAME,) if has_conditional_on_disk else ()
    )
    try:
        manifest=json.loads((output_dir/MANIFEST_NAME).read_text()); validate_agent07_artifact_manifest_contract(manifest,artifact_bytes={n:(output_dir/n).read_bytes() for n in committed_scientific_names})
    except FileNotFoundError: return _resume("MANIFEST_INCOMPLETE")
    except Exception: return _resume("ARTIFACT_MISMATCH")
    expected_committed_names = AGENT07_ARTIFACT_NAMES + (
        (AGENT07_CONDITIONAL_ARTIFACT_NAME,) if has_conditional_on_disk else ()
    )
    for name in expected_committed_names:
        artifact=state.artifacts.get(name)
        if artifact is None or not Path(artifact.reference.path).is_file() or sha256_file(artifact.reference.path)!=artifact.reference.hash: return _resume("ARTIFACT_MISMATCH")
    for entry in reversed(state.decision_log):
        if entry.stage==AGENT07_STAGE_NAME: return _resume("COMMITTED",AgentResult.from_dict(entry.result))
    return _resume("MANIFEST_INCOMPLETE")
