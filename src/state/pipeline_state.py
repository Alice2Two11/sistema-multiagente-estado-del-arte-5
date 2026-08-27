"""Modelo de datos del estado global del pipeline.

Este módulo define únicamente estructuras de datos y validaciones para
``PipelineState`` según la Decisión 3. No realiza lectura o escritura de
archivos y no implementa PREPARE, COMMIT ni RESUME; esas operaciones pertenecen
a ``state_store.py``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts.agent_input import ArtifactReference
from ..contracts.agent_result import (
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
)


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} no puede ser None.")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} debe ser una cadena.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} no puede estar vacío.")
    return normalized


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} debe ser un entero.")
    if value < 0:
        raise ValueError(f"{field_name} debe ser mayor o igual a 0.")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    value = _non_negative_int(value, field_name)
    if value < 1:
        raise ValueError(f"{field_name} debe ser mayor o igual a 1.")
    return value


def _normalize_named_mapping(
    values: Mapping[str, Any], field_name: str
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} debe ser un mapping.")
    normalized: dict[str, Any] = {}
    for raw_name, value in values.items():
        name = _required_text(raw_name, f"Nombre en {field_name}")
        normalized[name] = value
    return normalized


@dataclass(frozen=True, slots=True)
class PipelineIdentity:
    """Identidad y versión de esquema de una ejecución del pipeline."""

    experiment_id: str
    run_id: str
    created_at: str
    updated_at: str
    schema_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "run_id",
            "created_at",
            "updated_at",
            "schema_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(self, field_name), f"PipelineIdentity.{field_name}"
                ),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineIdentity":
        if not isinstance(data, Mapping):
            raise TypeError("identity debe ser un mapping.")
        return cls(
            experiment_id=data.get("experiment_id"),
            run_id=data.get("run_id"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            schema_version=data.get("schema_version"),
        )


@dataclass(frozen=True, slots=True)
class StageFingerprints:
    """Fingerprints persistidos para decidir reutilización de una etapa."""

    input: str | None = None
    config: str | None = None
    dependencies: str | None = None
    composite: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("input", "config", "dependencies", "composite"):
            object.__setattr__(
                self,
                field_name,
                _optional_text(
                    getattr(self, field_name), f"StageFingerprints.{field_name}"
                ),
            )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "input": self.input,
            "config": self.config,
            "dependencies": self.dependencies,
            "composite": self.composite,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageFingerprints":
        if not isinstance(data, Mapping):
            raise TypeError("fingerprints debe ser un mapping.")
        return cls(
            input=data.get("input"),
            config=data.get("config"),
            dependencies=data.get("dependencies"),
            composite=data.get("composite"),
        )


@dataclass(frozen=True, slots=True)
class StageState:
    """Estado técnico y de calidad de una etapa del pipeline."""

    execution_status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    quality_status: QualityStatus | None = None
    attempts_used: int = 0
    retrieval_rounds_used: int = 0
    fingerprints: StageFingerprints = field(default_factory=StageFingerprints)
    warnings: tuple[Mapping[str, Any], ...] = ()
    failure_reason_codes: tuple[str, ...] = ()
    requested_transition: RequestedTransition | None = None
    last_error: Mapping[str, Any] | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        execution_status = (
            self.execution_status
            if isinstance(self.execution_status, ExecutionStatus)
            else ExecutionStatus(self.execution_status)
        )
        quality_status = self.quality_status
        if quality_status is not None and not isinstance(quality_status, QualityStatus):
            quality_status = QualityStatus(quality_status)

        attempts_used = _non_negative_int(
            self.attempts_used, "StageState.attempts_used"
        )
        retrieval_rounds_used = _non_negative_int(
            self.retrieval_rounds_used, "StageState.retrieval_rounds_used"
        )
        fingerprints = (
            self.fingerprints
            if isinstance(self.fingerprints, StageFingerprints)
            else StageFingerprints.from_dict(self.fingerprints)
        )
        warnings = tuple(deepcopy(dict(item)) for item in self.warnings)
        failure_reason_codes = tuple(
            _required_text(item, "StageState.failure_reason_codes[]")
            for item in self.failure_reason_codes
        )
        requested_transition = self.requested_transition
        if requested_transition is not None and not isinstance(
            requested_transition, RequestedTransition
        ):
            requested_transition = RequestedTransition.from_dict(
                requested_transition
            )
        last_error = (
            None if self.last_error is None else deepcopy(dict(self.last_error))
        )
        updated_at = _optional_text(self.updated_at, "StageState.updated_at")

        object.__setattr__(self, "execution_status", execution_status)
        object.__setattr__(self, "quality_status", quality_status)
        object.__setattr__(self, "attempts_used", attempts_used)
        object.__setattr__(
            self, "retrieval_rounds_used", retrieval_rounds_used
        )
        object.__setattr__(self, "fingerprints", fingerprints)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "failure_reason_codes", failure_reason_codes)
        object.__setattr__(self, "requested_transition", requested_transition)
        object.__setattr__(self, "last_error", last_error)
        object.__setattr__(self, "updated_at", updated_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_status": self.execution_status.value,
            "quality_status": (
                None if self.quality_status is None else self.quality_status.value
            ),
            "attempts_used": self.attempts_used,
            "retrieval_rounds_used": self.retrieval_rounds_used,
            "fingerprints": self.fingerprints.to_dict(),
            "warnings": [deepcopy(dict(item)) for item in self.warnings],
            "failure_reason_codes": list(self.failure_reason_codes),
            "requested_transition": (
                None
                if self.requested_transition is None
                else self.requested_transition.to_dict()
            ),
            "last_error": (
                None if self.last_error is None else deepcopy(dict(self.last_error))
            ),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageState":
        if not isinstance(data, Mapping):
            raise TypeError("Cada estado de etapa debe ser un mapping.")
        return cls(
            execution_status=data.get(
                "execution_status", ExecutionStatus.NOT_STARTED.value
            ),
            quality_status=data.get("quality_status"),
            attempts_used=data.get("attempts_used", 0),
            retrieval_rounds_used=data.get("retrieval_rounds_used", 0),
            fingerprints=StageFingerprints.from_dict(
                data.get("fingerprints", {})
            ),
            warnings=tuple(data.get("warnings", ())),
            failure_reason_codes=tuple(data.get("failure_reason_codes", ())),
            requested_transition=(
                None
                if data.get("requested_transition") is None
                else RequestedTransition.from_dict(data["requested_transition"])
            ),
            last_error=data.get("last_error"),
            updated_at=data.get("updated_at"),
        )


@dataclass(frozen=True, slots=True)
class CycleState:
    """Estado del ciclo global writer_verifier."""

    rounds_used: int = 0
    max_rounds: int = 3
    status: str = "NOT_STARTED"
    last_return_reason: str | None = None
    unresolved_claims_count: int = 0
    # Frontera legacy explícita para la identidad estable de claims (ver
    # src/tools/draft_writing/claim_identity.py) -- NUNCA se infiere
    # retroactivamente por similitud de texto. "STABLE_UID_V1" es el
    # valor real para cualquier ciclo creado desde este cambio en
    # adelante (el default del constructor). "LEGACY" es lo que
    # from_dict() asigna explícitamente cuando el campo está AUSENTE en
    # un pipeline_state.json persistido antes de este cambio -- la
    # única señal usada es la presencia/ausencia real del campo en el
    # JSON, nunca una heurística sobre el contenido.
    claim_identity_contract_version: str = "STABLE_UID_V1"
    # Registro de migración LEGACY -> STABLE_UID_V1, único y auditable --
    # None en todos hasta que ocurre una migración real. Nunca se
    # reescriben una vez poblados (una sola migración por ciclo).
    claim_identity_migration_from: str | None = None
    claim_identity_migration_to: str | None = None
    claim_identity_migration_round: int | None = None
    claim_identity_migration_decision_id: str | None = None
    claim_identity_migration_mode: str | None = None

    _ALLOWED_STATUSES = frozenset(
        {
            "NOT_STARTED",
            "ACTIVE",
            "RESOLVED",
            "EXHAUSTED",
            "PENDING_MANUAL_REVIEW",
        }
    )

    def __post_init__(self) -> None:
        rounds_used = _non_negative_int(
            self.rounds_used, "CycleState.rounds_used"
        )
        max_rounds = _positive_int(self.max_rounds, "CycleState.max_rounds")
        if rounds_used > max_rounds:
            raise ValueError(
                "CycleState.rounds_used no puede superar max_rounds."
            )
        status = _required_text(self.status, "CycleState.status")
        if status not in self._ALLOWED_STATUSES:
            raise ValueError(f"CycleState.status desconocido: {status}.")
        unresolved = _non_negative_int(
            self.unresolved_claims_count,
            "CycleState.unresolved_claims_count",
        )
        last_return_reason = _optional_text(
            self.last_return_reason, "CycleState.last_return_reason"
        )
        contract_version = _required_text(
            self.claim_identity_contract_version, "CycleState.claim_identity_contract_version"
        )
        if contract_version not in {"LEGACY", "STABLE_UID_V1"}:
            raise ValueError(
                f"CycleState.claim_identity_contract_version desconocido: {contract_version}."
            )

        object.__setattr__(self, "rounds_used", rounds_used)
        object.__setattr__(self, "max_rounds", max_rounds)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "unresolved_claims_count", unresolved)
        object.__setattr__(self, "last_return_reason", last_return_reason)
        object.__setattr__(self, "claim_identity_contract_version", contract_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds_used": self.rounds_used,
            "max_rounds": self.max_rounds,
            "status": self.status,
            "last_return_reason": self.last_return_reason,
            "unresolved_claims_count": self.unresolved_claims_count,
            "claim_identity_contract_version": self.claim_identity_contract_version,
            "claim_identity_migration_from": self.claim_identity_migration_from,
            "claim_identity_migration_to": self.claim_identity_migration_to,
            "claim_identity_migration_round": self.claim_identity_migration_round,
            "claim_identity_migration_decision_id": self.claim_identity_migration_decision_id,
            "claim_identity_migration_mode": self.claim_identity_migration_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CycleState":
        if not isinstance(data, Mapping):
            raise TypeError("Cada ciclo debe ser un mapping.")
        return cls(
            rounds_used=data.get("rounds_used", 0),
            max_rounds=data.get("max_rounds", 3),
            status=data.get("status", "NOT_STARTED"),
            last_return_reason=data.get("last_return_reason"),
            unresolved_claims_count=data.get("unresolved_claims_count", 0),
            # Ausente en el JSON persistido -> LEGACY, explícito y
            # determinista -- nunca "STABLE_UID_V1" por default aquí,
            # aunque ese SÍ sea el default del constructor para un
            # CycleState nuevo (ver arriba).
            claim_identity_contract_version=data.get("claim_identity_contract_version", "LEGACY"),
            claim_identity_migration_from=data.get("claim_identity_migration_from"),
            claim_identity_migration_to=data.get("claim_identity_migration_to"),
            claim_identity_migration_round=data.get("claim_identity_migration_round"),
            claim_identity_migration_decision_id=data.get("claim_identity_migration_decision_id"),
            claim_identity_migration_mode=data.get("claim_identity_migration_mode"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactState:
    """Artefacto definitivo registrado en el estado global."""

    reference: ArtifactReference
    created_at: str

    def __post_init__(self) -> None:
        reference = (
            self.reference
            if isinstance(self.reference, ArtifactReference)
            else ArtifactReference.from_dict(self.reference)
        )
        object.__setattr__(self, "reference", reference)
        object.__setattr__(
            self,
            "created_at",
            _required_text(self.created_at, "ArtifactState.created_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.reference.to_dict(), "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactState":
        if not isinstance(data, Mapping):
            raise TypeError("Cada artefacto debe ser un mapping.")
        return cls(
            reference=ArtifactReference.from_dict(data),
            created_at=data.get("created_at"),
        )


@dataclass(frozen=True, slots=True)
class InvalidationState:
    """Invalidación global del MVP, limitada deliberadamente a FULL."""

    scope_type: str = "FULL"
    invalidated_stages: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scope_type = _required_text(
            self.scope_type, "InvalidationState.scope_type"
        )
        if scope_type != "FULL":
            raise ValueError("La invalidación del MVP solo admite scope_type='FULL'.")
        stages = tuple(
            _required_text(item, "InvalidationState.invalidated_stages[]")
            for item in self.invalidated_stages
        )
        reasons = tuple(
            _required_text(item, "InvalidationState.reasons[]")
            for item in self.reasons
        )
        object.__setattr__(self, "scope_type", scope_type)
        object.__setattr__(self, "invalidated_stages", stages)
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": {"type": self.scope_type},
            "invalidated_stages": list(self.invalidated_stages),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InvalidationState":
        if not isinstance(data, Mapping):
            raise TypeError("invalidation debe ser un mapping.")
        scope = data.get("scope", {"type": "FULL"})
        if not isinstance(scope, Mapping):
            raise TypeError("invalidation.scope debe ser un mapping.")
        return cls(
            scope_type=scope.get("type", "FULL"),
            invalidated_stages=tuple(data.get("invalidated_stages", ())),
            reasons=tuple(data.get("reasons", ())),
        )


@dataclass(frozen=True, slots=True)
class PendingExecution:
    """PREPARE persistido pendiente de EXECUTE/COMMIT."""

    decision_id: str
    target_stage: str
    intended_action: str
    attempt_number: int
    prepared_at: str

    def __post_init__(self) -> None:
        for field_name in (
            "decision_id",
            "target_stage",
            "intended_action",
            "prepared_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(self, field_name), f"PendingExecution.{field_name}"
                ),
            )
        object.__setattr__(
            self,
            "attempt_number",
            _positive_int(
                self.attempt_number, "PendingExecution.attempt_number"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "target_stage": self.target_stage,
            "intended_action": self.intended_action,
            "attempt_number": self.attempt_number,
            "prepared_at": self.prepared_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PendingExecution":
        if not isinstance(data, Mapping):
            raise TypeError("pending_execution debe ser un mapping.")
        return cls(
            decision_id=data.get("decision_id"),
            target_stage=data.get("target_stage"),
            intended_action=data.get("intended_action"),
            attempt_number=data.get("attempt_number"),
            prepared_at=data.get("prepared_at"),
        )


@dataclass(frozen=True, slots=True)
class HumanReviewState:
    """Estado de revisión manual en una frontera del pipeline."""

    status: str
    reviewer: str | None = None
    reviewed_at: str | None = None
    notes: str | None = None

    _ALLOWED_STATUSES = frozenset({"PENDING", "CONFIRMED"})

    def __post_init__(self) -> None:
        status = _required_text(self.status, "HumanReviewState.status")
        if status not in self._ALLOWED_STATUSES:
            raise ValueError(f"HumanReviewState.status desconocido: {status}.")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "reviewer",
            _optional_text(self.reviewer, "HumanReviewState.reviewer"),
        )
        object.__setattr__(
            self,
            "reviewed_at",
            _optional_text(self.reviewed_at, "HumanReviewState.reviewed_at"),
        )
        object.__setattr__(
            self,
            "notes",
            _optional_text(self.notes, "HumanReviewState.notes"),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HumanReviewState":
        if not isinstance(data, Mapping):
            raise TypeError("human_review debe ser un mapping.")
        return cls(
            status=data.get("status"),
            reviewer=data.get("reviewer"),
            reviewed_at=data.get("reviewed_at"),
            notes=data.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class DecisionLogEntry:
    """Entrada serializable del historial de decisiones.

    La política append-only no se aplica aquí; será responsabilidad de
    ``StateStore``.
    """

    decision_id: str
    timestamp: str
    agent: str
    stage: str
    attempt: int
    observations: Mapping[str, Any]
    decision: Mapping[str, Any]
    reason_codes: tuple[str, ...]
    requested_transition: RequestedTransition | None
    result: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in ("decision_id", "timestamp", "agent", "stage"):
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(self, field_name), f"DecisionLogEntry.{field_name}"
                ),
            )
        object.__setattr__(
            self,
            "attempt",
            _positive_int(self.attempt, "DecisionLogEntry.attempt"),
        )
        reason_codes = tuple(
            _required_text(item, "DecisionLogEntry.reason_codes[]")
            for item in self.reason_codes
        )
        requested_transition = self.requested_transition
        if requested_transition is not None and not isinstance(
            requested_transition, RequestedTransition
        ):
            requested_transition = RequestedTransition.from_dict(
                requested_transition
            )
        object.__setattr__(
            self, "observations", deepcopy(dict(self.observations))
        )
        object.__setattr__(self, "decision", deepcopy(dict(self.decision)))
        object.__setattr__(self, "reason_codes", reason_codes)
        object.__setattr__(self, "requested_transition", requested_transition)
        object.__setattr__(self, "result", deepcopy(dict(self.result)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "agent": self.agent,
            "stage": self.stage,
            "attempt": self.attempt,
            "observations": deepcopy(dict(self.observations)),
            "decision": deepcopy(dict(self.decision)),
            "reason_codes": list(self.reason_codes),
            "requested_transition": (
                None
                if self.requested_transition is None
                else self.requested_transition.to_dict()
            ),
            "result": deepcopy(dict(self.result)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionLogEntry":
        if not isinstance(data, Mapping):
            raise TypeError("Cada entrada de decision_log debe ser un mapping.")
        return cls(
            decision_id=data.get("decision_id"),
            timestamp=data.get("timestamp"),
            agent=data.get("agent"),
            stage=data.get("stage"),
            attempt=data.get("attempt"),
            observations=data.get("observations", {}),
            decision=data.get("decision", {}),
            reason_codes=tuple(data.get("reason_codes", ())),
            requested_transition=(
                None
                if data.get("requested_transition") is None
                else RequestedTransition.from_dict(data["requested_transition"])
            ),
            result=data.get("result", {}),
        )


@dataclass(frozen=True, slots=True)
class PipelineState:
    """Instantánea completa y serializable del estado global del pipeline."""

    identity: PipelineIdentity
    generation_config_snapshot: Mapping[str, Any] = field(default_factory=dict)
    stages: Mapping[str, StageState] = field(default_factory=dict)
    cycles: Mapping[str, CycleState] = field(default_factory=dict)
    artifacts: Mapping[str, ArtifactState] = field(default_factory=dict)
    decision_log: tuple[DecisionLogEntry, ...] = ()
    invalidation: InvalidationState = field(default_factory=InvalidationState)
    pending_execution: PendingExecution | None = None
    human_review: HumanReviewState | None = None

    def __post_init__(self) -> None:
        identity = (
            self.identity
            if isinstance(self.identity, PipelineIdentity)
            else PipelineIdentity.from_dict(self.identity)
        )

        raw_stages = _normalize_named_mapping(self.stages, "PipelineState.stages")
        stages = {
            name: (
                value
                if isinstance(value, StageState)
                else StageState.from_dict(value)
            )
            for name, value in raw_stages.items()
        }

        raw_cycles = _normalize_named_mapping(self.cycles, "PipelineState.cycles")
        cycles = {
            name: (
                value
                if isinstance(value, CycleState)
                else CycleState.from_dict(value)
            )
            for name, value in raw_cycles.items()
        }

        raw_artifacts = _normalize_named_mapping(
            self.artifacts, "PipelineState.artifacts"
        )
        artifacts = {
            name: (
                value
                if isinstance(value, ArtifactState)
                else ArtifactState.from_dict(value)
            )
            for name, value in raw_artifacts.items()
        }

        decision_log = tuple(
            entry
            if isinstance(entry, DecisionLogEntry)
            else DecisionLogEntry.from_dict(entry)
            for entry in self.decision_log
        )
        invalidation = (
            self.invalidation
            if isinstance(self.invalidation, InvalidationState)
            else InvalidationState.from_dict(self.invalidation)
        )
        pending_execution = self.pending_execution
        if pending_execution is not None and not isinstance(
            pending_execution, PendingExecution
        ):
            pending_execution = PendingExecution.from_dict(pending_execution)
        human_review = self.human_review
        if human_review is not None and not isinstance(
            human_review, HumanReviewState
        ):
            human_review = HumanReviewState.from_dict(human_review)

        object.__setattr__(self, "identity", identity)
        object.__setattr__(
            self,
            "generation_config_snapshot",
            deepcopy(dict(self.generation_config_snapshot)),
        )
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "decision_log", decision_log)
        object.__setattr__(self, "invalidation", invalidation)
        object.__setattr__(self, "pending_execution", pending_execution)
        object.__setattr__(self, "human_review", human_review)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "generation_config_snapshot": deepcopy(
                dict(self.generation_config_snapshot)
            ),
            "stages": {
                name: stage.to_dict() for name, stage in self.stages.items()
            },
            "cycles": {
                name: cycle.to_dict() for name, cycle in self.cycles.items()
            },
            "artifacts": {
                name: artifact.to_dict()
                for name, artifact in self.artifacts.items()
            },
            "decision_log": [entry.to_dict() for entry in self.decision_log],
            "invalidation": self.invalidation.to_dict(),
            "pending_execution": (
                None
                if self.pending_execution is None
                else self.pending_execution.to_dict()
            ),
            "human_review": (
                None if self.human_review is None else self.human_review.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineState":
        if not isinstance(data, Mapping):
            raise TypeError("pipeline_state debe ser un mapping.")
        if "identity" not in data:
            raise ValueError("pipeline_state debe incluir identity.")
        return cls(
            identity=PipelineIdentity.from_dict(data["identity"]),
            generation_config_snapshot=data.get(
                "generation_config_snapshot", {}
            ),
            stages={
                name: StageState.from_dict(value)
                for name, value in data.get("stages", {}).items()
            },
            cycles={
                name: CycleState.from_dict(value)
                for name, value in data.get("cycles", {}).items()
            },
            artifacts={
                name: ArtifactState.from_dict(value)
                for name, value in data.get("artifacts", {}).items()
            },
            decision_log=tuple(
                DecisionLogEntry.from_dict(entry)
                for entry in data.get("decision_log", ())
            ),
            invalidation=InvalidationState.from_dict(
                data.get("invalidation", {})
            ),
            pending_execution=(
                None
                if data.get("pending_execution") is None
                else PendingExecution.from_dict(data["pending_execution"])
            ),
            human_review=(
                None
                if data.get("human_review") is None
                else HumanReviewState.from_dict(data["human_review"])
            ),
        )
