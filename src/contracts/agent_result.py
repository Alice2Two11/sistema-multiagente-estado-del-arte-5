"""Contrato de salida común para los agentes del pipeline.

Este módulo implementa únicamente el contrato ``AgentResult`` aprobado en la
Decisión 3. Es independiente de Colab y de la configuración concreta del
proyecto. No valida el grafo de transiciones: el agente solicita una transición
y el futuro Orquestador será responsable de aceptarla o rechazarla.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .agent_input import ArtifactReference


class ExecutionStatus(str, Enum):
    """Estado técnico de ejecución de un agente."""

    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INVALIDATED = "INVALIDATED"


class QualityStatus(str, Enum):
    """Clasificación científica o funcional del resultado producido."""

    APPROVED = "APPROVED"
    APPROVED_WITH_WARNINGS = "APPROVED_WITH_WARNINGS"
    APPROVED_PENDING_MANUAL_REVIEW = "APPROVED_PENDING_MANUAL_REVIEW"
    APPROVED_AFTER_MANUAL_REVIEW = "APPROVED_AFTER_MANUAL_REVIEW"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"
    NEEDS_REVISION = "NEEDS_REVISION"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    REJECTED = "REJECTED"


class TransitionAction(str, Enum):
    """Acción solicitada por el agente al futuro Orquestador."""

    ADVANCE = "ADVANCE"
    RETRY = "RETRY"
    RETURN = "RETURN"
    HALT_STAGE = "HALT_STAGE"
    STOP_PIPELINE = "STOP_PIPELINE"


class WarningSeverity(str, Enum):
    """Severidad estructurada de una advertencia del agente."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


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


@dataclass(frozen=True, slots=True)
class DecisionInfo:
    """Decisión tomada internamente por el agente y su justificación."""

    code: str
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _required_text(self.code, "DecisionInfo.code"))
        object.__setattr__(
            self,
            "rationale",
            _required_text(self.rationale, "DecisionInfo.rationale"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionInfo":
        if not isinstance(data, Mapping):
            raise TypeError("decision debe ser un mapping.")
        return cls(code=data.get("code"), rationale=data.get("rationale"))


@dataclass(frozen=True, slots=True)
class AgentWarning:
    """Advertencia estructurada generada durante la ejecución."""

    code: str
    severity: WarningSeverity
    blocking: bool
    message: str

    def __post_init__(self) -> None:
        code = _required_text(self.code, "AgentWarning.code")
        message = _required_text(self.message, "AgentWarning.message")
        severity = (
            self.severity
            if isinstance(self.severity, WarningSeverity)
            else WarningSeverity(self.severity)
        )
        if not isinstance(self.blocking, bool):
            raise TypeError("AgentWarning.blocking debe ser booleano.")

        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "severity", severity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "blocking": self.blocking,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentWarning":
        if not isinstance(data, Mapping):
            raise TypeError("Cada warning debe ser un mapping.")
        return cls(
            code=data.get("code"),
            severity=data.get("severity"),
            blocking=data.get("blocking"),
            message=data.get("message"),
        )


@dataclass(frozen=True, slots=True)
class RequestedTransition:
    """Petición de transición emitida por el agente, todavía no validada."""

    action: TransitionAction
    target_stage: str | None = None
    reason_code: str = ""
    requires_human_confirmation: bool = False

    def __post_init__(self) -> None:
        action = (
            self.action
            if isinstance(self.action, TransitionAction)
            else TransitionAction(self.action)
        )
        target_stage = _optional_text(
            self.target_stage, "RequestedTransition.target_stage"
        )
        reason_code = _required_text(
            self.reason_code, "RequestedTransition.reason_code"
        )
        if not isinstance(self.requires_human_confirmation, bool):
            raise TypeError(
                "RequestedTransition.requires_human_confirmation debe ser booleano."
            )

        object.__setattr__(self, "action", action)
        object.__setattr__(self, "target_stage", target_stage)
        object.__setattr__(self, "reason_code", reason_code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "target_stage": self.target_stage,
            "reason_code": self.reason_code,
            "requires_human_confirmation": self.requires_human_confirmation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequestedTransition":
        if not isinstance(data, Mapping):
            raise TypeError("requested_transition debe ser un mapping.")
        return cls(
            action=data.get("action"),
            target_stage=data.get("target_stage"),
            reason_code=data.get("reason_code"),
            requires_human_confirmation=data.get(
                "requires_human_confirmation", False
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolUsage:
    """Contadores mínimos de recursos usados por el agente."""

    retrieval_rounds: int = 0
    llm_calls: int = 0
    validation_calls: int = 0

    def __post_init__(self) -> None:
        for field_name in ("retrieval_rounds", "llm_calls", "validation_calls"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"ToolUsage.{field_name} debe ser un entero.")
            if value < 0:
                raise ValueError(
                    f"ToolUsage.{field_name} debe ser mayor o igual a 0."
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "retrieval_rounds": self.retrieval_rounds,
            "llm_calls": self.llm_calls,
            "validation_calls": self.validation_calls,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolUsage":
        if not isinstance(data, Mapping):
            raise TypeError("tool_usage debe ser un mapping.")
        return cls(
            retrieval_rounds=data.get("retrieval_rounds", 0),
            llm_calls=data.get("llm_calls", 0),
            validation_calls=data.get("validation_calls", 0),
        )


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Resultado estandarizado producido por un agente.

    Los atributos no pueden reasignarse. Los mappings se copian
    defensivamente al construir y serializar la instancia; esto proporciona
    una instantánea defensiva, no inmutabilidad profunda de sus valores.
    """

    execution_status: ExecutionStatus
    quality_status: QualityStatus
    decision: DecisionInfo
    quality_metrics: Mapping[str, Any]
    warnings: tuple[AgentWarning, ...]
    requested_transition: RequestedTransition
    output_artifacts: Mapping[str, ArtifactReference]
    tool_usage: ToolUsage
    attempt_number: int
    started_at: str
    failure_reason_codes: tuple[str, ...] = ()
    completed_at: str | None = None
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        execution_status = (
            self.execution_status
            if isinstance(self.execution_status, ExecutionStatus)
            else ExecutionStatus(self.execution_status)
        )
        quality_status = (
            self.quality_status
            if isinstance(self.quality_status, QualityStatus)
            else QualityStatus(self.quality_status)
        )
        decision = (
            self.decision
            if isinstance(self.decision, DecisionInfo)
            else DecisionInfo.from_dict(self.decision)
        )
        requested_transition = (
            self.requested_transition
            if isinstance(self.requested_transition, RequestedTransition)
            else RequestedTransition.from_dict(self.requested_transition)
        )
        tool_usage = (
            self.tool_usage
            if isinstance(self.tool_usage, ToolUsage)
            else ToolUsage.from_dict(self.tool_usage)
        )

        if not isinstance(self.attempt_number, int) or isinstance(
            self.attempt_number, bool
        ):
            raise TypeError("AgentResult.attempt_number debe ser un entero.")
        if self.attempt_number < 1:
            raise ValueError(
                "AgentResult.attempt_number debe ser mayor o igual a 1."
            )

        started_at = _required_text(self.started_at, "AgentResult.started_at")
        completed_at = _optional_text(
            self.completed_at, "AgentResult.completed_at"
        )

        terminal_statuses = {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.INVALIDATED,
        }
        if execution_status in terminal_statuses and completed_at is None:
            raise ValueError(
                "AgentResult.completed_at es obligatorio para estados terminales."
            )

        if execution_status is ExecutionStatus.FAILED:
            if self.error is None:
                raise ValueError(
                    "AgentResult.error es obligatorio cuando execution_status=FAILED."
                )
            if not isinstance(self.error, Mapping):
                raise TypeError("AgentResult.error debe ser un mapping o None.")
            error = deepcopy(dict(self.error))
            if not error:
                raise ValueError(
                    "AgentResult.error no puede estar vacío cuando execution_status=FAILED."
                )
        else:
            if self.error is not None:
                raise ValueError(
                    "AgentResult.error solo puede definirse cuando execution_status=FAILED."
                )
            error = None

        failure_reason_codes = tuple(
            _required_text(code, "AgentResult.failure_reason_codes[]")
            for code in self.failure_reason_codes
        )

        warnings: list[AgentWarning] = []
        for warning in self.warnings:
            warnings.append(
                warning
                if isinstance(warning, AgentWarning)
                else AgentWarning.from_dict(warning)
            )

        artifacts: dict[str, ArtifactReference] = {}
        for name, reference in self.output_artifacts.items():
            if not isinstance(name, str):
                raise TypeError("Los nombres de output_artifacts deben ser cadenas.")
            normalized_name = name.strip()
            if not normalized_name:
                raise ValueError(
                    "Los nombres de output_artifacts no pueden estar vacíos."
                )
            artifacts[normalized_name] = (
                reference
                if isinstance(reference, ArtifactReference)
                else ArtifactReference.from_dict(reference)
            )

        object.__setattr__(self, "execution_status", execution_status)
        object.__setattr__(self, "quality_status", quality_status)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(
            self, "quality_metrics", deepcopy(dict(self.quality_metrics))
        )
        object.__setattr__(self, "warnings", tuple(warnings))
        object.__setattr__(self, "failure_reason_codes", failure_reason_codes)
        object.__setattr__(self, "requested_transition", requested_transition)
        object.__setattr__(self, "output_artifacts", artifacts)
        object.__setattr__(self, "tool_usage", tool_usage)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "completed_at", completed_at)
        object.__setattr__(self, "error", error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_status": self.execution_status.value,
            "quality_status": self.quality_status.value,
            "decision": self.decision.to_dict(),
            "quality_metrics": deepcopy(dict(self.quality_metrics)),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "failure_reason_codes": list(self.failure_reason_codes),
            "requested_transition": self.requested_transition.to_dict(),
            "output_artifacts": {
                name: reference.to_dict()
                for name, reference in self.output_artifacts.items()
            },
            "tool_usage": self.tool_usage.to_dict(),
            "attempt_number": self.attempt_number,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": deepcopy(dict(self.error)) if self.error is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentResult":
        if not isinstance(data, Mapping):
            raise TypeError("AgentResult debe reconstruirse desde un mapping.")

        return cls(
            execution_status=data.get("execution_status"),
            quality_status=data.get("quality_status"),
            decision=DecisionInfo.from_dict(data.get("decision", {})),
            quality_metrics=dict(data.get("quality_metrics", {})),
            warnings=tuple(
                AgentWarning.from_dict(item) for item in data.get("warnings", ())
            ),
            failure_reason_codes=tuple(data.get("failure_reason_codes", ())),
            requested_transition=RequestedTransition.from_dict(
                data.get("requested_transition", {})
            ),
            output_artifacts={
                name: ArtifactReference.from_dict(reference)
                for name, reference in data.get("output_artifacts", {}).items()
            },
            tool_usage=ToolUsage.from_dict(data.get("tool_usage", {})),
            attempt_number=data.get("attempt_number"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=(
                None
                if data.get("error") is None
                else dict(data.get("error"))
            ),
        )
