"""Contrato de entrada común para los agentes del pipeline.

Este módulo es independiente de Colab y de la configuración concreta del
proyecto. Define únicamente estructuras de datos y validaciones del contrato
``AgentInput`` aprobado en la Decisión 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ExecutionMode(str, Enum):
    """Modo de invocación de un agente."""

    FULL_RUN = "FULL_RUN"
    RECLASSIFY_ONLY = "RECLASSIFY_ONLY"


# Máximo de intentos por defecto para la comparación "es el intento
# final" (ver AgentInput.is_final_attempt) -- mismo valor que ya
# usaban las comparaciones "attempt_number == 2" dispersas en
# extraction_agent.py/draft_writing_agent.py antes de centralizarlas
# aquí. Cada etapa puede declarar su propio máximo distinto en su
# policy y pasarlo explícitamente a is_final_attempt(); este default
# solo cubre las etapas que, como hoy, no tienen un campo de policy
# dedicado para su número de intentos.
DEFAULT_MAX_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Referencia inmutable a un artefacto mediante ruta y hash."""

    path: str
    hash: str

    def __post_init__(self) -> None:
        normalized_path = str(self.path).strip()
        normalized_hash = str(self.hash).strip()

        if not normalized_path:
            raise ValueError("ArtifactReference.path no puede estar vacío.")
        if not normalized_hash:
            raise ValueError("ArtifactReference.hash no puede estar vacío.")

        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "hash", normalized_hash)

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "hash": self.hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactReference":
        if "path" not in data:
            raise ValueError("La dependencia debe incluir 'path'.")
        if "hash" not in data:
            raise ValueError("La dependencia debe incluir 'hash'.")
        return cls(path=str(data["path"]), hash=str(data["hash"]))


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Capacidades y recursos disponibles para una ejecución del agente."""

    allowed_tools: tuple[str, ...] = ()
    output_directory: str = ""
    runtime_resources: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tools = tuple(str(tool).strip() for tool in self.allowed_tools)
        if any(not tool for tool in tools):
            raise ValueError("allowed_tools no puede contener nombres vacíos.")

        output_directory = str(self.output_directory).strip()
        if not output_directory:
            raise ValueError("AgentContext.output_directory no puede estar vacío.")

        object.__setattr__(self, "allowed_tools", tools)
        object.__setattr__(self, "output_directory", output_directory)
        object.__setattr__(self, "runtime_resources", dict(self.runtime_resources))

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "output_directory": self.output_directory,
            "runtime_resources": dict(self.runtime_resources),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentContext":
        return cls(
            allowed_tools=tuple(data.get("allowed_tools", ())),
            output_directory=str(data.get("output_directory", "")),
            runtime_resources=dict(data.get("runtime_resources", {})),
        )


@dataclass(frozen=True, slots=True)
class PreviousAttemptSummary:
    """Diagnóstico resumido del intento anterior, sin copiar su resultado completo."""

    quality_status: str
    quality_metrics: Mapping[str, Any] = field(default_factory=dict)
    blocking_warnings: tuple[str, ...] = ()
    failure_reason_codes: tuple[str, ...] = ()
    previous_artifacts: Mapping[str, ArtifactReference] = field(default_factory=dict)

    def __post_init__(self) -> None:
        quality_status = str(self.quality_status).strip()
        if not quality_status:
            raise ValueError("PreviousAttemptSummary.quality_status no puede estar vacío.")

        warnings = tuple(str(item).strip() for item in self.blocking_warnings)
        reason_codes = tuple(str(item).strip() for item in self.failure_reason_codes)
        if any(not item for item in warnings):
            raise ValueError("blocking_warnings no puede contener valores vacíos.")
        if any(not item for item in reason_codes):
            raise ValueError("failure_reason_codes no puede contener valores vacíos.")

        artifacts = {
            str(name): (
                reference
                if isinstance(reference, ArtifactReference)
                else ArtifactReference.from_dict(reference)
            )
            for name, reference in self.previous_artifacts.items()
        }

        object.__setattr__(self, "quality_status", quality_status)
        object.__setattr__(self, "quality_metrics", dict(self.quality_metrics))
        object.__setattr__(self, "blocking_warnings", warnings)
        object.__setattr__(self, "failure_reason_codes", reason_codes)
        object.__setattr__(self, "previous_artifacts", artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_status": self.quality_status,
            "quality_metrics": dict(self.quality_metrics),
            "blocking_warnings": list(self.blocking_warnings),
            "failure_reason_codes": list(self.failure_reason_codes),
            "previous_artifacts": {
                name: reference.to_dict()
                for name, reference in self.previous_artifacts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreviousAttemptSummary":
        return cls(
            quality_status=str(data.get("quality_status", "")),
            quality_metrics=dict(data.get("quality_metrics", {})),
            blocking_warnings=tuple(data.get("blocking_warnings", ())),
            failure_reason_codes=tuple(data.get("failure_reason_codes", ())),
            previous_artifacts={
                str(name): ArtifactReference.from_dict(reference)
                for name, reference in data.get("previous_artifacts", {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class OrchestratorConstraints:
    """Restricciones impuestas por el Orquestador a una ejecución concreta."""

    disallowed_actions: tuple[str, ...] = ()
    allowed_terminal_outcomes: tuple[str, ...] = ()
    cycle_budget_exhausted: bool = False
    manual_review_available: bool = False

    def __post_init__(self) -> None:
        disallowed = tuple(str(item).strip() for item in self.disallowed_actions)
        outcomes = tuple(str(item).strip() for item in self.allowed_terminal_outcomes)
        if any(not item for item in disallowed):
            raise ValueError("disallowed_actions no puede contener valores vacíos.")
        if any(not item for item in outcomes):
            raise ValueError("allowed_terminal_outcomes no puede contener valores vacíos.")

        object.__setattr__(self, "disallowed_actions", disallowed)
        object.__setattr__(self, "allowed_terminal_outcomes", outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disallowed_actions": list(self.disallowed_actions),
            "allowed_terminal_outcomes": list(self.allowed_terminal_outcomes),
            "cycle_budget_exhausted": self.cycle_budget_exhausted,
            "manual_review_available": self.manual_review_available,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OrchestratorConstraints":
        return cls(
            disallowed_actions=tuple(data.get("disallowed_actions", ())),
            allowed_terminal_outcomes=tuple(
                data.get("allowed_terminal_outcomes", ())
            ),
            cycle_budget_exhausted=bool(data.get("cycle_budget_exhausted", False)),
            manual_review_available=bool(data.get("manual_review_available", False)),
        )


@dataclass(frozen=True, slots=True)
class AgentInput:
    """Entrada estandarizada para un agente de generación científica."""

    experiment_id: str
    run_id: str
    stage_name: str
    attempt_number: int
    mode: ExecutionMode
    agent_context: AgentContext
    dependencies: Mapping[str, ArtifactReference]
    policy: Mapping[str, Any]
    previous_attempt: PreviousAttemptSummary | None = None
    orchestrator_constraints: OrchestratorConstraints = field(
        default_factory=OrchestratorConstraints
    )

    def __post_init__(self) -> None:
        experiment_id = str(self.experiment_id).strip()
        run_id = str(self.run_id).strip()
        stage_name = str(self.stage_name).strip()

        if not experiment_id:
            raise ValueError("AgentInput.experiment_id no puede estar vacío.")
        if not run_id:
            raise ValueError("AgentInput.run_id no puede estar vacío.")
        if not stage_name:
            raise ValueError("AgentInput.stage_name no puede estar vacío.")
        if not isinstance(self.attempt_number, int) or isinstance(self.attempt_number, bool):
            raise TypeError("AgentInput.attempt_number debe ser un entero.")
        if self.attempt_number < 1:
            raise ValueError("AgentInput.attempt_number debe ser mayor o igual a 1.")

        mode = self.mode if isinstance(self.mode, ExecutionMode) else ExecutionMode(self.mode)
        context = (
            self.agent_context
            if isinstance(self.agent_context, AgentContext)
            else AgentContext.from_dict(self.agent_context)
        )
        constraints = (
            self.orchestrator_constraints
            if isinstance(self.orchestrator_constraints, OrchestratorConstraints)
            else OrchestratorConstraints.from_dict(self.orchestrator_constraints)
        )
        previous_attempt = self.previous_attempt
        if previous_attempt is not None and not isinstance(
            previous_attempt, PreviousAttemptSummary
        ):
            previous_attempt = PreviousAttemptSummary.from_dict(previous_attempt)

        dependencies = {
            str(name): (
                reference
                if isinstance(reference, ArtifactReference)
                else ArtifactReference.from_dict(reference)
            )
            for name, reference in self.dependencies.items()
        }
        if any(not name.strip() for name in dependencies):
            raise ValueError("Los nombres de dependencias no pueden estar vacíos.")

        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "stage_name", stage_name)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "agent_context", context)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "policy", dict(self.policy))
        object.__setattr__(self, "previous_attempt", previous_attempt)
        object.__setattr__(self, "orchestrator_constraints", constraints)

    def is_first_attempt(self) -> bool:
        """True si esta es la primera vez que el agente ejecuta esta
        etapa (``attempt_number == 1``). Fuente única de verdad para
        esta comparación -- evita que cada agente (extraction_agent.py,
        draft_writing_agent.py, etc.) repita la comparación numérica
        de forma independiente. Semántica idéntica a la comparación
        que reemplaza; no introduce ningún cambio de comportamiento."""

        return self.attempt_number == 1

    def is_final_attempt(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> bool:
        """True si este es el último intento permitido para esta etapa
        (``attempt_number >= max_attempts``) -- a partir de aquí el
        agente ya no debe reintentar y debe tomar una decisión
        definitiva (HALT/REJECTED/APPROVED_PENDING_MANUAL_REVIEW en
        vez de RETRY). ``max_attempts`` es explícito porque cada etapa
        declara su propio máximo de intentos en su policy; el default
        (``DEFAULT_MAX_ATTEMPTS = 2``) es el mismo valor que ya usaban
        las comparaciones ``attempt_number == 2`` que este método
        reemplaza -- ni aumenta ni reduce el número de intentos
        actual. Fuente única de verdad para esta comparación."""

        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts debe ser un entero mayor o igual a 1.")
        return self.attempt_number >= max_attempts

    def to_dict(self) -> dict[str, Any]:
        """Serializa el contrato a un diccionario compuesto por tipos básicos."""

        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "stage_name": self.stage_name,
            "attempt_number": self.attempt_number,
            "mode": self.mode.value,
            "agent_context": self.agent_context.to_dict(),
            "dependencies": {
                name: reference.to_dict()
                for name, reference in self.dependencies.items()
            },
            "policy": dict(self.policy),
            "previous_attempt": (
                self.previous_attempt.to_dict()
                if self.previous_attempt is not None
                else None
            ),
            "orchestrator_constraints": self.orchestrator_constraints.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentInput":
        """Reconstruye y valida un ``AgentInput`` desde un diccionario."""

        required_fields = (
            "experiment_id",
            "run_id",
            "stage_name",
            "attempt_number",
            "mode",
            "agent_context",
            "dependencies",
            "policy",
        )
        missing = [field_name for field_name in required_fields if field_name not in data]
        if missing:
            raise ValueError(
                "Faltan campos obligatorios en AgentInput: " + ", ".join(missing)
            )

        previous_attempt_data = data.get("previous_attempt")
        return cls(
            experiment_id=str(data["experiment_id"]),
            run_id=str(data["run_id"]),
            stage_name=str(data["stage_name"]),
            attempt_number=data["attempt_number"],
            mode=ExecutionMode(data["mode"]),
            agent_context=AgentContext.from_dict(data["agent_context"]),
            dependencies={
                str(name): ArtifactReference.from_dict(reference)
                for name, reference in data["dependencies"].items()
            },
            policy=dict(data["policy"]),
            previous_attempt=(
                PreviousAttemptSummary.from_dict(previous_attempt_data)
                if previous_attempt_data is not None
                else None
            ),
            orchestrator_constraints=OrchestratorConstraints.from_dict(
                data.get("orchestrator_constraints", {})
            ),
        )
