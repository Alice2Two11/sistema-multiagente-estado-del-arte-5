# ============================================================
# ALMACENAMIENTO Y PERSISTENCIA DEL ESTADO DEL PIPELINE
# Gestiona el estado durable de las etapas, ejecuciones pendientes,
# resultados comprometidos, fingerprints, ciclos y reanudación.
# ============================================================


from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import AgentResult, ExecutionStatus
from src.io.atomic_write import atomic_write_json
from src.state.pipeline_state import (
    ArtifactState,
    DecisionLogEntry,
    PendingExecution,
    PipelineIdentity,
    PipelineState,
    StageState,
)

# Representa el resultado de preparar una nueva ejecución,
# guardando el identificador de decisión y el estado actualizado del pipeline.
@dataclass(frozen=True, slots=True)
class PrepareResult:
    decision_id: str
    state: PipelineState

# Representa cómo debe resolverse una ejecución pendiente al reanudar el pipeline,
# incluyendo la acción tomada, el nuevo estado y un resultado ya comprometido si existe.
@dataclass(frozen=True, slots=True)
class ResumeResolution:
    action: str
    state: PipelineState
    committed_result: AgentResult | None = None

    # Valida que la acción de reanudación sea una de las opciones permitidas
    # y evita crear resultados de reanudación con estados desconocidos.
    def __post_init__(self) -> None:
        if self.action not in {"NO_PENDING", "COMMITTED", "REEXECUTE"}:
            raise ValueError(f"unsupported resume action: {self.action}")


class StateStore:
    """Persist and transactionally update a single pipeline-state document."""

    # Inicializa el almacenamiento persistente del pipeline,
    # definiendo la ruta del estado y el directorio donde se guardarán los resultados de agentes.
    def __init__(
        self,
        state_path: str | Path,
        *,
        agent_results_directory: str | Path | None = None,
    ) -> None:
        self.state_path = self._normalize_path(state_path, "state_path")
        # Si no se proporciona una carpeta específica para los resultados,
        # crea una automáticamente junto al archivo principal de estado.
        if agent_results_directory is None:
            agent_results_directory = (
                self.state_path.parent / f"{self.state_path.stem}_agent_results"
            )
        self.agent_results_directory = self._normalize_path(
            agent_results_directory,
            "agent_results_directory",
        )

    @staticmethod
    # Normaliza y valida las rutas utilizadas por StateStore,
    # asegurando que sean cadenas o Path válidos y que no estén vacíos.
    def _normalize_path(value: str | Path, field_name: str) -> Path:
        if isinstance(value, bool) or not isinstance(value, (str, Path)): # Rechaza valores booleanos o tipos que no sean rutas válidas.
            raise TypeError(f"{field_name} must be a string or pathlib.Path")
        path = Path(value)  # Convierte el valor recibido en un objeto Path.
        if not str(path).strip():
            raise ValueError(f"{field_name} must not be empty")
        return path

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # Valida que un valor sea una cadena de texto no vacía
    # y devuelve su contenido eliminando espacios innecesarios.
    @staticmethod
    def _validate_non_empty_string(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip(): # Rechaza valores que no sean texto o que estén vacíos.
            raise ValueError(f"{field_name} must be a non-empty string")
        return value.strip()

    # Inicializa por primera vez el estado persistente del pipeline.
    # Evita sobrescribir un estado existente, salvo que se autorice explícitamente.
    def initialize(self, state: PipelineState, *, overwrite: bool = False) -> PipelineState:
        if not isinstance(state, PipelineState):
            raise TypeError("state must be a PipelineState")
        if self.state_path.exists() and not overwrite:
            raise FileExistsError(f"pipeline state already exists: {self.state_path}")
        self.save(state)
        return state

    # Carga desde disco el estado persistente del pipeline
    # y lo reconstruye como un objeto PipelineState.
    def load(self) -> PipelineState:
        if not self.state_path.is_file():
            raise FileNotFoundError(f"pipeline state not found: {self.state_path}")
        with self.state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return PipelineState.from_dict(payload)

    # Guarda de forma persistente el estado actual del pipeline,
    # asegurando que sea un PipelineState válido antes de escribirlo en disco.
    def save(self, state: PipelineState) -> None:
        if not isinstance(state, PipelineState):
            raise TypeError("state must be a PipelineState")
        atomic_write_json(self.state_path, state.to_dict())

    # Prepara una nueva ejecución del pipeline, cargando el estado actual
    # y comprobando que no exista otra ejecución pendiente antes de continuar.
    def prepare_execution(
        self,
        *,
        target_stage: str,
        intended_action: str,
        attempt_number: int,
        decision_id: str | None = None,
        prepared_at: str | None = None,
    ) -> PrepareResult:
        state = self.load()     # Carga el estado persistido actual del pipeline.
        if state.pending_execution is not None:     # Impide preparar una nueva ejecución si ya existe otra pendiente.
            raise RuntimeError("a pending execution already exists")

        # Valida los datos básicos de la ejecución antes de crearla:
        # comprueba la etapa, la acción solicitada y que el número de intento sea válido.
        target_stage = self._validate_non_empty_string(target_stage, "target_stage")
        intended_action = self._validate_non_empty_string(
            intended_action,
            "intended_action",
        )
        if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
            raise TypeError("attempt_number must be an integer")
        if attempt_number < 1:
            raise ValueError("attempt_number must be greater than or equal to 1")

        # Define los identificadores y la fecha de preparación de la ejecución:
        # valida los valores recibidos o los genera automáticamente si no fueron proporcionados.
        decision_id = (
            self._validate_non_empty_string(decision_id, "decision_id")
            if decision_id is not None
            else str(uuid4())
        )
        prepared_at = (
            self._validate_non_empty_string(prepared_at, "prepared_at")
            if prepared_at is not None
            else self._now_iso()
        )

        # Construye la ejecución pendiente con todos sus datos,
        # la incorpora al estado del pipeline y la guarda
        pending = PendingExecution(
            decision_id=decision_id,
            target_stage=target_stage,
            intended_action=intended_action,
            attempt_number=attempt_number,
            prepared_at=prepared_at,
        )
        prepared_state = replace(state, pending_execution=pending)
        self.save(prepared_state)
        return PrepareResult(decision_id=decision_id, state=prepared_state)

    # Construye la ruta donde se guardará el resultado de un agente,
    # usando el decision_id como nombre único del archivo JSON.
    def _agent_result_path(self, decision_id: str) -> Path:
        decision_id = self._validate_non_empty_string(decision_id, "decision_id")
        return self.agent_results_directory / f"{decision_id}.json"

    # Guarda de forma persistente el resultado producido por una ejecución,
    # asociándolo con su decision_id para poder confirmarlo o recuperarlo después.
    def persist_agent_result(self, decision_id: str, result: AgentResult) -> Path:
        """Persist an EXECUTE result for later COMMIT or RESUME.
        EXECUTE itself remains outside ``StateStore``. This helper only stores
        the already produced result using the transaction decision identifier.
        """
        if not isinstance(result, AgentResult):
            raise TypeError("result must be an AgentResult")
        path = self._agent_result_path(decision_id)
        atomic_write_json(path, result.to_dict())
        return path

    # Busca y recupera un AgentResult previamente guardado a partir
    # de su decision_id; si no existe ningún archivo asociado, devuelve None.
    def find_persisted_agent_result(self, decision_id: str) -> AgentResult | None:
        path = self._agent_result_path(decision_id)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return AgentResult.from_dict(payload)

    # Guarda el resultado final de la etapa que acaba de ejecutarse.
    @staticmethod
    def _normalize_fingerprints(
        fingerprints: Any,
    ):
        from src.state.pipeline_state import StageFingerprints
        if isinstance(fingerprints, StageFingerprints):
            value = fingerprints
        elif isinstance(fingerprints, Mapping):
            value = StageFingerprints.from_dict(fingerprints)
        else:
            raise TypeError("fingerprints must be StageFingerprints or a mapping")
        required = (value.input, value.config, value.dependencies, value.composite)
        if any(not isinstance(item, str) or not item.strip() for item in required):
            raise ValueError("all stage fingerprints must be present and non-empty")
        return value

    # Guarda el resultado final de la etapa que acaba de ejecutarse
    # y verifica que coincida con la ejecución que estaba pendiente.
    #comprueba que el resultado pertenezca a la ejecución que estaba pendiente;
    #valida que la etapa, el intento y los fingerprints coincidan;
    #prepara la fecha en la que ese resultado va a quedar guardado oficialmente.
    def commit_execution(
        self,
        *,
        decision_id: str,
        result: AgentResult,
        stage_name: str,
        fingerprints: Any,
        observations: Mapping[str, Any] | None = None,
        committed_at: str | None = None,
    ) -> PipelineState:
        if not isinstance(result, AgentResult):
            raise TypeError("result must be an AgentResult")

        state = self.load()
        pending = state.pending_execution
        if pending is None:
            raise RuntimeError("no pending execution exists")

        decision_id = self._validate_non_empty_string(decision_id, "decision_id")
        stage_name = self._validate_non_empty_string(stage_name, "stage_name")

        if pending.decision_id != decision_id:
            raise ValueError("decision_id does not match pending_execution")
        if pending.target_stage != stage_name:
            raise ValueError("stage_name does not match pending_execution.target_stage")
        if pending.attempt_number != result.attempt_number:
            raise ValueError("AgentResult attempt_number does not match pending_execution")
        if result.execution_status not in {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
        }:
            raise ValueError("only COMPLETED or FAILED AgentResult values can be committed")

        stage_fingerprints = self._normalize_fingerprints(fingerprints)
        committed_at = (
            self._validate_non_empty_string(committed_at, "committed_at")
            if committed_at is not None
            else self._now_iso()
        )

        current_stage = state.stages.get(stage_name, StageState())
        updated_stage = replace(
            current_stage,
            execution_status=result.execution_status,
            quality_status=result.quality_status,
            attempts_used=current_stage.attempts_used + 1,
            retrieval_rounds_used=(
                current_stage.retrieval_rounds_used
                + result.tool_usage.retrieval_rounds
            ),
            fingerprints=stage_fingerprints,
            warnings=tuple(warning.to_dict() for warning in result.warnings),
            failure_reason_codes=result.failure_reason_codes,
            requested_transition=result.requested_transition,
            last_error=result.error,
            updated_at=committed_at,
        )
        # Actualiza el estado de la etapa y registra los artefactos generados.
        updated_stages = dict(state.stages)
        updated_stages[stage_name] = updated_stage
        updated_artifacts = dict(state.artifacts)
        for artifact_name, artifact_reference in result.output_artifacts.items():
            updated_artifacts[artifact_name] = ArtifactState(
                reference=artifact_reference,
                created_at=result.completed_at or committed_at,
            )
        # Registra en el historial qué etapa se ejecutó, qué decidió
        # y cuál fue el resultado obtenido.
        log_entry = DecisionLogEntry(
            decision_id=decision_id,
            timestamp=committed_at,
            agent=stage_name,
            stage=stage_name,
            attempt=result.attempt_number,
            observations=dict(observations or {}),
            decision=result.decision.to_dict(),
            reason_codes=result.failure_reason_codes,
            requested_transition=result.requested_transition.to_dict(),
            result=result.to_dict(),
        )
        # Construye el nuevo estado del pipeline con la ejecución ya confirmada
        # y elimina la ejecución pendiente porque ya terminó.
        committed_state = replace(
            state,
            identity=replace(state.identity, updated_at=committed_at),
            stages=updated_stages,
            artifacts=updated_artifacts,
            decision_log=state.decision_log + (log_entry,),
            pending_execution=None,
        )
        self.save(committed_state)
        return committed_state

    # Cancela una ejecución pendiente y deja el pipeline
    # sin ninguna etapa marcada como pendiente.
    def cancel_pending_execution(self) -> PipelineState:
        state = self.load()
        if state.pending_execution is None:
            return state
        cancelled_state = replace(state, pending_execution=None)
        self.save(cancelled_state)
        return cancelled_state

    # Resuelve qué hacer cuando el pipeline se reanuda
    # y encuentra una ejecución que había quedado pendiente.
    def resolve_resume(
        self,
        *,
        stage_name: str,
        fingerprints: Any,
        observations: Mapping[str, Any] | None = None,
    ) -> ResumeResolution:
        state = self.load()
        pending = state.pending_execution
        # Si no hay nada pendiente, no es necesario recuperar ni repetir nada.
        if pending is None:
            return ResumeResolution(action="NO_PENDING", state=state)
        result = self.find_persisted_agent_result(pending.decision_id)
        # Si no existe un resultado guardado, cancela la ejecución pendiente
        # y deja la etapa lista para ejecutarse nuevamente.
        if result is None:
            cancelled = self.cancel_pending_execution()
            return ResumeResolution(action="REEXECUTE", state=cancelled)

        # Si el resultado sí existe, lo confirma oficialmente mediante COMMIT.
        committed = self.commit_execution(
            decision_id=pending.decision_id,
            result=result,
            stage_name=stage_name,
            fingerprints=fingerprints,
            observations=observations,
        )
        return ResumeResolution(
            action="COMMITTED",
            state=committed,
            committed_result=result,
        )

    # Agrega una nueva decisión al historial del pipeline
    # y guarda el estado actualizado.
    def append_decision_log(
        self,
        entry: DecisionLogEntry,
    ) -> PipelineState:
        if not isinstance(entry, DecisionLogEntry):
            raise TypeError("entry must be a DecisionLogEntry")
        state = self.load()
        updated = replace(
            state,
            decision_log=state.decision_log + (entry,),
        )
        self.save(updated)
        return updated
