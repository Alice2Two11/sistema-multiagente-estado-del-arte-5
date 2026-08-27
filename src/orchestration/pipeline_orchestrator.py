"""Orquestador de las etapas 03-06 usando exclusivamente los componentes de ``src/``.
Diseño
------
El orquestador coordina las etapas operativas del pipeline implementadas en
``src/`` y reutiliza, para cada una de ellas, componentes especializados de
ejecución, validación, persistencia y control de estado.

Cada etapa orquestada expone principalmente:

1. Un constructor de ejecución ubicado en ``src/adapters/*_runtime.py`` que
   prepara el agente o capacidad correspondiente, construye su ``AgentInput``
   y conecta las dependencias necesarias a partir de la configuración del
   experimento activo y de los artefactos generados por etapas anteriores.

2. Un mecanismo de ejecución transaccional que controla el ciclo
   PREPARE → EXECUTE → persist → COMMIT sobre ``StateStore``. Esto permite
   registrar de forma consistente tanto ejecuciones exitosas como fallos,
   evitando que una excepción deje el estado del pipeline incompleto o sin
   trazabilidad.

Este módulo mantiene la especificación de las etapas, conecta sus
constructores con los mecanismos de ejecución y proporciona ``run_pipeline``
como bucle principal de orquestación. El avance no depende únicamente de
recorrer una lista fija de etapas, sino de las transiciones solicitadas por
cada ejecución mediante ``RequestedTransition``.

La lógica de decisión se encuentra separada en ``decision_engine.py``. Allí
se validan decisiones como ADVANCE, RETRY, RETURN, HALT_STAGE y STOP_PIPELINE,
además de la vigencia de resultados mediante fingerprints y la invalidación
en cascada cuando cambian datos de entrada. El orquestador aplica estas
decisiones sobre la ejecución real de los agentes y capacidades.

Durante la ejecución:

- utiliza el ``pipeline_state.json`` canónico del experimento activo para
  conservar el estado global del pipeline;
- puede reutilizar resultados previamente completados cuando sus fingerprints
  siguen vigentes, evitando ejecuciones innecesarias;
- vuelve a ejecutar una etapa cuando sus entradas o dependencias relevantes
  han cambiado;
- resuelve ejecuciones pendientes de runs anteriores antes de continuar;
- valida cada transición solicitada antes de aplicarla;
- conserva los resultados, errores, artefactos y decisiones necesarios para
  mantener trazabilidad entre etapas.

El flujo actualmente orquestado comprende:

03  → extracción estructurada de información científica;
03B → extracción cuantitativa como capacidad especializada;
04  → análisis temático transversal;
05  → generación del esquema del estado del arte;
06  → redacción del borrador;
07  → verificación científica, trazabilidad y recuperación adicional de
      evidencia cuando resulta necesaria;
08  → evaluación del estado del arte generado mediante métricas automáticas
      y comparación con el Ground Truth.

La etapa 03B se implementa como una capacidad especializada y no como un
agente autónomo independiente.

La verificación de la etapa 07 integra el Retriever y los mecanismos de
recuperación adicional de evidencia. Cuando la evidencia inicial de una
afirmación no resulta suficiente, el subsistema de Agentic Retrieval puede
decidir nuevas acciones de recuperación, como reformular la consulta o
ajustar la cantidad de resultados recuperados, antes de entregar la evidencia
final al agente verificador.

La etapa 08 consume los artefactos generados por el pipeline y realiza la
evaluación final. El Ground Truth permanece separado de las etapas de
generación, recuperación y verificación, y se utiliza únicamente durante esta
fase de evaluación.

Las etapas 00, 01 y 02 no forman parte del bucle interno de
``pipeline_orchestrator.py``. Se ejecutan previamente mediante notebooks y
preparan respectivamente la configuración del experimento, la memoria
documental y el índice vectorial/Retriever RAG necesarios para la ejecución
posterior.

Por tanto, el orquestador no ejecuta notebooks ni depende de sus celdas como
unidades de procesamiento. Los notebooks funcionan como puntos de entrada y
preparación experimental, mientras que la lógica multiagente de las etapas
03–08 está modularizada en archivos Python dentro de ``src/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Importa las estructuras y enumeraciones que estandarizan el resultado
# de ejecución de los agentes, incluyendo estado técnico, estado de calidad,
# advertencias, uso de herramientas, información de decisión y transiciones
# solicitadas entre las distintas etapas del pipeline.
from src.contracts.agent_result import (
    AgentResult,
    AgentWarning,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
    WarningSeverity,
)

# Importa el motor de decisiones del orquestador y las funciones que
# controlan el orden de las etapas, los límites de reintentos, la
# validación de transiciones, la detección de resultados vigentes y
# la invalidación de etapas cuando cambian sus dependencias. También
# incluye la lógica para gestionar retornos y ciclos entre etapas.
from src.orchestration import decision_engine as de
from src.orchestration.decision_engine import (
    CANONICAL_STAGE_ORDER,
    MAX_ATTEMPTS_DEFAULT,
    ValidatedTransition,
    apply_return_with_cycle,
    default_next_stage,
    invalidate_from,
    is_stage_fresh,
    resolve_cycle_if_active,
    validate_transition,
)
from src.state.fingerprints import build_stage_fingerprints
from src.state.pipeline_state import PipelineIdentity, PipelineState
from src.state.state_store import StateStore

DRAFT_STAGE_NAME = "06_agente_redactor"


# ---------------------------------------------------------------------------
# Resolución de rutas del experimento activo
# ---------------------------------------------------------------------------

def load_active_experiment(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    active_path = root / "active_experiment.json"
    if not active_path.is_file():
        raise FileNotFoundError(
            "No existe active_experiment.json en "
            f"{root}. Ejecuta primero la etapa 00 (bootstrap del proyecto)."
        )
    return json.loads(active_path.read_text(encoding="utf-8"))

# Determina la ruta del pipeline_state.json del experimento activo y devuelve también su experiment_id y run_id.
def resolve_state_path(project_dir: str | Path) -> tuple[Path, str, str]:
    """Devuelve (state_path, experiment_id, run_id) del experimento activo."""

    root = Path(project_dir).resolve()
    active = load_active_experiment(root)
    experiment_id = active["active_experiment_id"]
    run_id = active.get("run_id", experiment_id)
    experiment_dir = root / experiment_id
    state_path = (
        experiment_dir / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
    )
    return state_path, experiment_id, run_id


#Abre el estado persistente del pipeline y, si aún no existe, crea e inicializa un pipeline_state.json nuevo para el experimento activo.
def ensure_pipeline_state(project_dir: str | Path) -> StateStore:
    """Abre el ``StateStore`` canónico, inicializándolo si es la primera vez."""

    state_path, experiment_id, run_id = resolve_state_path(project_dir)
    store = StateStore(state_path)
    if not state_path.is_file():
        now = datetime.now(timezone.utc).isoformat()
        store.initialize(
            PipelineState(
                identity=PipelineIdentity(
                    experiment_id=experiment_id,
                    run_id=run_id,
                    created_at=now,
                    updated_at=now,
                    schema_version="1.0",
                )
            )
        )
    return store



# ---------------------------------------------------------------------------
# Constructores reales por etapa (envuelven los `build_real_*` existentes)
# ---------------------------------------------------------------------------


#Construye y prepara la ejecución real del Agente de Extracción: carga configuración y 
#credenciales, crea el runtime y el AgentInput, y devuelve el ExtractionAgent listo para 
#ejecutarse junto con su entrada. (arma todo lo necesario para poder ejecutar el Agente 
# 03 de Extracción.)
def _real_extraction_execution(project_dir: Path, attempt_number: int):
    from src.adapters.extraction_runtime import (
        build_agent_input,
        build_extraction_runtime,
        load_runtime_configuration,
        resolve_openai_api_key,
    )
    from src.agents.extraction_agent import ExtractionAgent

    api_key = resolve_openai_api_key(project_dir=project_dir, required=True)
    configuration = load_runtime_configuration(project_dir)
    runtime = build_extraction_runtime(configuration, api_key=api_key)
    agent_input = build_agent_input(
        configuration,
        attempt_number=attempt_number,
        runtime_resources={
            "df_chunks_clean": runtime.dataframe,
            "collection": runtime.collection,
        },
    )
    return ExtractionAgent(runtime.dependencies), agent_input


# Prepara la capacidad de extracción cuantitativa 03B y su entrada para 
# que el orquestador pueda ejecutarla.
def _real_quantitative_execution(project_dir: Path, attempt_number: int):
    from src.adapters.quantitative_extraction_runtime import (
        build_quantitative_agent_input,
        build_quantitative_capability,
        load_quantitative_configuration,
    )
    configuration = load_quantitative_configuration(project_dir)
    capability = build_quantitative_capability(configuration)
    agent_input = build_quantitative_agent_input(configuration)
    return capability, agent_input


# Prepara el Agente de Análisis Temático y su AgentInput para que
# el orquestador pueda ejecutar la etapa 04.
def _real_thematic_execution(project_dir: Path, attempt_number: int):
    from src.adapters.thematic_analysis_runtime import build_real_thematic_execution

    agent, agent_input, _configuration = build_real_thematic_execution(
        project_dir, attempt_number
    )
    return agent, agent_input


# Prepara el Generador de Esquema y su AgentInput para que
# el orquestador pueda ejecutar la etapa 05.
def _real_outline_execution(project_dir: Path, attempt_number: int):
    from src.adapters.outline_generation_runtime import build_real_outline_execution

    agent, agent_input, _configuration = build_real_outline_execution(
        project_dir, attempt_number
    )
    return agent, agent_input

   
#Decide si el Agente Redactor 06 debe escribir un borrador inicial o corregir 
#un borrador anterior después de la verificación 07.
def _resolve_draft_execution_mode(project_dir: Path, store) -> dict[str, Any] | None:
  
   # Comprueba el estado actual del ciclo entre el Redactor y el Verificador.
   # Si no existe un ciclo activo o todavía no se ha usado ninguna ronda,
   # la función concluye que 06 debe trabajar en modo INITIAL_DRAFT y
   # devuelve None sin buscar información adicional de revisión.
    state = store.load()
    cycle = state.cycles.get("writer_verifier")
    if cycle is None or cycle.status != "ACTIVE" or cycle.rounds_used == 0:
        return None

    import json as _json

    from src.tools.verification.cycle_round_persistence import (
        list_persisted_rounds,
        read_round_artifact,
        read_round_status,
        round_is_persisted,
    )

   # Recupera la información del experimento activo y localiza las rondas
   # de revisión que fueron persistidas por el ciclo 06 ↔ 07. Selecciona
   # la ronda más reciente y verifica que exista correctamente en disco
   # antes de intentar reconstruir una ejecución de revisión.
    active_experiment = _json.loads((project_dir / "active_experiment.json").read_text(encoding="utf-8"))
    experiment_id = active_experiment["active_experiment_id"]

    persisted_rounds = list_persisted_rounds(project_dir=project_dir, experiment_id=experiment_id)
    if not persisted_rounds:
        raise RuntimeError(
            "DRAFT_REVISION_ROUND_NOT_PERSISTED: el ciclo writer_verifier está "
            "ACTIVE pero no hay ninguna ronda persistida en writer_verifier_cycle/."
        )

    round_number = persisted_rounds[-1]
    if not round_is_persisted(project_dir=project_dir, experiment_id=experiment_id, round_number=round_number):
        raise RuntimeError(f"DRAFT_REVISION_ROUND_NOT_PERSISTED: round_{round_number:02d}")

    status = read_round_status(project_dir=project_dir, experiment_id=experiment_id, round_number=round_number)
    if status is None:
        raise RuntimeError(f"DRAFT_REVISION_ROUND_NOT_PERSISTED: round_{round_number:02d}")
    if status["status"] not in {"AWAITING_REVISION", "REVISION_COMPLETED"}:
        raise RuntimeError(
            f"DRAFT_REVISION_ROUND_UNEXPECTED_STATUS: round_{round_number:02d} está en "
            f"{status['status']!r}, se esperaba 'AWAITING_REVISION' o 'REVISION_COMPLETED'."
        )
   # Si la ronda ya está REVISION_COMPLETED, se reconstruye el mismo
   # AgentInput para que run_stage() la detecte como SKIPPED_FRESH
   # y evite ejecutar nuevamente una revisión ya completada.

    writer_revision_request = read_round_artifact(
        project_dir=project_dir, experiment_id=experiment_id, round_number=round_number,
        filename="writer_revision_request.json",
    )

    # Consistencia (punto 5): la ronda debe corresponder al MISMO experimento
    # y ronda activos -- si algo no coincide, no se arma un AgentInput de
    # revisión con datos inconsistentes.
    if writer_revision_request.get("experiment_id") != experiment_id:
        raise RuntimeError(
            "DRAFT_REVISION_EXPERIMENT_MISMATCH: writer_revision_request "
            f"pertenece a {writer_revision_request.get('experiment_id')!r}, "
            f"se esperaba {experiment_id!r}."
        )
    if int(writer_revision_request.get("round_number", -1)) != round_number:
        raise RuntimeError(
            "DRAFT_REVISION_ROUND_MISMATCH: writer_revision_request declara ronda "
            f"{writer_revision_request.get('round_number')!r}, se esperaba {round_number}."
        )

   # Carga el último borrador comprometido por el Redactor y comprueba,
   # mediante su fingerprint, que sea el mismo borrador que fue evaluado
   # por el Verificador. Si todo coincide, devuelve los datos necesarios
   # para construir la nueva ejecución de 06 en modo REVISION.
    experiment_dir = project_dir / experiment_id
    draft_json_path = experiment_dir / "05_outputs" / "05_draft" / "state_of_art_draft.json"
    if not draft_json_path.is_file():
        raise RuntimeError("DRAFT_REVISION_PREVIOUS_DRAFT_NOT_FOUND")
    previous_draft = _json.loads(draft_json_path.read_text(encoding="utf-8"))

    if previous_draft.get("source_draft_fingerprint") not in (
        None,
        writer_revision_request["source_draft_fingerprint"],
    ):
        raise RuntimeError(
            "DRAFT_REVISION_FINGERPRINT_MISMATCH: el borrador en disco no coincide "
            "con source_draft_fingerprint del writer_revision_request."
        )
    return {
        "mode": "REVISION",
        "writer_revision_request": writer_revision_request,
        "previous_draft": previous_draft,
        "round_number": round_number,
        "cycle_project_dir": str(project_dir),
        "experiment_id": experiment_id,
    }


# Prepara el Agente Redactor 06, determina si trabaja en modo inicial
# o revisión y devuelve el agente con su AgentInput listo para ejecutar.
def _real_draft_execution(project_dir: Path, attempt_number: int):
    from src.adapters.draft_writing_runtime import build_real_draft_execution

    project_dir = Path(project_dir)
    store = ensure_pipeline_state(project_dir)
    revision_overrides = _resolve_draft_execution_mode(project_dir, store)

    agent, agent_input, _configuration = build_real_draft_execution(
        project_dir, attempt_number, policy_overrides=revision_overrides
    )
    return agent, agent_input


# Prepara la etapa 07 de verificación y devuelve su ejecución
# lista para ser utilizada por el orquestador.
def _experimental_verification_execution(project_dir: Path, attempt_number: int):
    from src.adapters.verification_orchestrator_runtime import (
        build_experimental_verification_execution,
    )

    return build_experimental_verification_execution(project_dir, attempt_number)


# Prepara la etapa 08 de evaluación y devuelve su ejecución
# lista para ser utilizada por el orquestador.
def _experimental_evaluation_execution(project_dir: Path, attempt_number: int):
    from src.adapters.evaluation_stagespec_wiring import build_execution_for_stagespec

    return build_execution_for_stagespec(project_dir, attempt_number)


# Ejecuta la etapa 08 utilizando la implementación real del
# runtime de evaluación y devuelve su resultado al orquestador.
def _run_evaluation_stage(**kwargs):
    from src.adapters.evaluation_orchestrator_runtime import (
        _run_evaluation_stage as _real_run_evaluation_stage,
    )

    return _real_run_evaluation_stage(**kwargs)


# ---------------------------------------------------------------------------
# Protocolo transaccional para la etapa 06
# ---------------------------------------------------------------------------

# Ejecuta la etapa 06 mediante el ciclo PREPARE → EXECUTE → PERSIST → COMMIT.
# Si la redacción falla, convierte el error en un AgentResult FAILED para
# mantenerlo registrado en el estado del pipeline en lugar de perder la excepción.

def _draft_runtime_transaction(
    *,
    store: StateStore,
    build_execution: Callable[[], tuple[Any, Any]],
    attempt_number: int,
    observations: Mapping[str, Any] | None = None,
):
    from src.runtime.draft_writing_protocol import (
        DraftWritingTransactionResult,
        build_draft_fingerprints,
    )

    prepared = store.prepare_execution(
        target_stage=DRAFT_STAGE_NAME,
        intended_action="EXECUTE_DRAFT_WRITING",
        attempt_number=attempt_number,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        agent, agent_input = build_execution()
        result = agent.execute(agent_input)
        fingerprints = build_draft_fingerprints(agent_input)
    except Exception as exc:  # AgentResult FAILED
        text = str(exc)
        if isinstance(exc, FileNotFoundError):
            code = "DEPENDENCY_NOT_FOUND"
        elif "OPENAI_API_KEY" in text:
            code = "CREDENTIAL_NOT_FOUND"
        else:
            code = "RUNTIME_DEPENDENCY_FAILED"
        now = datetime.now(timezone.utc).isoformat()
        result = AgentResult(
            execution_status=ExecutionStatus.FAILED,
            quality_status=QualityStatus.REJECTED,
            decision=DecisionInfo(
                code="DRAFT_RUNTIME_FAILED",
                rationale="Falló la preparación de la etapa 06.",
            ),
            quality_metrics={"technical": {}, "scientific": {}},
            warnings=(
                AgentWarning(
                    code=code,
                    severity=WarningSeverity.ERROR,
                    blocking=True,
                    message=text,
                ),
            ),
            failure_reason_codes=(code,),
            requested_transition=RequestedTransition(
                action=TransitionAction.HALT_STAGE,
                target_stage=None,
                reason_code=code,
                requires_human_confirmation=False,
            ),
            output_artifacts={},
            tool_usage=ToolUsage(),
            attempt_number=attempt_number,
            started_at=started_at,
            completed_at=now,
            error={"type": type(exc).__name__, "message": text, "stage": DRAFT_STAGE_NAME},
        )
        fingerprints = build_stage_fingerprints(
            input_data={"stage_name": DRAFT_STAGE_NAME, "attempt_number": attempt_number},
            config_data={"runtime_resolution": "FAILED"},
            dependencies_data={},
        )

    persisted_path = store.persist_agent_result(prepared.decision_id, result)
    committed_state = store.commit_execution(
        decision_id=prepared.decision_id,
        result=result,
        stage_name=DRAFT_STAGE_NAME,
        fingerprints=fingerprints,
        observations=dict(observations or {}),
    )
    return DraftWritingTransactionResult(
        prepared, result, str(persisted_path), committed_state
    )


def _quantitative_runtime_transaction(
    *,
    store: StateStore,
    build_execution: Callable[[], tuple[Any, Any]],
    attempt_number: int,
    observations: Mapping[str, Any] | None = None,
):
    # execute_quantitative_runtime_transaction no admite attempt_number: la
    # etapa 03B siempre corre como attempt_number=1 en el wiring actual.
    from src.runtime.quantitative_extraction_protocol import (
        execute_quantitative_runtime_transaction,
    )

    return execute_quantitative_runtime_transaction(
        store=store, build_execution=build_execution, observations=observations
    )


# ---------------------------------------------------------------------------
# Ejecución dedicada de la etapa 07 
# ---------------------------------------------------------------------------
#
# Ejecuta y controla la etapa 07 de verificación, permitiendo reanudar
# ejecuciones previas, reutilizar resultados vigentes o repetir la etapa
# cuando existen cambios, inconsistencias o artefactos incompletos.


def _run_verification_stage(
    *,
    store: StateStore,
    project_dir: str | Path,
    spec: StageSpec,
    attempt_number: int = 1,
    observations: Mapping[str, Any] | None = None,
    force_rerun: bool = False,
) -> StageOutcome:
    from src.adapters.verification_notebook import (
        commit_executed_agent07,
        execute_prepared_agent07,
        prepare_agent07_execution,
        resume_agent07_execution,
    )

    project_dir = Path(project_dir)
    had_pending = store.load().pending_execution is not None

    try:
        dependencies, runtime_input = spec.build_execution(project_dir, attempt_number)
    except Exception as exc: 
       
        raise

    def _do_fresh_execution() -> AgentResult:
        prepared = prepare_agent07_execution(store=store, runtime_input=runtime_input)
        executed = execute_prepared_agent07(
            store=store, prepared=prepared, dependencies=dependencies
        )
        commit_executed_agent07(store=store, executed=executed)
        return executed.agent_result

    if force_rerun and not had_pending:
        result = _do_fresh_execution()
        status = (
            "COMMITTED"
            if result.execution_status == ExecutionStatus.COMPLETED
            else "FAILED"
        )
    else:
        resume = resume_agent07_execution(store=store, runtime_input=runtime_input)
        if resume.action == "COMMITTED":
            result = resume.committed_result
            status = "COMMITTED" if had_pending else "SKIPPED_FRESH"
        elif resume.action == "EXECUTED_NOT_COMMITTED":
            commit_executed_agent07(store=store, executed=resume.executed)
            result = resume.executed.agent_result
            status = "COMMITTED"
        elif resume.action in {
            "NO_COMMIT",
            "REEXECUTE",
            "FINGERPRINT_MISMATCH",
            "ARTIFACT_MISMATCH",
            "MANIFEST_INCOMPLETE",
        }:
            result = _do_fresh_execution()
            status = (
                "COMMITTED"
                if result.execution_status == ExecutionStatus.COMPLETED
                else "FAILED"
            )
        else:  
            raise RuntimeError(
                f"resume_agent07_execution devolvió una acción inesperada: {resume.action}"
            )

    state = store.load()
    attempts_used = state.stages[spec.key].attempts_used
    return _outcome_from_result(spec, result, status, attempts_used=attempts_used)



# ---------------------------------------------------------------------------
# Registro de etapas
# ---------------------------------------------------------------------------

# Define la ficha de configuración de cada etapa del pipeline, indicando
# cómo se construye, ejecuta, reanuda y valida, además de sus reglas
# especiales de reintentos, fingerprints y ejecución personalizada.
@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    build_execution: Callable[[Path, int], tuple[Any, Any]]
    runtime_transaction: Callable[..., Any]
    resolve_resume: Callable[..., Any]
   
    build_fingerprints: Callable[[Any], Any] | None = None
    max_attempt_number: int | None = None
   
    bypass_manual_review: bool = False
   
    custom_run: Callable[..., "StageOutcome"] | None = None


#registro central de etapas que conoce el orquestador. 
#En otras palabras, _stage_registry() le dice al sistema qué etapas existen 
#y qué funciones debe usar para preparar, ejecutar, reanudar y validar cada una.
# y las ejecuciones especiales utilizadas por las etapas 07 y 08.
def _stage_registry() -> list[StageSpec]:
   
    from src.runtime.extraction_protocol import (
        build_agent_input_fingerprints,
        execute_extraction_runtime_transaction,
        resolve_extraction_resume,
    )
    from src.runtime.outline_generation_protocol import (
        build_outline_fingerprints,
        execute_outline_runtime_transaction,
        resolve_outline_resume,
    )
    from src.runtime.quantitative_extraction_protocol import (
        build_quantitative_fingerprints,
        resolve_quantitative_resume,
    )
    from src.runtime.thematic_analysis_protocol import (
        build_thematic_fingerprints,
        execute_thematic_runtime_transaction,
        resolve_thematic_resume,
    )
    from src.runtime.draft_writing_protocol import (
        build_draft_fingerprints,
        resolve_draft_resume,
    )

    return [
        StageSpec(
            key="03_agente_extraccion_kb", #qué etapa es
            label="02 · Extracción de información científica", #cómo mostrarla
            build_execution=_real_extraction_execution, #cómo prepararla
            runtime_transaction=execute_extraction_runtime_transaction, #cómo ejecutarla
            resolve_resume=resolve_extraction_resume, #cómo reanudarla
            build_fingerprints=build_agent_input_fingerprints, #cómo comprobar si su resultado sigue vigente
            max_attempt_number=2, #permite hasta 2 intentos.
        ),
        StageSpec(
            key="03B_extraccion_cuantitativa_kb",
            label="03 · Extracción y normalización cuantitativa",
            build_execution=_real_quantitative_execution,
            runtime_transaction=_quantitative_runtime_transaction,
            resolve_resume=resolve_quantitative_resume,
            build_fingerprints=build_quantitative_fingerprints,
            max_attempt_number=1,
        ),
        StageSpec(
            key="04_agente_analisis_tematico",
            label="04 · Análisis temático",
            build_execution=_real_thematic_execution,
            runtime_transaction=execute_thematic_runtime_transaction,
            resolve_resume=resolve_thematic_resume,
            build_fingerprints=build_thematic_fingerprints,
        ),
        StageSpec(
            key="05_generador_esquema",
            label="05 · Generación del esquema",
            build_execution=_real_outline_execution,
            runtime_transaction=execute_outline_runtime_transaction,
            resolve_resume=resolve_outline_resume,
            build_fingerprints=build_outline_fingerprints,
        ),
        StageSpec(
            key=DRAFT_STAGE_NAME,
            label="06 · Redacción del borrador",
            build_execution=_real_draft_execution,
            runtime_transaction=_draft_runtime_transaction,
            resolve_resume=resolve_draft_resume,
            build_fingerprints=build_draft_fingerprints,
        ),
        StageSpec(
            key="07_agente_verificador",
            label="07 · Verificación y trazabilidad",
            build_execution=_experimental_verification_execution,
            runtime_transaction=None,  # ver custom_run
            resolve_resume=None,  # ver custom_run
            build_fingerprints=None,  # ver comentario en el campo del dataclass
            custom_run=_run_verification_stage,
        ),
        StageSpec(
            key="08_evaluacion_experimental",
            label="08 · Evaluación experimental",
            build_execution=_experimental_evaluation_execution,
            runtime_transaction=None,  # ver custom_run
            resolve_resume=None,  # ver custom_run
            build_fingerprints=None,  # ver comentario en el campo del dataclass
            custom_run=_run_evaluation_stage,
        ),
    ]


#Define el orden canónico de ejecución de las etapas 03 a 08 del pipeline.. 
STAGE_ORDER: tuple[str, ...] = (
    "03_agente_extraccion_kb",
    "03B_extraccion_cuantitativa_kb",
    "04_agente_analisis_tematico",
    "05_generador_esquema",
    DRAFT_STAGE_NAME,
    "07_agente_verificador",
    "08_evaluacion_experimental",
)


# ---------------------------------------------------------------------------
# Ejecución de una etapa y del pipeline completo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
# Resume el resultado de una etapa y la transición validada que el
# orquestador debe aplicar después de su ejecución.
class StageOutcome:
    key: str #Identificador interno de la etapa
    label: str
    status: str #Estado operativo de la ejecución SKIPPED_FRESH | COMMITTED (via resume) | COMMITTED | FAILED
    execution_status: str | None #Indica si la ejecución técnica terminó correctamente o falló.
    quality_status: str | None #Indica el resultado de calidad científica de la etapa, que es distinto del estado técnico.
    warnings: tuple[str, ...] #Lista de advertencias generadas durante la ejecución.
    error: Mapping[str, Any] | None #Información del error, si ocurrió alguno.
    attempt_number: int #intentos
    next_action: str     # ADVANCE | RETRY | RETURN | HALT_STAGE | STOP_PIPELINE
    target_stage: str | None
    reason_code: str


# Valida la transición solicitada por una etapa considerando su calidad,
# número de intentos y reglas del pipeline antes de aplicarla.
def _validated_transition_for(
    spec: StageSpec,
    *,
    requested_transition: RequestedTransition,
    quality_status: QualityStatus,
    attempts_used: int,
) -> ValidatedTransition:
    return validate_transition(
        current_stage=spec.key,
        requested_transition=requested_transition,
        quality_status=quality_status,
        attempts_used=attempts_used,
        max_attempts=spec.max_attempt_number or MAX_ATTEMPTS_DEFAULT,
        known_stages=frozenset(CANONICAL_STAGE_ORDER),
        bypass_manual_review=spec.bypass_manual_review,
    )


# Resume el resultado del agente y determina, después de validar su
# solicitud, qué acción debe seguir el orquestador en el pipeline.
def _outcome_from_result(
    spec: StageSpec, result: AgentResult, status: str, *, attempts_used: int
) -> StageOutcome:
    validated = _validated_transition_for(
        spec,
        requested_transition=result.requested_transition,
        quality_status=result.quality_status,
        attempts_used=attempts_used,
    )
    return StageOutcome(
        key=spec.key,
        label=spec.label,
        status=status,
        execution_status=result.execution_status.value,
        quality_status=result.quality_status.value,
        warnings=tuple(w.message for w in result.warnings),
        error=result.error,
        attempt_number=result.attempt_number,
        next_action=validated.action,
        target_stage=validated.target_stage,
        reason_code=validated.reason_code,
    )


# Recupera el resultado de una etapa ya completada y todavía vigente,
# reutiliza su transición anterior y construye el StageOutcome necesario
# para que el orquestador continúe sin ejecutar nuevamente la etapa.
#(usado para SKIPPED_FRESH)
def _outcome_from_committed_stage(spec: StageSpec, committed, *, status: str) -> StageOutcome:
    """Construye un StageOutcome a partir de un StageState ya comprometido
    (usado para SKIPPED_FRESH), reutilizando su ``requested_transition``
    histórica en vez de asumir ADVANCE por defecto."""

    requested = committed.requested_transition or RequestedTransition(
        action=TransitionAction.ADVANCE, reason_code="ASSUMED_ADVANCE_NO_HISTORY"
    )
    validated = _validated_transition_for(
        spec,
        requested_transition=requested,
        quality_status=committed.quality_status,
        attempts_used=committed.attempts_used,
    )
    return StageOutcome(
        key=spec.key,
        label=spec.label,
        status=status,
        execution_status=committed.execution_status.value,
        quality_status=(
            committed.quality_status.value if committed.quality_status else None
        ),
        warnings=tuple(w.get("message", "") for w in committed.warnings),
        error=committed.last_error,
        attempt_number=committed.attempts_used,
        next_action=validated.action,
        target_stage=validated.target_stage,
        reason_code=validated.reason_code,
    )


# Controla la ejecución de una etapa individual, decidiendo si debe
# ejecutarse, reanudarse, reutilizarse o usar una ejecución especial,
# respetando además los límites de intentos definidos para cada etapa.
def run_stage(
    *,
    store: StateStore,
    project_dir: str | Path,
    spec: StageSpec,
    attempt_number: int = 1,
    observations: Mapping[str, Any] | None = None,
    force_rerun: bool = False,
) -> StageOutcome:
    """Ejecuta una etapa del pipeline, resolviendo reanudaciones, reutilizando
      resultados vigentes mediante fingerprints o ejecutándola nuevamente cuando
      sea necesario. Si la etapa tiene un custom_run, delega en esa ejecución
      especial. Los errores no controlados se convierten en un StageOutcome FAILED
      para mantenerlos registrados y detener la etapa de forma segura.
    """
    project_dir = Path(project_dir)
    if spec.max_attempt_number is not None and attempt_number > spec.max_attempt_number:
        raise ValueError(
            f"{spec.key} admite como máximo attempt_number={spec.max_attempt_number}."
        )

# Envuelve toda la ejecución de la etapa en una red de seguridad.
# Si ocurre un error no controlado, lo convierte en un StageOutcome
# FAILED y solicita HALT_STAGE en lugar de dejar caer el pipeline.
    try:
        if spec.custom_run is not None:
            return spec.custom_run(
                store=store,
                project_dir=project_dir,
                spec=spec,
                attempt_number=attempt_number,
                observations=observations,
                force_rerun=force_rerun,
            )

        def build_execution() -> tuple[Any, Any]:
            return spec.build_execution(project_dir, attempt_number)

        state = store.load()

         # Si existe una ejecución pendiente de esta misma etapa, intenta reanudarla.
         # Si el resultado ya estaba comprometido, lo reutiliza; si debe reejecutarse,
         # libera el estado pendiente y continúa con la lógica normal de ejecución.
        if (
            state.pending_execution is not None
            and state.pending_execution.target_stage == spec.key
        ):
            _agent, agent_input = build_execution()
            resume = spec.resolve_resume(
                store=store, agent_input=agent_input, observations=observations
            )
            if resume.action == "COMMITTED":
                state = store.load()
                attempts_used = state.stages[spec.key].attempts_used
                return _outcome_from_result(
                    spec, resume.committed_result, "COMMITTED", attempts_used=attempts_used
                )
            # REEXECUTE o NO_PENDING: el pending quedó liberado; se sigue abajo.
            state = store.load()
      
        # Si la etapa ya fue completada, compara sus fingerprints para decidir
         # si el resultado sigue vigente y puede reutilizarse como SKIPPED_FRESH.
         # Si quedó obsoleto, vuelve a ejecutarla; después guarda el resultado y
         # convierte cualquier error no controlado en un StageOutcome FAILED.
        committed = state.stages.get(spec.key)
        if (
            committed is not None
            and committed.execution_status == ExecutionStatus.COMPLETED
            and not force_rerun
        ):
            if spec.build_fingerprints is not None:
                _agent, agent_input = build_execution()
                current_fingerprints = spec.build_fingerprints(agent_input)
                if is_stage_fresh(committed, current_fingerprints):
                    return _outcome_from_committed_stage(
                        spec, committed, status="SKIPPED_FRESH"
                    )
                # Fingerprints obsoletos: no se salta aunque no se haya pedido
                # force_rerun explícito.
                observations = dict(observations or {})
                observations["orchestrator_note"] = (
                    "stage_stale_fingerprint_mismatch_reexecuting"
                )
            else:
                # Sin build_fingerprints disponible: se conserva el chequeo
                # antiguo (solo COMPLETED, sin comparar vigencia).
                return _outcome_from_committed_stage(
                    spec, committed, status="SKIPPED_FRESH"
                )

        transaction = spec.runtime_transaction(
            store=store,
            build_execution=build_execution,
            attempt_number=attempt_number,
            observations=observations,
        )
        status = (
            "COMMITTED"
            if transaction.agent_result.execution_status == ExecutionStatus.COMPLETED
            else "FAILED"
        )
        state = store.load()
        attempts_used = state.stages[spec.key].attempts_used
        return _outcome_from_result(
            spec, transaction.agent_result, status, attempts_used=attempts_used
        )
    except Exception as exc:  
        return StageOutcome(
            key=spec.key,
            label=spec.label,
            status="FAILED",
            execution_status=None,
            quality_status=None,
            warnings=(),
            error={"type": type(exc).__name__, "message": str(exc)},
            attempt_number=attempt_number,
            next_action="HALT_STAGE",
            target_stage=None,
            reason_code=f"UNCAUGHT_EXCEPTION:{type(exc).__name__}",
        )


# Resuelve una ejecución pendiente de otra etapa antes de continuar
# con la etapa actual, evitando que el pipeline quede bloqueado.
def _reconcile_pending_execution_for_other_stage(
    *,
    store,
    project_dir: Path,
    registry: Mapping[str, "StageSpec"],
    current_stage: str,
    attempt_numbers: Mapping[str, int],
    observations: Mapping[str, Any] | None,
) -> tuple[list["StageOutcome"], bool]:
    """Reconcilia una ejecución pendiente de otra etapa antes de continuar con
      la etapa actual. Usa el protocolo oficial de esa etapa mediante run_stage()
      para resolverla correctamente y evita que el slot global pending_execution
      bloquee el pipeline. Devuelve los resultados generados y si debe detenerse.
    """
    state = store.load()
    pending = state.pending_execution
    if pending is None or pending.target_stage == current_stage:
        return [], False

    pending_stage_key = pending.target_stage
    if pending_stage_key not in registry:
        return (
            [
                StageOutcome(
                    key=current_stage,
                    label=registry[current_stage].label if current_stage in registry else current_stage,
                    status="FAILED",
                    execution_status=None,
                    quality_status=None,
                    warnings=(),
                    error={
                        "type": "PendingExecutionUnknownTargetStage",
                        "message": (
                            f"pending_execution.target_stage={pending_stage_key!r} "
                            "no está en el registro de etapas -- no se puede "
                            "reconciliar automáticamente."
                        ),
                    },
                    attempt_number=0,
                    next_action="HALT_STAGE",
                    target_stage=None,
                    reason_code="PENDING_EXECUTION_UNKNOWN_TARGET_STAGE",
                )
            ],
            True,
        )

    pending_spec = registry[pending_stage_key]
    reconcile_outcome = run_stage(
        store=store,
        project_dir=project_dir,
        spec=pending_spec,
        attempt_number=attempt_numbers.get(pending_stage_key, 1),
        observations=observations,
        force_rerun=False,
    )

    state = store.load()
    if state.pending_execution is not None:
        return [reconcile_outcome], True

    return [reconcile_outcome], False



# Aplica la transición solicitada por la etapa que acaba de ejecutarse.
# Según el resultado, puede avanzar a la siguiente etapa, repetir la actual,
# regresar a una etapa anterior o detener el pipeline. También actualiza los
# contadores de intentos y controla el ciclo Redactor ↔ Verificador, deteniendo
# la ejecución si se supera el número máximo de rondas permitido.
def _apply_stage_transition(
    outcome: "StageOutcome",
    *,
    store,
    stage_key: str,
    attempt_number: int,
    attempt_numbers: dict[str, int],
    until: str | None,
    outcomes: list["StageOutcome"],
) -> tuple[str | None, bool]:
    """Interpreta la acción indicada por una etapa y decide cuál debe ser
      la siguiente etapa del pipeline o si la ejecución debe detenerse.
      Aplica la misma lógica para ADVANCE, RETRY, RETURN y HALT/STOP,
      incluso cuando la etapa fue reconciliada fuera del flujo normal.
    """

    if until is not None and stage_key == until:
        return None, True

    if outcome.next_action == "ADVANCE":
        if outcome.target_stage is None:
            return None, True  # pipeline completo
        if stage_key == de.WRITER_VERIFIER_TRIGGER_STAGE:
            resolve_cycle_if_active(store)
        return outcome.target_stage, False

    if outcome.next_action == "RETRY":
        attempt_numbers[stage_key] = attempt_number + 1
        return stage_key, False

    if outcome.next_action == "RETURN":
        cycle_result = apply_return_with_cycle(
            store,
            from_stage=stage_key,
            target_stage=outcome.target_stage,
            reason=f"INVALIDATED_BY_RETURN_FROM_{stage_key}",
        )
        if cycle_result.cycle_exhausted:
            outcomes.append(
                StageOutcome(
                    key=stage_key,
                    label=f"(ciclo {de.WRITER_VERIFIER_CYCLE_NAME} agotado)",
                    status="CYCLE_EXHAUSTED",
                    execution_status=None,
                    quality_status=None,
                    warnings=(),
                    error=None,
                    attempt_number=attempt_number,
                    next_action="HALT_STAGE",
                    target_stage=None,
                    reason_code="WRITER_VERIFIER_CYCLE_EXHAUSTED",
                )
            )
            return None, True
        for stage_key_to_clear in CANONICAL_STAGE_ORDER[
            CANONICAL_STAGE_ORDER.index(outcome.target_stage) :
        ]:
            attempt_numbers.pop(stage_key_to_clear, None)
        return outcome.target_stage, False

    # HALT_STAGE o STOP_PIPELINE: se detiene el bucle.
    return None, True


from src.orchestration.decision_log_frontier import (
    _causally_connects,
    _segment_decision_log,
    _reconstruct_authoritative_frontier,
)


# Comprueba si el pipeline ya terminó previamente con HALT_STAGE o
# STOP_PIPELINE. Si existe ese estado terminal, recupera su resultado
# histórico y evita ejecutar nuevamente las etapas, salvo que se solicite
# explícitamente un punto de inicio o una reejecución forzada.
def _check_already_terminal_state(
    *, store, registry: Mapping[str, "StageSpec"], start_stage: str | None, force_rerun: bool
) -> "StageOutcome | None":
    """Detecta si el pipeline ya quedó detenido de forma terminal según la
      decisión autoritativa registrada en el decision_log. Si la última decisión
      válida fue HALT_STAGE o STOP_PIPELINE, reconstruye ese resultado terminal
      desde el AgentResult histórico correspondiente y evita reiniciar etapas
      innecesariamente o mezclarlo con estados posteriores inconsistentes.
    """

    if start_stage is not None or force_rerun:
        return None

    state = store.load()
    frontier_entry = _reconstruct_authoritative_frontier(state.decision_log)
    if frontier_entry is None:
        return None

    frontier_transition = frontier_entry.requested_transition
    if (
        frontier_transition is None
        or frontier_transition.action not in (TransitionAction.HALT_STAGE, TransitionAction.STOP_PIPELINE)
        or frontier_entry.stage not in registry
    ):
        return None

    frontier_result = AgentResult.from_dict(frontier_entry.result)
    outcome = _outcome_from_result(
        registry[frontier_entry.stage], frontier_result, "ALREADY_TERMINAL", attempts_used=frontier_entry.attempt
    )

    if outcome.next_action not in ("HALT_STAGE", "STOP_PIPELINE"):
        return None

    return outcome



def run_pipeline(
    project_dir: str | Path,
    *,
    start_stage: str | None = None, #desde qué etapa empezar
    until: str | None = None, #Permite detener el pipeline después de una etapa específica.
    attempt_numbers: Mapping[str, int] | None = None, #indicar el número de intento actual de cada etapa.
    force_rerun: bool = False, #forzar la reejecución de la etapa inicial aunque ya tenga resultado válido.
    max_iterations: int = 50, #límite de seguridad para evitar ciclos infinitos.
    observations: Mapping[str, Any] | None = None,
) -> list[StageOutcome]: #lista con los resultados de las etapas procesadas.
    """Ejecuta el pipeline siguiendo las transiciones solicitadas por cada etapa:
       ADVANCE para avanzar, RETRY para repetir, RETURN para volver e invalidar
       etapas posteriores, y HALT_STAGE/STOP_PIPELINE para detenerse. También permite
       parar en una etapa específica con `until` y aplicar `force_rerun` solo al inicio.
    """
    attempt_numbers = dict(attempt_numbers or {})
    store = ensure_pipeline_state(project_dir)
    registry = {spec.key: spec for spec in _stage_registry()}

    if until is not None and until not in STAGE_ORDER:
        raise ValueError(f"Etapa desconocida en 'until': {until}")

    current_stage = start_stage or STAGE_ORDER[0] #empieza por la primera del pipeline.
    outcomes: list[StageOutcome] = []
    force_rerun_current = force_rerun

    terminal_outcome = _check_already_terminal_state(
        store=store, registry=registry, start_stage=start_stage, force_rerun=force_rerun #Comprueba si una ejecución anterior ya dejó el pipeline detenido definitivamente.
    )
    if terminal_outcome is not None:
        _print_outcome(terminal_outcome)
        return [terminal_outcome]
   #_______________________________
   #bucle principal del orquestador
   #_______________________________
    for _ in range(max_iterations):
       #se llegó a una etapa no ejecutable ordena detener el pipeline.
        if current_stage not in registry:
            outcomes.append(
                StageOutcome(
                    key=current_stage,
                    label=f"(sin StageSpec ejecutable todavía: {current_stage})",
                    status="REACHED_UNREGISTERED_STAGE",
                    execution_status=None,
                    quality_status=None,
                    warnings=(),
                    error=None,
                    attempt_number=0,
                    next_action="STOP_PIPELINE",
                    target_stage=None,
                    reason_code="STAGE_NOT_REGISTERED",
                )
            )
            break

        spec = registry[current_stage] #Obtiene la ficha StageSpec de la etapa actual.

        reconcile_outcomes, must_stop = _reconcile_pending_execution_for_other_stage( #Comprueba si quedó una ejecución pendiente de otra etapa anterior.
            store=store, project_dir=project_dir, registry=registry, current_stage=current_stage,
            attempt_numbers=attempt_numbers, observations=observations,
        )
        if reconcile_outcomes: #Si efectivamente tuvo que resolver algo pendiente...
            reconcile_outcome = reconcile_outcomes[0] #obtiene el resultado de esa reconciliación.
            outcomes.append(reconcile_outcome) #lo guarda
            _print_outcome(reconcile_outcome)
            if must_stop:
                break
            reconciled_stage_key = reconcile_outcome.key #Identifica qué etapa fue la que se acaba de reconciliar.
            reconciled_attempt_number = attempt_numbers.get(reconciled_stage_key, 1) #Obtiene su número de intento.
            new_stage, should_stop = _apply_stage_transition( #Interpreta qué pidió hacer esa etapa: ADVANCE RETRY RETURN HALT_STAGE STOP_PIPELINE  
                reconcile_outcome, store=store, stage_key=reconciled_stage_key,
                attempt_number=reconciled_attempt_number, attempt_numbers=attempt_numbers,
                until=until, outcomes=outcomes,
            )
            if should_stop:
                break
            current_stage = new_stage
            continue
        
       #Si no había ninguna ejecución pendiente, llega al flujo normal:
        attempt_number = attempt_numbers.get(current_stage, 1) #Obtiene el intento actual de la etapa.
        outcome = run_stage( #ejecuta la etapa actual o decide reutilizar/reanudar su resultado.
            store=store, #estado del pipeline
            project_dir=project_dir, #ruta del proyecto
            spec=spec, #StageSpec
            attempt_number=attempt_number, #número de intento
            observations=observations, #observaciones
            force_rerun=force_rerun_current, #force_rerun
        )
        force_rerun_current = False
        outcomes.append(outcome)
        _print_outcome(outcome)

        new_stage, should_stop = _apply_stage_transition( #interpreta qué transición pidió la etapa que acaba de ejecutarse. (04 → ADVANCE → 05,07 → RETURN → 06)
            outcome, store=store, stage_key=current_stage, attempt_number=attempt_number,
            attempt_numbers=attempt_numbers, until=until, outcomes=outcomes,
        )
        if should_stop:
            break
        current_stage = new_stage
    else:
        raise RuntimeError(
            "run_pipeline alcanzó max_iterations sin converger a un estado "
            "terminal; posible ciclo ADVANCE/RETURN entre etapas."
        )

    return outcomes


#mostrar en pantalla un resumen legible del resultado de una etapa.
def _print_outcome(outcome: StageOutcome) -> None:
    print(
        f"[{outcome.status:24s}] {outcome.label:45s} " #estado #nombre de la etapa
        f"execution={outcome.execution_status} quality={outcome.quality_status} " #estado técnico (complete) y de calidad (approved)
        f"next={outcome.next_action}->{outcome.target_stage}"
    )
    for warning in outcome.warnings:
        print(f"    warning: {warning}")
    if outcome.error:
        print(f"    error: {outcome.error}")



# ---------------------------------------------------------------------------
# CLI para uso directo en Colab: `python -m src.orchestration.pipeline_orchestrator`
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir", required=True, help="Ruta a PROJECT_DIR (contiene active_experiment.json)."
    )
    parser.add_argument(
        "--until",
        default=None,
        choices=STAGE_ORDER,
        help="Detenerse tras completar esta etapa (por defecto corre hasta 06).",
    )
    parser.add_argument(
        "--start-stage",
        default=None,
        choices=STAGE_ORDER,
        help=(
            "Empezar el recorrido directamente en esta etapa, en vez de "
            "STAGE_ORDER[0] -- ejecuta ÚNICAMENTE esta etapa y las que "
            "resulten de sus transiciones reales (nunca las anteriores). "
            "Con start-stage explícito, el chequeo de estado ya-terminal "
            "se omite deliberadamente (se respeta la petición explícita "
            "del llamador, igual que --force-rerun) -- si la etapa ya "
            "está COMPLETED y vigente (fingerprints sin cambios), sigue "
            "reconociéndose SKIPPED_FRESH con normalidad; si su último "
            "commit fue FAILED (ej. HALT_STAGE), esto la reintenta con "
            "un decision_id nuevo, SIN --force-rerun y sin tocar ninguna "
            "etapa previa."
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Reejecuta la etapa inicial aunque ya esté COMPLETED y vigente en pipeline_state.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    outcomes = run_pipeline(
        args.project_dir,
        start_stage=args.start_stage,
        until=args.until,
        force_rerun=args.force_rerun,
    )
    return 0 if all(o.status not in {"FAILED", "REACHED_UNREGISTERED_STAGE"} for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
