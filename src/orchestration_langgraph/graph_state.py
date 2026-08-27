"""Estado de coordinación de LangGraph para MAIN 5.

Deliberadamente mínimo: solo lo necesario para que el grafo sepa qué etapa
está corriendo, cuántos intentos lleva cada una, y hacia dónde ir después.
NUNCA copia artefactos científicos, fingerprints, ni el contenido de
``pipeline_state.json`` -- eso sigue viviendo exclusivamente en
``StateStore`` (ver ``src/orchestration/stage_execution.py``), consultado
bajo demanda por cada nodo, no duplicado aquí.
"""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

from src.orchestration.stage_execution import StageOutcome


class GraphState(TypedDict, total=False):
    project_dir: str
    # Etapa por la que entra el grafo en esta invocación (equivalente a
    # `start_stage` de `run_pipeline`) -- None/ausente = la primera del
    # orden canónico.
    entry_stage: str | None
    until: str | None
    force_rerun: bool
    attempt_numbers: dict[str, int]
    last_stage_outcome: StageOutcome | None
    outcomes: list[StageOutcome]
    observations: Mapping[str, Any] | None
    # Nombre del nodo destino (o "__end__"), ya resuelto por
    # `routing.resolve_transition()` dentro del propio nodo que acaba de
    # correr -- la función de edge condicional de LangGraph solo lo lee,
    # nunca vuelve a decidir ni a mutar StateStore (ver routing.py).
    route_target: str
