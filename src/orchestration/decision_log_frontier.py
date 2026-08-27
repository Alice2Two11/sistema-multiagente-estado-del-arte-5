"""Reconstrucción causal del ``decision_log`` real de un experimento --
compartida entre el orquestador (``pipeline_orchestrator.py``, para
reconocer un estado terminal ya comprometido en un restart) y los
adaptadores que resuelven dependencias upstream comprometidas (ej.
``agent06_verification_handoff.py``, para que 07 encuentre el ÚLTIMO
06 causalmente válido, no simplemente el más reciente cronológicamente
ni el ``StageState`` mutable vigente -- ambos pueden reflejar una
ejecución espuria posterior a un ``HALT_STAGE`` terminal ya
comprometido).

``decision_log`` es append-only y puede contener múltiples tramos/
epochs de ejecución acumulados durante todo un experimento real
(reintentos, restarts, invalidaciones, ciclos 06↔07 repetidos). Ni
``decision_log[-1]`` ni ``state.stages[etapa]`` (el ``StageState``
VIGENTE, que se sobrescribe con cada nuevo commit para esa etapa, sin
importar si es causalmente legítimo) son fuentes de verdad confiables
por sí solas para "qué fue lo último que realmente pasó" -- las
funciones de este módulo reconstruyen esa cadena causal real a partir
de las transiciones (``action``/``target_stage``) que cada decisión
realmente solicitó.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.contracts.agent_result import AgentResult, ExecutionStatus, TransitionAction
from src.orchestration.decision_engine import CANONICAL_STAGE_ORDER

if TYPE_CHECKING:
    from src.state.pipeline_state import DecisionLogEntry


def _causally_connects(predecessor: "DecisionLogEntry", entry: "DecisionLogEntry") -> bool:
    """True si ``entry`` es la continuación legítima de ``predecessor``
    según la transición REAL que ``predecessor`` solicitó -- el único
    criterio es semántico (acción + ``target_stage``/misma etapa), nunca
    el nombre de una etapa concreta ni el texto de ningún
    ``reason_code``:

        ADVANCE / RETURN -> ``entry.stage`` debe ser el
            ``target_stage`` de la transición de ``predecessor``.
        RETRY             -> ``entry.stage`` debe ser la MISMA etapa
            que ``predecessor`` (un reintento).
        HALT_STAGE / STOP_PIPELINE -> nunca hay continuación legítima
            (``target_stage`` siempre ``None``) -- cualquier ``entry``
            aquí se considera desconectado, sin importar cuál sea."""

    predecessor_transition = predecessor.requested_transition
    if predecessor_transition is None:
        return False
    if predecessor_transition.action in (TransitionAction.ADVANCE, TransitionAction.RETURN):
        return entry.stage == predecessor_transition.target_stage
    if predecessor_transition.action == TransitionAction.RETRY:
        return entry.stage == predecessor.stage
    return False


def _segment_decision_log(
    decision_log: tuple["DecisionLogEntry", ...],
) -> list[list["DecisionLogEntry"]]:
    """Divide ``decision_log`` en tramos MAXIMALES causalmente
    conectados, en orden cronológico. Un ``decision_log`` real puede
    contener múltiples tramos/epochs -- ejecuciones, reintentos,
    restarts e invalidaciones acumuladas durante todo un experimento --
    que no forman una única cadena continua desde la primera entrada.
    Cada tramo nuevo empieza donde la entrada actual NO es continuación
    legítima de la última entrada del tramo anterior
    (``_causally_connects`` devuelve ``False``)."""

    if not decision_log:
        return []

    segments: list[list["DecisionLogEntry"]] = [[decision_log[0]]]
    for entry in decision_log[1:]:
        current_segment = segments[-1]
        if _causally_connects(current_segment[-1], entry):
            current_segment.append(entry)
        else:
            segments.append([entry])
    return segments


def _reconstruct_authoritative_frontier(
    decision_log: tuple["DecisionLogEntry", ...],
    *,
    stage_order: tuple[str, ...] = CANONICAL_STAGE_ORDER,
) -> "DecisionLogEntry | None":
    """Reconstruye la ÚLTIMA decisión CAUSALMENTE conectada del
    ``decision_log`` -- ni ``decision_log[-1]`` (cronológicamente más
    reciente, pero puede ser una ejecución espuria) ni asumir que TODO
    ``decision_log`` es una única cadena desde ``decision_log[0]`` (un
    experimento real acumula múltiples tramos/epochs de ejecución).

    Corrección sobre la versión anterior (ver historial): caminar hacia
    atrás descartando EN CASCADA cualquier tramo cuyo predecesor haya
    terminado en HALT_STAGE/STOP_PIPELINE falla en un caso real
    confirmado con datos productivos: una secuencia como

        tramo A: ... -> 06/ADVANCE->07          (avanza hasta 07)
        tramo B: 06/HALT_STAGE                  (un fallo técnico real
                                                   de 06, ej. dependencia
                                                   de red -- NO de 07)
        tramo C: 06/ADVANCE -> 07/HALT_STAGE     (06 se reintentó, esta
                                                   vez avanzó de verdad
                                                   hasta 07 y ESE es el
                                                   HALT real)
        tramo D: 06/HALT_STAGE                   (espurio, posterior al
                                                   HALT real de C)
        tramo E: 06/HALT_STAGE                   (espurio también)

    La cascada ciega (D espurio -> se descarta -> C también se descarta
    porque SU predecesor B terminó en HALT -> ...) terminaba
    seleccionando el HALT de B (o D/E), NUNCA el de C -- que es el que
    realmente importa, porque C alcanzó una etapa MÁS AVANZADA
    (``07_agente_verificador``) que B, D y E (que nunca pasan de
    ``06_agente_redactor``).

    Algoritmo corregido: se camina el ``decision_log`` HACIA ATRÁS,
    tramo por tramo (``_segment_decision_log``), y se prefiere el tramo
    MÁS RECIENTE cuya última entrada sea TERMINAL
    (``HALT_STAGE``/``STOP_PIPELINE``) Y cuyo "alcance" -- la posición de
    su última etapa en ``stage_order``, el orden canónico REAL del
    pipeline, ya usado en otras partes de este mismo módulo para razonar
    sobre progreso (ver ``RETURN``, más abajo) -- sea ESTRICTAMENTE
    MAYOR que el de cualquier tramo terminal ya considerado (los
    posteriores, cronológicamente, en el recorrido hacia atrás). Un
    tramo terminal que NO alcanzó más lejos que uno ya visto (como D o
    E frente a C, o B frente a C) queda descartado -- no por su relación
    de adyacencia inmediata, sino porque no representa ningún avance
    real que debiera desplazar al tramo que sí llegó más lejos.

    El criterio sigue siendo puramente estructural/semántico: la
    posición de una etapa en ``stage_order`` (el orden canónico del
    pipeline, ya existente y usado en otras partes del código -- no un
    nombre de etapa concreto verificado por texto) y la acción/target de
    cada transición real. Ningún ``reason_code`` se inspecciona.

    Devuelve ``None`` si ``decision_log`` está vacío o si ningún tramo
    terminó en una transición terminal."""

    segments = _segment_decision_log(decision_log)
    if not segments:
        return None

    def _stage_reach(stage: str) -> int:
        return stage_order.index(stage) if stage in stage_order else -1

    best_entry: "DecisionLogEntry | None" = None
    best_reach = -1
    for segment in reversed(segments):
        last_entry = segment[-1]
        transition = last_entry.requested_transition
        is_terminal = transition is not None and transition.action in (
            TransitionAction.HALT_STAGE, TransitionAction.STOP_PIPELINE,
        )
        if not is_terminal:
            continue
        reach = _stage_reach(last_entry.stage)
        if reach > best_reach:
            best_entry = last_entry
            best_reach = reach

    return best_entry


def authoritative_decision_log_entry_for_stage(
    decision_log: tuple["DecisionLogEntry", ...], stage_name: str,
) -> "DecisionLogEntry | None":
    """Encuentra la ÚLTIMA entrada CAUSALMENTE VÁLIDA de ``stage_name``
    en ``decision_log`` -- ni ``state.stages[stage_name]`` (el
    ``StageState`` vigente, que una ejecución espuria posterior puede
    sobrescribir sin ser una continuación legítima) ni
    ``[e for e in decision_log if e.stage==stage_name][-1]`` (la última
    entrada CRONOLÓGICA de esa etapa, que puede ser esa misma ejecución
    espuria).

    Reutiliza ``_segment_decision_log``/``_reconstruct_authoritative_
    frontier`` (mismas funciones, mismo criterio, ya usado para
    reconocer un ``HALT_STAGE`` terminal en un restart): se determina
    primero el TRAMO autoritativo (el que contiene el frontier terminal
    real, si existe; si el log no terminó en nada terminal, el tramo
    más reciente, porque el pipeline sigue "en vuelo") y, dentro de ese
    tramo -- y de todo lo que llevó causalmente hasta él -- se busca la
    última entrada de ``stage_name``.

    Ejemplo real que motivó esta función: ``decision_log`` con
    ``06/ADVANCE->07`` (causalmente válido) seguido de ``07/HALT_STAGE``
    (terminal) seguido de un ``06/HALT_STAGE`` ESPURIO (posterior al
    terminal, sin ninguna transición que lo señalara). El ``StageState``
    vigente de 06 y la última entrada cronológica de 06 en el log
    reflejan ese espurio (``FAILED``) -- pero el 06 que REALMENTE
    condujo a 07 (el único con el que 07 debe poder trabajar en un
    retry explícito) es el de la entrada ``ADVANCE->07``, que esta
    función sí encuentra.

    Devuelve ``None`` si ``stage_name`` nunca aparece en ninguna parte
    causalmente válida del log."""

    segments = _segment_decision_log(decision_log)
    if not segments:
        return None

    frontier_entry = _reconstruct_authoritative_frontier(decision_log)
    if frontier_entry is not None:
        authoritative_segments = segments
        for index, segment in enumerate(segments):
            if frontier_entry in segment:
                authoritative_segments = segments[: index + 1]
                break
    else:
        # Sin ningún terminal en el log -- el pipeline sigue "en vuelo";
        # el tramo causalmente conectado más reciente es el autoritativo.
        authoritative_segments = segments

    for segment in reversed(authoritative_segments):
        for entry in reversed(segment):
            if entry.stage == stage_name:
                return entry
    return None


def committed_predecessor_for_stage(
    decision_log: tuple["DecisionLogEntry", ...], *, predecessor: str, target: str,
) -> "DecisionLogEntry | None":
    """Encuentra el ÚLTIMO commit de ``predecessor`` que CAUSALMENTE
    habilitó una continuación hacia ``target`` -- una pregunta distinta
    de la que responde ``authoritative_decision_log_entry_for_stage``.

    Esa otra función responde "¿cuál es el estado terminal VIGENTE de
    esta etapa, para reconocer un restart?" -- correcta para eso, pero
    equivocada para resolver una dependencia upstream: el estado
    "vigente" de una etapa puede ser una ejecución espuria posterior
    (ej. un ``06/HALT_STAGE`` que apareció después de que 07 ya hubiera
    llegado a su propio ``HALT_STAGE`` terminal, sin que ninguna
    transición señalara esa reejecución de 06). Ese espurio nunca fue
    "el commit que habilitó a 07" -- 07 ya se había ejecutado, con OTRO
    commit de 06, ANTES de que ese espurio existiera.

    Esta función responde la pregunta correcta para un hand-off
    upstream: "¿cuál fue el último commit de ``predecessor`` que
    REALMENTE avanzó hacia ``target``?" -- exige, sobre la MISMA
    entrada:

        stage == predecessor
        execution_status == COMPLETED
        requested_transition.action == ADVANCE
        requested_transition.target_stage == target

    y además exige que esa entrada pertenezca a un tramo causalmente
    válido (ver ``_segment_decision_log``) -- una entrada que en el
    papel cumple los tres requisitos de arriba pero que en sí misma es
    huérfana (nunca debió ejecutarse) no cuenta. Se camina
    ``decision_log`` hacia atrás: si ``predecessor`` avanzó hacia
    ``target`` más de una vez a lo largo del experimento (ej. tras un
    ciclo RETURN legítimo), importa la ocurrencia más reciente, no la
    primera.

    Devuelve ``None`` si ninguna entrada cumple todo lo anterior."""

    segments = _segment_decision_log(decision_log)
    valid_entry_ids = {id(entry) for segment in segments for entry in segment}

    for entry in reversed(decision_log):
        if id(entry) not in valid_entry_ids:
            continue
        if entry.stage != predecessor:
            continue
        # El campo requested_transition de nivel superior puede venir en
        # None en algunos constructores de DecisionLogEntry que no pasan
        # por StateStore.commit_execution (que sí lo pobla siempre desde
        # result.requested_transition) -- se usa como fuente primaria,
        # pero si falta, se cae al mismo dato dentro de entry.result
        # (la copia real y completa del AgentResult comprometido), que
        # es la fuente de verdad definitiva en cualquier caso.
        transition = entry.requested_transition
        if transition is not None:
            action = transition.action
            target_stage = transition.target_stage
        else:
            raw_transition = entry.result.get("requested_transition") or {}
            action_raw = raw_transition.get("action")
            action = TransitionAction(action_raw) if action_raw is not None else None
            target_stage = raw_transition.get("target_stage")
        if action != TransitionAction.ADVANCE or target_stage != target:
            continue
        if AgentResult.from_dict(entry.result).execution_status != ExecutionStatus.COMPLETED:
            continue
        return entry

    return None
