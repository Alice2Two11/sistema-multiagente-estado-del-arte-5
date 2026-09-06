"""
Agentic Retrieval de la Etapa 07 -- Bloque 2
Controla el ciclo de recuperación adicional de evidencia antes de verificar el claim.

Flujo general:

    claim
      -> RETRIEVE
         Recupera evidencia usando inicialmente el texto del claim
         como consulta y el valor top_k_initial.

      -> GRADE_EVIDENCE
         Evalúa automáticamente si la evidencia recuperada es suficiente
         para respaldar el claim.

      -> ¿La evidencia es suficiente?
           sí -> ACCEPT_EVIDENCE
                 El controller acepta automáticamente la evidencia.
                 En este caso el planner no interviene.

           no -> ¿Se agotó el presupuesto de recuperación?

                    sí -> ¿La evidencia disponible es mínimamente viable?
                              sí -> ACCEPT_EVIDENCE
                                    Se acepta lo mejor que se consiguió.

                              no -> FINISH_UNRESOLVED
                                    El claim queda sin evidencia suficiente.

                    no -> el planner decide cómo mejorar la recuperación:
                           - REWRITE_QUERY:
                             reformula la consulta para buscar evidencia
                             desde otra formulación del claim.

                           - ADJUST_TOP_K:
                             modifica la cantidad de resultados recuperados.

                           - ACCEPT_EVIDENCE:
                             solo puede elegirse después de haber intentado
                             mejorar la recuperación y únicamente si la
                             evidencia disponible es mínimamente viable.

      -> se ejecuta la acción elegida
      -> se realiza una nueva recuperación
      -> se vuelve a evaluar la evidencia
      -> el ciclo continúa hasta aceptar evidencia o finalizar sin resolver.


Decisión del planner:

El planner solo interviene cuando la evidencia es insuficiente y todavía
queda presupuesto para realizar nuevas recuperaciones.

En la primera insuficiencia debe intentar mejorar la búsqueda, por lo que
solo puede elegir entre:

    REWRITE_QUERY
    ADJUST_TOP_K

No puede aceptar inmediatamente la evidencia recuperada.

Después de al menos un intento de mejora, también puede elegir
ACCEPT_EVIDENCE si lo recuperado alcanza el mínimo necesario para continuar.

Las acciones RETRIEVE y GRADE_EVIDENCE no son decisiones del planner:
el controller las ejecuta obligatoriamente como parte del ciclo.


Prompt y parseo:

Este módulo no reutiliza react_prompting.py porque dicho componente está
acoplado al dominio de post-verificación y valida acciones diferentes.

Por eso define su propio prompt y parser mínimo. El planner debe responder
con una estructura JSON controlada, sin razonamientos libres. Si la salida
es inválida o no cumple el contrato esperado, el sistema falla de forma
segura en lugar de ejecutar una acción no válida.


Presupuesto:

El ciclo utiliza remaining_retrieval_budget como único contador de
recuperaciones adicionales disponibles.

Este valor deriva de max_additional_retrieval_requests y es compartido con
la lógica de verificación del claim. No se crea un contador adicional para
Agentic Retrieval, evitando límites redundantes o inconsistentes.


Alcance del bloque:

Este módulo implementa únicamente el controller del ciclo de Agentic
Retrieval y sus pruebas.

No modifica:
    - verification_runtime.py
    - verification_agent.py
    - el retriever
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from src.config.agentic_retrieval_policy_config import GRADE_REASON_CODES

# ---------------------------------------------------------------------------
# Acciones que el planner puede elegir durante Agentic Retrieval.
# Solo puede seleccionar entre reformular la consulta, ajustar el top_k
# o aceptar la evidencia disponible cuando esta ya sea mínimamente viable.
#
# FINISH_UNRESOLVED no es una decisión del planner: el controller la aplica
# automáticamente cuando la evidencia sigue siendo insuficiente y ya no queda
# presupuesto para realizar nuevas recuperaciones.
# ---------------------------------------------------------------------------
AGENTIC_RETRIEVAL_ACTIONS = ("REWRITE_QUERY", "ADJUST_TOP_K", "ACCEPT_EVIDENCE")
FINISH_UNRESOLVED = "FINISH_UNRESOLVED"
AGENTIC_PLANNER_FAILED = "AGENTIC_PLANNER_FAILED"

# decision_basis indica la razón concreta en la que se basa la decisión del planner.
# Solo admite motivos predefinidos por el grader: poca cantidad de candidatos,
# baja diversidad de fuentes, baja relevancia, cobertura insuficiente o aceptación
# deliberada de evidencia que, aunque tenga vacíos, sigue siendo mínimamente viable.
# No permite explicaciones libres ni razonamientos abiertos del LLM.
AGENTIC_DECISION_BASIS_VALUES = (
    "EVIDENCE_INSUFFICIENT_LOW_CANDIDATE_COUNT",
    "EVIDENCE_INSUFFICIENT_LOW_SOURCE_DIVERSITY",
    "EVIDENCE_INSUFFICIENT_LOW_RELEVANCE",
    "EVIDENCE_INSUFFICIENT_LOW_COVERAGE",
    "EVIDENCE_ACCEPTABLE_DESPITE_GAPS",
)

# Valida que la acción elegida por el planner pertenezca al conjunto de
# acciones permitidas para Agentic Retrieval. Si recibe una acción no válida,
# detiene el proceso con un error; si es válida, devuelve la misma acción.
def validate_selected_action(value: str) -> str:
    if value not in AGENTIC_RETRIEVAL_ACTIONS:
        raise ValueError(
            f"selected_action debe ser una de {AGENTIC_RETRIEVAL_ACTIONS}, recibido {value!r}."
        )
    return value


def validate_decision_basis(value: str) -> str:
    if value not in AGENTIC_DECISION_BASIS_VALUES:
        raise ValueError(
            f"decision_basis debe ser uno de {AGENTIC_DECISION_BASIS_VALUES}, recibido {value!r}."
        )
    return value



# Representa el estado observable del ciclo de Agentic Retrieval para un claim
# en un instante concreto. Todos sus valores provienen de resultados reales
# de recuperación y evaluación de evidencia, no de estados inferidos o inventados.

@dataclass(frozen=True)
class AgenticRetrievalObservation:
    claim_id: str
    claim_text: str
    current_query: str
    retrieval_round: int  # 0 = resultado de la recuperación inicial
    current_top_k: int
    effective_top_k_max: int
    remaining_retrieval_budget: int
    candidate_count: int
    evidence_ids: tuple[str, ...]
    max_relevance_score: float
    grade_result: str  # "SUFFICIENT" | "INSUFFICIENT"
    reason_codes: tuple[str, ...]
    minimum_viable_evidence: bool
    query_rewrite_count: int

    # Valida que todos los campos que deben representar cantidades sean enteros reales.
    # La comprobación excluye explícitamente los valores booleanos porque en Python
    # bool hereda de int y, sin esta validación, True o False podrían aceptarse
    # incorrectamente como valores numéricos.
    def __post_init__(self) -> None:
        # --- Tipos estrictos: bool es subclase de int en Python ---
        for int_field_name, int_field_value in (
            ("retrieval_round", self.retrieval_round),
            ("current_top_k", self.current_top_k),
            ("effective_top_k_max", self.effective_top_k_max),
            ("remaining_retrieval_budget", self.remaining_retrieval_budget),
            ("candidate_count", self.candidate_count),
            ("query_rewrite_count", self.query_rewrite_count),
        ):
            if isinstance(int_field_value, bool) or not isinstance(int_field_value, int):
                raise TypeError(
                    f"{int_field_name} debe ser int, recibido "
                    f"{type(int_field_value).__name__} ({int_field_value!r})."
                )

        # Valida que los valores numéricos del estado sean coherentes con las reglas del ciclo
        # de recuperación y que no representen estados imposibles del controller.
        if self.retrieval_round < 0: # no pueden ser negativos, porque representan contadores acumulados.
            raise ValueError(f"retrieval_round debe ser >= 0, recibido {self.retrieval_round!r}")
        if self.remaining_retrieval_budget < 0: # no pueden ser negativos, porque representan contadores acumulados.
            raise ValueError(
                f"remaining_retrieval_budget debe ser >= 0, recibido {self.remaining_retrieval_budget!r}"
            )
        if self.candidate_count < 0: # no pueden ser negativos, porque representan contadores acumulados.
            raise ValueError(f"candidate_count debe ser >= 0, recibido {self.candidate_count!r}")
        if self.current_top_k <= 0: # no pueden ser negativos, porque representan contadores acumulados.
            raise ValueError(f"current_top_k debe ser > 0, recibido {self.current_top_k!r}")
        if self.effective_top_k_max <= 0: # no pueden ser negativos, porque representan contadores acumulados.
            raise ValueError(f"effective_top_k_max debe ser > 0, recibido {self.effective_top_k_max!r}")
        if self.query_rewrite_count < 0: # no pueden ser negativos, porque representan contadores acumulados.
            raise ValueError(f"query_rewrite_count debe ser >= 0, recibido {self.query_rewrite_count!r}")
        if self.current_top_k > self.effective_top_k_max: # nunca puede superar el máximo permitido para la recuperación.
            raise ValueError(
                f"current_top_k ({self.current_top_k}) no puede exceder "
                f"effective_top_k_max ({self.effective_top_k_max})."
            )
        if self.query_rewrite_count > self.retrieval_round:
            raise ValueError(
                f"query_rewrite_count ({self.query_rewrite_count}) no puede exceder "
                f"retrieval_round ({self.retrieval_round}) -- cada REWRITE_QUERY incrementa "
                "ambos en 1, cada ADJUST_TOP_K solo incrementa retrieval_round; "
                "query_rewrite_count > retrieval_round es un estado imposible según las "
                "transiciones del controller."
            )
        if self.candidate_count > self.current_top_k:
            raise ValueError(
                f"candidate_count ({self.candidate_count}) no puede exceder current_top_k "
                f"({self.current_top_k}) -- confirmado contra Agent07ChromaRetriever."
                "retrieve_more (verification_incremental_retriever.py): "
                "'if len(selected) >= self.top_k: break' trunca siempre el conjunto que "
                "alimenta grade_evidence a top_k. No confundir con effective_top_k_max/"
                "fetch_k, que acotan la petición bruta a Chroma, no el pool materializado."
            )

        if (
            not isinstance(self.max_relevance_score, (int, float))
            or isinstance(self.max_relevance_score, bool)
            or not (0.0 <= float(self.max_relevance_score) <= 1.0)
        ):
            raise ValueError(
                f"max_relevance_score debe estar en [0.0, 1.0], recibido {self.max_relevance_score!r}."
            )

        # Valida que claim_id, claim_text y current_query sean cadenas de texto reales
        # y no valores de otros tipos convertibles implícitamente a str, como bool, int,
        # float, None o listas.
        #
        # También comprueba que ninguna de estas cadenas esté vacía o contenga solo espacios.
        # Esta validación es especialmente importante para current_query, porque posteriormente
        # puede ser modificada por la acción REWRITE_QUERY y debe mantenerse siempre como texto válido.
        for str_field_name, str_field_value in (
            ("claim_id", self.claim_id),
            ("claim_text", self.claim_text),
            ("current_query", self.current_query),
        ):
            if not isinstance(str_field_value, str):
                raise TypeError(
                    f"{str_field_name} debe ser str real, recibido "
                    f"{type(str_field_value).__name__} ({str_field_value!r})."
                )
            if not str_field_value.strip():
                raise ValueError(f"{str_field_name} no puede estar vacío.")

        # Comprueba que, mientras no se haya ejecutado ninguna acción REWRITE_QUERY,
        # la consulta utilizada siga siendo exactamente el texto original del claim.
        # Las otras transiciones válidas, como la recuperación inicial o ADJUST_TOP_K,
        # pueden cambiar la ronda o la cantidad de resultados recuperados, pero no modifican
        # el contenido de current_query.
        if self.query_rewrite_count == 0 and self.current_query != self.claim_text:
            raise ValueError(
                "query_rewrite_count=0 implica current_query == claim_text -- "
                f"recibido current_query={self.current_query!r} distinto de "
                f"claim_text={self.claim_text!r}. Ninguna transición válida puede "
                "producir este estado sin haber pasado por REWRITE_QUERY."
            )

        # Valida que grade_result solo pueda ser SUFFICIENT o INSUFFICIENT.
        #
        # También comprueba que reason_codes sea una tupla de códigos válidos producidos
        # por el grader. Cada elemento debe ser una cadena incluida en GRADE_REASON_CODES,
        # reutilizando así el mismo vocabulario de motivos definido en el Bloque 1.
        #
        # Finalmente, verifica que no existan códigos repetidos, porque cada motivo de
        # insuficiencia debe registrarse como máximo una vez.
        if self.grade_result not in ("SUFFICIENT", "INSUFFICIENT"):
            raise ValueError(f"grade_result inválido: {self.grade_result!r}")
        if not isinstance(self.reason_codes, tuple):
            raise TypeError(
                f"reason_codes debe ser tuple real, recibido {type(self.reason_codes).__name__} "
                f"({self.reason_codes!r})."
            )
        for code in self.reason_codes:
            if not isinstance(code, str):
                raise TypeError(
                    f"reason_codes contiene un elemento no-str: {code!r} ({type(code).__name__})."
                )
            if code not in GRADE_REASON_CODES:
                raise ValueError(
                    f"reason_codes contiene {code!r}, fuera de GRADE_REASON_CODES {GRADE_REASON_CODES} "
                    "(Bloque 1) -- se reutiliza el vocabulario del grader, no se duplica."
                )
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError(
                f"reason_codes contiene duplicados: {self.reason_codes!r} -- cada reason_code "
                "debe aparecer como máximo una vez."
            )


        # Comprueba que el resultado del grader sea coherente con los motivos registrados:
        # si la evidencia es SUFFICIENT, no debe existir ningún reason_code;
        # si es INSUFFICIENT, debe existir al menos un motivo que explique la insuficiencia.
        if self.grade_result == "SUFFICIENT" and self.reason_codes:
            raise ValueError(
                "grade_result=SUFFICIENT debe tener reason_codes vacío -- "
                f"recibido {self.reason_codes!r}."
            )
        if self.grade_result == "INSUFFICIENT" and not self.reason_codes:
            raise ValueError(
                "grade_result=INSUFFICIENT debe tener al menos un reason_code -- "
                "recibido reason_codes vacío."
            )
        # valida que minimum_viable_evidence sea un booleano real.
        # Este campo controla si una evidencia insuficiente pero todavía mínimamente útil
        # puede ser aceptada cuando se agota el presupuesto de recuperación, por lo que
        # valores de otros tipos podrían provocar una aceptación incorrecta.
        if not isinstance(self.minimum_viable_evidence, bool):
            raise TypeError(
                f"minimum_viable_evidence debe ser bool real, recibido "
                f"{type(self.minimum_viable_evidence).__name__} ({self.minimum_viable_evidence!r}) -- "
                "un str/int truthy podría producir ACCEPT_EVIDENCE indebidamente al agotarse "
                "el presupuesto."
            )

        # Comprueba la coherencia entre el resultado del grader y el criterio de evidencia
        # mínimamente viable. Como SUFFICIENT exige condiciones más estrictas que
        # minimum_viable_evidence, toda evidencia clasificada como SUFFICIENT debe cumplir
        # necesariamente también el mínimo requerido para ser considerada viable.
        if self.grade_result == "SUFFICIENT" and self.minimum_viable_evidence is not True:
            raise ValueError(
                "grade_result=SUFFICIENT implica minimum_viable_evidence=True -- "
                f"recibido minimum_viable_evidence={self.minimum_viable_evidence!r}. "
                "SUFFICIENT es un criterio más estricto que minimum_viable_evidence por "
                "diseño; esta combinación es un estado imposible."
            )

        # Endurecimiento del contrato de evidence_ids -- será esencial para
        # trazabilidad en Bloque 3: tuple real (no list), cada elemento str
        # real no vacío, todos únicos.
        if not isinstance(self.evidence_ids, tuple):
            raise TypeError(
                f"evidence_ids debe ser tuple real, recibido {type(self.evidence_ids).__name__} "
                f"({self.evidence_ids!r})."
            )
        for evidence_id in self.evidence_ids:
            if not isinstance(evidence_id, str):
                raise TypeError(
                    f"evidence_ids contiene un elemento no-str: {evidence_id!r} "
                    f"({type(evidence_id).__name__})."
                )
            if not evidence_id.strip():
                raise ValueError("evidence_ids no puede contener IDs vacíos.")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError(
                f"evidence_ids contiene IDs duplicados: {self.evidence_ids!r} -- "
                "todos los IDs deben ser únicos."
            )

        # evidence_ids representa TODOS los candidatos materializados en
        # esta ronda (mismo conjunto que candidate_count cuenta en
        # grade_evidence, Bloque 1) -- no un subconjunto. Ambos deben
        # coincidir en cardinalidad.
        if self.candidate_count != len(self.evidence_ids):
            raise ValueError(
                f"candidate_count ({self.candidate_count}) debe ser igual a "
                f"len(evidence_ids) ({len(self.evidence_ids)}) -- evidence_ids representa "
                "todos los candidatos materializados en esta ronda, el mismo conjunto que "
                "candidate_count cuenta (Bloque 1, grade_evidence)."
            )

        # minimum_viable_evidence=True requiere evidencia real -- no puede
        # afirmarse viabilidad sobre un conjunto vacío. Por consecuencia
        # (SUFFICIENT ya implica minimum_viable_evidence=True, arriba),
        # SUFFICIENT también implica candidate_count >= 1 -- ambos chequeos
        # explícitos para fail-closed directo y mensajes claros.
        if self.minimum_viable_evidence and self.candidate_count < 1:
            raise ValueError(
                "minimum_viable_evidence=True requiere candidate_count >= 1 -- "
                f"recibido candidate_count={self.candidate_count!r}. No puede haber "
                "evidencia mínimamente viable sobre un conjunto vacío."
            )
        if self.grade_result == "SUFFICIENT" and self.candidate_count < 1:
            raise ValueError(
                "grade_result=SUFFICIENT requiere candidate_count >= 1 -- "
                f"recibido candidate_count={self.candidate_count!r}."
            )

        # candidate_count/evidence_ids/max_relevance_score describen el
        # mismo resultado materializado del retrieval -- confirmado contra
        # el grader real de Bloque 1 (grade_evidence: max_relevance_score
        # = max(scores) if scores else 0.0): si no hay candidatos, el
        # score máximo es 0.0, nunca positivo. No se impone la inversa
        # (max_relevance_score=0.0 no implica candidate_count=0 --
        # candidatos con relevancia 0 son posibles).
        if self.candidate_count == 0 and self.max_relevance_score != 0.0:
            raise ValueError(
                f"candidate_count=0 requiere max_relevance_score=0.0 -- recibido "
                f"{self.max_relevance_score!r}. Sin candidatos no puede existir un score "
                "máximo positivo procedente de ellos (Bloque 1, grade_evidence)."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "current_query": self.current_query,
            "retrieval_round": self.retrieval_round,
            "current_top_k": self.current_top_k,
            "effective_top_k_max": self.effective_top_k_max,
            "remaining_retrieval_budget": self.remaining_retrieval_budget,
            "candidate_count": self.candidate_count,
            "evidence_ids": list(self.evidence_ids),
            "max_relevance_score": self.max_relevance_score,
            "grade_result": self.grade_result,
            "reason_codes": list(self.reason_codes),
            "minimum_viable_evidence": self.minimum_viable_evidence,
            "query_rewrite_count": self.query_rewrite_count,
        }


# ---------------------------------------------------------------------------
# Gate de decisión 
# ---------------------------------------------------------------------------

# Determina si el controller puede resolver el estado actual sin consultar al planner.
#
# Si la evidencia ya fue calificada como SUFFICIENT, la acepta automáticamente.
# Si la evidencia sigue siendo INSUFFICIENT pero ya no queda presupuesto para
# nuevas recuperaciones, el controller toma la decisión final:
# acepta la evidencia si todavía es mínimamente viable o finaliza el claim
# como no resuelto si no alcanza ese mínimo.
#
# Solo devuelve None cuando la evidencia es insuficiente y aún queda presupuesto,
# porque en ese caso sí debe intervenir el planner para decidir cómo mejorar la búsqueda.
def determine_forced_outcome(observation: AgenticRetrievalObservation) -> str | None:
    """Casos que el controller resuelve SIN consultar al planner:
    - SUFFICIENT: ACCEPT_EVIDENCE automático (no hay decisión que tomar).
    - Presupuesto agotado (INSUFFICIENT): ACCEPT_EVIDENCE si
      minimum_viable_evidence, FINISH_UNRESOLVED si no -- NUNCA
      convertido en una elección del planner."""
    if observation.grade_result == "SUFFICIENT":
        return "ACCEPT_EVIDENCE"
    if observation.remaining_retrieval_budget <= 0:
        return "ACCEPT_EVIDENCE" if observation.minimum_viable_evidence else FINISH_UNRESOLVED
    return None

# Define qué acciones puede elegir el planner cuando la evidencia sigue siendo
# insuficiente, todavía queda presupuesto y no existe un resultado forzado.
#
# ADJUST_TOP_K solo se permite si aún es posible aumentar la cantidad de resultados
# recuperados sin superar effective_top_k_max.
#
# REWRITE_QUERY permanece disponible mientras exista presupuesto, porque reformular
# la consulta es otra forma válida de intentar encontrar mejor evidencia.
#
# ACCEPT_EVIDENCE no puede elegirse en la primera insuficiencia: el sistema obliga
# primero a intentar mejorar la recuperación. Solo se habilita a partir de la segunda
# evaluación, cuando ya hubo al menos un intento de mejora y la evidencia disponible
# sigue siendo mínimamente viable.
#
# Si la evidencia ya es SUFFICIENT o el presupuesto se agotó, devuelve una tupla vacía,
# porque esos casos los resuelve directamente determine_forced_outcome().
def compute_allowed_actions(observation: AgenticRetrievalObservation) -> tuple[str, ...]:
    if observation.grade_result == "SUFFICIENT" or observation.remaining_retrieval_budget <= 0:
        return ()
        
    actions: list[str] = []
    if observation.current_top_k < observation.effective_top_k_max:
        actions.append("ADJUST_TOP_K")
    actions.append("REWRITE_QUERY")

    if observation.retrieval_round >= 1 and observation.minimum_viable_evidence:
        actions.append("ACCEPT_EVIDENCE")
    return tuple(actions)


# ---------------------------------------------------------------------------
# Planner -- prompt mínimo + parseo fail-closed, autocontenido.
# ---------------------------------------------------------------------------

# Define el error que se lanza cuando la respuesta del planner no puede convertirse
# en una decisión válida, incluso después de intentar corregir o volver a interpretar
# su salida según el formato esperado.
class AgenticPlannerResponseError(ValueError):
    """La respuesta del planner no pudo interpretarse como una decisión
    válida tras agotar los reintentos de parseo."""
MAX_PLANNER_PARSE_RETRIES = 2

# Construye el prompt que se enviará al planner para que elija una única acción
# válida de Agentic Retrieval a partir del estado actual del claim.
def build_agentic_planner_prompt(
    *, observation: AgenticRetrievalObservation, allowed_actions: tuple[str, ...]
) -> str:
    if not allowed_actions:
        raise ValueError(
            "build_agentic_planner_prompt requiere allowed_actions no vacío -- "
            "si no hay ninguna acción autorizada, el controller no debe invocar "
            "al planner (cierre determinista vía determine_forced_outcome)."
        )
    for action in allowed_actions:
        validate_selected_action(action)
    return (
        "Eres el planner de recuperación agentic para UN claim científico, "
        "Stage 07 (pre-verificación).\n"
        "Debes elegir EXACTAMENTE UNA acción de la lista permitida. No "
        "expliques tu razonamiento en texto libre.\n\n"
        f"OBSERVATION:\n{json.dumps(observation.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        f"ACCIONES PERMITIDAS (elige solo una, ninguna otra es válida):\n"
        f"{json.dumps(list(allowed_actions), ensure_ascii=False)}\n\n"
        f"VALORES VÁLIDOS DE decision_basis:\n"
        f"{json.dumps(list(AGENTIC_DECISION_BASIS_VALUES), ensure_ascii=False)}\n\n"
        "Responde ÚNICAMENTE con un objeto JSON de exactamente estas dos claves, "
        "sin texto adicional, sin markdown:\n"
        '{"selected_action": "<una de las acciones permitidas>", '
        '"decision_basis": "<uno de los valores válidos>"}'
    )

# Valida que la acción elegida por el planner sea coherente con la razón usada
# para justificarla, no solo que ambos valores pertenezcan a sus enums permitidos.
def _validate_action_decision_basis_coherence(
    *, selected_action: str, decision_basis: str, observation_reason_codes: tuple[str, ...]
) -> None:
    """Coherencia obligatoria entre la acción elegida y su justificación
    -- ambos enums ya se validaron por separado, esto valida la
    COMBINACIÓN:
    - ACCEPT_EVIDENCE únicamente con EVIDENCE_ACCEPTABLE_DESPITE_GAPS.
    - REWRITE_QUERY/ADJUST_TOP_K (acciones de mejora) únicamente con un
      decision_basis cuyo reason code correspondiente esté REALMENTE
      presente en observation_reason_codes -- no basta con que el
      decision_basis pertenezca al enum general."""
    if selected_action == "ACCEPT_EVIDENCE":
        if decision_basis != "EVIDENCE_ACCEPTABLE_DESPITE_GAPS":
            raise AgenticPlannerResponseError(
                f"PLANNER_RESPONSE_INCOHERENT_DECISION_BASIS: ACCEPT_EVIDENCE solo admite "
                f"decision_basis='EVIDENCE_ACCEPTABLE_DESPITE_GAPS', recibido {decision_basis!r}."
            )
        return

    # Valida la coherencia de las acciones de mejora REWRITE_QUERY y ADJUST_TOP_K
    # REWRITE_QUERY / ADJUST_TOP_K (acciones de mejora)
    if decision_basis == "EVIDENCE_ACCEPTABLE_DESPITE_GAPS":
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_INCOHERENT_DECISION_BASIS: {selected_action!r} no puede "
            "justificarse con 'EVIDENCE_ACCEPTABLE_DESPITE_GAPS' -- esa justificación es "
            "exclusiva de ACCEPT_EVIDENCE."
        )
    corresponding_reason_code = decision_basis.removeprefix("EVIDENCE_INSUFFICIENT_")
    if corresponding_reason_code not in observation_reason_codes:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_INCOHERENT_DECISION_BASIS: decision_basis={decision_basis!r} "
            f"implica reason_code={corresponding_reason_code!r}, que no está presente en "
            f"la Observation actual (reason_codes={observation_reason_codes!r})."
        )


# Interpreta y valida la respuesta devuelta por el planner antes de aceptar su decisión.
# Primero exige que la salida sea únicamente un objeto JSON válido, sin texto,
# markdown ni contenido adicional.
# Después comprueba que el JSON contenga exactamente dos campos:
# selected_action y decision_basis, sin claves extra como rationale.
# Luego valida que:
# - selected_action sea una acción reconocida por Agentic Retrieval;
# - esa acción esté permitida específicamente en el estado actual del ciclo;
# - decision_basis pertenezca al conjunto cerrado de justificaciones válidas.
# Finalmente verifica que la combinación entre acción y justificación sea coherente
# con los reason_codes realmente observados en la evidencia actual.
def parse_agentic_planner_response(
    raw_text: str, *, allowed_actions: tuple[str, ...], observation_reason_codes: tuple[str, ...] = (),
) -> dict[str, str]:
    text = (raw_text or "").strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise AgenticPlannerResponseError(
            "PLANNER_RESPONSE_NOT_PURE_JSON_OBJECT: debe ser exactamente un objeto JSON."
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgenticPlannerResponseError(f"PLANNER_RESPONSE_INVALID_JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise AgenticPlannerResponseError("PLANNER_RESPONSE_ROOT_NOT_OBJECT")
    expected_keys = {"selected_action", "decision_basis"}
    if set(payload.keys()) != expected_keys:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_UNEXPECTED_KEYS: se esperaban exactamente "
            f"{sorted(expected_keys)}, se recibió {sorted(payload.keys())} -- "
            "no se acepta 'rationale' ni ningún otro campo."
        )
    selected_action = payload["selected_action"]
    decision_basis = payload["decision_basis"]
    if not isinstance(selected_action, str) or selected_action not in AGENTIC_RETRIEVAL_ACTIONS:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_INVALID_SELECTED_ACTION: {selected_action!r} "
            "no pertenece a AGENTIC_RETRIEVAL_ACTIONS."
        )
    if selected_action not in allowed_actions:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_ACTION_NOT_ALLOWED: {selected_action!r} no está "
            f"en el conjunto autorizado {allowed_actions!r}."
        )
    if not isinstance(decision_basis, str) or decision_basis not in AGENTIC_DECISION_BASIS_VALUES:
        raise AgenticPlannerResponseError(
            f"PLANNER_RESPONSE_INVALID_DECISION_BASIS: {decision_basis!r} "
            "no pertenece a AGENTIC_DECISION_BASIS_VALUES."
        )
    _validate_action_decision_basis_coherence(
        selected_action=selected_action, decision_basis=decision_basis,
        observation_reason_codes=observation_reason_codes,
    )
    return {"selected_action": selected_action, "decision_basis": decision_basis}

# Invoca al planner y valida su respuesta, permitiendo varios intentos cuando
# la salida no cumple el formato o las reglas esperadas.
#
# En cada intento:
# - llama al planner con el mismo prompt;
# - intenta parsear y validar la respuesta;
# - si la respuesta es válida, la devuelve inmediatamente;
# - si es inválida, guarda el error y vuelve a intentarlo.
#
# max_retries indica cuántos reintentos adicionales se permiten después del
# primer intento, por lo que attempts = max_retries + 1.

def invoke_agentic_planner_with_retry(
    *,
    invoke_fn: Callable[[str], str],
    prompt: str,
    allowed_actions: tuple[str, ...],
    observation_reason_codes: tuple[str, ...] = (),
    max_retries: int = MAX_PLANNER_PARSE_RETRIES,
) -> dict[str, str]:
    last_error: Exception | None = None
    attempts = max(1, max_retries + 1)
    for _ in range(attempts):
        raw = invoke_fn(prompt)
        try:
            return parse_agentic_planner_response(
                raw, allowed_actions=allowed_actions, observation_reason_codes=observation_reason_codes,
            )
        except AgenticPlannerResponseError as exc:
            last_error = exc
            continue
    raise AgenticPlannerResponseError(
        f"PLANNER_RESPONSE_RETRIES_EXHAUSTED tras {attempts} intentos: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Validación de la transición producida por execute_action_fn -- el
# presupuesto compartido es la garantía de terminación del ciclo, así
# que no basta con confiar en que la tool devuelva algo razonable.
# ---------------------------------------------------------------------------

AGENTIC_TRANSITION_INVALID = "AGENTIC_TRANSITION_INVALID"

# Excepción para acciones válidas que no pueden materializarse.
# Por ejemplo, REWRITE_QUERY puede estar permitido por el controller, pero la
# herramienta puede no encontrar términos nuevos suficientes para generar una
# reformulación real de la consulta.
class AgenticRetrievalActionUnavailable(Exception):
    """E2E-BUG-01 (contract fix): excepción tipada de integración -- la
    acción seleccionada era legal según ``compute_allowed_actions`` para
    la Observation actual, pero no puede ejecutarse con los datos
    concretos disponibles (ej. ``generate_query_rewrite`` sin
    vocabulario nuevo genuino que incorporar). NO significa fallo
    técnico global, claim unsupported, presupuesto agotado, transición
    inválida ni fallo del planner -- el executor (Bloque 4/runtime) es
    responsable de traducir la condición legítima específica
    (``QUERY_REWRITE_UNAVAILABLE``) a esta excepción; cualquier otro
    error debe seguir propagándose sin conversión."""


EXECUTION_STATUS_EXECUTED = "EXECUTED"
EXECUTION_STATUS_ACTION_UNAVAILABLE = "ACTION_UNAVAILABLE"
EXECUTION_STATUS_TERMINAL = "TERMINAL"


# Valida que una acción de mejora haya producido exactamente los cambios permitidos
# en el estado del ciclo. Compara la Observation anterior y posterior para impedir
# que una tool modifique campos que no le corresponden.
def _validate_improvement_transition(
    *, action: str, before: AgenticRetrievalObservation, after: AgenticRetrievalObservation
) -> None:
    """Fail-closed: para REWRITE_QUERY/ADJUST_TOP_K (las únicas
    acciones que ejecuta una tool real vía execute_action_fn -- nunca
    ACCEPT_EVIDENCE, que termina el ciclo antes de llegar aquí),
    verifica que la Observation resultante cumpla exactamente el
    contrato de esa acción -- incluidos los campos que NINGUNA acción
    de mejora puede tocar (identidad del claim y el tope estructural
    de top_k), no solo los que cada una sí modifica."""
    # Invariantes compartidos: ninguna acción de mejora puede alterar
    # la identidad del claim ni el tope estructural de top_k.
    if after.claim_id != before.claim_id:
        raise ValueError(
            f"{action}: claim_id_after ({after.claim_id!r}) debe ser igual a "
            f"claim_id_before ({before.claim_id!r})."
        )
    if after.claim_text != before.claim_text:
        raise ValueError(
            f"{action}: claim_text_after debe ser igual a claim_text_before -- "
            "ninguna acción de mejora reescribe el claim."
        )
    if after.effective_top_k_max != before.effective_top_k_max:
        raise ValueError(
            f"{action}: effective_top_k_max_after ({after.effective_top_k_max}) debe ser "
            f"igual a effective_top_k_max_before ({before.effective_top_k_max}) -- es un "
            "tope estructural fijo del ciclo, no algo que la tool pueda modificar."
        )

    if after.retrieval_round != before.retrieval_round + 1:
        raise ValueError(
            f"{action}: retrieval_round_after ({after.retrieval_round}) debe ser "
            f"retrieval_round_before + 1 ({before.retrieval_round + 1})."
        )
    if after.remaining_retrieval_budget != before.remaining_retrieval_budget - 1:
        raise ValueError(
            f"{action}: remaining_retrieval_budget_after ({after.remaining_retrieval_budget}) "
            f"debe ser remaining_retrieval_budget_before - 1 ({before.remaining_retrieval_budget - 1}) "
            "-- el presupuesto compartido es la garantía de terminación del ciclo."
        )

    # REWRITE_QUERY debe cambiar únicamente la consulta de búsqueda:
    # genera una query distinta, incrementa el contador de reescrituras
    # y mantiene sin cambios el top_k actual.
    if action == "REWRITE_QUERY":
        if after.current_query == before.current_query:
            raise ValueError("REWRITE_QUERY: current_query_after debe ser distinto de current_query_before.")
        if after.query_rewrite_count != before.query_rewrite_count + 1:
            raise ValueError(
                f"REWRITE_QUERY: query_rewrite_count_after ({after.query_rewrite_count}) debe ser "
                f"query_rewrite_count_before + 1 ({before.query_rewrite_count + 1})."
            )
        if after.current_top_k != before.current_top_k:
            raise ValueError(
                f"REWRITE_QUERY: current_top_k_after ({after.current_top_k}) debe ser igual a "
                f"current_top_k_before ({before.current_top_k}) -- REWRITE_QUERY no toca top_k."
            )
    # ADJUST_TOP_K mantiene intacta la consulta y aumenta únicamente la cantidad
    # de resultados solicitados, sin superar el máximo permitido ni modificar
    # el contador de reescrituras.
    elif action == "ADJUST_TOP_K":
        if after.current_query != before.current_query:
            raise ValueError(
                "ADJUST_TOP_K: current_query_after debe ser EXACTAMENTE igual a "
                "current_query_before -- ADJUST_TOP_K no toca la query."
            )
        if after.current_top_k <= before.current_top_k:
            raise ValueError(
                f"ADJUST_TOP_K: current_top_k_after ({after.current_top_k}) debe ser mayor que "
                f"current_top_k_before ({before.current_top_k})."
            )
        if after.current_top_k > before.effective_top_k_max:
            raise ValueError(
                f"ADJUST_TOP_K: current_top_k_after ({after.current_top_k}) no puede exceder "
                f"effective_top_k_max ({before.effective_top_k_max})."
            )
        if after.query_rewrite_count != before.query_rewrite_count:
            raise ValueError(
                f"ADJUST_TOP_K: query_rewrite_count_after ({after.query_rewrite_count}) debe ser "
                f"igual a query_rewrite_count_before ({before.query_rewrite_count}) -- "
                "ADJUST_TOP_K no toca la query."
            )
    else:
        raise ValueError(f"_validate_improvement_transition: acción inesperada {action!r}.")

# Genera automáticamente la justificación cuando solo existe una acción posible
# y, por tanto, no es necesario consultar al planner.
def _infer_deterministic_decision_basis(observation: AgenticRetrievalObservation) -> str:
    """Cuando solo hay 1 acción disponible, Python ejecuta directamente
    sin consultar al planner -- pero el step sigue registrando un
    decision_basis, derivado determinísticamente del primer reason_code
    presente en la Observation (mismo vocabulario, prefijo
    "EVIDENCE_INSUFFICIENT_", reutilizado de Bloque 1 sin duplicar)."""
    if not observation.reason_codes:
        raise ValueError(
            "No se puede inferir decision_basis determinista sin reason_codes -- "
            "esto no debería ocurrir para grade_result=INSUFFICIENT (invariante ya "
            "validada en __post_init__)."
        )
    return "EVIDENCE_INSUFFICIENT_" + observation.reason_codes[0]


# ---------------------------------------------------------------------------
# Ciclo
# ---------------------------------------------------------------------------


@dataclass
class AgenticRetrievalResult:
    claim_id: str
    outcome: str  # "ACCEPT_EVIDENCE" | FINISH_UNRESOLVED | AGENTIC_PLANNER_FAILED | AGENTIC_TRANSITION_INVALID
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_observation: AgenticRetrievalObservation | None = None

# Ejecuta el ciclo completo de Agentic Retrieval para un claim, partiendo de
# una recuperación inicial ya evaluada, hasta llegar a una condición terminal:
# aceptar la evidencia, finalizar sin resolver, detectar fallo del planner
# o rechazar una transición inválida.
def run_agentic_retrieval_cycle(
    *,
    initial_observation: AgenticRetrievalObservation,
    invoke_planner_fn: Callable[[str], str],
    execute_action_fn: Callable[[str, str, AgenticRetrievalObservation], AgenticRetrievalObservation],
) -> AgenticRetrievalResult:
    observation = initial_observation
    steps: list[dict[str, Any]] = []
    unavailable_actions_for_current_observation: set[str] = set()
    
    # Repite el ciclo de decisión, ejecución y reevaluación hasta alcanzar
    # una condición terminal.
    while True:
        forced_outcome = determine_forced_outcome(observation)
        if forced_outcome is not None:
            return AgenticRetrievalResult(
                claim_id=observation.claim_id, outcome=forced_outcome, steps=steps, final_observation=observation,
            )
        allowed_actions = tuple(
            a for a in compute_allowed_actions(observation)
            if a not in unavailable_actions_for_current_observation
        )
        if not allowed_actions:
            return AgenticRetrievalResult(
                claim_id=observation.claim_id, outcome=FINISH_UNRESOLVED, steps=steps, final_observation=observation,
            )
        if len(allowed_actions) == 1:
            selected_action = allowed_actions[0]
            decision_basis = _infer_deterministic_decision_basis(observation)
            planner_invoked = False
        else:
            prompt = build_agentic_planner_prompt(observation=observation, allowed_actions=allowed_actions)
            try:
                decision = invoke_agentic_planner_with_retry(
                    invoke_fn=invoke_planner_fn, prompt=prompt, allowed_actions=allowed_actions,
                    observation_reason_codes=observation.reason_codes,
                )
            except AgenticPlannerResponseError:
                return AgenticRetrievalResult(
                    claim_id=observation.claim_id, outcome=AGENTIC_PLANNER_FAILED, steps=steps, final_observation=observation,
                )
            selected_action = decision["selected_action"]
            decision_basis = decision["decision_basis"]
            planner_invoked = True

        step_number = len(steps) + 1

        # Si la acción elegida es ACCEPT_EVIDENCE, registra la decisión como terminal
        # y finaliza el ciclo devolviendo la evidencia disponible como aceptada.
        # Ejecuta la acción seleccionada, como REWRITE_QUERY o ADJUST_TOP_K,
        # junto con la nueva recuperación y evaluación de evidencia asociadas.
        if selected_action == "ACCEPT_EVIDENCE":
            steps.append({
                "step_number": step_number,
                "selected_action": selected_action,
                "decision_basis": decision_basis,
                "planner_invoked": planner_invoked,
                "execution_status": EXECUTION_STATUS_TERMINAL,
            })
            return AgenticRetrievalResult(
                claim_id=observation.claim_id, outcome="ACCEPT_EVIDENCE", steps=steps, final_observation=observation,
            )
        observation_before = observation
        try:
            observation_after = execute_action_fn(selected_action, decision_basis, observation_before)
        except AgenticRetrievalActionUnavailable as exc:
            print(
                "AGENTIC_ACTION_UNAVAILABLE_CAUGHT",
                type(exc).__module__,
                type(exc).__qualname__,
                repr(exc),
            )
            steps.append({
                "step_number": step_number,
                "selected_action": selected_action,
                "decision_basis": decision_basis,
                "planner_invoked": planner_invoked,
                "execution_status": EXECUTION_STATUS_ACTION_UNAVAILABLE,
            })
            unavailable_actions_for_current_observation.add(selected_action)
            continue

        # Registra que la acción de mejora se ejecutó correctamente, guardando el número
        # de paso, la acción seleccionada, su justificación, si intervino el planner
        # y el estado final de ejecución.
        steps.append({
            "step_number": step_number,
            "selected_action": selected_action,
            "decision_basis": decision_basis,
            "planner_invoked": planner_invoked,
            "execution_status": EXECUTION_STATUS_EXECUTED,
        })

        # Si la transición viola alguna regla del ciclo, finaliza con
        # AGENTIC_TRANSITION_INVALID y conserva como estado final la Observation anterior,
        # evitando propagar un estado inconsistente.
        try:
            _validate_improvement_transition(action=selected_action, before=observation_before, after=observation_after)
        except ValueError:
            return AgenticRetrievalResult(
                claim_id=observation_before.claim_id, outcome=AGENTIC_TRANSITION_INVALID,
                steps=steps, final_observation=observation_before,
            )
        observation = observation_after
        unavailable_actions_for_current_observation = set()
