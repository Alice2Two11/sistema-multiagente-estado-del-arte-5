"""Resolución fail-closed de ``pipeline_outcome`` (``SUCCESS`` |
``PARTIAL_HALT``) que decide si 07 es evaluable por 08 -- y con qué
metadatos.

Dos tipos de entrada científica admitidos, exactamente:

1. 07 ``COMPLETED`` + ``APPROVED``/``APPROVED_WITH_WARNINGS`` +
   ``ADVANCE -> 08`` => ``pipeline_outcome = "SUCCESS"``.
2. 07 ``COMPLETED`` + ``NEEDS_REVISION`` + ``HALT_STAGE`` por una de las
   TRES conclusiones científicas legítimas que ``classify_verification_
   transition`` puede alcanzar tras procesar realmente cada claim (ver
   ``SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES`` más abajo) =>
   ``pipeline_outcome = "PARTIAL_HALT"`` -- pero SOLO evaluable si se
   autoriza explícitamente (``allow_partial_halt=True``, nunca inferido
   ni activado por defecto): esto es lo que separa "camino explícito de
   evaluación" de "falsear una transición histórica 07->08" -- 07 SIGUE
   habiendo hecho ``HALT_STAGE`` en su propio ``decision_log``, nunca se
   reescribe ni reinterpreta como ``ADVANCE``.

Cualquier otra combinación -- en particular ``execution_status ==
FAILED`` (fallo técnico real de 07: runtime/contract/artifact errors) o
un ``HALT_STAGE`` NO reconocido como científico (ej.
``AGENT07_NO_CLAIMS``, ``AGENT07_MALFORMED_CLAIM``,
``AGENT07_MIXED_CLAIM_IDENTITY_CONTRACT`` -- problemas estructurales/de
datos, no conclusiones científicas sobre claims ya procesados) -- nunca
es evaluable, sin excepción.

Esta función es de SOLO LECTURA sobre ``decision_log`` -- nunca
modifica el estado de 06 ni de 07."""

from __future__ import annotations

from typing import Any

from src.contracts.agent_result import AgentResult, ExecutionStatus, TransitionAction
from src.orchestration.decision_log_frontier import authoritative_decision_log_entry_for_stage
from src.state.state_store import StateStore

AGENT07_STAGE_NAME = "07_agente_verificador"
AGENT08_STAGE_NAME = "08_evaluacion_experimental"

# Whitelist explícita -- únicamente reason_codes de HALT_STAGE que
# representan una conclusión CIENTÍFICA legítima, alcanzada DESPUÉS de
# procesar realmente cada claim -- nunca un problema estructural/de
# datos (esos producen HALT_STAGE ANTES de llegar a procesar claim por
# claim: AGENT07_NO_CLAIMS, AGENT07_MALFORMED_CLAIM,
# AGENT07_MIXED_CLAIM_IDENTITY_CONTRACT, AGENT07_MISSING_CLAIM_UID,
# AGENT07_UNKNOWN_ELIGIBILITY, AGENT07_CLAIM_UID_CONTRACT_VIOLATION --
# deliberadamente NUNCA en esta lista). Nunca se adivina por texto/
# similitud: un reason_code fuera de esta lista simplemente no es
# evaluable como PARTIAL_HALT. Las tres corresponden exactamente a las
# tres ramas de HALT_STAGE que ``classify_verification_transition``
# (``src/tools/verification/writer_revision_cycle.py``) puede alcanzar
# tras clasificar realmente cada claim:
#   - rondas agotadas con issues corregibles restantes.
#   - al menos un claim con elegibilidad bloqueante real
#     (MANUAL_REVIEW_REQUIRED / NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE).
#   - al menos un claim marcado corregible pero sin soporte de
#     corrección utilizable (evidencia/propuesta insuficiente).
SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES = frozenset({
    "WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED",
    "AGENT07_NON_CORRECTABLE_ISSUE",
    "AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT",
})


def resolve_pipeline_outcome_for_evaluation(
    *, store: StateStore, allow_partial_halt: bool,
) -> dict[str, Any]:
    """Devuelve el dict de metadatos de evaluabilidad para 08, o lanza
    ``ValueError`` (fail-closed, reason code explícito) si 07 no es
    evaluable en absoluto.

    Lee la entrada CAUSALMENTE VÁLIDA de 07 en ``decision_log`` (nunca
    ``state.stages`` directo, que puede reflejar una ejecución espuria
    posterior -- mismo criterio ya usado para 06 en ``agent06_
    verification_handoff.py``)."""

    state = store.load()
    entry = authoritative_decision_log_entry_for_stage(state.decision_log, AGENT07_STAGE_NAME)
    if entry is None:
        raise ValueError("AGENT08_UPSTREAM_07_NOT_COMMITTED")

    result = AgentResult.from_dict(entry.result)
    if result.execution_status != ExecutionStatus.COMPLETED:
        # Fallo técnico real de 07 (runtime/contract/artifact errors) --
        # nunca evaluable, sin excepción.
        raise ValueError("AGENT08_UPSTREAM_07_TECHNICAL_FAILURE")

    transition = result.requested_transition

    if transition.action == TransitionAction.ADVANCE and transition.target_stage == AGENT08_STAGE_NAME:
        return {
            "pipeline_outcome": "SUCCESS",
            "verification_approved": True,
            "autonomous_convergence": True,
            "human_review_required": False,
            "approved_for_publication": True,
            "usable_for_evaluation": True,
            "agent07_halt_reason": None,
            "agent07_reason_code": transition.reason_code,
            "agent07_decision_id": entry.decision_id,
        }

    if (
        transition.action == TransitionAction.HALT_STAGE
        and transition.reason_code in SCIENTIFIC_EXHAUSTION_HALT_REASON_CODES
    ):
        if not allow_partial_halt:
            raise ValueError("AGENT08_PARTIAL_HALT_NOT_EXPLICITLY_AUTHORIZED")
        cycle = state.cycles.get("writer_verifier")
        return {
            "pipeline_outcome": "PARTIAL_HALT",
            "verification_approved": False,
            "autonomous_convergence": False,
            "human_review_required": True,
            # Requisito explícito: PARTIAL_HALT nunca puede quedar
            # aprobado para publicación, aunque sea evaluable.
            "approved_for_publication": False,
            "usable_for_evaluation": True,
            # Causa real del halt de 07, tal cual quedó en su propio
            # decision_log -- nunca reformulada ni suavizada.
            "agent07_halt_reason": transition.reason_code,
            "agent07_reason_code": transition.reason_code,
            "agent07_decision_id": entry.decision_id,
            "rounds_used": cycle.rounds_used if cycle is not None else None,
            "max_rounds": cycle.max_rounds if cycle is not None else None,
        }

    raise ValueError(
        f"AGENT08_UPSTREAM_07_NOT_EVALUABLE:{transition.action.value}:{transition.reason_code}"
    )
