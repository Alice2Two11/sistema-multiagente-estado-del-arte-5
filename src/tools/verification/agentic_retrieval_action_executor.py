"""Agentic Retrieval (Stage 07, pre-verificación) -- Bloque 4
(corregido): executor real de acciones REWRITE_QUERY/ADJUST_TOP_K.

Implementa exactamente la interfaz ya congelada de Bloque 2 (ampliada
en la ronda anterior con ``decision_basis``, autorizada explícitamente):

    execute_action_fn(selected_action, decision_basis, observation)
    -> nueva AgenticRetrievalObservation

Orquesta, sin reimplementar ninguna lógica ya cerrada:
    Bloque 3 (``generate_query_rewrite``) para REWRITE_QUERY
    Bloque 1 (``config.next_top_k``) para el valor de ADJUST_TOP_K
    ``Agent07ChromaRetriever.retrieve_more`` REAL (con los overrides
        de Bloque 4) para la recuperación
    Bloque 1 (``grade_evidence``/``is_minimum_viable_evidence``) para
        construir la nueva Observation
    Bloque 2 (``AGENTIC_DECISION_BASIS_VALUES``/``validate_decision_
        basis``) para validar decision_basis -- REUTILIZADO, no
        duplicado (corrección de esta ronda).

Correcciones sobre la versión anterior (ronda de cierre de Bloque 4):
1. Coherencia contexto/Observation: antes de ejecutar cualquier acción,
   se exige ``observation.claim_id == self._claim_id``,
   ``observation.claim_text == self._claim_text``, y que
   ``observation.candidate_count``/``observation.evidence_ids``
   correspondan EXACTAMENTE a ``self._current_candidates`` (no solo
   misma longitud) -- garantiza que la Observation que decidió el
   planner es la misma evidencia que la acción realmente usará.
2. ``decision_basis`` se valida contra el enum real de Bloque 2
   (``AGENTIC_DECISION_BASIS_VALUES``/``validate_decision_basis``,
   importados, no duplicados) antes de convertirlo a ``rewrite_reason``
   -- rechaza ``EVIDENCE_ACCEPTABLE_DESPITE_GAPS`` (exclusivo de
   ACCEPT_EVIDENCE) y cualquier valor sin el prefijo
   ``EVIDENCE_INSUFFICIENT_``; el reason_code derivado debe pertenecer
   a ``observation.reason_codes``.
3. ``rewrite_trace`` solo registra un rewrite DESPUÉS de que retrieval +
   grade + construcción de la Observation tuvieron éxito completo -- si
   el retriever falla tras generar el rewrite, la traza permanece sin
   ese intento (nunca mezcla intentos fallidos con rewrites realmente
   ejecutados).

``evidence_ids`` -- auditoría confirmada (Bloque 4, punto G): el
retriever real deduplica candidatos por ``(source_filename, chunk_id)``
(``seen_pairs`` en ``verification_incremental_retriever.py``);
``chunk_id`` NO es globalmente único entre distintos papers. Stage 07
posteriormente crea su propio ``evidence_id`` canónico
(``evidence_selection.py``, ``f"E{n:02d}"``) para la selección
científica -- DISTINTO y POSTERIOR, nunca confundido ni reutilizado
aquí. Se usa la identidad compuesta, mismo criterio que el dedupe
interno del retriever:

    evidence_ids = f"{source_filename}::{chunk_id}"

``decision_basis`` llega EXACTAMENTE como lo resolvió el controller de
Bloque 2 (respuesta real validada del planner, o el valor determinista
de Python cuando el planner no fue consultado) -- se usa directamente
como ``rewrite_reason`` de Bloque 3 (tras validarse), sin re-derivarlo
desde ``observation.reason_codes``.

Estado explícito por instancia (nunca globals): cada
``AgenticRetrievalActionExecutor`` mantiene su propio contexto por
claim (candidatos actuales, fuentes autorizadas, retriever, umbrales,
traza de rewrites) -- auditable, sin variables compartidas entre
instancias/claims.

Presupuesto: cada acción consume exactamente 1
``retrieval_round``/``remaining_retrieval_budget`` (contrato ya
cerrado de Bloque 2) -- este módulo NO crea ningún contador nuevo, y
NO consume todavía el presupuesto interno de
``VerificationAgent.verify_claim`` (ese acoplamiento pertenece al
wiring posterior, fuera de Bloque 4)."""

from __future__ import annotations

from typing import Any

from src.config.agentic_retrieval_policy_config import (
    DEFAULT_GRADER_THRESHOLDS,
    DEFAULT_MINIMUM_VIABLE_THRESHOLDS,
    next_top_k,
)
from src.tools.verification.agentic_retrieval_controller import (
    AgenticRetrievalActionUnavailable,
    AgenticRetrievalObservation,
    validate_decision_basis,
)
from src.tools.verification.agentic_retrieval_grader import grade_evidence, is_minimum_viable_evidence
from src.tools.verification.agentic_retrieval_query_rewrite import QueryRewriteError, generate_query_rewrite


class ActionExecutorError(ValueError):
    """Fail-closed: incoherencia entre el contexto del executor y la
    Observation recibida, o decision_basis inválido/incoherente."""


def _build_evidence_ids(candidates: list[dict[str, Any]]) -> tuple[str, ...]:
    """Identidad compuesta ``source_filename::chunk_id`` -- mismo
    criterio que el dedupe interno del retriever (``seen_pairs``), NUNCA
    confundido con el ``evidence_id`` canónico posterior de Stage 07
    (``evidence_selection.py``)."""
    return tuple(f"{c['source_filename']}::{c['chunk_id']}" for c in candidates)


class AgenticRetrievalActionExecutor:
    """Contexto explícito y auditable por claim -- construye el callable
    ``execute_action_fn`` que consume ``run_agentic_retrieval_cycle``
    (Bloque 2)."""

    def __init__(
        self,
        *,
        retriever,
        allowed_source_filenames: frozenset[str] | set[str],
        claim_id: str,
        claim_text: str,
        initial_candidates: list[dict[str, Any]],
        grader_thresholds: dict | None = None,
        minimum_viable_thresholds: dict | None = None,
    ) -> None:
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ActionExecutorError(
                f"claim_id debe ser str real no vacío, recibido {claim_id!r} ({type(claim_id).__name__})."
            )
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise ActionExecutorError(
                f"claim_text debe ser str real no vacío, recibido {claim_text!r} ({type(claim_text).__name__})."
            )
        if not isinstance(allowed_source_filenames, (frozenset, set)) or not allowed_source_filenames:
            raise ActionExecutorError(
                "allowed_source_filenames debe ser frozenset/set no vacío -- "
                "frontera dura, no un default silencioso."
            )
        for source in allowed_source_filenames:
            if not isinstance(source, str) or not source.strip():
                raise ActionExecutorError(f"allowed_source_filenames contiene un elemento inválido: {source!r}.")
        if not isinstance(initial_candidates, list):
            raise ActionExecutorError(
                f"initial_candidates debe ser list, recibido {type(initial_candidates).__name__}."
            )

        self._retriever = retriever
        self._allowed_source_filenames = frozenset(allowed_source_filenames)
        self._claim_id = claim_id
        self._claim_text = claim_text
        self._current_candidates: list[dict[str, Any]] = list(initial_candidates)
        self._grader_thresholds = grader_thresholds or DEFAULT_GRADER_THRESHOLDS
        self._minimum_viable_thresholds = minimum_viable_thresholds or DEFAULT_MINIMUM_VIABLE_THRESHOLDS
        # Traza auditable de rewrites -- solo registra intentos que
        # completaron el ciclo con éxito (rewrite + retrieval + grade +
        # Observation construida) -- ver _execute_rewrite_query.
        self.rewrite_trace: list[dict[str, Any]] = []

    @property
    def current_candidates(self) -> tuple[dict[str, Any], ...]:
        """Integration accessor read-only (Bloque 5) -- copia defensiva
        de los candidatos actuales, con schema real completo
        (source_filename/chunk_id/text/native_scores_by_retriever).
        NO usar ``executor._current_candidates`` productivamente; esta
        es la única API pública para consumir los candidatos finales
        tras el ciclo. No modifica la lógica de Bloque 4."""
        return tuple(dict(c) for c in self._current_candidates)

    def _require_context_matches_observation(self, observation: AgenticRetrievalObservation) -> None:
        """Fail-closed: la Observation que decidió el planner debe
        corresponder EXACTAMENTE al contexto que este executor
        mantiene -- mismo claim, misma evidencia actual (no solo misma
        cantidad)."""
        if observation.claim_id != self._claim_id:
            raise ActionExecutorError(
                f"observation.claim_id ({observation.claim_id!r}) no coincide con "
                f"el contexto del executor ({self._claim_id!r})."
            )
        if observation.claim_text != self._claim_text:
            raise ActionExecutorError(
                "observation.claim_text no coincide con el contexto del executor -- "
                "posible cruce entre claims distintos."
            )
        if observation.candidate_count != len(self._current_candidates):
            raise ActionExecutorError(
                f"observation.candidate_count ({observation.candidate_count}) no coincide con "
                f"len(self._current_candidates) ({len(self._current_candidates)}) -- la "
                "Observation no corresponde a la evidencia actual del executor."
            )
        expected_evidence_ids = _build_evidence_ids(self._current_candidates)
        if observation.evidence_ids != expected_evidence_ids:
            raise ActionExecutorError(
                f"observation.evidence_ids ({observation.evidence_ids!r}) no coincide "
                f"exactamente con los candidatos actuales del executor "
                f"({expected_evidence_ids!r}) -- la Observation que decidió el planner "
                "no es la misma evidencia que la acción usaría."
            )

    def _require_valid_decision_basis(
        self, decision_basis: str, observation: AgenticRetrievalObservation
    ) -> str:
        """Frontera pública -- no confía ciegamente en el caller.
        Reutiliza el enum/validador real de Bloque 2 (sin duplicar
        vocabulario). Rechaza ``EVIDENCE_ACCEPTABLE_DESPITE_GAPS``
        (exclusivo de ACCEPT_EVIDENCE) y cualquier valor sin el
        prefijo esperado para acciones de mejora; el reason_code
        derivado debe pertenecer realmente a
        ``observation.reason_codes``. Retorna el ``rewrite_reason``
        (reason_code desnudo) ya validado."""
        try:
            validate_decision_basis(decision_basis)
        except ValueError as exc:
            raise ActionExecutorError(f"decision_basis inválido: {exc}") from exc
        if not decision_basis.startswith("EVIDENCE_INSUFFICIENT_"):
            raise ActionExecutorError(
                f"decision_basis={decision_basis!r} no corresponde a una acción de mejora "
                "-- REWRITE_QUERY/ADJUST_TOP_K exigen el prefijo 'EVIDENCE_INSUFFICIENT_' "
                "(ej. 'EVIDENCE_ACCEPTABLE_DESPITE_GAPS' es exclusivo de ACCEPT_EVIDENCE)."
            )
        reason_code = decision_basis.removeprefix("EVIDENCE_INSUFFICIENT_")
        if reason_code not in observation.reason_codes:
            raise ActionExecutorError(
                f"decision_basis={decision_basis!r} implica reason_code={reason_code!r}, "
                f"que no está presente en observation.reason_codes={observation.reason_codes!r} "
                "-- el executor no confía ciegamente en el caller."
            )
        return reason_code

    def __call__(
        self, selected_action: str, decision_basis: str, observation: AgenticRetrievalObservation
    ) -> AgenticRetrievalObservation:
        self._require_context_matches_observation(observation)
        rewrite_reason = self._require_valid_decision_basis(decision_basis, observation)

        if selected_action == "REWRITE_QUERY":
            return self._execute_rewrite_query(rewrite_reason, observation)
        if selected_action == "ADJUST_TOP_K":
            return self._execute_adjust_top_k(observation)
        raise ActionExecutorError(
            f"AgenticRetrievalActionExecutor no implementa la acción {selected_action!r} "
            "-- solo REWRITE_QUERY/ADJUST_TOP_K (Bloque 4)."
        )

    def _execute_rewrite_query(
        self, rewrite_reason: str, observation: AgenticRetrievalObservation
    ) -> AgenticRetrievalObservation:
        try:
            rewrite = generate_query_rewrite(
                claim_text=self._claim_text,
                current_query=observation.current_query,
                reason_codes=observation.reason_codes,
                rewrite_reason=rewrite_reason,
                candidates=self._current_candidates,
                authorized_sources=self._allowed_source_filenames,
            )
        except QueryRewriteError as exc:
            # E2E-BUG-01 (contract fix): SOLO la condición legítima
            # "sin vocabulario nuevo genuino que incorporar" se traduce a
            # ACTION_UNAVAILABLE -- REWRITE_QUERY era legal según la
            # Observation, pero no ejecutable con estos datos concretos.
            # Cualquier otro QueryRewriteError (inputs inválidos,
            # authorized_sources vacío, violación de contrato) sigue
            # propagándose como error técnico/contractual real, sin
            # conversión -- distinguido por el único código estable
            # disponible en el mensaje (no existe todavía un tipo/campo
            # estructurado separado en Bloque 3 para esta condición
            # específica).
            if str(exc).startswith("QUERY_REWRITE_UNAVAILABLE"):
                raise AgenticRetrievalActionUnavailable(str(exc)) from exc
            raise

        effective_query = rewrite["rewritten_query"]
        new_observation = self._run_retrieval_and_build_observation(
            observation=observation,
            effective_query=effective_query,
            effective_top_k=observation.current_top_k,
            new_query_rewrite_count=observation.query_rewrite_count + 1,
        )
        # Operación atómica (corrección de esta ronda): solo se registra
        # en la traza DESPUÉS de que retrieval + grade + construcción de
        # la Observation completaron con éxito -- si algo de eso lanza,
        # esta línea nunca se alcanza y rewrite_trace queda sin este
        # intento.
        self.rewrite_trace.append(rewrite)
        return new_observation

    def _execute_adjust_top_k(
        self, observation: AgenticRetrievalObservation
    ) -> AgenticRetrievalObservation:
        # El planner NUNCA elige el número -- Python lo decide vía la
        # política ya cerrada de Bloque 1.
        new_top_k = next_top_k(
            current_top_k=observation.current_top_k,
            effective_top_k_max=observation.effective_top_k_max,
        )
        return self._run_retrieval_and_build_observation(
            observation=observation,
            effective_query=observation.current_query,
            effective_top_k=new_top_k,
            new_query_rewrite_count=observation.query_rewrite_count,
        )

    def _run_retrieval_and_build_observation(
        self,
        *,
        observation: AgenticRetrievalObservation,
        effective_query: str,
        effective_top_k: int,
        new_query_rewrite_count: int,
    ) -> AgenticRetrievalObservation:
        result = self._retriever.retrieve_more({
            "claim_id": self._claim_id,
            "claim_context": {"claim_text": self._claim_text},
            "allowed_source_filenames": tuple(self._allowed_source_filenames),
            "query_override": effective_query,
            "top_k_override": effective_top_k,
        })
        candidates = list(result["selected_candidates"])

        grade = grade_evidence(
            claim_text=self._claim_text, candidates=candidates, thresholds=self._grader_thresholds,
        )
        minimum_viable = is_minimum_viable_evidence(
            candidates=candidates,
            thresholds=self._minimum_viable_thresholds,
            authorized_sources=self._allowed_source_filenames,
        )

        new_observation = AgenticRetrievalObservation(
            claim_id=observation.claim_id,
            claim_text=observation.claim_text,
            current_query=effective_query,
            retrieval_round=observation.retrieval_round + 1,
            current_top_k=effective_top_k,
            effective_top_k_max=observation.effective_top_k_max,
            remaining_retrieval_budget=observation.remaining_retrieval_budget - 1,
            candidate_count=grade["candidate_count"],
            evidence_ids=_build_evidence_ids(candidates),
            max_relevance_score=grade["max_relevance_score"],
            grade_result=grade["grade_result"],
            reason_codes=grade["reason_codes"],
            minimum_viable_evidence=minimum_viable,
            query_rewrite_count=new_query_rewrite_count,
        )
        # Contexto actualizado SOLO tras construir la Observation con
        # éxito -- si AgenticRetrievalObservation.__post_init__ falla
        # (invariante violado), self._current_candidates no se
        # actualiza, manteniendo coherencia con la última Observation
        # realmente válida.
        self._current_candidates = candidates
        return new_observation
