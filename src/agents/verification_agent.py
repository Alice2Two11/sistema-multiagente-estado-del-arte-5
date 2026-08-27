# ============================================================
# 07 - AGENTE VERIFICADOR DE TRAZABILIDAD Y SOPORTE CIENTÍFICO
# Verifica cada afirmación del borrador contra la evidencia disponible,
# detecta problemas de soporte y estima el riesgo de alucinación.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

from src.tools.verification.prompting import (
    build_verification_messages,
    normalize_verification_llm_response,
    parse_verification_response,
)
from src.tools.verification.validation import (
    ClaimRetrievalTool,
    allowed_verdicts_for_claim,
    compute_hallucination_risk,
    derive_semantic_issue_codes,
    determine_final_correction_eligibility,
    deterministic_precheck,
    select_evidence_for_scientific_judgment,
    validate_claim_verification_context,
    validate_llm_verification_response,
    validate_additional_retrieval_delta,
)


class VerificationLLM(Protocol):
    def invoke(self, messages: Sequence[Mapping[str, str]]) -> Any: ...

# Define la estructura del resultado de verificar cada afirmación científica.
# Guarda el veredicto, la evidencia, los problemas detectados, el riesgo
# de alucinación, las decisiones tomadas y la trazabilidad de la verificación.
@dataclass(frozen=True, slots=True)
class ClaimVerificationResult:
    claim_id: str #identifican la afirmación que se está verificando.
    claim_type: str #indica qué tipo de afirmación es.
    scientific_judgment_required: bool #dice si hace falta realmente un juicio científico.
    execution_status: str #indican si la verificación pudo ejecutarse correctamente
    technical_status: str
    technical_issue_codes: tuple[str, ...] #registra errores técnicos.
    scientific_judgment_status: str #estado del proceso de evaluación científica.
    scientific_verdict: str #veredicto final sobre la afirmación, por ejemplo si está respaldada o no.
    support_level: str #cuánto respaldo tiene.
    deterministic_issue_codes: tuple[str, ...] #problemas detectados mediante reglas, sin depender del LLM
    semantic_issue_codes: tuple[str, ...] #problemas detectados al analizar el significado
    eligible_evidence: tuple[dict[str, Any], ...] #evidencia que sí puede utilizarse.
    deterministically_discarded_evidence: tuple[dict[str, Any], ...] #evidencia eliminada automáticamente por las reglas
    evidence_used: tuple[dict[str, Any], ...] #evidencia utilizada para decidir.
    evidence_rejected: tuple[dict[str, Any], ...] #evidencia considerada pero descartada.
    contradiction_assessment: dict[str, Any] #registra si existe evidencia contradictoria.
    numeric_assessment: str #comprueba los datos numéricos.
    attribution_assessment: str #comprueba que una afirmación esté atribuida correctamente a su fuente.
    extrapolation_assessment: str #comprueba si el texto está afirmando más de lo que permite la evidencia.
    hallucination_risk: str #estima el riesgo de que el claim contenga información no respaldada.
    llm_correction_recommendation: bool #indica si el LLM recomienda corregirlo.
    final_correction_eligibility: str #decide finalmente si puede corregirse automáticamente.
    manual_review_required: bool #indica si debe revisarlo una persona.
    reason_codes: tuple[str, ...] #guarda las razones concretas de la decisión.
    tool_usage: dict[str, Any] #registra qué herramientas se utilizaron.
    decision_trace: tuple[str, ...] #guarda paso a paso qué decisiones tomó el verificador.
    raw_attempts: tuple[dict[str, Any], ...] #conserva los intentos realizados.
    result_contract_valid: bool #comprueba que el resultado tenga la estructura esperada.
    scientific_validation_ok: bool #indica si científicamente pasó la validación.
    validation_ok: bool #indica si el resultado completo es válido.
    claim_uid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerificationAgent:
    def __init__(self, *, llm: VerificationLLM | None, retrieval_tool: ClaimRetrievalTool | None = None) -> None:
        self.llm = llm
        self.retrieval_tool = retrieval_tool

    # Organiza cómo fue utilizada cada evidencia durante la verificación:
    # soporte, contradicción u otro uso permitido, conservando su trazabilidad.
    @staticmethod
    def _evidence_usage(rows: Sequence[Mapping[str, Any]], ids: Sequence[str], role: str,
                        contradiction_ids: Sequence[str] = ()) -> tuple[dict[str, Any], ...]:
        by_id = {str(row["evidence_id"]): row for row in rows}
        contradiction_set = set(contradiction_ids)
        out = []
        # Recorre las evidencias seleccionadas y determina qué papel tuvo cada una:
        # contradicción, otro uso permitido o el rol asignado como soporte.
        for evidence_id in ids:
            row = by_id[evidence_id]
            if evidence_id in contradiction_set:
                usage_role = "CONTRADICTION"
            elif row.get("usage_allowed") != "SUPPORT":
                usage_role = row.get("usage_allowed")
            else:
                usage_role = role

            # Guarda la trazabilidad básica de cada evidencia utilizada,
            # indicando su origen, autorización y función dentro de la verificación.
            out.append({
                "evidence_id": evidence_id,
                "source_filename": row["source_filename"],
                "chunk_id": row["chunk_id"],
                "authorized_for_section": bool(row["authorized_for_section"]),
                "retrieval_origin": row.get("retrieval_origin", ""),
                "usage_role": usage_role,
            })
        return tuple(out)

    @staticmethod
    # Define los datos que se registrarán sobre el uso de herramientas
    # durante la verificación de cada claim.
    def _tool_usage(*, considered: Sequence[str], selected: Sequence[str], retrieval_requests: int,
                    retrieval_rounds: int, evidence_selected: int, llm_calls: int,
                    format_attempts: int, schema_validation_attempts: int,
                    scientific_judgment_attempts: int, format_retries: int,
                    schema_retries: int) -> dict[str, Any]:
        considered_names = tuple(dict.fromkeys(considered))
        selected_names = tuple(dict.fromkeys(selected))
        return {
            "tool_names_considered": considered_names, 
            "tool_names_selected": selected_names, 
            "tools_considered": len(considered_names), #herramientas que el agente consideró usar
            "tools_selected": len(selected_names), #herramientas que realmente usó.
            "retrieval_requested": retrieval_requests, #cuántas veces pidió evidencia adicional
            "retrieval_rounds": retrieval_rounds, #cuántas rondas de recuperación se ejecutaron.
            "evidence_selected": evidence_selected, #cuántas evidencias quedaron seleccionadas.
            "llm_calls": llm_calls, #cuántas llamadas se hicieron al LLM.
            "format_attempts": format_attempts, #cuántas veces se intentó interpretar el formato de respuesta.
            "schema_validation_attempts": schema_validation_attempts, #cuántas veces se validó que la respuesta tuviera la estructura correcta
            "scientific_judgment_attempts": scientific_judgment_attempts, #cuántos intentos reales de juicio científico hubo
            "format_retries": format_retries, #cuántos reintentos ocurrieron por formato incorrecto
            "schema_retries": schema_retries, #cuántos reintentos ocurrieron porque la respuesta no cumplió el esquema
            "total_response_retries": format_retries + schema_retries,
        }

    # sirve porque puede haber recuperación adicional de evidencia. Cuando llegan nuevos resultados, 
    # el agente tiene que sumar lo nuevo a lo que ya tenía sin repetir evidencias, unde dos listas
    @staticmethod
    def _stable_union(old: Sequence[Any], new: Sequence[Any]) -> tuple[Any, ...]:
        out: list[Any] = []
        fingerprints: set[str] = set()
        for item in list(old) + list(new): #datos que ya existían y datos nuevos que llegaron.
            fp = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if fp not in fingerprints:
                fingerprints.add(fp)
                out.append(dict(item) if isinstance(item, Mapping) else item)
        return tuple(out)

    # Toma la información vieja y nueva de una misma evidencia y construye un único registro actualizado.
    # conservando sus datos originales y agregando lo recuperado en rondas posteriores.
    @staticmethod
    def _merge_candidate(previous: Mapping[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(previous)
        union_fields = ("retrieval_sources", "query_ids", "all_native_ranks", "text_variants", "contradiction_signals")
        for field in union_fields: # Une, sin duplicados, los campos que pueden acumular información cuando la misma evidencia aparece en varias rondas.
            merged[field] = VerificationAgent._stable_union(previous.get(field, ()) or (), delta.get(field, ()) or ())
        for field in ("native_ranks_by_retriever", "native_scores_by_retriever", "native_score_types_by_retriever"): #mezcla la información de ranking y puntuación que ya tenía con la nueva
            mapping = dict(previous.get(field, {}) or {})
            mapping.update(dict(delta.get(field, {}) or {}))
            merged[field] = {key: mapping[key] for key in sorted(mapping)}
        if "first_seen_round" in previous or "first_seen_round" in delta: ## Conserva la primera ronda en la que apareció la evidencia.
            vals = [int(v) for v in (previous.get("first_seen_round"), delta.get("first_seen_round")) if v is not None]
            merged["first_seen_round"] = min(vals)
        if "last_seen_round" in previous or "last_seen_round" in delta: # Conserva la última ronda en la que volvió a aparecer la evidencia.
            vals = [int(v) for v in (previous.get("last_seen_round"), delta.get("last_seen_round")) if v is not None]
            merged["last_seen_round"] = max(vals)
        for field in ("source_filename", "chunk_id", "authorized_for_section", "outside_section_sources",
                      "usage_allowed", "is_inherited", "retrieval_scope", "canonical_text", "contractual_text"): # Mantiene los datos originales de procedencia y autorización
            if field in previous:
                merged[field] = previous[field]
        if previous.get("text"): # Conserva preferentemente el texto que ya tenía la evidencia; usa el nuevo solo si antes no existía.
            merged["text"] = previous["text"]
        elif delta.get("text"):
            merged["text"] = delta["text"]
        scores = [v for v in (previous.get("fused_rrf_score"), delta.get("fused_rrf_score")) if isinstance(v, (int, float))] # Conserva el mayor score RRF registrado para esa evidencia.
        if scores: #significa: si encontró al menos un score válido, guarda el más alto.
            merged["fused_rrf_score"] = max(scores)
        for key, value in delta.items():
            if key not in set(union_fields) | {"native_ranks_by_retriever", "native_scores_by_retriever", "native_score_types_by_retriever", "first_seen_round", "last_seen_round", "source_filename", "chunk_id", "authorized_for_section", "outside_section_sources", "usage_allowed", "is_inherited", "retrieval_scope", "canonical_text", "contractual_text", "text", "fused_rrf_score"}:
                merged[key] = value
        return merged

    # Integra los resultados de una recuperación adicional con lo que el
    # verificador ya había recuperado, sin perder historial ni duplicar evidencia.
    @staticmethod
    def _merge_retrieval_delta(previous: Mapping[str, Any], delta: Mapping[str, Any], *,
                               allowed_source_pairs: Sequence[Sequence[str]] = (),
                               retrieval_mode: str = "SECTION_SCOPED") -> dict[str, Any]:
        # Valida que la nueva recuperación tenga la estructura esperada
        # y crea una copia de los resultados anteriores para poder actualizarlos.
        delta = validate_additional_retrieval_delta(delta, strict=True)
        merged = dict(previous)
        # Suma los contadores de la recuperación anterior y de la nueva ronda para mantener el total acumulado del proceso de búsqueda.
        for field in ("rounds_executed", "total_candidates_seen", "total_unique_candidates_seen", "queries_executed_total", "new_unique_pairs_seen"):
            merged[field] = int(previous.get(field, 0) or 0) + int(delta.get(field, 0) or 0)
        # Une el historial anterior y el nuevo sin repetir información.
        for field in ("queries", "discarded_candidates", "retrieval_trace", "contradiction_signals", "technical_issue_codes"):
            merged[field] = VerificationAgent._stable_union(previous.get(field, ()) or (), delta.get(field, ()) or ())
        # Organiza la evidencia ya seleccionada usando como clave la combinación paper + chunk para identificar cada candidato de forma única.
        by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for row in previous.get("selected_candidates", ()) or ():
            pair = (str(row["source_filename"]), str(row["chunk_id"]))
            by_pair[pair] = dict(row)
        # Ordena los candidatos nuevos recuperados usando paper, chunk y contenido completo como criterio de orden.
        delta_rows = sorted(
            (dict(row) for row in (delta.get("selected_candidates", ()) or ())),
            key=lambda row: (str(row["source_filename"]), str(row["chunk_id"]),
                             json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)),
        )
        # Convierte la lista de pares autorizados en un conjunto de combinaciones válidas paper + chunk.
        allowed_pairs = {(str(item[0]), str(item[1])) for item in allowed_source_pairs if len(item) == 2}
        # Revisa cada evidencia nueva: si ya existía, la actualiza; si es nueva, define si puede usarse como soporte o solo como contraste.
        for row in delta_rows:
            pair = (str(row["source_filename"]), str(row["chunk_id"]))
            if pair in by_pair:
                by_pair[pair] = VerificationAgent._merge_candidate(by_pair[pair], row)
            else:
                candidate = dict(row)
                authorized = pair in allowed_pairs
                candidate["authorized_for_section"] = authorized
                candidate["outside_section_sources"] = not authorized
                candidate["usage_allowed"] = "SUPPORT" if authorized else "CONTRAST"
                candidate["is_inherited"] = False
                candidate["retrieval_scope"] = retrieval_mode
                by_pair[pair] = candidate
        merged["selected_candidates"] = tuple(by_pair[pair] for pair in sorted(by_pair))
        # Actualiza el estado más reciente de la recuperación usando los valores que llegaron en la nueva ronda.
        for field in ("coverage_after", "stop_reason", "technical_status", "queries_remaining"):
            if field in delta:
                merged[field] = delta[field]
        # Cuenta cuántos candidatos únicos quedaron y crea el conjunto de evidencias que ya venían heredadas de antes.
        merged["total_unique_candidates_retained"] = len(by_pair)
        inherited_pairs = {
            (str(row.get("source_filename", "")), str(row.get("chunk_id", "")))
            for row in previous.get("inherited_evidence", ()) or () if isinstance(row, Mapping)
        }
        # Cuenta cuántas evidencias nuevas quedaron seleccionadas y prepara los estados de cobertura para compararlos.
        merged["new_unique_pairs_selected"] = sum(1 for pair in by_pair if pair not in inherited_pairs)
        global_before = previous.get("coverage_before", {}) or {}
        previous_after = previous.get("coverage_after", {}) or {}
        merged_after = merged.get("coverage_after", {}) or {}
        # Indica si esta ronda encontró la evidencia que faltaba y logró completar la cobertura del claim.                     
        merged["structural_coverage_improved_this_delta"] = bool(
            isinstance(merged_after, Mapping) and merged_after.get("structural_coverage_ok")
            and not (isinstance(previous_after, Mapping) and previous_after.get("structural_coverage_ok"))
        )
        merged["structural_coverage_improved"] = bool( #mira la mejora global, no solo la última ronda.
            isinstance(merged_after, Mapping) and merged_after.get("structural_coverage_ok")
            and not (isinstance(global_before, Mapping) and global_before.get("structural_coverage_ok"))
        )
        return merged

    # Inicia la verificación de un claim, valida su contexto, prepara la trazabilidad y selecciona la evidencia inicial.
    def verify_claim(self, context: Mapping[str, Any]) -> ClaimVerificationResult:
        current = validate_claim_verification_context(context)
        trace = ["DETERMINISTIC_PRECHECK"]
        raw_attempts: list[dict[str, Any]] = [] #ntentos crudos del LLM o del retriever
        technical_issues: list[str] = [] #guarda errores técnicos.
        considered = ["DETERMINISTIC_PRECHECK", "EVIDENCE_SELECTOR"] #herramientas que el agente contempla usar.
        selected_tools = ["DETERMINISTIC_PRECHECK", "EVIDENCE_SELECTOR"] #herramientas que ya decidió usar en este punto.
        llm_calls = format_attempts = schema_attempts = scientific_attempts = 0
        retrieval_requests = format_retries = schema_retries = 0

         # Ejecuta la revisión inicial con reglas, selecciona la evidencia útil y registra que esa evidencia ya quedó preparada para el análisis.  
        precheck = deterministic_precheck(current)
        selection = select_evidence_for_scientific_judgment(current)
        trace.append("EVIDENCE_SELECTED")

       
        if not precheck["scientific_judgment_required"]:
            # Devuelve el resultado final cuando el claim no necesita una evaluación científica adicional.
            return self._terminal(current, precheck, selection, "NOT_APPLICABLE", "NONE", technical_issues,
                                  trace + ["SCIENTIFIC_JUDGMENT_NOT_REQUIRED"], raw_attempts, considered,
                                  selected_tools, llm_calls, format_attempts, schema_attempts, scientific_attempts,
                                  retrieval_requests, format_retries, schema_retries)
        # Finaliza la verificación sin usar el LLM cuando la revisión inicial ya determina que no debe continuar. (deterministic_precheck())
        if precheck["terminal_without_llm"]:
            return self._terminal(current, precheck, selection, "NOT_EVALUATED", "NONE", technical_issues,
                                  trace + ["DETERMINISTIC_TERMINAL"], raw_attempts, considered, selected_tools,
                                  llm_calls, format_attempts, schema_attempts, scientific_attempts,
                                  retrieval_requests, format_retries, schema_retries)

        # Registra que el agente podría usar el LLM y recuperación adicional.
        # Si el LLM no está disponible, bloquea el juicio científico y termina la verificación.
        considered.extend(["LLM_JUDGE", "ADDITIONAL_RETRIEVER"])
        if self.llm is None:
            technical_issues.append("LLM_UNAVAILABLE")
            precheck = {**precheck, "technical_status": "LLM_UNAVAILABLE", "scientific_judgment_status": "BLOCKED"}
            return self._terminal(current, precheck, selection, "NOT_EVALUATED", "NONE", technical_issues,
                                  trace + ["LLM_UNAVAILABLE"], raw_attempts, considered, selected_tools,
                                  llm_calls, format_attempts, schema_attempts, scientific_attempts,
                                  retrieval_requests, format_retries, schema_retries)
        # Lee de la política los límites de intentos permitidos y prepara las variables para registrar errores y una respuesta válida.
        max_calls = int(current["policy"]["max_llm_attempts_per_claim"])
        max_format = int(current["policy"]["max_format_repair_attempts"])
        max_retrieval = int(current["policy"]["max_additional_retrieval_requests"])
        previous_errors: list[str] = []
        validated: dict[str, Any] | None = None

        while llm_calls < max_calls:
            allowed = allowed_verdicts_for_claim(current, precheck)
            messages = build_verification_messages(current, eligible_evidence=selection.eligible_evidence,
                                                    allowed_verdicts=allowed, previous_errors=previous_errors)
            selected_tools.append("LLM_JUDGE")
            trace.append("LLM_JUDGMENT_REQUESTED")
            llm_calls += 1
            # Intenta llamar al LLM; si la llamada falla, registra el error, bloquea el juicio científico y finaliza el claim sin evaluarlo.
            try:
                raw = self.llm.invoke(messages)
            except Exception as exc:
                technical_issues.append("LLM_INVOCATION_FAILED")
                raw_attempts.append({"attempt_number": llm_calls, "raw_text": "", "parse_status": "INVOCATION_FAILED",
                                     "schema_errors": (), "validation_errors": (type(exc).__name__,),
                                     "normalized_response": None})
                precheck = {**precheck, "technical_status": "LLM_INVOCATION_FAILED", "scientific_judgment_status": "BLOCKED"}
                return self._terminal(current, precheck, selection, "NOT_EVALUATED", "NONE", technical_issues,
                                      trace + ["LLM_INVOCATION_FAILED"], raw_attempts, considered, selected_tools,
                                      llm_calls, format_attempts, schema_attempts, scientific_attempts,
                                      retrieval_requests, format_retries, schema_retries)
                
            # Normaliza la respuesta cruda del LLM y prepara el registro de este intento antes de validar su formato y contenido.
            normalized_raw = normalize_verification_llm_response(raw)
            attempt = {"attempt_number": llm_calls, "raw_text": normalized_raw, "parse_status": "PENDING",
                       "schema_errors": (), "validation_errors": (), "normalized_response": None}
            format_attempts += 1

            # Intenta interpretar la respuesta del LLM. Si el formato es incorrecto (validate_llm_verification_response), registra el error y vuelve a intentarlo si todavía hay presupuesto.
            try:
                parsed = parse_verification_response(normalized_raw)
                attempt["parse_status"] = "PARSED"
            except ValueError as exc:
                attempt["parse_status"] = "INVALID_FORMAT"
                attempt["validation_errors"] = (str(exc),)
                raw_attempts.append(attempt)
                previous_errors = [str(exc)]
                format_retries += 1
                trace.append("FORMAT_RETRY")
                if format_retries > max_format:
                    break
                continue

            # Valida que la respuesta del LLM, además de poder leerse, cumpla las reglas esperadas para ese claim (llm_response_validation.py)
            schema_attempts += 1
            try:
                validated = validate_llm_verification_response(parsed, context=current,
                    eligible_evidence=selection.eligible_evidence, allowed_verdicts=allowed)
            except ValueError as exc:
                attempt["parse_status"] = "SCHEMA_INVALID"
                attempt["validation_errors"] = (str(exc),)
                raw_attempts.append(attempt)
                previous_errors = [str(exc)]
                schema_retries += 1
                trace.append("SCHEMA_RETRY")
                if str(exc) == "ADDITIONAL_RETRIEVAL_WITHOUT_BUDGET":
                    technical_issues.append("ADDITIONAL_RETRIEVAL_BUDGET_EXHAUSTED")
                    precheck = {**precheck, "technical_status": "ADDITIONAL_RETRIEVAL_BUDGET_EXHAUSTED", "scientific_judgment_status": "BLOCKED"}
                    return self._terminal(current, precheck, selection, "NOT_EVALUATED", "NONE", technical_issues,
                                          trace + ["ADDITIONAL_RETRIEVAL_BUDGET_EXHAUSTED"], raw_attempts,
                                          considered, selected_tools, llm_calls, format_attempts, schema_attempts,
                                          scientific_attempts, retrieval_requests, format_retries, schema_retries)
                continue
            # Registra que hubo un intento válido de juicio científico, guarda la respuesta validada y deja constancia en la trazabilidad.
            scientific_attempts += 1
            attempt["normalized_response"] = validated
            raw_attempts.append(attempt)
            trace.append("RESPONSE_VALIDATED")

            # Si el LLM pide buscar más evidencia, primero verifica que todavía queden recuperaciones adicionales disponibles.
            if validated["additional_retrieval_needed"]:
                if retrieval_requests >= max_retrieval:
                    technical_issues.append("ADDITIONAL_RETRIEVAL_BUDGET_EXHAUSTED")
                    precheck = {**precheck, "technical_status": "ADDITIONAL_RETRIEVAL_BUDGET_EXHAUSTED", "scientific_judgment_status": "BLOCKED"}
                    return self._terminal(current, precheck, selection, "NOT_EVALUATED", "NONE", technical_issues,
                                          trace + ["ADDITIONAL_RETRIEVAL_BUDGET_EXHAUSTED"], raw_attempts,
                                          considered, selected_tools, llm_calls, format_attempts, schema_attempts,
                                          scientific_attempts, retrieval_requests, format_retries, schema_retries)
                # Si el LLM necesita más evidencia pero el recuperador adicional no está disponible, bloquea la evaluación y termina el claim.
                if self.retrieval_tool is None:
                    technical_issues.append("ADDITIONAL_RETRIEVER_UNAVAILABLE")
                    precheck = {**precheck, "technical_status": "ADDITIONAL_RETRIEVER_UNAVAILABLE", "scientific_judgment_status": "BLOCKED"}
                    return self._terminal(current, precheck, selection, "NOT_EVALUATED", "NONE", technical_issues,
                                          trace + ["ADDITIONAL_RETRIEVER_UNAVAILABLE"], raw_attempts,
                                          considered, selected_tools, llm_calls, format_attempts, schema_attempts,
                                          scientific_attempts, retrieval_requests, format_retries, schema_retries)
                
                # Registra que se usará el recuperador adicional, prepara la solicitud de búsqueda y deja constancia en la trazabilidad.
                selected_tools.append("ADDITIONAL_RETRIEVER")
                retrieval_requests += 1
                missing = current.get("deterministic_validation", {}).get("missing_structural_elements", ())
                request = {
                    "claim_context": current,
                    "retrieval_reason_codes": tuple(validated["reason_codes"]),
                    "remaining_budget": max_retrieval - retrieval_requests,
                    "eligible_evidence": selection.eligible_evidence,
                    "missing_structural_elements": tuple(missing),
                    "originating_llm_response": validated,
                }
                trace.append("ADDITIONAL_RETRIEVAL_REQUESTED:" + ",".join(validated["reason_codes"]))

                # Intenta ejecutar la recuperación adicional de evidencia. Si falla, registra el error técnico y finaliza el claim sin evaluarlo.
                try:
                    retrieval_result = self.retrieval_tool.retrieve_more(request)
                
                except Exception as exc:
                    technical_issues.append(
                        "ADDITIONAL_RETRIEVAL_FAILED"
                    )
                
                    request_hash = hashlib.sha256(
                        json.dumps(
                            request,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str
                        ).encode("utf-8")
                    ).hexdigest()
                
                    raw_attempts.append({
                        "tool_name": "ADDITIONAL_RETRIEVER",
                        "request_number": retrieval_requests,
                        "exception_type": type(exc).__name__,
                        "exception_message_hash": hashlib.sha256(
                            str(exc).encode("utf-8")
                        ).hexdigest(),
                        "retrieval_reason_codes": tuple(
                            validated["reason_codes"]
                        ),
                        "request_hash": request_hash,
                        "parse_status": "TOOL_INVOCATION_FAILED",
                        "raw_text": "",
                        "schema_errors": (),
                        "validation_errors": (
                            type(exc).__name__,
                        ),
                        "normalized_response": None,
                    })
                
                    precheck = {
                        **precheck,
                        "technical_status": "ADDITIONAL_RETRIEVAL_FAILED",
                        "scientific_judgment_status": "BLOCKED"
                    }
                
                    return self._terminal(
                        current,
                        precheck,
                        selection,
                        "NOT_EVALUATED",
                        "NONE",
                        technical_issues,
                        trace + ["ADDITIONAL_RETRIEVAL_FAILED"],
                        raw_attempts,
                        considered,
                        selected_tools,
                        llm_calls,
                        format_attempts,
                        schema_attempts,
                        scientific_attempts,
                        retrieval_requests,
                        format_retries,
                        schema_retries
                    )

                # Valida que la nueva evidencia recuperada tenga la estructura esperada y la fusiona con la evidencia que ya tenía el verificador.
                try:
                    validated_delta = validate_additional_retrieval_delta(retrieval_result, strict=True)
                    fused_retrieval = self._merge_retrieval_delta(
                        current.get("retrieval_result", {}), validated_delta,
                        allowed_source_pairs=current.get("allowed_source_pairs", ()),
                        retrieval_mode=str(current.get("retrieval_result", {}).get("retrieval_mode", "SECTION_SCOPED")),
                    )
                # Si la nueva recuperación viene con una estructura inválida,
                # registra el error y termina el claim como no evaluado.
                except ValueError as exc:
                    technical_issues.append("ADDITIONAL_RETRIEVAL_FAILED")
                
                    request_hash = hashlib.sha256(
                        json.dumps(
                            request,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str
                        ).encode("utf-8")
                    ).hexdigest()
                
                    raw_attempts.append({
                        "tool_name": "ADDITIONAL_RETRIEVER",
                        "request_number": retrieval_requests,
                        "exception_type": "RetrievalDeltaValidationError",
                        "exception_message_hash": hashlib.sha256(
                            str(exc).encode("utf-8")
                        ).hexdigest(),
                        "retrieval_reason_codes": tuple(
                            validated["reason_codes"]
                        ),
                        "request_hash": request_hash,
                        "parse_status": "DELTA_INVALID",
                        "raw_text": "",
                        "schema_errors": (str(exc),),
                        "validation_errors": (str(exc),),
                        "normalized_response": None
                    })
                
                    precheck = {
                        **precheck,
                        "technical_status": "ADDITIONAL_RETRIEVAL_FAILED",
                        "scientific_judgment_status": "BLOCKED"
                    }
                
                    return self._terminal(
                        current,
                        precheck,
                        selection,
                        "NOT_EVALUATED",
                        "NONE",
                        technical_issues,
                        trace + ["ADDITIONAL_RETRIEVAL_DELTA_INVALID"],
                        raw_attempts,
                        considered,
                        selected_tools,
                        llm_calls,
                        format_attempts,
                        schema_attempts,
                        scientific_attempts,
                        retrieval_requests,
                        format_retries,
                        schema_retries
                    )
                    
                updated = dict(current)
                updated["retrieval_result"] = fused_retrieval

                # Si la nueva recuperación trae una validación determinística actualizada,
                # la incorpora al contexto del claim.
                if "deterministic_validation" in validated_delta:
                    merged_validation = dict(updated["deterministic_validation"])
                    merged_validation.update(validated_delta["deterministic_validation"])
                    updated["deterministic_validation"] = merged_validation
                
                # Descuenta una búsqueda disponible y vuelve a revisar el claim
                # con la evidencia nueva antes de continuar.
                ac = dict(updated["attempt_context"])
                ac["remaining_retrieval_requests"] = max(0,int(ac.get("remaining_retrieval_requests", 0)) - 1)
                updated["attempt_context"] = ac
                current = validate_claim_verification_context(updated)
                precheck = deterministic_precheck(current)
                selection = select_evidence_for_scientific_judgment(current)
                
                trace.extend(["DETERMINISTIC_PRECHECK_AFTER_RETRIEVAL","EVIDENCE_RESELECTED"])

                # Actualiza las reglas determinísticas con la nueva recuperación, descuenta una búsqueda disponible y vuelve a evaluar el claim.
                if "deterministic_validation" in validated_delta:
                    merged_validation = dict(updated["deterministic_validation"])
                    merged_validation.update(validated_delta["deterministic_validation"])
                    updated["deterministic_validation"] = merged_validation
                
                # Actualiza el número de búsquedas disponibles y vuelve a revisarel claim usando la evidencia recién recuperada.    
                ac = dict(updated["attempt_context"])
                ac["remaining_retrieval_requests"] = max(0, int(ac.get("remaining_retrieval_requests", 0)) - 1)
                updated["attempt_context"] = ac
                current = validate_claim_verification_context(updated)
                precheck = deterministic_precheck(current)
                selection = select_evidence_for_scientific_judgment(current)
                trace.extend(["DETERMINISTIC_PRECHECK_AFTER_RETRIEVAL", "EVIDENCE_RESELECTED"])

                # Si la nueva evidencia ya permite cerrar el claim sin usar el LLM, finaliza la verificación; si no, vuelve a intentar el juicio científico.
                if precheck["terminal_without_llm"]:
                    return self._terminal(current, precheck, selection, "NOT_EVALUATED", "NONE", technical_issues,
                                          trace + ["DETERMINISTIC_TERMINAL_AFTER_RETRIEVAL"], raw_attempts,
                                          considered, selected_tools, llm_calls, format_attempts, schema_attempts,
                                          scientific_attempts, retrieval_requests, format_retries, schema_retries)
                validated = None
                continue
            break
            
        # # Si después de todos los intentos no se obtuvo una respuesta válida del LLM, bloquea el juicio científico y deja el claim como no evaluado.
        if validated is None:
            technical_issues.append("LLM_VALIDATION_ATTEMPTS_EXHAUSTED")
            precheck = {**precheck, "technical_status": "LLM_VALIDATION_ATTEMPTS_EXHAUSTED",
                        "scientific_judgment_status": "BLOCKED"}
            return self._terminal(current, precheck, selection, "NOT_EVALUATED", "NONE", technical_issues,
                                  trace + ["SCIENTIFIC_JUDGMENT_BLOCKED"], raw_attempts, considered,
                                  selected_tools, llm_calls, format_attempts, schema_attempts, scientific_attempts,
                                  retrieval_requests, format_retries, schema_retries)

        # Procesa el resultado validado del LLM: identifica problemas semánticos,
        # clasifica la evidencia, calcula el riesgo de alucinación
        # y decide si el claim puede corregirse automáticamente.
        semantic = derive_semantic_issue_codes(validated)

        used = self._evidence_usage(
            selection.eligible_evidence,
            validated["evidence_ids_used"],
            "SUPPORT",
            validated["contradiction_evidence_ids"]
        )
        rejected = self._evidence_usage(
            selection.eligible_evidence,
            validated["evidence_ids_rejected"],
            "REJECTED"
        )
        risk = compute_hallucination_risk(
            deterministic_issue_codes=precheck["deterministic_issue_codes"],
            semantic_issue_codes=semantic,
            validated_response=validated,
            eligible_evidence=selection.eligible_evidence,
            technical_status="OK"
        )
        localized = bool(
            current.get("attempt_context", {}).get(
                "correction_localized",
                False
            )
        )
        correction = determine_final_correction_eligibility(
            verdict=validated["verdict"],
            deterministic_issue_codes=precheck["deterministic_issue_codes"],
            semantic_issue_codes=semantic,
            llm_recommendation=validated["llm_correction_recommendation"],
            manual_review_required=validated["manual_review_required"],
            eligible_evidence=selection.eligible_evidence,
            evidence_ids_used=validated["evidence_ids_used"],
            correction_localized=localized
        )
        manual_review_required = (
            bool(validated["manual_review_required"])
            or correction == "MANUAL_REVIEW_REQUIRED"
        )
        scientific_ok = validated["verdict"] == "SUPPORTED" and risk == "LOW"
        trace.append(f"VERDICT_{validated['verdict']}")
       
        reason_codes = tuple(sorted(set(validated["reason_codes"]) | set(precheck.get("retrieval_reason_codes", ()))))
        # Construye y devuelve el resultado final de la verificación del claim,
        # incluyendo veredicto, evidencia, riesgos, corrección y trazabilidad.
        return ClaimVerificationResult(
            current["claim_id"],current["claim_type"],
            True,"COMPLETED","OK",
            tuple(sorted(set(technical_issues))),
            "COMPLETED",validated["verdict"],
            validated["support_level"],
        
            tuple(precheck["deterministic_issue_codes"]),
            semantic,
            selection.eligible_evidence,
            selection.deterministically_discarded_evidence,
            used,
            rejected,
        
            {
                "type": validated["contradiction_type"],
                "evidence_ids": validated["contradiction_evidence_ids"]
            },
        
            validated["numeric_assessment"],
            validated["attribution_assessment"],
            validated["extrapolation_assessment"],
            risk,
            validated["llm_correction_recommendation"],
            correction,
            manual_review_required,
            reason_codes,
        
            self._tool_usage(
                considered=considered,
                selected=selected_tools,
                retrieval_requests=retrieval_requests,
                retrieval_rounds=int(
                    current.get("retrieval_result", {}).get("rounds_executed",0)),
                evidence_selected=len(selection.eligible_evidence),
                llm_calls=llm_calls,
                format_attempts=format_attempts,
                schema_validation_attempts=schema_attempts,
                scientific_judgment_attempts=scientific_attempts,
                format_retries=format_retries,
                schema_retries=schema_retries
            ),
        
            tuple(trace),
            tuple(raw_attempts),
        
            True,scientific_ok,True,current.get("claim_uid", ""),
        )


    # Construye el resultado final cuando el claim termina sin un juicio
    # científico completo del LLM, conservando el motivo, riesgo y trazabilidad.
    def _terminal(
        self,
        current,
        precheck,
        selection,
        verdict,
        support_level,
        technical_issues,
        trace,
        raw_attempts,
        considered,
        selected_tools,
        llm_calls,
        format_attempts,
        schema_attempts,
        scientific_attempts,
        retrieval_requests,
        format_retries,
        schema_retries
    ):
        deterministic = tuple(
            precheck["deterministic_issue_codes"]
        )
    
        reason_codes = tuple(
            sorted(set(deterministic) | set(precheck.get("retrieval_reason_codes",())))
        )
    
        risk = compute_hallucination_risk(
            deterministic_issue_codes=deterministic,
            semantic_issue_codes=(),
            validated_response=None,
            eligible_evidence=selection.eligible_evidence,
            technical_status=str(precheck.get("technical_status","OK"))
        )
    
        manual = (
            verdict == "NOT_EVALUATED" and bool(precheck["scientific_judgment_required"])
        )
    
        correction = determine_final_correction_eligibility(
            verdict=verdict,
            deterministic_issue_codes=deterministic,
            semantic_issue_codes=(),
            llm_recommendation=False,
            manual_review_required=manual,
            eligible_evidence=selection.eligible_evidence
        )
    
        scientific_ok = verdict == "NOT_APPLICABLE"
    
        return ClaimVerificationResult(
            current["claim_id"],
            current["claim_type"],
            bool(precheck["scientific_judgment_required"]),
            "COMPLETED",
            str(precheck.get("technical_status", "OK")),
            tuple(sorted(set(technical_issues))),
            str(precheck["scientific_judgment_status"]),
            verdict,support_level,deterministic,(),
            selection.eligible_evidence,
            selection.deterministically_discarded_evidence,(),(),
            {"type": "NONE","evidence_ids": ()},
            (
                "UNSUPPORTED"
                if "UNSUPPORTED_NUMERIC_VALUE" in deterministic
                else "NOT_APPLICABLE"
            ),
            "NOT_APPLICABLE",
            "NOT_APPLICABLE",
            risk,
            False,
            correction,
            manual,
            reason_codes,
            self._tool_usage(
                considered=considered,
                selected=selected_tools,
                retrieval_requests=retrieval_requests,
                retrieval_rounds=int(
                    current.get(
                        "retrieval_result",
                        {}
                    ).get(
                        "rounds_executed",
                        0
                    )
                ),
                evidence_selected=len(
                    selection.eligible_evidence
                ),
                llm_calls=llm_calls,
                format_attempts=format_attempts,
                schema_validation_attempts=schema_attempts,
                scientific_judgment_attempts=scientific_attempts,
                format_retries=format_retries,
                schema_retries=schema_retries
            ),
            tuple(trace),
            tuple(dict(x) for x in raw_attempts),
            True,
            scientific_ok,
            True,
            current.get("claim_uid", ""),
        )
