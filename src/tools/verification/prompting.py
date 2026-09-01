"""Prompts versionados y parsing estricto para juicio científico por claim.

Este módulo no contiene clientes OpenAI. Trabaja con texto JSON y contratos
internos inyectables.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from src.config.verification_policy_config import (
    ATTRIBUTION_ASSESSMENTS,
    ATTRIBUTION_RELATIONS,
    CONTRADICTION_TYPES,
    CORRECTION_CHANGE_SCOPES,
    CORRECTION_REASON_CODES,
    CORRECTION_SEMANTIC_CHANGE_LEVELS,
    EXTRAPOLATION_ASSESSMENTS,
    NUMERIC_ASSESSMENTS,
    SCIENTIFIC_VERDICTS,
    SUPPORT_LEVELS,
    SEMANTIC_REASON_CODES,
    ADDITIONAL_RETRIEVAL_REASON_CODES,
)

VERIFICATION_RESPONSE_FIELDS = (
    "claim_id",
    "verdict",
    "support_level",
    "evidence_ids_used",
    "evidence_ids_rejected",
    "rationale",
    "contradiction_type",
    "contradiction_evidence_ids",
    "numeric_assessment",
    "attribution_assessment",
    "extrapolation_assessment",
    "confidence",
    "additional_retrieval_needed",
    "llm_correction_recommendation",
    "manual_review_required",
    "reason_codes",
)


def build_verification_messages(
    context: Mapping[str, Any],
    *,
    eligible_evidence: Sequence[Mapping[str, Any]],
    allowed_verdicts: Sequence[str],
    previous_errors: Sequence[str] = (),
) -> tuple[dict[str, str], dict[str, str]]:
    policy = context["policy"]
    system = (
        f"Prompt {policy['verification_system_prompt_version']}. "
        "Evalúa un único claim usando exclusivamente la evidencia visible. "
        "No uses conocimiento externo ni inventes fuentes, autores, métricas o valores. "
        "La falta de evidencia no equivale a falsedad. Distingue desacuerdo entre papers "
        "de conflicto claim-evidence. Usa solo evidence_id entregados. No reescribas el claim. "
        "Devuelve únicamente un objeto JSON sin Markdown y conserva el idioma del claim en rationale. "
        "Criterio explícito para llm_correction_recommendation (no lo dejes en False por defecto sin "
        "evaluarlo): marcá true cuando el veredicto sea PARTIALLY_SUPPORTED o CONTRADICTED y el ajuste "
        "necesario sea LOCALIZADO -- es decir, se pueda resolver acotando, matizando o corrigiendo una "
        "parte puntual del claim con la evidencia ya visible, sin introducir información nueva ni "
        "reescribir la afirmación completa. Marcá manual_review_required=true (no "
        "llm_correction_recommendation) solo cuando el problema NO sea localizado -- por ejemplo, la "
        "evidencia disponible contradice el claim de forma sustancial, hay múltiples fuentes en desacuerdo "
        "real entre sí sobre el mismo punto, o el ajuste necesario cambiaría el sentido central de la "
        "afirmación. Ambos campos son independientes: podés marcar llm_correction_recommendation=true y "
        "manual_review_required=false en el mismo claim si el problema es puntual y corregible."
    )
    evidence_payload = []
    for row in eligible_evidence:
        evidence_payload.append({
            "evidence_id": row["evidence_id"],
            "source_filename": row["source_filename"],
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "authorized_for_section": bool(row.get("authorized_for_section", False)),
            "usage_allowed": row.get("usage_allowed", "SUPPORT"),
        })
    schema = {
        "claim_id": context["claim_id"],
        "verdict": f"one of {tuple(allowed_verdicts)}",
        "support_level": f"one of {SUPPORT_LEVELS}",
        "evidence_ids_used": [],
        "evidence_ids_rejected": [],
        "rationale": "string",
        "contradiction_type": f"one of {CONTRADICTION_TYPES}",
        "contradiction_evidence_ids": [],
        "numeric_assessment": f"one of {NUMERIC_ASSESSMENTS}",
        "attribution_assessment": f"one of {ATTRIBUTION_ASSESSMENTS}",
        "extrapolation_assessment": f"one of {EXTRAPOLATION_ASSESSMENTS}",
        "confidence": "LOW|MEDIUM|HIGH",
        "additional_retrieval_needed": False,
        "llm_correction_recommendation": False,
        "manual_review_required": False,
        "reason_codes": [],
    }
    user_payload = {
        "prompt_version": policy["verification_user_prompt_version"],
        "claim": {
            "claim_id": context["claim_id"],
            "section_id": context["section_id"],
            "section_title": context["section_title"],
            "claim_text": context["claim_text"],
            "claim_type": context["claim_type"],
        },
        "deterministic_findings": context["deterministic_validation"],
        "eligible_evidence": evidence_payload,
        "allowed_verdicts_for_this_claim": list(allowed_verdicts),
        "related_claim_ids": list(context.get("related_claim_ids", ())),
        "related_claims": list(context.get("related_claims", ())),
        "allowed_semantic_reason_codes": list(SEMANTIC_REASON_CODES),
        "allowed_additional_retrieval_reason_codes": list(ADDITIONAL_RETRIEVAL_REASON_CODES),
        "previous_errors": list(previous_errors),
        "response_schema": schema,
    }
    return {"role": "system", "content": system}, {
        "role": "user",
        "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
    }


def normalize_verification_llm_response(raw_response: Any) -> str | dict[str, Any]:
    """Normalize the adapter boundary before strict JSON/schema validation.

    LangChain chat models return ``BaseMessage``/``AIMessage`` instances whose
    JSON payload lives in ``.content``.  The scientific contract still receives
    exactly the same string or mapping; this function only unwraps the transport
    object and does not alter the payload, rubric, or validation rules.
    """
    value = raw_response
    if not isinstance(value, (str, Mapping)) and hasattr(value, "content"):
        value = getattr(value, "content")

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return value
    return ""


def parse_verification_response(raw_response: Any) -> dict[str, Any]:
    normalized = normalize_verification_llm_response(raw_response)
    if isinstance(normalized, Mapping):
        return dict(normalized)
    if not normalized.strip():
        raise ValueError("LLM_RESPONSE_EMPTY")
    text = normalized.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("LLM_RESPONSE_NOT_PURE_JSON_OBJECT")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM_RESPONSE_INVALID_JSON:{exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("LLM_RESPONSE_ROOT_NOT_OBJECT")
    return value

# Phase 5 correction prompts. These are separate from scientific judgment prompts.
from src.tools.verification.validation import canonical_correction_evidence_text as _canonical_correction_evidence_text
CORRECTION_RESPONSE_FIELDS = (
    "claim_id", "correction_decision", "action_type", "target_text", "replacement_text",
    "evidence_ids", "reason_codes", "change_scope", "semantic_change_level",
    "old_citation_refs", "new_citation_refs", "old_numeric_pairs", "new_numeric_pairs",
    "metric_context", "unit_context", "old_attribution_elements", "new_attribution_elements",
    "attribution_relation", "new_entities", "new_attributions", "new_conditions",
    "new_technical_terms", "citation_text_span", "llm_correction_recommendation",
)

def build_correction_messages(
    context: Mapping[str, Any], *, eligible_evidence: Sequence[Mapping[str, Any]],
    previous_errors: Sequence[str] = (),
) -> tuple[dict[str, str], dict[str, str]]:
    policy = context["policy"]
    system = (
        f"Prompt {policy['correction_system_prompt_version']}. Propón como máximo una modificación "
        "localizada para un claim. Usa solo evidencia autorizada visible. No agregues hechos, citas, "
        "entidades, números, condiciones ni atribuciones no respaldadas. No reescribas la sección ni "
        "devuelvas un claim completo libre. Conserva el idioma. Devuelve solo JSON. "
        "REQUEST_MANUAL_REVIEW y NO_CHANGE son decisiones, no action_type. "
        "llm_correction_recommendation es un booleano JSON (true/false), NUNCA el texto de "
        "correction_decision -- son dos campos distintos con tipos distintos: correction_decision es "
        "un string (PROPOSE_CHANGE/NO_CORRECTION/DEFER_TO_MANUAL_REVIEW/NOT_CORRECTABLE), "
        "llm_correction_recommendation es exclusivamente true o false. "
        "change_scope y semantic_change_level deben ser EXACTAMENTE uno de los valores listados en "
        "allowed_change_scopes/allowed_semantic_change_levels del payload (todo en mayúsculas, ej. "
        "TOKEN/PHRASE/CLAUSE/SENTENCE/MULTI_SENTENCE y NONE/MINIMAL/MODERATE/SUBSTANTIAL) -- nunca una "
        "palabra libre como 'local' o 'minor'. reason_codes debe ser un subconjunto EXACTO de "
        "allowed_reason_codes -- nunca el nombre de un action_type (ej. NARROW_SCOPE es un "
        "action_type, no un reason_code válido; no lo repitas dentro de reason_codes) ni un código "
        "inventado. metric_context y unit_context son SIEMPRE string JSON, nunca null: usa \"\" "
        "(string vacío) cuando no aplica, no el valor null. "
        "action_type sigue la regla CONTRARIA a metric_context/unit_context: cuando "
        "correction_decision NO es PROPOSE_CHANGE (es decir, es NO_CORRECTION, "
        "DEFER_TO_MANUAL_REVIEW o NOT_CORRECTABLE), action_type debe ser el valor JSON null "
        "-- NUNCA \"\" (string vacío) y nunca el nombre de una acción. Solo cuando "
        "correction_decision=='PROPOSE_CHANGE' action_type es un string, y debe ser "
        "exactamente uno de allowed_action_types. \"\" no es un valor válido para action_type "
        "en ningún caso. "
        "old_citation_refs y new_citation_refs son listas de OBJETOS, nunca de strings ni de "
        "evidence_id sueltos: cada elemento debe ser {\"source_filename\": <string>, "
        "\"chunk_id\": <string>}, copiados exactamente de source_filename/chunk_id en "
        "eligible_evidence -- nunca [\"E03\"] ni el evidence_id como texto. "
        "evidence_ids es un campo DISTINTO y NO sigue el formato de old_citation_refs/"
        "new_citation_refs: es una lista de IDs simples en string (ej. [\"E03\", \"E07\"]), "
        "copiados del campo evidence_id de eligible_evidence -- NUNCA objetos "
        "{source_filename, chunk_id}. No confundas los dos campos: evidence_ids identifica "
        "QUÉ evidencia se usó (IDs sueltos); old_citation_refs/new_citation_refs describe "
        "citas textuales dentro del claim (objetos con source_filename+chunk_id). "
        "old_numeric_pairs y new_numeric_pairs son listas de PARES [valor_antiguo, "
        "valor_nuevo], cada par una lista de EXACTAMENTE 2 strings -- nunca un objeto "
        "{\"old_value\":...} ni una lista de un solo elemento. "
        "citation_text_span sigue la MISMA regla que action_type (JSON null, nunca \"\" "
        "string vacío) y por la MISMA razón que metric_context/unit_context usan \"\" -- son "
        "reglas opuestas para dos campos distintos, no la confundas: cuando action_type NO es "
        "REPLACE_CITATION (incluyendo cuando correction_decision no es PROPOSE_CHANGE), "
        "citation_text_span debe ser el valor JSON null -- NUNCA \"\" ni ningún string. Solo "
        "cuando action_type=='REPLACE_CITATION', citation_text_span es un OBJETO (nunca un "
        "string, ni siquiera el texto de la cita) con las mismas claves que claim_span_in_"
        "section: {\"coordinate_base\": \"CLAIM_TEXT\", \"coordinate_system\": "
        "\"PYTHON_CODEPOINT_OFFSETS\", \"start\": <int>, \"end\": <int>, \"text\": <string>}, "
        "donde \"text\" es un fragmento copiado EXACTO (carácter por carácter) del claim "
        "original que contenga o esté junto al marcador de la cita referenciada en "
        "old_citation_refs. \"\" no es válido para citation_text_span en ningún caso. "
        "attribution_relation sigue la MISMA regla: debe ser el valor JSON null siempre que "
        "action_type != 'CORRECT_ATTRIBUTION' (incluyendo cuando correction_decision no es "
        "PROPOSE_CHANGE) -- NUNCA \"\" ni ningún string. Solo cuando action_type=='CORRECT_"
        "ATTRIBUTION', attribution_relation es un string no vacío tomado de allowed_attribution_"
        "relations. "
        "change_scope tiene un valor válido adicional según el caso: cuando "
        "correction_decision NO es PROPOSE_CHANGE, change_scope debe ser exactamente \"NONE\" "
        "(igual que semantic_change_level); cuando correction_decision SÍ es PROPOSE_CHANGE, "
        "change_scope debe ser uno de allowed_change_scopes (TOKEN/PHRASE/CLAUSE/SENTENCE/"
        "MULTI_SENTENCE) y NUNCA \"NONE\" -- \"NONE\" solo es válido para change_scope cuando "
        "no hay corrección. "
        "target_text se reemplaza mecánicamente por replacement_text dentro del claim original "
        "(splice de caracteres, sin limpieza posterior de espacios ni puntuación). Para "
        "REMOVE_UNSUPPORTED_FRAGMENT (replacement_text=\"\"), target_text DEBE incluir el "
        "espacio, coma o conector adyacente necesario para que el texto restante quede "
        "gramaticalmente limpio tras la eliminación -- nunca dejes doble espacio ni un signo de "
        "puntuación huérfano (ej. \" ,\" o \"  \"). Antes de responder, verifica mentalmente cómo "
        "queda el texto tras pegar original[:start]+replacement_text+original[end:] y ajusta los "
        "límites de target_text si el resultado no seria una oración limpia."
    )
    evidence = [{
        "evidence_id": x["evidence_id"], "source_filename": x["source_filename"],
        "chunk_id": x["chunk_id"], "text": _canonical_correction_evidence_text(x),
        "authorized_for_section": x.get("authorized_for_section"), "usage_role": x.get("usage_role"),
    } for x in eligible_evidence]
    payload = {
        "prompt_version": policy["correction_user_prompt_version"],
        "claim": {"claim_id": context["claim_id"], "section_id": context["section_id"],
                  "original_claim_text": context["original_claim_text"]},
        "source_verdict": context.get("scientific_verdict"),
        "source_issue_codes": sorted(set(context.get("deterministic_issue_codes", ())) | set(context.get("semantic_issue_codes", ()))),
        "eligible_evidence": evidence,
        "allowed_correction_decisions": ["NO_CORRECTION", "PROPOSE_CHANGE", "DEFER_TO_MANUAL_REVIEW", "NOT_CORRECTABLE"],
        "allowed_action_types": ["REMOVE_UNSUPPORTED_FRAGMENT", "REPLACE_NUMERIC_VALUE", "CORRECT_ATTRIBUTION", "NARROW_SCOPE", "ADD_QUALIFICATION", "REPLACE_CITATION", "SPLIT_CLAIM"],
        "allowed_change_scopes": list(CORRECTION_CHANGE_SCOPES),
        "allowed_semantic_change_levels": list(CORRECTION_SEMANTIC_CHANGE_LEVELS),
        "allowed_reason_codes": list(CORRECTION_REASON_CODES),
        "allowed_attribution_relations": list(ATTRIBUTION_RELATIONS),
        "previous_errors": list(previous_errors),
        "response_fields": list(CORRECTION_RESPONSE_FIELDS),
        "correction_contract": {
            "claim_id_must_equal": context["claim_id"],
            "recommendation_coherence": (
                "llm_correction_recommendation must be the JSON boolean true when "
                "correction_decision=='PROPOSE_CHANGE', and the JSON boolean false for every other "
                "correction_decision (NO_CORRECTION, DEFER_TO_MANUAL_REVIEW, NOT_CORRECTABLE). "
                "Never set llm_correction_recommendation to a string -- it is not the same field as "
                "correction_decision and must never contain its value."
            ),
            "citation_span_rule": "REPLACE_CITATION requires citation_text_span linked to old_citation_refs by a contractual source/chunk marker",
            "citation_text_span_null_rule": (
                "citation_text_span must be the JSON value null whenever action_type != "
                "'REPLACE_CITATION' -- never \"\" (empty string). Same rule as action_type, "
                "opposite of metric_context/unit_context. When action_type == "
                "'REPLACE_CITATION', it must be an object {coordinate_base, coordinate_system, "
                "start, end, text} -- never a bare string, not even the citation text itself."
            ),
            "attribution_relation_null_rule": (
                "attribution_relation must be the JSON value null whenever action_type != "
                "'CORRECT_ATTRIBUTION' -- never \"\" (empty string). Same rule as action_type/"
                "citation_text_span. When action_type == 'CORRECT_ATTRIBUTION', it must be a "
                "non-empty string from allowed_attribution_relations."
            ),
            "target_text_removal_whitespace_rule": (
                "target_text is spliced mechanically as original[:start]+replacement_text+"
                "original[end:], with no automatic whitespace/punctuation cleanup. For "
                "REMOVE_UNSUPPORTED_FRAGMENT (replacement_text=''), target_text must include "
                "the adjacent space/connector so the spliced result has no double space and no "
                "orphaned punctuation."
            ),
            "narrow_scope_rule": "NARROW_SCOPE requires non-empty new_conditions supported by authorized evidence and cannot alter citations, attribution, or numeric values",
            "format_retry_semantics": "max_correction_format_repair_attempts counts extra calls after the first format-invalid response",
            "reason_codes_rule": (
                "reason_codes must be a subset of allowed_reason_codes exactly as listed -- action_type "
                "values (e.g. NARROW_SCOPE, ADD_QUALIFICATION) are never valid reason_codes."
            ),
            "metric_unit_context_rule": (
                "metric_context and unit_context are always a JSON string, never null -- use \"\" "
                "(empty string) when there is no metric/unit context, not the JSON value null."
            ),
            "action_type_rule": (
                "action_type is the OPPOSITE of metric_context/unit_context: it must be the JSON "
                "value null whenever correction_decision != 'PROPOSE_CHANGE', and must be a "
                "non-empty string from allowed_action_types when correction_decision == "
                "'PROPOSE_CHANGE'. \"\" (empty string) is never valid for action_type."
            ),
            "citation_refs_shape_rule": (
                "old_citation_refs and new_citation_refs are lists of objects "
                "{\"source_filename\": str, \"chunk_id\": str} copied from eligible_evidence -- "
                "never a bare evidence_id string like \"E03\"."
            ),
            "numeric_pairs_shape_rule": (
                "old_numeric_pairs and new_numeric_pairs are lists of 2-element string pairs "
                "[old_value, new_value] -- never an object with keys like old_value/new_value."
            ),
            "evidence_ids_shape_rule": (
                "evidence_ids is a list of bare string IDs (e.g. \"E03\"), copied from the "
                "evidence_id field of eligible_evidence. It is NOT the same shape as "
                "old_citation_refs/new_citation_refs -- never put {source_filename, chunk_id} "
                "objects into evidence_ids."
            ),
            "change_scope_none_rule": (
                "change_scope must be exactly \"NONE\" when correction_decision != "
                "'PROPOSE_CHANGE' (matching semantic_change_level). When correction_decision "
                "== 'PROPOSE_CHANGE', change_scope must be one of allowed_change_scopes and "
                "must never be \"NONE\"."
            ),
        },
    }
    return {"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}

def parse_correction_response(raw_text: Any) -> dict[str, Any]:
    """Desenvuelve la respuesta cruda del LLM antes de exigirle forma JSON
    pura -- igual que ``parse_verification_response`` ya hacía. Antes de
    este arreglo, ``raw_text`` recibía el objeto ``AIMessage`` de LangChain
    sin desenvolver (nunca un ``str``), así que
    ``isinstance(raw_text, str)`` siempre daba ``False`` y el chequeo
    ``CORRECTION_RESPONSE_EMPTY`` se disparaba en TODOS los intentos, sin
    importar qué tan bien formado viniera el JSON real dentro de
    ``.content`` -- confirmado con datos crudos reales de
    experimento_paper_13 (bug preexistente, expuesto recién ahora porque
    antes casi nunca se llegaba a ejercitar este paso)."""

    normalized = normalize_verification_llm_response(raw_text)
    if isinstance(normalized, Mapping):
        return dict(normalized)
    if not normalized.strip():
        raise ValueError("CORRECTION_RESPONSE_EMPTY")
    text = normalized.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("CORRECTION_RESPONSE_NOT_PURE_JSON_OBJECT")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"CORRECTION_RESPONSE_INVALID_JSON:{exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("CORRECTION_RESPONSE_ROOT_NOT_OBJECT")
    return value

# Phase 6.3: prompts independientes para reverificación virtual previa a aplicación.
AGENT07_REVERIFICATION_SYSTEM_V1 = "AGENT07_REVERIFICATION_SYSTEM_V1"
AGENT07_REVERIFICATION_USER_V1 = "AGENT07_REVERIFICATION_USER_V1"
REVERIFICATION_RESPONSE_FIELDS = (
    "correction_id", "claim_id", "proposed_verdict", "support_level",
    "evidence_ids_used", "observed_issue_codes", "target_issues_resolved",
    "supported_meaning_preserved", "intended_semantic_change_valid",
    "unintended_semantic_change_absent", "scope_change_valid",
    "numeric_change_valid", "attribution_change_valid", "citation_change_valid",
    "manual_review_recommended", "reason_codes", "rationale", "confidence",
)

def build_reverification_messages(context: Mapping[str, Any], *, previous_errors: Sequence[str] = ()) -> tuple[dict[str, str], dict[str, str]]:
    policy = context["policy"]
    system = (
        f"Prompt {policy['reverification_system_prompt_version']}. "
        "Realiza una reverificación virtual independiente previa a aplicación. Evalúa únicamente "
        "el claim virtual propuesto y compáralo con el original. Usa solo la evidencia entregada; "
        "no uses conocimiento externo, no solicites retrieval, no propongas nuevas correcciones y "
        "no decidas ACCEPT_FOR_07C. Devuelve exclusivamente JSON puro."
    )
    evidence = [{
        "evidence_id": row["evidence_id"],
        "source_filename": row["source_filename"],
        "chunk_id": row["chunk_id"],
        "text": row.get("canonical_text") or row.get("contractual_text") or row.get("text") or "",
        "authorized_for_section": bool(row.get("authorized_for_section", False)),
        "usage_role": row.get("usage_role", "SUPPORT"),
    } for row in context["authorized_evidence"]]
    payload = {
        "prompt_version": policy["reverification_user_prompt_version"],
        "verification_mode": "REVERIFICATION",
        "correction_id": context["correction_id"],
        "claim_id": context["claim_id"],
        "section_id": context["section_id"],
        "original_claim_text": context["original_claim_text"],
        "claim_text": context["claim_text"],
        "source_verdict": context["source_verdict"],
        "source_issue_codes": list(context["source_issue_codes"]),
        "target_issue_codes": list(context["target_issue_codes"]),
        "correction_action_type": context["correction_action_type"],
        "allowed_evidence_ids": list(context["allowed_evidence_ids"]),
        "authorized_evidence": evidence,
        "retrieval_allowed": False,
        "retrieval_rounds": 0,
        "previous_errors": list(previous_errors),
        "response_fields": list(REVERIFICATION_RESPONSE_FIELDS),
    }
    return {"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}

def parse_reverification_response(raw_text: str) -> dict[str, Any]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("REVERIFICATION_OUTPUT_INVALID_JSON")
    text = raw_text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("REVERIFICATION_OUTPUT_INVALID_JSON")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("REVERIFICATION_OUTPUT_INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("REVERIFICATION_OUTPUT_SCHEMA_INVALID")
    return value


# Phase 6.3R prompt/schema: action-specific assessments and frozen context fingerprint.
REVERIFICATION_RESPONSE_FIELDS = (
    "correction_id", "claim_id", "proposed_verdict", "support_level",
    "evidence_ids_used", "observed_issue_codes", "target_issues_resolved",
    "supported_meaning_preserved", "intended_semantic_change_valid",
    "unintended_semantic_change_absent", "scope_assessment",
    "numeric_assessment", "attribution_assessment", "citation_assessment",
    "manual_review_recommended", "reason_codes", "rationale", "confidence",
)

def build_reverification_messages(context: Mapping[str, Any], *, previous_errors: Sequence[str] = ()) -> tuple[dict[str, str], dict[str, str]]:
    policy = context["policy"]
    system = {
        "role": "system",
        "content": (
            f"Prompt {policy['reverification_system_prompt_version']}. "
            "Evalúa únicamente el claim virtual propuesto con la evidencia congelada entregada. "
            "No uses conocimiento externo, no solicites retrieval, no propongas correcciones y no "
            "decidas aceptación para 07C. Devuelve un único objeto JSON puro. "
            "Los assessments usan VALID, INVALID o NOT_APPLICABLE y deben respetar la acción."
        ),
    }
    evidence = []
    for row in context["authorized_evidence"]:
        evidence.append({
            "evidence_id": row["evidence_id"],
            "source_filename": row["source_filename"],
            "chunk_id": row["chunk_id"],
            "authorized_for_section": row["authorized_for_section"],
            "usage_role": row["usage_role"],
            "text": row.get("canonical_text") or row.get("contractual_text") or row.get("text") or "",
        })
    payload = {
        "prompt_version": policy["reverification_user_prompt_version"],
        "reverification_context_fingerprint": context["reverification_context_fingerprint"],
        "verification_mode": "REVERIFICATION",
        "correction_id": context["correction_id"],
        "claim_id": context["claim_id"],
        "section_id": context["section_id"],
        "original_claim_text": context["original_claim_text"],
        "virtual_proposed_claim_text": context["claim_text"],
        "source_verdict": context["source_verdict"],
        "source_issue_codes": list(context["source_issue_codes"]),
        "target_issue_codes": list(context["target_issue_codes"]),
        "correction_action_type": context["correction_action_type"],
        "allowed_evidence_ids": list(context["allowed_evidence_ids"]),
        "authorized_evidence": evidence,
        "retrieval_allowed": False,
        "retrieval_rounds": 0,
        "allowed_observed_issue_codes": list(policy["reverification_observed_issue_codes"]),
        "allowed_reason_codes": list(policy["reverification_llm_reason_codes"]),
        "assessment_values": ["VALID", "INVALID", "NOT_APPLICABLE"],
        "previous_errors": list(previous_errors),
        "response_fields": list(REVERIFICATION_RESPONSE_FIELDS),
    }
    return system, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}
