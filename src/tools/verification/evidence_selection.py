"""Selección de evidencia y contexto de verificación de claims (Agente 07).

Extraído mecánicamente de src.tools.verification.validation (Bloque C,
C2) -- sin cambios de comportamiento, lógica, firmas ni mensajes de
error respecto al código original. validation.py reexporta estos
símbolos para preservar el contrato de importación existente.

Replica localmente el subconjunto del bloque "Phase 4" (ver
validation.py) que este cluster realmente usa (_dataclass, _asdict,
_Protocol, _get_phase4_policy) -- el bloque completo permanece en
validation.py porque también lo usan otros clusters/bloques
contaminados que no se movieron en este Bloque C.
"""
from __future__ import annotations

from dataclasses import dataclass as _dataclass, asdict as _asdict
from typing import Any, Mapping, Protocol as _Protocol

from src.config.verification_policy_config import (
    CLAIM_TYPES,
    VERIFICATION_INTENSITIES,
    get_verification_input_policy as _get_phase4_policy,
)


class ClaimRetrievalTool(_Protocol):
    """Interfaz inyectable para solicitar evidencia adicional por claim."""

    def retrieve_more(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


@_dataclass(frozen=True, slots=True)
class EvidenceSelection:
    eligible_evidence: tuple[dict[str, Any], ...]
    deterministically_discarded_evidence: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


_REQUIRED_LLM_FIELDS = {
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
}


def validate_claim_verification_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        raise ValueError("CLAIM_VERIFICATION_INPUT_NOT_MAPPING")
    required = (
        "claim_id", "claim_id_origin", "section_id", "section_title", "claim_text",
        "claim_type", "verification_intensity", "supporting_citations",
        "inherited_evidence_assessment", "retrieval_result", "deterministic_validation",
        "allowed_source_pairs", "policy", "attempt_context",
    )
    missing = [key for key in required if key not in context]
    if missing:
        raise ValueError(f"CLAIM_VERIFICATION_INPUT_FIELDS_MISSING:{','.join(missing)}")
    value = dict(context)
    for key in ("claim_id", "section_id", "claim_text", "claim_type", "verification_intensity"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"CLAIM_VERIFICATION_INPUT_INVALID:{key}")
        value[key] = value[key].strip()
    if value["claim_type"] not in CLAIM_TYPES:
        raise ValueError(f"CLAIM_VERIFICATION_INPUT_UNKNOWN_CLAIM_TYPE:{value['claim_type']}")
    if value["verification_intensity"] not in VERIFICATION_INTENSITIES:
        raise ValueError(f"CLAIM_VERIFICATION_INPUT_UNKNOWN_INTENSITY:{value['verification_intensity']}")
    if type(value["supporting_citations"]) not in (list, tuple):
        raise ValueError("CLAIM_VERIFICATION_INPUT_INVALID:supporting_citations")
    for item in value["supporting_citations"]:
        if not isinstance(item, Mapping):
            raise ValueError("CLAIM_VERIFICATION_INPUT_INVALID:supporting_citations_item")
        if not str(item.get("source_filename", "")).strip() or not str(item.get("chunk_id", "")).strip():
            raise ValueError("CLAIM_VERIFICATION_INPUT_INVALID:supporting_citation_identity")
    if type(value["allowed_source_pairs"]) not in (list, tuple):
        raise ValueError("CLAIM_VERIFICATION_INPUT_INVALID:allowed_source_pairs")
    normalized_pairs = []
    for item in value["allowed_source_pairs"]:
        if type(item) not in (list, tuple) or len(item) != 2:
            raise ValueError("CLAIM_VERIFICATION_INPUT_INVALID:allowed_source_pair")
        source, chunk = item
        if not isinstance(source, str) or not source.strip() or not isinstance(chunk, str) or not chunk.strip():
            raise ValueError("CLAIM_VERIFICATION_INPUT_INVALID:allowed_source_pair")
        normalized_pairs.append((source.strip(), chunk.strip()))
    value["allowed_source_pairs"] = tuple(dict.fromkeys(normalized_pairs))
    for key in ("inherited_evidence_assessment", "retrieval_result", "deterministic_validation", "attempt_context"):
        if not isinstance(value[key], Mapping):
            raise ValueError(f"CLAIM_VERIFICATION_INPUT_INVALID:{key}")
        value[key] = dict(value[key])
    value["policy"] = _get_phase4_policy(value["policy"])

    related = value.get("related_claims", ())
    legacy_ids = value.get("related_claim_ids", ())
    if related and legacy_ids:
        # Normalize legacy parallel structures only when they are exactly aligned.
        if type(related) not in (list, tuple) or type(legacy_ids) not in (list, tuple) or len(related) != len(legacy_ids):
            raise ValueError("RELATED_CLAIMS_IDS_CONTEXT_INCOMPLETE")
        if all(isinstance(item, str) for item in related):
            related = tuple({"claim_id": cid, "claim_text": text} for cid, text in zip(legacy_ids, related))
    elif legacy_ids and not related:
        raise ValueError("RELATED_CLAIMS_IDS_CONTEXT_INCOMPLETE")
    if related:
        if type(related) not in (list, tuple):
            raise ValueError("RELATED_CLAIMS_INVALID")
        normalized_related = []
        seen_ids = set()
        for item in related:
            if not isinstance(item, Mapping):
                raise ValueError("RELATED_CLAIMS_INVALID_ITEM")
            rid = item.get("claim_id")
            text = item.get("claim_text")
            if not isinstance(rid, str) or not rid.strip() or not isinstance(text, str) or not text.strip():
                raise ValueError("RELATED_CLAIMS_INVALID_ITEM")
            rid = rid.strip()
            if rid == value["claim_id"]:
                raise ValueError("RELATED_CLAIM_CANNOT_REFERENCE_SELF")
            if rid in seen_ids:
                raise ValueError("DUPLICATE_RELATED_CLAIM_ID")
            seen_ids.add(rid)
            normalized_related.append({"claim_id": rid, "claim_text": text.strip()})
        value["related_claims"] = tuple(normalized_related)
        value["related_claim_ids"] = tuple(item["claim_id"] for item in normalized_related)
    else:
        value["related_claims"] = ()
        value["related_claim_ids"] = ()
    return value




def canonical_correction_evidence_text(row: Mapping[str, Any]) -> str:
    """Texto contractual único para todas las validaciones de corrección.

    Prioridad congelada: canonical_text -> contractual_text -> text.
    """
    if not isinstance(row, Mapping):
        return ""
    for field in ("canonical_text", "contractual_text", "text"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""

def _canonical_evidence_rows(context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inherited = context.get("inherited_evidence_assessment", {})
    retrieval = context.get("retrieval_result", {})
    raw_rows = []
    for row in inherited.get("evidence_rows", ()):
        raw_rows.append((0, "INHERITED_DIRECT", row))
    for row in inherited.get("additional_evidence_rows", ()):
        raw_rows.append((1, "INHERITED_ADDITIONAL", row))
    for row in retrieval.get("selected_candidates", ()):
        raw_rows.append((2, "RETRIEVED", row))

    allowed_pairs = {tuple(item) for item in context.get("allowed_source_pairs", ())}
    eligible: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for priority, origin, raw in raw_rows:
        row = dict(raw)
        source = str(row.get("source_filename", "")).strip()
        chunk = str(row.get("chunk_id", "")).strip()
        canonical = row.get("canonical_text") or row.get("contractual_text") or row.get("text")
        reasons: list[str] = []
        if not source or not chunk:
            reasons.append("MISSING_DOCUMENT_IDENTITY")
        if not isinstance(canonical, str) or not canonical.strip():
            reasons.append("CANONICAL_TEXT_UNAVAILABLE")
        if row.get("text_match_status") == "CANDIDATE_TEXT_VARIATION" and not row.get("canonical_text"):
            reasons.append("CANONICAL_TEXT_UNAVAILABLE")
        technical_invalid = bool(row.get("technical_invalid", False))
        if technical_invalid:
            reasons.append("TECHNICALLY_INVALID_EVIDENCE")
        pair = (source, chunk)
        outside = bool(row.get("outside_section_sources", False))
        authorized = pair in allowed_pairs and not outside
        role = "CONTEXT" if outside else "SUPPORT"
        if outside:
            role = "CONTRAST" if row.get("retrieval_scope") == "CORPUS_WIDE_CONTRADICTION" else "CONTEXT"
        if reasons:
            discarded.append({**row, "discard_reason_codes": tuple(sorted(set(reasons)))})
            continue
        if pair in seen:
            continue
        seen.add(pair)
        eligible.append({
            **row,
            "source_filename": source,
            "chunk_id": chunk,
            "text": canonical.strip(),
            "authorized_for_section": authorized,
            "outside_section_sources": outside,
            "usage_allowed": role,
            "retrieval_origin": origin,
            "_priority": priority,
        })
    eligible.sort(key=lambda row: (
        row["_priority"],
        0 if row["authorized_for_section"] else 1,
        -float(row.get("fused_rrf_score", 0.0) or 0.0),
        row["source_filename"],
        row["chunk_id"],
    ))
    return eligible, discarded


def select_evidence_for_scientific_judgment(context: Mapping[str, Any]) -> EvidenceSelection:
    value = validate_claim_verification_context(context)
    policy = value["policy"]
    eligible, discarded = _canonical_evidence_rows(value)
    max_chunks = int(policy["max_llm_evidence_chunks_per_claim"])
    max_chars = int(policy["max_total_evidence_chars"])
    max_per_source = int(policy["max_llm_evidence_per_source"])
    max_contrast = int(policy["max_contrast_evidence_chunks"])
    selected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    contrast_count = 0
    chars = 0
    for row in eligible:
        reasons: list[str] = []
        if len(selected) >= max_chunks:
            reasons.append("MAX_LLM_EVIDENCE_CHUNKS_REACHED")
        if source_counts.get(row["source_filename"], 0) >= max_per_source:
            reasons.append("MAX_LLM_EVIDENCE_PER_SOURCE_REACHED")
        is_contrast = row["usage_allowed"] in {"CONTRAST", "CONTEXT"}
        if is_contrast and contrast_count >= max_contrast:
            reasons.append("MAX_CONTRAST_EVIDENCE_REACHED")
        text_len = len(row["text"])
        if chars + text_len > max_chars:
            reasons.append("MAX_TOTAL_EVIDENCE_CHARS_REACHED")
        if reasons:
            discarded.append({**{k: v for k, v in row.items() if k != "_priority"}, "discard_reason_codes": tuple(reasons)})
            continue
        clean = {k: v for k, v in row.items() if k != "_priority"}
        clean["evidence_id"] = f"E{len(selected)+1:02d}"
        selected.append(clean)
        source_counts[row["source_filename"]] = source_counts.get(row["source_filename"], 0) + 1
        contrast_count += int(is_contrast)
        chars += text_len
    return EvidenceSelection(tuple(selected), tuple(discarded))


def deterministic_precheck(context: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_claim_verification_context(context)
    deterministic = dict(value["deterministic_validation"])
    claim_type = value["claim_type"]
    issues: list[str] = list(deterministic.get("deterministic_issue_codes", ()))
    # Códigos que pertenecen al vocabulario RETRIEVAL_REASON_CODES, no a
    # DETERMINISTIC_ISSUE_CODES -- nunca deben mezclarse con "issues"
    # (que se convierte en deterministic_issue_codes más abajo). Se
    # acumulan aquí, aparte, y se devuelven como retrieval_reason_codes
    # para que el llamador los incorpore a reason_codes -- el campo
    # cuyo vocabulario permitido SÍ incluye RETRIEVAL_REASON_CODES (ver
    # validate_claim_verification_result_contract).
    retrieval_reason_codes: list[str] = []
    judgment_required = claim_type not in {"ORGANIZATIONAL", "TRANSITIONAL"}
    terminal_verdict = ""

    if not judgment_required:
        terminal_verdict = "NOT_APPLICABLE"
    citation_valid = bool(deterministic.get("citation_valid", True))
    identity_valid = bool(deterministic.get("document_identity_valid", True))
    authorization_valid = bool(deterministic.get("authorization_valid", True))
    if not citation_valid:
        issues.append("INVALID_CITATION")
    if not identity_valid:
        issues.append("DOCUMENT_IDENTITY_INVALID")
    if not authorization_valid:
        issues.append("UNAUTHORIZED_SOURCE")

    numeric_valid = bool(deterministic.get("numeric_pairs_valid", True))
    retrieval = value.get("retrieval_result", {})
    retrieval_budget = int(value["attempt_context"].get("remaining_retrieval_requests", 0))
    retrieval_possible = retrieval_budget > 0 and not bool(retrieval.get("terminal_technical_blocker", False))
    numeric_terminal = False
    if claim_type == "QUANTITATIVE" and not numeric_valid:
        if retrieval_possible:
            retrieval_reason_codes.append("QUANTITATIVE_COVERAGE_INCOMPLETE")
        else:
            issues.append("UNSUPPORTED_NUMERIC_VALUE")
            numeric_terminal = True

    technical_blockers = tuple(deterministic.get("technical_blockers", ()))
    technical_status = "OK"
    judgment_status = "PENDING" if judgment_required else "NOT_REQUIRED"
    if technical_blockers and not select_evidence_for_scientific_judgment(value).eligible_evidence:
        technical_status = "RETRIEVAL_BLOCKED"
        judgment_status = "BLOCKED"
        terminal_verdict = "NOT_EVALUATED"
        issues.append("RETRIEVAL_TECHNICAL_BLOCKER")

    if judgment_required and (not citation_valid or not identity_valid or not authorization_valid or numeric_terminal):
        judgment_status = "COMPLETED"
        terminal_verdict = "NOT_EVALUATED"

    return {
        "scientific_judgment_required": judgment_required,
        "execution_status": "COMPLETED",
        "technical_status": technical_status,
        "scientific_judgment_status": judgment_status,
        "scientific_verdict": terminal_verdict or "NOT_EVALUATED",
        "deterministic_issue_codes": tuple(sorted(set(issues))),
        "retrieval_reason_codes": tuple(sorted(set(retrieval_reason_codes))),
        "retrieval_possible": retrieval_possible,
        "terminal_without_llm": bool(terminal_verdict) or judgment_status in {"BLOCKED", "COMPLETED"} and bool(issues),
    }


def allowed_verdicts_for_claim(context: Mapping[str, Any], precheck: Mapping[str, Any]) -> tuple[str, ...]:
    if not precheck["scientific_judgment_required"]:
        return ("NOT_APPLICABLE",)
    if precheck["scientific_judgment_status"] == "BLOCKED":
        return ("NOT_EVALUATED",)
    issues = set(precheck["deterministic_issue_codes"])
    if issues & {"INVALID_CITATION", "DOCUMENT_IDENTITY_INVALID", "UNAUTHORIZED_SOURCE", "UNSUPPORTED_NUMERIC_VALUE"}:
        return ("NOT_EVALUATED",)
    allowed = ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "NOT_VERIFIABLE"]
    return tuple(allowed)
