"""Validación de propuestas de corrección numérica/cuantitativa (Agente 07).

Extraído mecánicamente de src.tools.verification.validation (Bloque C,
C5) -- sin cambios de comportamiento, lógica, firmas ni mensajes de
error respecto al código original (incluye el import diferido local
`.corrections`, sin modificar). validation.py reexporta estos
símbolos para preservar el contrato de importación existente.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


def validate_correction_proposal_response(value: Mapping[str, Any], *, allowed_evidence_ids: tuple[str, ...]) -> dict[str, Any]:
    from .corrections import validate_correction_response
    return validate_correction_response(value, allowed_evidence_ids=allowed_evidence_ids)

def validate_correction_text_integrity(text: str) -> tuple[str, ...]:
    from .corrections import validate_text_integrity
    return validate_text_integrity(text)


# Phase 5R: reutilización de literales cuantitativos estrictos.
def _normalize_decimal_literal(value: str) -> str:
    value = value.strip().replace(",", ".")
    try:
        from decimal import Decimal
        d = Decimal(value)
        return format(d.normalize(), "f")
    except Exception:
        return value

def extract_quantitative_pairs_strict(text: str) -> tuple[tuple[str, str], ...]:
    """Extrae pares valor-unidad con límites explícitos; soporta %, ms y símbolos."""
    pattern = re.compile(r"(?<![\w\d.,])([+-]?\d+(?:[.,]\d+)?)\s*([%‰°µμ/\w-]+)(?![\w])", re.UNICODE)
    pairs=[]
    for m in pattern.finditer(text):
        pair=(_normalize_decimal_literal(m.group(1)), m.group(2).casefold())
        if pair not in pairs: pairs.append(pair)
    return tuple(pairs)

def quantitative_pair_supported(text: str, pair: tuple[str, str]) -> bool:
    expected=(_normalize_decimal_literal(pair[0]), pair[1].strip().casefold())
    return expected in extract_quantitative_pairs_strict(text)

def metric_context_supported(text: str, metric_context: str) -> bool:
    terms=[t.casefold() for t in re.findall(r"[\w-]+", metric_context, re.UNICODE) if t.strip()]
    tokens={t.casefold() for t in re.findall(r"[\w-]+", text, re.UNICODE)}
    return bool(terms) and all(term in tokens for term in terms)
