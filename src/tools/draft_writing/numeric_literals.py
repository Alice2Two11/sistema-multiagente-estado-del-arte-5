from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_NUMERIC_VALUE_RE = re.compile(
    r"^\s*(?P<sign>[+-]?)(?P<integer>\d+)(?:[.,](?P<fraction>\d+))?(?P<percent>\s*%)?\s*$"
)


@dataclass(frozen=True)
class NumericLiteral:
    sign: str
    integer: str
    fraction: str | None
    is_percentage: bool

    @property
    def canonical_value(self) -> str:
        decimal = self.integer
        if self.fraction is not None:
            decimal = f"{decimal}.{self.fraction}"
        if self.sign:
            decimal = f"{self.sign}{decimal}"
        return f"{decimal}%" if self.is_percentage else decimal


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def parse_numeric_literal(value: Any) -> NumericLiteral | None:
    match = _NUMERIC_VALUE_RE.fullmatch(_safe_text(value))
    if match is None:
        return None
    return NumericLiteral(
        sign=match.group("sign") or "",
        integer=match.group("integer"),
        fraction=match.group("fraction"),
        is_percentage=bool(match.group("percent")),
    )


def _literal_pattern(literal: NumericLiteral) -> re.Pattern[str]:
    sign = re.escape(literal.sign)
    integer = re.escape(literal.integer)
    decimal = integer
    if literal.fraction is not None:
        decimal = rf"{integer}[.,]{re.escape(literal.fraction)}"

    percentage = r"\s*%" if literal.is_percentage else r"(?!\s*%)"
    return re.compile(
        rf"(?<![\w.,]){sign}{decimal}{percentage}(?![\w.,])"
    )


def numeric_literal_exists_in_text(value: Any, text: Any) -> bool:
    """
    Match a numeric literal as a complete token.

    Decimal comma and decimal point are treated as equivalent. Percentage
    values only match percentage tokens, and non-percentage values do not
    match percentage tokens. No semantic conversions, rounding or unit
    transformations are performed.
    """
    literal = parse_numeric_literal(value)
    if literal is None:
        return False
    return _literal_pattern(literal).search(_safe_text(text)) is not None


def historical_number_exists_in_text(value: Any, text: Any) -> bool:
    """Exact reproduction of the current validator's comma-normalized substring rule."""
    token = "".join("." if character == "," else character for character in _safe_text(value))
    normalized_text = "".join("." if character == "," else character for character in _safe_text(text))
    return token in normalized_text
