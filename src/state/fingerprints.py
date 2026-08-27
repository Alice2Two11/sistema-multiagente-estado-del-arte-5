"""Deterministic SHA-256 fingerprints for pipeline stages.

This module only canonicalizes values and calculates hashes. It does not decide
whether a stage should execute and does not modify ``PipelineState``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .pipeline_state import StageFingerprints


_DEFAULT_BLOCK_SIZE = 1024 * 1024


def canonicalize(value: Any) -> Any:
    """Convert a supported value into a deterministic JSON-compatible value.

    Supported inputs are ``None``, scalar JSON values, ``Path``, enums,
    dataclass instances, mappings, lists, tuples, sets and frozensets.
    Unsupported values raise ``TypeError``; no implicit ``str(value)`` fallback
    is used.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Los floats no finitos no pueden canonicalizarse.")
        return value

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, Enum):
        return canonicalize(value.value)

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: canonicalize(getattr(value, item.name))
            for item in fields(value)
        }

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Las claves de los mappings deben ser cadenas.")
            if key in normalized:
                raise ValueError(f"Clave duplicada tras normalización: {key!r}.")
            normalized[key] = canonicalize(item)
        return {key: normalized[key] for key in sorted(normalized)}

    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]

    if isinstance(value, (set, frozenset)):
        canonical_items = [canonicalize(item) for item in value]
        return sorted(canonical_items, key=stable_json_dumps)

    raise TypeError(
        f"Tipo no soportado para canonicalización: {type(value).__name__}."
    )


def stable_json_dumps(value: Any) -> str:
    """Serialize a supported value using stable JSON formatting."""

    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of bytes."""

    if not isinstance(data, bytes):
        raise TypeError("data debe ser bytes.")
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str, encoding: str = "utf-8") -> str:
    """Return the hexadecimal SHA-256 digest of text."""

    if not isinstance(text, str):
        raise TypeError("text debe ser una cadena.")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("encoding debe ser una cadena no vacía.")
    return sha256_bytes(text.encode(encoding))


def sha256_file(path: str | Path, block_size: int = _DEFAULT_BLOCK_SIZE) -> str:
    """Hash a regular file by reading it in blocks."""

    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise TypeError("block_size debe ser un entero.")
    if block_size < 1:
        raise ValueError("block_size debe ser mayor o igual a 1.")

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No existe un archivo regular: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_mapping(value: Mapping[str, Any]) -> str:
    """Calculate a deterministic fingerprint for a mapping."""

    if not isinstance(value, Mapping):
        raise TypeError("value debe ser un mapping.")
    return sha256_text(stable_json_dumps(value))


def build_stage_fingerprints(
    *,
    input_data: Mapping[str, Any],
    config_data: Mapping[str, Any],
    dependencies_data: Mapping[str, Any],
) -> StageFingerprints:
    """Build input, config, dependencies and composite fingerprints."""

    input_fingerprint = fingerprint_mapping(input_data)
    config_fingerprint = fingerprint_mapping(config_data)
    dependencies_fingerprint = fingerprint_mapping(dependencies_data)
    composite_fingerprint = fingerprint_mapping(
        {
            "input": input_fingerprint,
            "config": config_fingerprint,
            "dependencies": dependencies_fingerprint,
        }
    )
    return StageFingerprints(
        input=input_fingerprint,
        config=config_fingerprint,
        dependencies=dependencies_fingerprint,
        composite=composite_fingerprint,
    )


def fingerprints_match(
    left: StageFingerprints | Mapping[str, Any],
    right: StageFingerprints | Mapping[str, Any],
) -> bool:
    """Return whether two complete stage-fingerprint snapshots match."""

    left_value = (
        left if isinstance(left, StageFingerprints) else StageFingerprints.from_dict(left)
    )
    right_value = (
        right
        if isinstance(right, StageFingerprints)
        else StageFingerprints.from_dict(right)
    )
    return left_value == right_value
