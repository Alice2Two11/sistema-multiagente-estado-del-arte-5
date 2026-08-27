"""Atomic file-writing helpers for pipeline artifacts.

This module is independent from Colab and project-specific configuration.
It writes a temporary file in the destination directory, validates it,
atomically replaces the destination with ``os.replace()``, and computes the
SHA-256 hash only from the definitive file.

No backup, deletion policy, or PipelineState update is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


Validator = Callable[[Path], None]


@dataclass(frozen=True, slots=True)
class AtomicWriteResult:
    """Metadata for a successfully written definitive file."""

    path: str
    hash: str
    size_bytes: int


def _normalize_destination(destination: str | Path) -> Path:
    if isinstance(destination, bool) or not isinstance(destination, (str, Path)):
        raise TypeError("destination must be a string or pathlib.Path")

    destination_path = Path(destination)
    if not str(destination_path).strip():
        raise ValueError("destination must not be empty")

    if destination_path.exists() and destination_path.is_dir():
        raise IsADirectoryError(f"destination is a directory: {destination_path}")

    return destination_path


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _flush_and_try_fsync(handle: Any) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except (AttributeError, OSError):
        # Some filesystems or file-like implementations may not support fsync.
        pass


def _validate_regular_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"temporary file was not created: {path}")
    if not path.is_file():
        raise ValueError(f"temporary path is not a regular file: {path}")


def _atomic_write(
    destination: str | Path,
    writer: Callable[[Any], None],
    *,
    binary: bool,
    validator: Validator | None = None,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> AtomicWriteResult:
    destination_path = _normalize_destination(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = destination_path.suffix
    prefix = f".{destination_path.name}."
    temp_path: Path | None = None

    try:
        mode = "wb" if binary else "w"
        with tempfile.NamedTemporaryFile(
            mode=mode,
            dir=destination_path.parent,
            prefix=prefix,
            suffix=f"{suffix}.tmp",
            delete=False,
            encoding=None if binary else encoding,
            newline=None if binary else newline,
        ) as handle:
            temp_path = Path(handle.name)
            writer(handle)
            _flush_and_try_fsync(handle)

        _validate_regular_file(temp_path)

        if validator is not None:
            validator(temp_path)

        os.replace(temp_path, destination_path)
        temp_path = None

        _validate_regular_file(destination_path)
        definitive_hash = _sha256_file(destination_path)
        size_bytes = destination_path.stat().st_size

        return AtomicWriteResult(
            path=str(destination_path),
            hash=definitive_hash,
            size_bytes=size_bytes,
        )
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def atomic_write_bytes(
    destination: str | Path,
    data: bytes | bytearray | memoryview,
    *,
    validator: Validator | None = None,
) -> AtomicWriteResult:
    """Atomically write binary data."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes, bytearray, or memoryview")

    payload = bytes(data)
    return _atomic_write(
        destination,
        lambda handle: handle.write(payload),
        binary=True,
        validator=validator,
    )


def atomic_write_text(
    destination: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
    validator: Validator | None = None,
) -> AtomicWriteResult:
    """Atomically write text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("encoding must be a non-empty string")

    return _atomic_write(
        destination,
        lambda handle: handle.write(text),
        binary=False,
        validator=validator,
        encoding=encoding,
        newline=newline,
    )


def _validate_json_file(path: Path, *, encoding: str) -> None:
    with path.open("r", encoding=encoding) as handle:
        json.load(handle)


def atomic_write_json(
    destination: str | Path,
    data: Any,
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> AtomicWriteResult:
    """Serialize and atomically write one valid JSON document."""

    if isinstance(indent, bool) or (indent is not None and not isinstance(indent, int)):
        raise TypeError("indent must be an integer or None")
    if indent is not None and indent < 0:
        raise ValueError("indent must be greater than or equal to 0")

    def writer(handle: Any) -> None:
        json.dump(
            data,
            handle,
            ensure_ascii=ensure_ascii,
            indent=indent,
            sort_keys=sort_keys,
            allow_nan=False,
        )
        handle.write("\n")

    return _atomic_write(
        destination,
        writer,
        binary=False,
        validator=lambda path: _validate_json_file(path, encoding=encoding),
        encoding=encoding,
    )


def _validate_jsonl_file(path: Path, *, encoding: str) -> None:
    with path.open("r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {line_number}")
            json.loads(line)


def atomic_write_jsonl(
    destination: str | Path,
    records: Iterable[Any],
    *,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    sort_keys: bool = True,
) -> AtomicWriteResult:
    """Serialize and atomically write an iterable as JSON Lines."""

    if isinstance(records, (str, bytes, bytearray, memoryview, Mapping)):
        raise TypeError("records must be an iterable of JSON values, not a scalar or mapping")

    try:
        iterator = iter(records)
    except TypeError as exc:
        raise TypeError("records must be iterable") from exc

    def writer(handle: Any) -> None:
        for record in iterator:
            line = json.dumps(
                record,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.write(line)
            handle.write("\n")

    return _atomic_write(
        destination,
        writer,
        binary=False,
        validator=lambda path: _validate_jsonl_file(path, encoding=encoding),
        encoding=encoding,
    )


def _normalize_csv_rows(
    rows: Iterable[Mapping[str, Any] | Sequence[Any]],
    fieldnames: Sequence[str] | None,
) -> tuple[list[Mapping[str, Any] | Sequence[Any]], tuple[str, ...] | None, bool]:
    if isinstance(rows, (str, bytes, bytearray, memoryview, Mapping)):
        raise TypeError("rows must be an iterable of row mappings or row sequences")

    try:
        materialized = list(rows)
    except TypeError as exc:
        raise TypeError("rows must be iterable") from exc

    normalized_fieldnames: tuple[str, ...] | None = None
    if fieldnames is not None:
        if isinstance(fieldnames, (str, bytes)):
            raise TypeError("fieldnames must be a sequence of non-empty strings")
        normalized_fieldnames = tuple(fieldnames)
        if not normalized_fieldnames:
            raise ValueError("fieldnames must not be empty")
        for name in normalized_fieldnames:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("each field name must be a non-empty string")

    if not materialized:
        if normalized_fieldnames is None:
            raise ValueError("fieldnames are required when rows are empty")
        return materialized, normalized_fieldnames, True

    first_is_mapping = isinstance(materialized[0], Mapping)
    for row in materialized:
        if isinstance(row, Mapping) != first_is_mapping:
            raise TypeError("all CSV rows must use the same representation")
        if not isinstance(row, Mapping) and (
            isinstance(row, (str, bytes, bytearray, memoryview))
            or not isinstance(row, Sequence)
        ):
            raise TypeError("CSV sequence rows must be non-string sequences")

    if first_is_mapping:
        if normalized_fieldnames is None:
            normalized_fieldnames = tuple(str(key) for key in materialized[0].keys())
            if not normalized_fieldnames:
                raise ValueError("mapping rows must contain at least one field")
            for name in normalized_fieldnames:
                if not name.strip():
                    raise ValueError("derived field names must be non-empty")
        return materialized, normalized_fieldnames, True

    return materialized, normalized_fieldnames, False


def _validate_csv_file(
    path: Path,
    *,
    encoding: str,
    expected_rows: int,
    has_header: bool,
) -> None:
    with path.open("r", encoding=encoding, newline="") as handle:
        parsed_rows = list(csv.reader(handle))

    expected_total = expected_rows + (1 if has_header else 0)
    if len(parsed_rows) != expected_total:
        raise ValueError(
            f"CSV validation failed: expected {expected_total} rows, "
            f"found {len(parsed_rows)}"
        )


def atomic_write_csv(
    destination: str | Path,
    rows: Iterable[Mapping[str, Any] | Sequence[Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    encoding: str = "utf-8",
    dialect: str = "excel",
) -> AtomicWriteResult:
    """Atomically write CSV rows.

    Mapping rows produce a header and use ``fieldnames`` or the first row's key
    order. Sequence rows are written without a header unless ``fieldnames`` is
    provided, in which case that sequence is written as the header.
    """

    normalized_rows, normalized_fieldnames, mapping_rows = _normalize_csv_rows(
        rows,
        fieldnames,
    )

    def writer(handle: Any) -> None:
        if mapping_rows:
            assert normalized_fieldnames is not None
            csv_writer = csv.DictWriter(
                handle,
                fieldnames=normalized_fieldnames,
                dialect=dialect,
                extrasaction="raise",
            )
            csv_writer.writeheader()
            csv_writer.writerows(normalized_rows)
        else:
            csv_writer = csv.writer(handle, dialect=dialect)
            if normalized_fieldnames is not None:
                csv_writer.writerow(normalized_fieldnames)
            csv_writer.writerows(normalized_rows)

    has_header = mapping_rows or normalized_fieldnames is not None
    return _atomic_write(
        destination,
        writer,
        binary=False,
        validator=lambda path: _validate_csv_file(
            path,
            encoding=encoding,
            expected_rows=len(normalized_rows),
            has_header=has_header,
        ),
        encoding=encoding,
        newline="",
    )
