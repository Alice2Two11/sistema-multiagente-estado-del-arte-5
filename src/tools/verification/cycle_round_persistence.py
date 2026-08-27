"""Persistencia por rondas del ciclo correctivo `06 <-> 07`
(``writer_verifier_cycle/round_NN/``).

Ciclo de vida transaccional de UNA ronda, en dos fases explícitas que NO
se pisan entre sí:

- **Fase 07** (``create_round_awaiting_revision``): crea la ronda de
  forma atómica (staging + rename del directorio completo) con los
  artefactos de 07 (borrador de entrada, resultado de 07,
  ``writer_revision_request.json``, transición, fingerprints). Estado
  final: ``AWAITING_REVISION``.
- **Fase 06** (``complete_round_revision``): COMPLETA esa misma ronda ya
  creada -- nunca la vuelve a crear. Valida que la ronda exista, que
  pertenezca al mismo experimento/ciclo/número de ronda, y que el hash del
  ``writer_revision_request`` coincida exactamente con el que 07 dejó al
  crearla. Escribe los artefactos de 06 en un staging interno a la ronda,
  los mueve uno por uno (rename atómico por archivo) y actualiza el
  archivo de estado AL FINAL (también vía staging + rename atómico de ese
  archivo puntual). Estado final: ``REVISION_COMPLETED``. Un segundo
  intento de completar la misma ronda falla explícitamente.

Ninguna excepción se silencia en ningún punto: ``FileExistsError`` por
sobrescritura, ``RuntimeError`` por inconsistencia (experimento/ronda/hash
distintos, ronda ya completada), o cualquier error de escritura/
serialización se propaga tal cual.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

CYCLE_DIRECTORY_NAME = "writer_verifier_cycle"
ROUND_STATUS_FILENAME = "_round_status.json"

STATUS_AWAITING_REVISION = "AWAITING_REVISION"
STATUS_REVISION_COMPLETED = "REVISION_COMPLETED"
STATUS_REVERIFIED = "REVERIFIED"


def round_directory(project_dir: str | Path, experiment_id: str, round_number: int) -> Path:
    return (
        Path(project_dir)
        / experiment_id
        / "05_outputs"
        / CYCLE_DIRECTORY_NAME
        / f"round_{round_number:02d}"
    )


def _serialize(content: Any) -> str:
    if isinstance(content, (dict, list, tuple)):
        return json.dumps(content, ensure_ascii=False, indent=2, default=str)
    return str(content)


def _stable_hash(value: Any) -> str:
    from src.state.fingerprints import fingerprint_mapping

    if isinstance(value, dict):
        return fingerprint_mapping(value)
    return fingerprint_mapping({"value": value})


def read_round_status(*, project_dir: str | Path, experiment_id: str, round_number: int) -> dict[str, Any] | None:
    status_path = round_directory(project_dir, experiment_id, round_number) / ROUND_STATUS_FILENAME
    if not status_path.is_file():
        return None
    return json.loads(status_path.read_text(encoding="utf-8"))


def create_round_awaiting_revision(
    *,
    project_dir: str | Path,
    experiment_id: str,
    cycle_id: str,
    round_number: int,
    writer_revision_request: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Path]:
    """Fase 07: crea atómicamente ``round_NN`` (staging + rename del
    directorio completo). Lanza ``FileExistsError`` si la ronda ya existe
    -- nunca se sobrescribe. ``artifacts`` debe incluir
    ``"writer_revision_request.json": writer_revision_request`` (se exige
    explícitamente para poder fijar el hash de referencia que
    ``complete_round_revision`` validará después)."""

    if "writer_revision_request.json" not in artifacts:
        raise ValueError(
            "create_round_awaiting_revision requiere 'writer_revision_request.json' "
            "en artifacts -- es el ancla de consistencia para la fase 06."
        )
    if not artifacts:
        raise ValueError("create_round_awaiting_revision requiere al menos un artefacto.")

    final_dir = round_directory(project_dir, experiment_id, round_number)
    if final_dir.exists():
        raise FileExistsError(
            f"La ronda {round_number} ya está persistida en {final_dir} "
            "-- no se sobrescriben rondas anteriores."
        )

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_dir.parent / f".staging_round_{round_number:02d}_{uuid.uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        for filename, content in artifacts.items():
            (staging_dir / filename).write_text(_serialize(content), encoding="utf-8")

        status = {
            "status": STATUS_AWAITING_REVISION,
            "experiment_id": experiment_id,
            "cycle_id": cycle_id,
            "round_number": round_number,
            "writer_revision_request_hash": _stable_hash(writer_revision_request),
        }
        (staging_dir / ROUND_STATUS_FILENAME).write_text(_serialize(status), encoding="utf-8")

        missing = [name for name in list(artifacts) + [ROUND_STATUS_FILENAME] if not (staging_dir / name).is_file()]
        if missing:
            raise OSError(f"Artefactos faltantes tras escribir el staging de la ronda {round_number}: {missing}")

        staging_dir.rename(final_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {filename: final_dir / filename for filename in artifacts}


def complete_round_revision(
    *,
    project_dir: str | Path,
    experiment_id: str,
    cycle_id: str,
    round_number: int,
    writer_revision_request: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Path]:
    """Fase 06: COMPLETA una ronda que 07 ya creó (nunca la crea). Exige
    que la ronda exista y esté en ``AWAITING_REVISION``; valida
    experimento/ciclo/ronda y el hash EXACTO del ``writer_revision_request``
    contra el que 07 dejó al crear la ronda -- 06 no puede consumir un
    request de otra ronda, otro experimento, u otro request alterado (ni
    siquiera un cambio de un solo campo, porque el hash cambia). Rechaza
    artefactos que ya existan en la ronda. No deja la ronda a medio
    completar: los artefactos de 06 se escriben primero en un staging
    interno, se mueven uno por uno (rename atómico), y el estado se
    actualiza a ``REVISION_COMPLETED`` como ÚLTIMO paso, también vía
    escritura atómica de ese único archivo. Un segundo intento sobre la
    misma ronda ya completada lanza ``RuntimeError`` explícito."""

    if not artifacts:
        raise ValueError("complete_round_revision requiere al menos un artefacto.")

    final_dir = round_directory(project_dir, experiment_id, round_number)
    if not final_dir.is_dir():
        raise FileNotFoundError(
            f"DRAFT_REVISION_ROUND_NOT_FOUND: la ronda {round_number} no existe en {final_dir} "
            "-- 07 debe crearla primero con create_round_awaiting_revision."
        )

    status = read_round_status(project_dir=project_dir, experiment_id=experiment_id, round_number=round_number)
    if status is None:
        raise RuntimeError(f"DRAFT_REVISION_ROUND_STATUS_MISSING: round_{round_number:02d} sin archivo de estado.")

    if status["status"] == STATUS_REVISION_COMPLETED:
        raise RuntimeError(
            f"DRAFT_REVISION_ROUND_ALREADY_COMPLETED: round_{round_number:02d} ya fue completada por 06."
        )
    if status["status"] != STATUS_AWAITING_REVISION:
        raise RuntimeError(
            f"DRAFT_REVISION_ROUND_UNEXPECTED_STATUS: round_{round_number:02d} está en "
            f"{status['status']!r}, se esperaba {STATUS_AWAITING_REVISION!r}."
        )
    if status["experiment_id"] != experiment_id:
        raise RuntimeError(
            f"DRAFT_REVISION_EXPERIMENT_MISMATCH: la ronda pertenece a {status['experiment_id']!r}, "
            f"se esperaba {experiment_id!r}."
        )
    if status["cycle_id"] != cycle_id:
        raise RuntimeError(
            f"DRAFT_REVISION_CYCLE_MISMATCH: la ronda pertenece al ciclo {status['cycle_id']!r}, "
            f"se esperaba {cycle_id!r}."
        )
    if int(status["round_number"]) != round_number:
        raise RuntimeError(
            f"DRAFT_REVISION_ROUND_MISMATCH: la ronda persistida declara "
            f"{status['round_number']!r}, se esperaba {round_number}."
        )
    actual_hash = _stable_hash(writer_revision_request)
    if status["writer_revision_request_hash"] != actual_hash:
        raise RuntimeError(
            "DRAFT_REVISION_REQUEST_HASH_MISMATCH: el writer_revision_request que recibió 06 "
            "no coincide con el que 07 dejó al crear la ronda (otro contenido, aunque sea la "
            "misma ronda/experimento) -- no se completa la ronda con un request alterado."
        )

    existing = [name for name in artifacts if (final_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"La ronda {round_number} ya tiene artefactos de 06 persistidos: {existing} "
            "-- no se sobrescriben."
        )

    completion_staging = final_dir / f".staging_completion_{uuid.uuid4().hex[:8]}"
    completion_staging.mkdir(parents=True, exist_ok=False)

    try:
        for filename, content in artifacts.items():
            (completion_staging / filename).write_text(_serialize(content), encoding="utf-8")

        missing = [name for name in artifacts if not (completion_staging / name).is_file()]
        if missing:
            raise OSError(f"Artefactos faltantes tras escribir el staging de completado: {missing}")

        moved: list[str] = []
        try:
            for filename in artifacts:
                (completion_staging / filename).rename(final_dir / filename)
                moved.append(filename)
        except BaseException:
            # Revertir lo ya movido para no dejar la ronda a medio
            # completar (algunos artefactos de 06 sueltos sin estado
            # REVISION_COMPLETED).
            for filename in moved:
                moved_path = final_dir / filename
                if moved_path.exists():
                    moved_path.unlink()
            raise

        new_status = {**status, "status": STATUS_REVISION_COMPLETED}
        tmp_status_path = final_dir / f"._round_status_tmp_{uuid.uuid4().hex[:8]}.json"
        tmp_status_path.write_text(_serialize(new_status), encoding="utf-8")
        tmp_status_path.replace(final_dir / ROUND_STATUS_FILENAME)
    finally:
        shutil.rmtree(completion_staging, ignore_errors=True)

    return {filename: final_dir / filename for filename in artifacts}


def list_persisted_rounds(*, project_dir: str | Path, experiment_id: str) -> list[int]:
    base = Path(project_dir) / experiment_id / "05_outputs" / CYCLE_DIRECTORY_NAME
    if not base.is_dir():
        return []
    rounds = []
    for entry in base.iterdir():
        if entry.is_dir() and entry.name.startswith("round_"):
            try:
                rounds.append(int(entry.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(rounds)


def read_round_artifact(*, project_dir: str | Path, experiment_id: str, round_number: int, filename: str) -> Any:
    path = round_directory(project_dir, experiment_id, round_number) / filename
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def round_is_persisted(*, project_dir: str | Path, experiment_id: str, round_number: int) -> bool:
    return round_directory(project_dir, experiment_id, round_number).is_dir()
