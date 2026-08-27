"""Identity migration bootstrap para ciclos LEGACY ya AGOTADOS.

Los parches 14-18 solo permiten la migración LEGACY -> STABLE_UID_V1
cuando 06 produce una nueva ronda de revisión con identidad -- pero un
experimento que ya usó todas sus rondas científicas (``rounds_used >=
max_rounds``) no puede consumir otra ronda solo para adquirir
identidad estable. Este módulo cubre EXCLUSIVAMENTE ese caso operativo:
una migración técnica de una sola vez, fuera del conteo de rondas
científicas, que nunca reescribe contenido ni historial.

Diferencia con la migración normal (patch 17): esa migración ocurre
DENTRO de una ronda real de revisión (``migration_mode=
"EXPLICIT_SIGNAL_FROM_06"``, incrementa ``rounds_used`` como cualquier
RETURN real). Este bootstrap (``migration_mode="EXPLICIT_FRESH_UID_
MINT"``) es una operación administrativa que NUNCA toca ``rounds_used``
ni ``max_rounds``, no es una ronda del ciclo writer_verifier, y compromete
un nuevo resultado de 06 con ``ADVANCE`` directo a 07 (nunca ``RETURN``)
-- de modo que 07 pueda verificar el draft ya con identidad estable, sin
que esto autorice una ronda de corrección adicional.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.adapters.agent06_verification_handoff import (
    AGENT06_REQUIRED_ARTIFACTS,
    resolve_committed_agent06_artifacts,
)
from src.contracts.agent_input import ArtifactReference
from src.contracts.agent_result import (
    AgentResult,
    DecisionInfo,
    ExecutionStatus,
    QualityStatus,
    RequestedTransition,
    ToolUsage,
    TransitionAction,
)
from src.state.fingerprints import sha256_file
from src.state.pipeline_state import CycleState
from src.state.state_store import StateStore
from src.tools.draft_writing.claim_identity import default_mint_claim_uid
from src.tools.verification.corrections import fingerprint_text

AGENT06_STAGE_NAME = "06_agente_redactor"
AGENT07_STAGE_NAME = "07_agente_verificador"
MIGRATION_MODE = "EXPLICIT_FRESH_UID_MINT"


@dataclass(frozen=True, slots=True)
class ClaimIdentityBootstrapResult:
    bootstrapped: bool
    already_stable: bool
    decision_id: str | None
    claims_migrated: int


def bootstrap_legacy_claim_identity_for_exhausted_cycle(
    *, store: StateStore, project_dir: str | Path, mint_uid=default_mint_claim_uid,
) -> ClaimIdentityBootstrapResult:
    """Migración técnica de una sola vez, EXCLUSIVA para un ciclo
    ``writer_verifier`` ya agotado (``rounds_used >= max_rounds``) cuyo
    contrato de identidad sigue sin declararse como ``STABLE_UID_V1``
    (es decir, sigue en ``"LEGACY"`` -- el valor que ``CycleState.
    from_dict`` asigna explícitamente cuando el campo está ausente en un
    ``pipeline_state.json`` de un experimento anterior a este contrato).

    Precondiciones (fail-closed, ``ValueError`` con reason code explícito):
    - Debe existir un ciclo ``writer_verifier`` real.
    - ``rounds_used >= max_rounds`` -- solo aplica a ciclos agotados;
      si aún quedan rondas, la migración normal (patch 17, dentro de
      una ronda real) es la vía correcta, no esta.
    - El contrato declarado debe ser ``"LEGACY"`` o ``"STABLE_UID_V1"``
      (cualquier otro valor es un error de datos, no algo que este
      bootstrap deba interpretar).

    Idempotencia: si el contrato ya es ``"STABLE_UID_V1"``, no mintea
    nada nuevo, no crea ningún commit -- devuelve
    ``already_stable=True`` sin tocar absolutamente nada.

    Efecto (solo si LEGACY y agotado):
    1. Toma el draft canónico LEGACY actualmente publicado (vía
       ``resolve_committed_agent06_artifacts`` -- la misma resolución
       causal real que usa 07, nunca ``state.stages`` directo).
    2. Copia BYTE A BYTE los 8 artefactos requeridos a un nuevo
       directorio (``05_draft/claim_identity_bootstrap/``) -- ningún
       archivo histórico se modifica ni se sobrescribe.
    3. Asigna, únicamente en la copia de ``state_of_art_draft.json``, a
       CADA claim: ``claim_uid`` (UUID4 nuevo), ``claim_version=1``,
       ``parent_claim_uids=[]``, ``claim_text_fingerprint`` real (del
       texto exacto, sin reescribirlo), ``created_round``/
       ``updated_round`` = ``rounds_used`` (el punto donde el
       experimento quedó agotado). Nunca se reconstruye una identidad
       "adivinada" -- cada claim recibe una identidad genuinamente
       nueva, sin relación con ninguna ronda histórica (``parent_
       claim_uids=[]``, nunca similitud).
    4. Compromete este resultado como una ejecución NUEVA y real de 06
       (mismo mecanismo que cualquier otro commit: ``prepare_execution``
       / ``persist_agent_result`` / ``commit_execution``), con
       ``requested_transition=ADVANCE->07`` (nunca ``RETURN`` -- esto no
       es una ronda de corrección) y ``decision.code=
       "AGENT06_CLAIM_IDENTITY_BOOTSTRAP"`` -- auditable como migración
       técnica, no como revisión científica.
    5. Actualiza ``CycleState``: ``claim_identity_contract_version=
       "STABLE_UID_V1"`` + el registro de migración completo
       (``from``, ``to``, ``round``, ``decision_id``, ``migration_mode=
       "EXPLICIT_FRESH_UID_MINT"``) -- ``rounds_used``/``max_rounds``
       SIN TOCAR, byte a byte los mismos valores que antes."""

    state = store.load()
    cycle = state.cycles.get("writer_verifier")
    if cycle is None:
        raise ValueError("AGENT06_BOOTSTRAP_NO_CYCLE")
    if cycle.rounds_used < cycle.max_rounds:
        raise ValueError("AGENT06_BOOTSTRAP_CYCLE_NOT_EXHAUSTED")

    if cycle.claim_identity_contract_version == "STABLE_UID_V1":
        return ClaimIdentityBootstrapResult(
            bootstrapped=False, already_stable=True, decision_id=None, claims_migrated=0,
        )
    if cycle.claim_identity_contract_version != "LEGACY":
        raise ValueError(
            f"AGENT06_BOOTSTRAP_UNKNOWN_CONTRACT:{cycle.claim_identity_contract_version}"
        )

    project_dir = Path(project_dir)
    _, committed_result, paths, _ = resolve_committed_agent06_artifacts(
        store=store, stage_name=AGENT06_STAGE_NAME,
    )

    bootstrap_dir = paths["state_of_art_draft.json"].parent / "claim_identity_bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    new_paths: dict[str, Path] = {}
    for name in AGENT06_REQUIRED_ARTIFACTS:
        source = paths[name]
        target = bootstrap_dir / name
        if name == "state_of_art_draft.json":
            draft = json.loads(source.read_text(encoding="utf-8"))
            claims_migrated = 0
            for section in draft.get("sections", []):
                for claim in section.get("claims", []):
                    claim_text = str(claim.get("claim") or claim.get("claim_text") or "")
                    claim["claim_uid"] = mint_uid()
                    claim["claim_version"] = 1
                    claim["parent_claim_uids"] = []
                    claim["claim_text_fingerprint"] = fingerprint_text(claim_text)
                    claim["created_round"] = cycle.rounds_used
                    claim["updated_round"] = cycle.rounds_used
                    claims_migrated += 1
            target.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
        else:
            shutil.copy2(source, target)
        new_paths[name] = target

    refs = {
        name: ArtifactReference(path=str(path), hash=sha256_file(path))
        for name, path in new_paths.items()
    }

    now = datetime.now(timezone.utc).isoformat()
    next_attempt = (
        state.stages[AGENT06_STAGE_NAME].attempts_used + 1
        if AGENT06_STAGE_NAME in state.stages
        else 1
    )
    result = AgentResult(
        execution_status=ExecutionStatus.COMPLETED,
        quality_status=QualityStatus.APPROVED,
        decision=DecisionInfo(
            code="AGENT06_CLAIM_IDENTITY_BOOTSTRAP",
            rationale=(
                "Migración técnica administrativa: identidad STABLE_UID_V1 asignada "
                "a un draft LEGACY ya agotado (rounds_used>=max_rounds), sin ronda "
                "científica adicional ni cambio de contenido."
            ),
        ),
        quality_metrics={},
        warnings=(),
        requested_transition=RequestedTransition(
            action=TransitionAction.ADVANCE, target_stage=AGENT07_STAGE_NAME,
            reason_code="AGENT06_CLAIM_IDENTITY_BOOTSTRAP",
        ),
        output_artifacts=refs,
        tool_usage=ToolUsage(),
        attempt_number=next_attempt,
        started_at=now,
        completed_at=now,
    )

    prepared = store.prepare_execution(
        target_stage=AGENT06_STAGE_NAME, intended_action="EXECUTE", attempt_number=next_attempt,
    )
    store.persist_agent_result(prepared.decision_id, result)
    from src.state.fingerprints import build_stage_fingerprints

    store.commit_execution(
        decision_id=prepared.decision_id, result=result, stage_name=AGENT06_STAGE_NAME,
        fingerprints=build_stage_fingerprints(
            input_data={"migration": MIGRATION_MODE, "decision_id": prepared.decision_id},
            config_data={}, dependencies_data={},
        ),
        observations={},
    )

    from dataclasses import replace as _dc_replace

    migrated_cycle = _dc_replace(
        cycle,
        claim_identity_contract_version="STABLE_UID_V1",
        claim_identity_migration_from="LEGACY",
        claim_identity_migration_to="STABLE_UID_V1",
        claim_identity_migration_round=cycle.rounds_used,
        claim_identity_migration_decision_id=prepared.decision_id,
        claim_identity_migration_mode=MIGRATION_MODE,
    )
    post_state = store.load()
    post_state = _dc_replace(
        post_state, cycles={**post_state.cycles, "writer_verifier": migrated_cycle},
        identity=_dc_replace(post_state.identity, updated_at=datetime.now(timezone.utc).isoformat()),
    )
    store.save(post_state)

    return ClaimIdentityBootstrapResult(
        bootstrapped=True, already_stable=False, decision_id=prepared.decision_id,
        claims_migrated=claims_migrated,
    )
