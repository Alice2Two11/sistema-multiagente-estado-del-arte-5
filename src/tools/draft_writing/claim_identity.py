"""Identidad estable de claims entre rondas del ciclo 06<->07.

Diseño confirmado (ver investigación real: ``claim_id`` es posicional --
``f"{section_id}_C{idx}"`` -- y se recalcula cada vez que una sección se
regenera vía LLM en modo REVISION, aunque el claim regenerado no tenga
relación real con el que ocupaba esa posición antes). Este módulo
introduce una identidad PERSISTENTE (``claim_uid``, UUID opaco) que
sobrevive reescritura de texto, cambio de posición, e inserción/
eliminación de otros claims en la misma sección -- ``claim_id`` se
conserva únicamente como ETIQUETA legible, recalculada cada ronda, nunca
como clave de comparación entre rondas.

La correspondencia entre un claim nuevo y su historial se declara
EXPLÍCITAMENTE por quien la conoce de verdad -- 06 (vía el LLM, dentro
de un contrato de respuesta estructurada validado) -- nunca se infiere
después por similitud de texto ni ninguna otra heurística.

Cuatro acciones de identidad, mutuamente excluyentes:

    CONTINUE     -- exactamente 1 padre; reutiliza su claim_uid.
    NEW          -- 0 padres; mintea un claim_uid nuevo.
    SPLIT_CHILD  -- exactamente 1 padre; mintea un claim_uid NUEVO por
                    cada hijo (un padre puede tener varios hijos, cada
                    uno con su propia llamada a resolve_claim_identity).
    MERGE        -- 2 o más padres; mintea un claim_uid nuevo.

Caso especial (requisito explícito): cuando el ``writer_revision_
request`` de 07 señala un claim con ``claim_uid`` concreto (un issue
dirigido a ESE claim en particular), la identidad de ese claim en la
respuesta del LLM queda FORZADA -- el LLM debe declarar
``action=CONTINUE`` con ``parent_claim_uids=(ese_uid,)`` exactamente.
Si declara cualquier otra cosa (otro uid, u otra acción), la respuesta
se rechaza explícitamente -- nunca se sobrescribe en silencio lo que el
LLM declaró.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

CLAIM_IDENTITY_ACTIONS = ("CONTINUE", "NEW", "SPLIT_CHILD", "MERGE")


def default_mint_claim_uid() -> str:
    """Generador real de claim_uid: UUID4 aleatorio y opaco -- nunca un
    hash de contenido/contexto (confirmado explícitamente: el
    fingerprint de texto es un campo APARTE, solo para auditoría, no
    para identidad)."""
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class ClaimIdentityDeclaration:
    """Lo que el LLM (o, en el caso forzado, el propio sistema) declara
    sobre la identidad de UN claim generado en una ronda de revisión.
    Se valida en construcción -- nunca se acepta una combinación
    acción/padres inconsistente."""

    action: str
    parent_claim_uids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in CLAIM_IDENTITY_ACTIONS:
            raise ValueError(f"CLAIM_IDENTITY_UNKNOWN_ACTION:{self.action}")
        n = len(self.parent_claim_uids)
        if len(set(self.parent_claim_uids)) != n:
            raise ValueError("CLAIM_IDENTITY_DUPLICATE_PARENT_UIDS")
        if self.action == "CONTINUE" and n != 1:
            raise ValueError("CLAIM_IDENTITY_CONTINUE_REQUIRES_ONE_PARENT")
        if self.action == "NEW" and n != 0:
            raise ValueError("CLAIM_IDENTITY_NEW_REQUIRES_NO_PARENTS")
        if self.action == "SPLIT_CHILD" and n != 1:
            raise ValueError("CLAIM_IDENTITY_SPLIT_CHILD_REQUIRES_ONE_PARENT")
        if self.action == "MERGE" and n < 2:
            raise ValueError("CLAIM_IDENTITY_MERGE_REQUIRES_TWO_OR_MORE_PARENTS")


@dataclass(frozen=True, slots=True)
class ClaimIdentityRecord:
    """Registro persistido por claim/versión -- lo que se escribe en el
    draft (``state_of_art_draft.json`` / ``revised_draft.json``) junto a
    cada claim. Mínimo pedido explícitamente: ``claim_uid``,
    ``claim_version``, ``claim_id``, ``parent_claim_uids``,
    ``claim_text_fingerprint``, ``created_round``, ``updated_round``."""

    claim_uid: str
    claim_version: int
    claim_id: str
    parent_claim_uids: tuple[str, ...]
    claim_text_fingerprint: str
    created_round: int
    updated_round: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_uid": self.claim_uid,
            "claim_version": self.claim_version,
            "claim_id": self.claim_id,
            "parent_claim_uids": list(self.parent_claim_uids),
            "claim_text_fingerprint": self.claim_text_fingerprint,
            "created_round": self.created_round,
            "updated_round": self.updated_round,
        }


def resolve_claim_identity(
    *,
    declaration: ClaimIdentityDeclaration,
    claim_text: str,
    claim_id: str,
    previous_claims_by_uid: Mapping[str, ClaimIdentityRecord],
    forced_parent_uid: str | None,
    round_number: int,
    text_fingerprint: Callable[[str], str],
    mint_uid: Callable[[], str] = default_mint_claim_uid,
) -> ClaimIdentityRecord:
    """Resuelve el ``ClaimIdentityRecord`` real de un claim generado en
    una ronda de revisión, a partir de lo que 06 (el LLM) declaró.

    ``forced_parent_uid``: si no es ``None``, ESTE claim corresponde a un
    issue de ``writer_revision_request`` que señalaba un ``claim_uid``
    concreto -- la identidad queda forzada. La declaración del LLM debe
    coincidir EXACTAMENTE (``action=="CONTINUE"`` y
    ``parent_claim_uids==(forced_parent_uid,)``); cualquier otra cosa
    falla cerrado con ``CLAIM_IDENTITY_FORCED_UID_MISMATCH`` -- nunca se
    sobrescribe en silencio lo que el LLM declaró (ver docstring del
    módulo)."""

    if forced_parent_uid is not None:
        if declaration.action != "CONTINUE" or declaration.parent_claim_uids != (forced_parent_uid,):
            raise ValueError(
                "CLAIM_IDENTITY_FORCED_UID_MISMATCH: el writer_revision_request señalaba "
                f"claim_uid={forced_parent_uid!r} para este claim, pero el LLM declaró "
                f"action={declaration.action!r} parent_claim_uids={declaration.parent_claim_uids!r}"
            )

    for parent_uid in declaration.parent_claim_uids:
        if parent_uid not in previous_claims_by_uid:
            raise ValueError(f"CLAIM_IDENTITY_PARENT_NOT_FOUND:{parent_uid}")

    fingerprint = text_fingerprint(claim_text)

    if declaration.action == "CONTINUE":
        parent = previous_claims_by_uid[declaration.parent_claim_uids[0]]
        return ClaimIdentityRecord(
            claim_uid=parent.claim_uid,
            claim_version=parent.claim_version + 1,
            claim_id=claim_id,
            parent_claim_uids=(parent.claim_uid,),
            claim_text_fingerprint=fingerprint,
            created_round=parent.created_round,
            updated_round=round_number,
        )

    if declaration.action == "NEW":
        return ClaimIdentityRecord(
            claim_uid=mint_uid(),
            claim_version=1,
            claim_id=claim_id,
            parent_claim_uids=(),
            claim_text_fingerprint=fingerprint,
            created_round=round_number,
            updated_round=round_number,
        )

    if declaration.action == "SPLIT_CHILD":
        # Un padre puede tener varios hijos -- cada uno llega aquí en su
        # propia llamada (una por claim generado), y cada uno mintea su
        # PROPIO claim_uid nuevo -- ninguno hereda el del padre.
        return ClaimIdentityRecord(
            claim_uid=mint_uid(),
            claim_version=1,
            claim_id=claim_id,
            parent_claim_uids=declaration.parent_claim_uids,
            claim_text_fingerprint=fingerprint,
            created_round=round_number,
            updated_round=round_number,
        )

    if declaration.action == "MERGE":
        return ClaimIdentityRecord(
            claim_uid=mint_uid(),
            claim_version=1,
            claim_id=claim_id,
            parent_claim_uids=declaration.parent_claim_uids,
            claim_text_fingerprint=fingerprint,
            created_round=round_number,
            updated_round=round_number,
        )

    raise ValueError(f"CLAIM_IDENTITY_UNKNOWN_ACTION:{declaration.action}")  # pragma: no cover
