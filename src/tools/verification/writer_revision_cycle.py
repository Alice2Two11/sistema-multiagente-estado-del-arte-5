"""Parte B1+B2 del ciclo correctivo real `06 ↔ 07` — versión endurecida.

Deriva TODO de ``ClaimVerificationResult`` real (``src/agents/
verification_agent.py``, campo ``final_correction_eligibility`` — los 5
valores reales, confirmados en ``src/tools/verification/validation.py``,
``determine_final_correction_eligibility``): ``NO_CORRECTION_NEEDED``,
``MANUAL_REVIEW_REQUIRED``, ``NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE``,
``AUTO_CORRECTION_ELIGIBLE``, ``POTENTIALLY_AUTO_CORRECTABLE``. No se
inventa ninguna categoría nueva.

No usa un LLM para decidir la transición — es una función pura sobre los
resultados de verificación ya calculados.

Correcciones de esta ronda (fail-closed, sin aprobar/retornar sin
evidencia suficiente):

1. Un claim con elegibilidad corregible pero SIN evidencia utilizable
   (``evidence_used`` vacío y sin ``correction_proposal`` con soporte) ya
   NO produce RETURN — produce HALT_STAGE con
   ``AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT``. Un solo claim así bloquea
   todo el lote (mismo criterio que un claim en revisión manual): no se
   envían correcciones parciales mientras exista un problema que 06 no
   puede resolver de forma segura.
2. Cualquier claim con ``final_correction_eligibility`` ausente o fuera de
   los 5 valores reales conocidos produce HALT_STAGE con
   ``AGENT07_UNKNOWN_ELIGIBILITY`` — nunca cae en ADVANCE por omisión.
3. ``severity`` ya no colapsa todo lo que no es ``HIGH`` en ``medium``:
   se conserva el valor real de ``hallucination_risk`` tal cual lo produce
   07 (normalizado a minúsculas únicamente — transformación de formato,
   no de contenido), sin inventar una escala nueva.
4. ``requested_change`` usa la propuesta correctiva real
   (``claim["correction_proposal"]["requested_change"]``) cuando el
   runtime de 07 la provee. El texto genérico queda como fallback
   EXPLÍCITO, marcado con ``"requested_change_is_fallback": True``, y solo
   se usa cuando ya se confirmó evidencia suficiente (nunca sustituye a
   una propuesta real disponible).

Validaciones nuevas (ver ``_validate_claims_shape``/
``build_writer_revision_request``): ``claim_id`` presente en cada claim;
elegibilidad conocida; consistencia entre ``correctable_claim_uids``
devueltos por ``classify_verification_transition`` y los claims recibidos;
``source_draft_path`` obligatorio (no ``None``) en el artefacto final;
``issue_id`` estable (derivado de ``claim_id``, no del orden de
iteración); rechazo explícito si el artefacto quedaría con ``issues``
vacío.

Alcance de esta entrega (documentado, no oculto): estas dos piezas siguen
siendo el núcleo puro, listo para conectarse a ``verification_notebook.py``
(que hoy construye ``RequestedTransition(action=ADVANCE if completed else
HALT_STAGE, target_stage=None, ...)`` — confirmado por lectura directa, sin
ninguna rama RETURN).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Valores reales confirmados de ClaimVerificationResult.final_correction_eligibility
NO_CORRECTION_NEEDED = "NO_CORRECTION_NEEDED"
MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE = "NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE"
AUTO_CORRECTION_ELIGIBLE = "AUTO_CORRECTION_ELIGIBLE"
POTENTIALLY_AUTO_CORRECTABLE = "POTENTIALLY_AUTO_CORRECTABLE"

KNOWN_ELIGIBILITIES = frozenset(
    {
        NO_CORRECTION_NEEDED,
        MANUAL_REVIEW_REQUIRED,
        NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE,
        AUTO_CORRECTION_ELIGIBLE,
        POTENTIALLY_AUTO_CORRECTABLE,
    }
)
CORRECTABLE_ELIGIBILITIES = frozenset({AUTO_CORRECTION_ELIGIBLE, POTENTIALLY_AUTO_CORRECTABLE})
BLOCKING_ELIGIBILITIES = frozenset(
    {MANUAL_REVIEW_REQUIRED, NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE}
)


def _has_usable_correction_support(claim: dict[str, Any]) -> bool:
    """Fail-closed: hay soporte suficiente para pedir una corrección solo
    si existe evidencia real usada (``evidence_used`` no vacío) O una
    propuesta correctiva real con su propio respaldo de evidencia
    (``correction_proposal.supporting_evidence`` no vacío). Ninguno de los
    dos se infiere ni se inventa — se lee tal cual lo entrega 07."""

    if claim.get("evidence_used"):
        return True
    proposal = claim.get("correction_proposal")
    if isinstance(proposal, dict) and proposal.get("supporting_evidence"):
        return True
    return False


CLAIM_IDENTITY_CONTRACT_VERSIONS = ("LEGACY", "STABLE_UID_V1")


def _resolve_claim_identity_contract_version(
    claims: list[dict[str, Any]], *, declared_contract_version: str | None,
    migration_signal: bool = False,
) -> tuple[str | None, str | None, bool, bool]:
    """Resolución HÍBRIDA del contrato de identidad para ESTA ronda --
    nunca deriva ciegamente de los claims cuando ya existe un contrato
    declarado para el ciclo/experimento (``CycleState.claim_identity_
    contract_version``), porque eso permitiría un DOWNGRADE (o UPGRADE)
    SILENCIOSO: un experimento ya ``STABLE_UID_V1`` que por un bug de
    serialización/propagación pierde TODOS sus ``claim_uid`` no debe
    reinterpretarse como ``LEGACY`` -- debe fallar cerrado, porque esa
    pérdida es en sí misma la señal de un defecto real.

    ``migration_signal``: bandera EXPLÍCITA de 06 (nunca inferida de la
    mera presencia de ``claim_uid``) indicando que este es el round
    donde 06 empezó deliberadamente a producir claims bajo el contrato
    ``STABLE_UID_V1``. Solo con esta señal, y solo cuando el contrato
    declarado es ``LEGACY``, ocurre la migración de una sola vez.

    Cero claims sin contrato declarado (``declared_contract_version is
    None`` y ``claims`` vacío): NUNCA se infiere ni se persiste nada --
    el contrato queda indeterminado (``None, None, False, False``) y se
    deja que ``classify_verification_transition`` maneje
    ``AGENT07_NO_CLAIMS`` sin crear una frontera de identidad a partir
    de un lote vacío.

    Devuelve ``(contract_version, error_reason_code, newly_inferred,
    migrated)``."""

    if declared_contract_version is not None:
        if declared_contract_version not in CLAIM_IDENTITY_CONTRACT_VERSIONS:
            return None, "AGENT07_UNKNOWN_CLAIM_IDENTITY_CONTRACT_VERSION", False, False
        with_uid = sum(1 for c in claims if c.get("claim_uid"))

        if declared_contract_version == "STABLE_UID_V1":
            # STABLE_UID_V1 -> LEGACY está prohibido sin excepción, con o
            # sin migration_signal: una vez migrado, nunca se retrocede.
            if with_uid != len(claims):
                return None, "AGENT07_CLAIM_UID_CONTRACT_VIOLATION", False, False
            return "STABLE_UID_V1", None, False, False

        # declared_contract_version == "LEGACY"
        if migration_signal:
            # Migración deliberada: 06 señaló explícitamente que este
            # round produce claims bajo el nuevo contrato. Se exige que
            # TODOS los claims de este round ya tengan claim_uid -- una
            # migración parcial (señal presente pero UIDs incompletos)
            # falla cerrado igual que cualquier otra violación.
            if with_uid != len(claims):
                return None, "AGENT07_CLAIM_UID_CONTRACT_VIOLATION", False, False
            return "STABLE_UID_V1", None, False, True

        # Sin señal: la aparición de CUALQUIER claim_uid bajo LEGACY sin
        # migration_signal nunca se interpreta como una migración
        # implícita -- exige la frontera explícita.
        if with_uid != 0:
            return None, "AGENT07_CLAIM_UID_CONTRACT_VIOLATION", False, False
        return "LEGACY", None, False, False

    # Sin contrato declarado todavía -- primera frontera real: se infiere
    # UNA VEZ, pero NUNCA a partir de un lote vacío (requisito A) --
    # AGENT07_NO_CLAIMS debe poder manejarse aguas arriba sin que esto
    # cree una frontera de identidad prematura.
    if not claims:
        return None, None, False, False
    with_uid = sum(1 for c in claims if c.get("claim_uid"))
    if with_uid == len(claims):
        return "STABLE_UID_V1", None, True, False
    if with_uid == 0:
        return "LEGACY", None, True, False
    return None, "AGENT07_MIXED_CLAIM_IDENTITY_CONTRACT", False, False


def _effective_correction_identity(claim: dict[str, Any], contract_version: str) -> str:
    """La identidad que gobierna ``correctable_claim_uids``/
    ``blocking_claim_uids``/``writer_revision_request`` para ESTE claim,
    según el contrato vigente de la ronda:

    - ``STABLE_UID_V1``: ``claim_uid`` (persistente, real identidad
      cross-round -- validado no vacío por ``_validate_claims_shape``
      antes de llegar aquí).
    - ``LEGACY``: ``claim_id`` (posicional, comportamiento histórico --
      válido únicamente DENTRO de esta misma ronda, nunca se compara
      entre rondas)."""

    if contract_version == "STABLE_UID_V1":
        return claim["claim_uid"]
    return claim["claim_id"]


def _validate_claims_shape(claims: list[dict[str, Any]], *, contract_version: str) -> str | None:
    """Devuelve un reason_code de bloqueo si algún claim está malformado
    (sin ``claim_id``, o -- bajo ``STABLE_UID_V1`` -- sin ``claim_uid``)
    o tiene elegibilidad desconocida; ``None`` si todos son válidos."""

    for claim in claims:
        if not claim.get("claim_id"):
            return "AGENT07_MALFORMED_CLAIM"
        if contract_version == "STABLE_UID_V1" and not claim.get("claim_uid"):
            return "AGENT07_MISSING_CLAIM_UID"
        if claim.get("final_correction_eligibility") not in KNOWN_ELIGIBILITIES:
            return "AGENT07_UNKNOWN_ELIGIBILITY"
    return None


# -----------------------------------------------------------------------
# B1. Política determinista de transición
# -----------------------------------------------------------------------


def classify_verification_transition(
    *,
    claims: list[dict[str, Any]],
    technical_status: str,
    rounds_used: int,
    max_rounds: int,
    declared_contract_version: str | None = None,
    migration_signal: bool = False,
) -> dict[str, Any]:
    """Deriva ADVANCE/RETURN/HALT_STAGE exclusivamente de los datos reales
    de cada claim (``claims``: lista de dicts con forma de
    ``ClaimVerificationResult.to_dict()``, más opcionalmente
    ``correction_proposal`` si el runtime de 07 lo produce).

    Identidad cross-round (``correctable_claim_uids``/``blocking_claim_
    uids``): bajo el contrato ``STABLE_UID_V1`` (todos los claims traen
    ``claim_uid`` real), esa es la identidad que gobierna la selección
    -- nunca ``claim_id`` (posicional, puede referirse a un claim
    distinto entre rondas si la sección se regeneró). Bajo ``LEGACY``
    (ningún claim trae ``claim_uid``) se mantiene el comportamiento
    posicional histórico. Una MEZCLA de ambos en el mismo lote nunca se
    resuelve en silencio -- ver ``AGENT07_MIXED_CLAIM_IDENTITY_
    CONTRACT`` más abajo.

    ``declared_contract_version``: el valor actual de ``CycleState.
    claim_identity_contract_version`` para este ciclo, o ``None`` si el
    ciclo aún no existe. Cuando NO es ``None``, es NORMATIVO -- los
    claims se validan contra él (ver ``_resolve_claim_identity_
    contract_version``); un experimento ya ``STABLE_UID_V1`` que pierde
    todos o algunos de sus ``claim_uid`` nunca se reinterpreta como
    ``LEGACY`` en silencio, falla cerrado con ``AGENT07_CLAIM_UID_
    CONTRACT_VIOLATION``. Cuando es ``None`` (primera frontera real), se
    infiere una vez de los propios claims -- el resultado incluye
    ``claim_identity_contract_version_newly_inferred=True`` para que el
    llamador sepa que debe persistirlo.

    Prioridad cuando coinciden varias condiciones (mayor a menor):
    1. Fallo técnico (``technical_status != "COMPLETED"``) -> HALT_STAGE.
    2. Artefactos incompletos (``claims`` vacío) -> HALT_STAGE.
    3. Contrato de identidad mezclado (algunos claims con ``claim_uid``,
       otros sin él) -> HALT_STAGE. Nunca se elige LEGACY o STABLE_UID_V1
       en silencio para un lote ambiguo.
    4. Cualquier claim malformado (sin ``claim_id``, o sin ``claim_uid``
       bajo ``STABLE_UID_V1``) o con elegibilidad desconocida ->
       HALT_STAGE. Nunca se aprueba ni se retorna un lote con datos que
       no se pueden interpretar con certeza.
    5. Rondas agotadas -> HALT_STAGE, incluso si aún quedan problemas
       corregibles con evidencia.
    6. Cualquier claim en ``BLOCKING_ELIGIBILITIES`` -> HALT_STAGE.
    7. Cualquier claim con elegibilidad corregible pero SIN soporte de
       corrección utilizable (``_has_usable_correction_support`` es
       False) -> HALT_STAGE con ``AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT``.
       Fail-closed: no se solicita una corrección sin evidencia real.
    8. Al menos un claim corregible CON soporte (y ninguno de 1-7) ->
       RETURN a 06.
    9. Todos los claims en ``NO_CORRECTION_NEEDED`` -> ADVANCE a 08.

    Devuelve ``{"action", "reason_code", "correctable_claim_uids",
    "blocking_claim_uids", "claim_identity_contract_version",
    "claim_identity_contract_version_newly_inferred", "rationale"}``.

    ADVERTENCIA DE NOMBRE -- ``correctable_claim_uids``/``blocking_
    claim_uids`` NO son siempre UUIDs reales: bajo ``STABLE_UID_V1`` sí
    lo son (identidad persistente cross-round); bajo ``LEGACY`` estas
    tuplas contienen ``claim_id`` (la etiqueta posicional), reutilizado
    como identidad SOLO dentro de esta misma ronda -- nunca comparable
    entre rondas bajo ese contrato. El nombre se mantuvo uniforme
    (en vez de introducir ``correctable_claim_keys`` como alternativa
    neutral) para no forzar un segundo cambio de firma sobre todos los
    consumidores ya actualizados en el parche anterior; en su lugar,
    esta advertencia documenta la semántica exacta. Siempre verificar
    ``claim_identity_contract_version`` en el mismo dict antes de tratar
    estos valores como identidad persistente.
    """

    if technical_status != "COMPLETED":
        return {
            "action": "HALT_STAGE",
            "reason_code": "AGENT07_TECHNICAL_FAILURE",
            "correctable_claim_uids": (),
            "blocking_claim_uids": (),
            "claim_identity_contract_version": None,
            "claim_identity_contract_version_newly_inferred": False,
            "claim_identity_contract_migrated": False,
            "rationale": f"Fallo técnico de 07: technical_status={technical_status!r}.",
        }

    if not claims:
        return {
            "action": "HALT_STAGE",
            "reason_code": "AGENT07_NO_CLAIMS",
            "correctable_claim_uids": (),
            "blocking_claim_uids": (),
            "claim_identity_contract_version": None,
            "claim_identity_contract_version_newly_inferred": False,
            "claim_identity_contract_migrated": False,
            "rationale": "No hay claims verificados — artefactos incompletos.",
        }

    contract_version, mixed_reason, newly_inferred, migrated = _resolve_claim_identity_contract_version(
        claims, declared_contract_version=declared_contract_version, migration_signal=migration_signal,
    )
    if mixed_reason is not None:
        return {
            "action": "HALT_STAGE",
            "reason_code": mixed_reason,
            "correctable_claim_uids": (),
            "blocking_claim_uids": (),
            "claim_identity_contract_version": None,
            "claim_identity_contract_version_newly_inferred": False,
            "claim_identity_contract_migrated": False,
            "rationale": (
                "Lote de claims con contrato de identidad mezclado o en violación del "
                "contrato declarado. Nunca se elige/reinterpreta LEGACY o STABLE_UID_V1 "
                "en silencio."
            ),
        }

    malformed_reason = _validate_claims_shape(claims, contract_version=contract_version)
    if malformed_reason is not None:
        return {
            "action": "HALT_STAGE",
            "reason_code": malformed_reason,
            "correctable_claim_uids": (),
            "blocking_claim_uids": (),
            "claim_identity_contract_version": contract_version,
            "claim_identity_contract_version_newly_inferred": newly_inferred,
            "claim_identity_contract_migrated": migrated,
            "rationale": (
                "Al menos un claim no tiene claim_id"
                if malformed_reason == "AGENT07_MALFORMED_CLAIM"
                else "Al menos un claim tiene final_correction_eligibility ausente o desconocida."
                if malformed_reason == "AGENT07_UNKNOWN_ELIGIBILITY"
                else "Al menos un claim no tiene claim_uid bajo el contrato STABLE_UID_V1."
            ),
        }

    if rounds_used >= max_rounds:
        return {
            "action": "HALT_STAGE",
            "reason_code": "WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED",
            "correctable_claim_uids": tuple(
                _effective_correction_identity(c, contract_version)
                for c in claims
                if c.get("final_correction_eligibility") in CORRECTABLE_ELIGIBILITIES
            ),
            "blocking_claim_uids": (),
            "claim_identity_contract_version": contract_version,
            "claim_identity_contract_version_newly_inferred": newly_inferred,
            "claim_identity_contract_migrated": migrated,
            "rationale": f"Se agotaron las {max_rounds} rondas permitidas ({rounds_used} usadas).",
        }

    blocking_claim_uids = tuple(
        _effective_correction_identity(c, contract_version)
        for c in claims
        if c.get("final_correction_eligibility") in BLOCKING_ELIGIBILITIES
    )
    if blocking_claim_uids:
        return {
            "action": "HALT_STAGE",
            "reason_code": "AGENT07_NON_CORRECTABLE_ISSUE",
            "correctable_claim_uids": (),
            "blocking_claim_uids": blocking_claim_uids,
            "claim_identity_contract_version": contract_version,
            "claim_identity_contract_version_newly_inferred": newly_inferred,
            "claim_identity_contract_migrated": migrated,
            "rationale": (
                f"{len(blocking_claim_uids)} claim(s) requieren revisión manual o "
                f"no tienen evidencia disponible para corregirse: {list(blocking_claim_uids)}."
            ),
        }

    correctable_candidates = [
        c for c in claims if c.get("final_correction_eligibility") in CORRECTABLE_ELIGIBILITIES
    ]
    insufficient_evidence_ids = tuple(
        _effective_correction_identity(c, contract_version)
        for c in correctable_candidates if not _has_usable_correction_support(c)
    )
    if insufficient_evidence_ids:
        return {
            "action": "HALT_STAGE",
            "reason_code": "AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT",
            "correctable_claim_uids": (),
            "blocking_claim_uids": insufficient_evidence_ids,
            "claim_identity_contract_version": contract_version,
            "claim_identity_contract_version_newly_inferred": newly_inferred,
            "claim_identity_contract_migrated": migrated,
            "rationale": (
                f"{len(insufficient_evidence_ids)} claim(s) están marcados como corregibles "
                "pero no tienen evidencia ni propuesta correctiva utilizable: "
                f"{list(insufficient_evidence_ids)}."
            ),
        }

    correctable_claim_uids = tuple(
        _effective_correction_identity(c, contract_version) for c in correctable_candidates
    )
    if correctable_claim_uids:
        return {
            "action": "RETURN",
            "reason_code": "AGENT07_CORRECTABLE_ISSUES",
            "correctable_claim_uids": correctable_claim_uids,
            "blocking_claim_uids": (),
            "claim_identity_contract_version": contract_version,
            "claim_identity_contract_version_newly_inferred": newly_inferred,
            "claim_identity_contract_migrated": migrated,
            "rationale": f"{len(correctable_claim_uids)} claim(s) corregibles con evidencia: {list(correctable_claim_uids)}.",
        }

    return {
        "action": "ADVANCE",
        "reason_code": "AGENT07_ALL_CLAIMS_APPROVED",
        "correctable_claim_uids": (),
        "blocking_claim_uids": (),
        "claim_identity_contract_version": contract_version,
            "claim_identity_contract_version_newly_inferred": newly_inferred,
            "claim_identity_contract_migrated": migrated,
        "rationale": "Todos los claims están en NO_CORRECTION_NEEDED.",
    }


# -----------------------------------------------------------------------
# B2. Artefacto de retroalimentación (writer_revision_request.json)
# -----------------------------------------------------------------------

REVISION_REQUEST_SCHEMA_VERSION = "writer_revision_request_v1"

_PROBLEM_TYPE_BY_ELIGIBILITY = {
    AUTO_CORRECTION_ELIGIBLE: "AUTO_CORRECTABLE",
    POTENTIALLY_AUTO_CORRECTABLE: "POTENTIALLY_CORRECTABLE",
}

_FALLBACK_REQUESTED_CHANGE = (
    "Ajustar el claim para que sea consistente exclusivamente con la "
    "evidencia citada; no introducir información fuera de evidence_used."
)


def _issue_from_claim(claim: dict[str, Any]) -> dict[str, Any]:
    eligibility = claim.get("final_correction_eligibility")
    evidence_used = tuple(claim.get("evidence_used") or ())
    citations = tuple(
        f"[{row.get('source_filename')} | {row.get('chunk_id')}]" for row in evidence_used
    )
    proposal = claim.get("correction_proposal")
    proposal_change = (
        proposal.get("requested_change") if isinstance(proposal, dict) else None
    )
    uses_fallback = not proposal_change

    hallucination_risk = claim.get("hallucination_risk")
    severity = str(hallucination_risk).lower() if hallucination_risk else None

    return {
        # issue_id estable: derivado de claim_id, NO del orden de iteración.
        "issue_id": f"issue_{claim['claim_id']}",
        "claim_id": claim["claim_id"],
        # claim_uid: identidad estable real (ver claim_identity.py) --
        # cuando está presente, 06 debe preservarla determinísticamente
        # al corregir este claim (ver enforce_forced_claim_uid_continuations).
        # Ausente ("") solo para claims legacy sin identidad estable
        # todavía -- nunca se inventa uno.
        "claim_uid": claim.get("claim_uid") or "",
        "section_id": claim.get("section_id"),
        "claim_text": claim.get("claim_text"),
        "problem_type": _PROBLEM_TYPE_BY_ELIGIBILITY.get(eligibility, eligibility),
        "verdict": claim.get("scientific_verdict"),
        "severity": severity,  # valor real de hallucination_risk, solo normalizado a minúsculas
        "hallucination_risk": hallucination_risk,
        "correction_needed": True,
        "source_filename": evidence_used[0].get("source_filename") if evidence_used else None,
        "chunk_id": evidence_used[0].get("chunk_id") if evidence_used else None,
        "evidence_text": evidence_used[0].get("text") if evidence_used else None,
        "citation": citations[0] if citations else None,
        "requested_change": proposal_change or _FALLBACK_REQUESTED_CHANGE,
        "requested_change_is_fallback": uses_fallback,
        "constraints": (
            "No modificar claims aprobados de otras secciones. "
            "No usar Ground Truth. No usar conocimiento externo al corpus."
        ),
        "correctable": eligibility in CORRECTABLE_ELIGIBILITIES,
    }


def build_writer_revision_request(
    *,
    experiment_id: str,
    cycle_id: str,
    round_number: int,
    source_draft_path: str,
    source_draft_fingerprint: str,
    verification_fingerprint: str,
    claims: list[dict[str, Any]],
    correctable_claim_uids: tuple[str, ...],
    claim_identity_contract_version: str,
    transition_reason: str,
) -> dict[str, Any]:
    """Construye ``writer_revision_request.json`` DERIVADO exclusivamente
    de los claims reales marcados como corregibles CON soporte suficiente
    — nunca desde Ground Truth ni desde conocimiento externo a lo que 07
    ya recuperó. Un ``issue`` por identidad corregible, sin duplicados.

    ``correctable_claim_uids``/``claim_identity_contract_version`` deben
    ser EXACTAMENTE los que devolvió ``classify_verification_transition``
    para este mismo lote de ``claims`` -- esta función indexa y valida
    por la MISMA identidad efectiva (``claim_uid`` bajo ``STABLE_UID_V1``,
    ``claim_id`` bajo ``LEGACY``), nunca por ``claim_id`` directamente
    salvo que el contrato sea ``LEGACY``.

    Validaciones (fail-closed, lanzan ``ValueError`` con reason code
    explícito en el mensaje):
    - ``source_draft_path`` obligatorio (no vacío/``None``).
    - ``claim_identity_contract_version`` debe ser uno de los valores
      conocidos.
    - Cada identidad en ``correctable_claim_uids`` debe existir entre
      ``claims`` (consistencia entre lo que dijo
      ``classify_verification_transition`` y lo que realmente se
      recibió).
    - Cada claim referenciado debe tener soporte de corrección utilizable
      (mismo criterio fail-closed que B1 — defensa en profundidad si esta
      función se llama directamente, sin pasar por B1).
    - El artefacto final no puede quedar con ``issues`` vacío.
    """

    if not source_draft_path:
        raise ValueError("AGENT07_REVISION_REQUEST_MALFORMED: source_draft_path es obligatorio.")
    if claim_identity_contract_version not in CLAIM_IDENTITY_CONTRACT_VERSIONS:
        raise ValueError(
            "AGENT07_REVISION_REQUEST_MALFORMED: claim_identity_contract_version "
            f"desconocido: {claim_identity_contract_version!r}."
        )

    claims_by_identity = {
        _effective_correction_identity(c, claim_identity_contract_version): c
        for c in claims
        if c.get("claim_id")
    }
    correctable_set = set(correctable_claim_uids)

    missing_ids = correctable_set - set(claims_by_identity)
    if missing_ids:
        raise ValueError(
            "AGENT07_REVISION_REQUEST_MALFORMED: correctable_claim_uids "
            f"referencia identidades inexistentes en claims: {sorted(missing_ids)}."
        )

    issues = []
    for identity in correctable_claim_uids:  # orden de correctable_claim_uids, sin reordenar por conveniencia
        claim = claims_by_identity[identity]
        if not _has_usable_correction_support(claim):
            raise ValueError(
                "AGENT07_CORRECTION_EVIDENCE_INSUFFICIENT: "
                f"claim identificado por {identity!r} no tiene evidencia ni propuesta correctiva utilizable."
            )
        issues.append(_issue_from_claim(claim))

    if not issues:
        raise ValueError("AGENT07_REVISION_REQUEST_MALFORMED: no hay issues que enviar a 06.")

    return {
        "schema_version": REVISION_REQUEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "cycle_id": cycle_id,
        "round_number": round_number,
        "source_draft_path": source_draft_path,
        "source_draft_fingerprint": source_draft_fingerprint,
        "verification_fingerprint": verification_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transition_reason": transition_reason,
        "claim_identity_contract_version": claim_identity_contract_version,
        "summary": f"{len(issues)} observación(es) corregible(s) de verificación.",
        "issues": issues,
    }
