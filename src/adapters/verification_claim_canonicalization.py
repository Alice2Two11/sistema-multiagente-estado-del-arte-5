"""Adaptador canónico entre las DOS estructuras productivas reales que
puede tener ``Agent07RuntimeResult.provisional_bundle`` y el modelo que
``classify_verification_transition``/``build_writer_revision_request``
(``src/tools/verification/writer_revision_cycle.py``) necesitan.

Hallazgo (investigación productiva real, no de fixture)
-----------------------------------------------------------
``build_provisional_verification_traceability_bundle`` (la función
productiva real de ``validation.py``) NO produce
``claim_verification_records`` — produce ``claim_traceability_rows`` +
``claim_evidence_traceability_rows`` + ``correction_traceability_rows``,
tres colecciones separadas que hay que UNIR por ``claim_id``/
``correction_id`` para reconstruir lo que
``classify_verification_transition`` necesita de cada claim. La forma
``claim_verification_records`` (``{"section_id":...,
"claim_verification_result": {...}}``) SÍ existe, pero es la forma cruda
de ``run_agent07_in_memory`` ANTES de pasar por el ``bundle_builder`` —
nunca la forma que sale de la transformación productiva real. Ambas son
reales; ninguna es un defecto — son dos puntos distintos del mismo
pipeline. Este módulo unifica ambas en un solo modelo canónico, fail-closed.

Modelo canónico (lo que ``classify_verification_transition``/
``build_writer_revision_request`` consumen, confirmado leyendo ambas
funciones completas):
    claim_id, section_id, claim_text, scientific_verdict,
    hallucination_risk, final_correction_eligibility, evidence_used
    (tupla de {"source_filename","chunk_id",...}),
    correction_proposal ({"requested_change","supporting_evidence"} o
    None), manual_review_required, llm_correction_recommendation.

Política de ``usage_role`` (investigación, no un bug de código)
-----------------------------------------------------------------
``_resolve_usage_role`` (``src/adapters/claim_verification_context.py``)
y el filtro de ``propose_correction`` (``src/tools/verification/
corrections.py``: ``usage_role in {"SUPPORT","NUMERIC","ATTRIBUTION"}``)
YA ESTÁN alineados correctamente. ``ELIGIBLE`` es el resultado honesto y
deliberado cuando la evidencia de origen (chunks heredados de 06) no trae
un rol concreto — no es una etapa faltante ni un defecto. La única
transformación productiva real que SÍ asigna un rol concreto
(``"SUPPORT"``) es el recorrido de RAG independiente de 07
(``_independent_retrieve_claim``, activado con
``dependencies.retriever_binding`` + ``dependencies.retrieval_tool`` +
``dependencies.retriever_binding is not None``). Esta es la vía Caso A:
existe una función productiva real; se usa donde corresponde (dentro del
runtime de 07, no aquí). Este módulo NO reasigna ``usage_role`` — se
limita a leer, para el evidence_used del canónico, lo que la fila de
trazabilidad ya trae (``usage_role``/``authorized_for_section``), sin
inventar ni convertir nada.
"""

from __future__ import annotations

from typing import Any, Mapping

CANONICAL_CLAIM_FIELDS = (
    "claim_id",
    "claim_uid",
    "section_id",
    "claim_text",
    "scientific_verdict",
    "hallucination_risk",
    "final_correction_eligibility",
    "evidence_used",
    "correction_proposal",
    "manual_review_required",
    "llm_correction_recommendation",
)


def _canonicalize_from_claim_verification_records(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Formato ANTIGUO/crudo (``claim_verification_records``): ya casi
    coincide con el canónico (es ``ClaimVerificationResult.to_dict()`` +
    ``section_id``) — solo se proyecta a los campos canónicos exactos, sin
    inventar valores para los que falten (fail-closed: un campo obligatorio
    ausente se queda ausente, y ``_validate_claims_shape`` aguas abajo lo
    rechaza como corresponde)."""

    records = bundle.get("claim_verification_records") or ()
    claims = []
    for record in records:
        verification = dict(record.get("claim_verification_result") or {})
        claims.append(
            {
                "claim_id": verification.get("claim_id"),
                "claim_uid": verification.get("claim_uid") or "",
                "section_id": record.get("section_id"),
                "claim_text": verification.get("claim_text"),
                "scientific_verdict": verification.get("scientific_verdict"),
                "hallucination_risk": verification.get("hallucination_risk"),
                "final_correction_eligibility": verification.get("final_correction_eligibility"),
                "evidence_used": verification.get("evidence_used") or (),
                "correction_proposal": verification.get("correction_proposal"),
                "manual_review_required": verification.get("manual_review_required"),
                "llm_correction_recommendation": verification.get("llm_correction_recommendation"),
            }
        )
    return claims


def _derive_final_correction_eligibility(
    *, claim_row: Mapping[str, Any], own_corrections: list[Mapping[str, Any]]
) -> str | None:
    """Deriva la elegibilidad fail-closed a partir de lo que la fila de
    trazabilidad y sus correcciones asociadas REALMENTE dicen — sin
    inventar un valor cuando los datos no alcanzan para determinarlo
    (devuelve ``None``, que ``_validate_claims_shape`` rechaza aguas
    abajo, exactamente el comportamiento fail-closed pedido).

    Reglas, en este orden:
    1. ``manual_review_required`` en la fila -> ``MANUAL_REVIEW_REQUIRED``
       (la fila lo dice explícitamente; máxima prioridad, sin excepción).
    2. ``propose_correction`` (función productiva real) SIEMPRE crea una
       fila de corrección para cada claim, incluso cuando no había nada
       que corregir — esa fila "placeholder" tiene
       ``is_scientific_correction_action=False`` y ``action_type=None``
       (confirmado empíricamente: un claim SUPPORTED sin issues produce
       igualmente ``proposal_status="NOT_PROPOSED"``, pero
       ``is_scientific_correction_action=False``). Sin filtrar esas
       filas, ``NOT_PROPOSED`` se confundía con un intento real de
       corrección rechazado, clasificando erróneamente claims ya
       soportados como ``NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE``
       (bloqueante) en vez de ``NO_CORRECTION_NEEDED``. Se filtran aquí
       antes de evaluar el resto de las reglas.
    3. Sin ninguna corrección CIENTÍFICA real asociada -> ``NO_CORRECTION_NEEDED``.
    4. Con corrección(es) científica(s): si alguna quedó
       ``ACCEPTED_FOR_REVERIFICATION`` -> ``AUTO_CORRECTION_ELIGIBLE`` (hay
       soporte determinista verificado, no solo una recomendación).
    5. Si todas las científicas quedaron
       ``REJECTED``/``DEFERRED``/``NOT_PROPOSED`` sin ninguna aceptada
       -> ``NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE``.
    6. Cualquier otra combinación no contemplada -> ``None`` (fail-closed:
       no se inventa una elegibilidad para una forma de datos no prevista).
    """

    if claim_row.get("manual_review_required"):
        return "MANUAL_REVIEW_REQUIRED"

    scientific_corrections = [c for c in own_corrections if c.get("is_scientific_correction_action")]

    if not scientific_corrections:
        return "NO_CORRECTION_NEEDED"

    statuses = {c.get("proposal_status") for c in scientific_corrections}
    if "ACCEPTED_FOR_REVERIFICATION" in statuses:
        return "AUTO_CORRECTION_ELIGIBLE"
    if statuses and statuses.issubset({"REJECTED", "DEFERRED", "NOT_PROPOSED"}):
        return "NOT_CORRECTABLE_WITH_AVAILABLE_EVIDENCE"
    return None


def _canonicalize_from_claim_traceability_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Formato PRODUCTIVO REAL (``claim_traceability_rows`` +
    ``claim_evidence_traceability_rows`` + ``correction_traceability_rows``,
    la salida real de ``build_provisional_verification_traceability_bundle``).
    Une las tres colecciones por ``claim_id``/``correction_id`` en el
    modelo canónico único. No lee ``claim_verification_records`` — esa
    clave no existe en esta forma del bundle."""

    claim_rows = bundle.get("claim_traceability_rows") or ()
    evidence_rows = bundle.get("claim_evidence_traceability_rows") or ()
    correction_rows = bundle.get("correction_traceability_rows") or ()

    claims = []
    for claim_row in claim_rows:
        claim_id = claim_row.get("claim_id")

        own_evidence = tuple(
            {
                "evidence_id": e.get("evidence_id"),
                "source_filename": e.get("source_filename"),
                "chunk_id": e.get("chunk_id"),
                "usage_role": e.get("usage_role"),
                "authorized_for_section": e.get("authorized_for_section"),
            }
            for e in evidence_rows
            if e.get("claim_id") == claim_id and e.get("used_in_original_verification")
        )

        own_corrections = [c for c in correction_rows if c.get("claim_id") == claim_id]
        scientific_own_corrections = [c for c in own_corrections if c.get("is_scientific_correction_action")]

        correction_proposal = None
        if scientific_own_corrections:
            # Determinista: prioriza la única corrección aceptada si existe;
            # si no hay ninguna aceptada, toma la primera por correction_id
            # ordenado (sin depender del orden de iteración del bundle).
            # Se excluyen las filas "placeholder" (is_scientific_correction_action
            # False) que propose_correction crea igual para claims sin nada
            # que corregir -- ver _derive_final_correction_eligibility.
            accepted = [c for c in scientific_own_corrections if c.get("proposal_status") == "ACCEPTED_FOR_REVERIFICATION"]
            chosen = accepted[0] if accepted else sorted(scientific_own_corrections, key=lambda c: str(c.get("correction_id")))[0]
            requested_change = chosen.get("replacement_text") or chosen.get("proposed_claim_text")
            supporting_evidence = tuple(
                {
                    "evidence_id": e.get("evidence_id"),
                    "source_filename": e.get("source_filename"),
                    "chunk_id": e.get("chunk_id"),
                }
                for e in bundle.get("correction_evidence_traceability_rows") or ()
                if e.get("correction_id") == chosen.get("correction_id") and e.get("used_in_correction")
            )
            correction_proposal = {
                "requested_change": requested_change,
                "supporting_evidence": supporting_evidence,
            }

        eligibility = _derive_final_correction_eligibility(claim_row=claim_row, own_corrections=own_corrections)

        claims.append(
            {
                "claim_id": claim_id,
                "claim_uid": claim_row.get("claim_uid") or "",
                "section_id": claim_row.get("section_id"),
                "claim_text": claim_row.get("original_claim_text"),
                "scientific_verdict": claim_row.get("source_verdict"),
                "hallucination_risk": claim_row.get("source_hallucination_risk"),
                "final_correction_eligibility": eligibility,
                "evidence_used": own_evidence,
                "correction_proposal": correction_proposal,
                "manual_review_required": claim_row.get("manual_review_required"),
                "llm_correction_recommendation": claim_row.get("terminal_correction_recommendation"),
            }
        )
    return claims


def canonicalize_claims_for_transition(provisional_bundle: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Punto único de entrada: detecta cuál de las DOS formas productivas
    reales trae el bundle y las convierte AMBAS al mismo modelo canónico
    antes de clasificar — nunca dos ramas de lógica de clasificación
    distintas aguas abajo. Bundle vacío/``None`` -> lista vacía (fail-closed:
    ``classify_verification_transition`` ya trata una lista vacía como
    ``AGENT07_NO_CLAIMS``)."""

    bundle = provisional_bundle or {}

    if "claim_verification_records" in bundle:
        return _canonicalize_from_claim_verification_records(bundle)
    if "claim_traceability_rows" in bundle:
        return _canonicalize_from_claim_traceability_rows(bundle)
    return []
