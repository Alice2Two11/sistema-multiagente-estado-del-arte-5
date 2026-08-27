"""Reparación dirigida de longitud total del borrador (Stage 06),
EXCLUSIVAMENTE para el contrato ``canonical_sentences_v2`` (Evidence
Handles) -- nunca para legacy, dado que Evidence Handles es una
construcción exclusivamente V2. Cuando la policy declara legacy, esta
reparación simplemente no se invoca (ver ``draft_writing_agent.py``);
el borrador sigue el camino NEEDS_REVISION/RETRY histórico sin cambios.

Nunca inventa claims, citas, números ni evidencia -- opera
EXCLUSIVAMENTE reutilizando la MISMA evidencia ya recuperada para cada
sección (``evidence_map``), y el MISMO mecanismo de validación V2 ya
existente (``validate_and_parse_sentences_v2``, ``v2_numeric_support_
errors``, ``materialize_initial_section_v2``) -- ninguna ruta de
validación nueva ni paralela. Nunca usa Ground Truth -- el prompt de
reparación, igual que el prompt V2 original, solo referencia
``EVIDENCIA_DISPONIBLE`` (misma evidencia ya retrieved por la sección).

Nunca "expansión determinista" que rellene con frases repetidas: el
prompt exige contenido científico sustentado, y si el LLM no puede
ampliar de forma sustantiva, devuelve la MISMA sección sin cambios --
lo cual esta función detecta como fallo de reparación (el word_count
no se movió en la dirección correcta), nunca como éxito."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .canonical_sentences import (
    build_evidence_handle_map,
    materialize_initial_section_v2,
    v2_numeric_support_errors,
    validate_and_parse_sentences_v2,
)
from .validation import count_content_words, section_allows_no_sources


def build_length_repair_prompt(
    section: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    current_text: str,
    mode: str,
    word_delta: int,
) -> str:
    """``mode``: ``"expand"`` | ``"condense"``. Muestra la evidencia con
    los MISMOS Evidence Handles (E1, E2, ...) que ya usa el contrato
    V2, el texto ACTUAL de la sección, y pide expandir o condensar
    EXCLUSIVAMENTE con esa evidencia -- nunca introduce fuentes ni
    afirmaciones nuevas fuera de lo ya recuperado, y nunca menciona ni
    usa Ground Truth."""

    evidence_handles = [
        {
            "handle": f"E{i + 1}",
            "source_filename": row["source_filename"],
            "chunk_id": row["chunk_id"],
            "text": row.get("text", ""),
        }
        for i, row in enumerate(evidence)
    ]
    section_id = str(section.get("section_id", ""))

    if mode == "expand":
        instruction = (
            f"El texto actual de esta sección contribuye a un déficit de aproximadamente "
            f"{word_delta} palabras respecto al mínimo contractual del estado del arte completo. "
            "Amplía el desarrollo de síntesis, comparación o análisis crítico de esta sección "
            "EXCLUSIVAMENTE con la evidencia ya listada en EVIDENCIA_DISPONIBLE -- nunca "
            "introduzcas datasets, métricas, autores, años ni resultados que no aparezcan en esa "
            "evidencia, y nunca uses Ground Truth ni conocimiento externo. No repitas ni "
            "parafrasees las oraciones existentes solo para alcanzar la extensión -- cada oración "
            "nueva debe aportar contenido científico real (comparación, matiz, limitación, "
            "conexión entre fuentes) sustentado en la evidencia disponible. Si la evidencia "
            "disponible no permite ampliar de forma sustantiva, devuelve exactamente las mismas "
            "oraciones que ya existían -- nunca rellenes con texto redundante para alcanzar el "
            "conteo."
        )
    else:
        instruction = (
            f"El texto actual de esta sección contribuye a un exceso de aproximadamente "
            f"{word_delta} palabras respecto al máximo contractual del estado del arte completo. "
            "Condensa esta sección eliminando redundancia y reiteración -- pero conserva TODAS "
            "las afirmaciones con soporte real en la evidencia y sus citas correspondientes. "
            "Nunca elimines una afirmación sustantiva sin soporte documental, y nunca uses "
            "Ground Truth ni conocimiento externo."
        )

    return (
        "Reformula ÚNICAMENTE la sección indicada, dentro del mismo contrato "
        "canonical_sentences_v2 (evidence handles) ya usado para generarla -- nunca inventes "
        "fuentes, valores ni afirmaciones fuera de la evidencia listada. No uses Ground Truth.\n"
        f"{instruction}\n"
        f"TEXTO ACTUAL DE LA SECCIÓN:\n{current_text}\n"
        "EVIDENCIA_DISPONIBLE (referencia cada una ÚNICAMENTE por su \"handle\"):\n"
        f"{json.dumps(evidence_handles, ensure_ascii=False, indent=2)}\n"
        "FORMATO EXACTO (el único permitido):\n"
        "{\n"
        f'  "section_id": "{section_id}",\n'
        '  "sentences": [\n'
        '    {"text": "Una sola oración, sin identificadores técnicos dentro.", "supporting_evidence_ids": ["E1"]}\n'
        "  ]\n"
        "}\n"
        "Devuelve ÚNICAMENTE JSON válido -- sin fences de Markdown, sin texto antes ni después."
    )


def attempt_section_length_repair(
    section: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    current_section: Mapping[str, Any],
    mode: str,
    word_delta: int,
    runtime: Any,
) -> tuple[dict[str, Any], bool]:
    """Invoca el LLM UNA VEZ para expandir/condensar la sección --
    valida con el MISMO mecanismo V2 (``validate_and_parse_sentences_
    v2`` + ``v2_numeric_support_errors``), materializa solo si es
    válido. Devuelve ``(section_dict, success)`` -- ``success=False``
    si el LLM no produjo una respuesta válida, si introdujo un error
    numérico, o si el resultado no se movió en la dirección correcta
    (expand: debe aumentar el conteo; condense: debe disminuirlo) --
    en cualquiera de esos casos se devuelve la sección ORIGINAL sin
    modificar, nunca una versión parcialmente aplicada."""

    current_text = str(current_section.get("draft_text", ""))
    sid = str(section.get("section_id", ""))
    prompt = build_length_repair_prompt(section, evidence, current_text, mode, word_delta)

    try:
        raw = runtime.invoke(prompt)
        payload = runtime.parse(raw)
    except Exception:
        return dict(current_section), False

    evidence_handle_map = build_evidence_handle_map(evidence)
    parse_result = validate_and_parse_sentences_v2(payload, evidence_handle_map, expected_section_id=sid)
    if not parse_result["validation_ok"]:
        return dict(current_section), False

    numeric_errors = v2_numeric_support_errors(parse_result["sentences"], evidence)
    if numeric_errors:
        return dict(current_section), False

    materialized = materialize_initial_section_v2(parse_result["sentences"], sid)
    new_word_count = count_content_words(materialized["draft_text"])
    old_word_count = count_content_words(current_text)

    if mode == "expand" and new_word_count <= old_word_count:
        return dict(current_section), False
    if mode == "condense" and new_word_count >= old_word_count:
        return dict(current_section), False

    repaired = dict(current_section)
    repaired["draft_text"] = materialized["draft_text"]
    repaired["claims"] = materialized["claims"]
    repaired["section_validation"] = {"validation_ok": True, "errors": []}
    return repaired, True


def attempt_directed_length_repair(
    generated: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    evidence_map: Mapping[str, Sequence[Mapping[str, Any]]],
    policy: Mapping[str, Any],
    runtime: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Orquesta la reparación dirigida sobre TODAS las secciones que
    tengan evidencia real disponible y no sean source-free
    organizacionales (esas nunca tienen evidencia que ampliar/
    condensar). Recorre en el orden en que aparecen las secciones,
    deteniéndose en cuanto el total queda dentro de
    ``[configured_min_total_words, configured_max_total_words]``.

    Fail-closed por diseño: si ninguna sección candidata tiene
    evidencia suficiente para moverse en la dirección correcta, la
    función simplemente no logra cerrar el déficit/exceso -- nunca
    fuerza un resultado, nunca rellena. El llamador (``draft_writing_
    agent.py``) es quien decide qué hacer con
    ``repair_metadata["final_total_words"]`` seguir o no dentro de
    rango.

    Devuelve ``(repaired_generated, repair_metadata)`` con
    ``repair_metadata = {"attempted": bool, "sections_repaired": [...],
    "sections_skipped_no_evidence": [...], "final_total_words": int}``.
    """

    by_id = {str(item.get("section_id", "")): dict(item) for item in generated}
    section_lookup = {str(sec.get("section_id", "")): sec for sec in sections}

    total_words = sum(count_content_words(item.get("draft_text", "")) for item in generated)
    target_total = int(policy["target_total_words"])
    configured_min = int(policy["min_total_words"])
    configured_max = int(policy["max_total_words"])

    if configured_min <= total_words <= configured_max:
        return list(generated), {
            "attempted": False, "sections_repaired": [],
            "sections_skipped_no_evidence": [], "final_total_words": total_words,
        }

    mode = "expand" if total_words < configured_min else "condense"
    attempted_any = False
    sections_repaired: list[str] = []
    sections_skipped: list[str] = []

    candidate_ids = [
        sid for sid, sec in section_lookup.items()
        if not section_allows_no_sources(sec) and evidence_map.get(sid)
    ]

    for sid in candidate_ids:
        current = by_id.get(sid)
        if current is None:
            continue

        remaining_deficit = configured_min - total_words
        remaining_excess = total_words - configured_max
        if mode == "expand" and remaining_deficit <= 0:
            break
        if mode == "condense" and remaining_excess <= 0:
            break

        word_delta = remaining_deficit if mode == "expand" else remaining_excess
        current_words = count_content_words(current.get("draft_text", ""))
        attempted_any = True

        repaired_section, success = attempt_section_length_repair(
            section_lookup[sid], evidence_map.get(sid, []), current, mode, word_delta, runtime,
        )
        if success:
            new_words = count_content_words(repaired_section.get("draft_text", ""))
            total_words += new_words - current_words
            by_id[sid] = repaired_section
            sections_repaired.append(sid)
        else:
            sections_skipped.append(sid)

    repaired_generated = [
        by_id.get(str(item.get("section_id", "")), item) for item in generated
    ]
    return repaired_generated, {
        "attempted": attempted_any,
        "sections_repaired": sections_repaired,
        "sections_skipped_no_evidence": sections_skipped,
        "final_total_words": total_words,
    }
