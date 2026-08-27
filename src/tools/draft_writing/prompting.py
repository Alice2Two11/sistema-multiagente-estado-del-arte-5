from __future__ import annotations
import json
from .retrieval import safe_str


def language_instruction(output_language):
    normalized = safe_str(output_language).casefold()
    if normalized in {"es", "español", "espanol", "spanish"}:
        return "Redacta en español académico."
    if normalized in {"en", "inglés", "ingles", "english"}:
        return "Write in academic English."
    return f"Redacta todos los campos en {output_language}."


def assign_section_budgets(outline_sections, target_total_words):
    section_count = max(len(outline_sections), 1)
    base_target = max(80, int(int(target_total_words) / section_count))
    budgets = {}
    for section in outline_sections:
        section_id = safe_str(section.get("section_id"))
        budgets[section_id] = {
            "target_words": base_target,
            "minimum_words": max(50, int(base_target * 0.65)),
            "maximum_words": max(90, int(base_target * 1.40)),
        }
    return budgets


def build_source_free_organizational_section(section, output_language="español"):
    section_id = safe_str(section.get("section_id"))
    section_title = safe_str(section.get("section_title"))
    normalized_language = safe_str(output_language).casefold()
    if normalized_language in {"es", "español", "espanol", "spanish", "español académico"}:
        text = (
            "Esta sección presenta el alcance y la organización de la revisión. "
            "Su función es orientar la lectura y establecer la transición hacia "
            "el análisis de la evidencia científica desarrollado en las secciones siguientes."
        )
    elif normalized_language in {"en", "inglés", "ingles", "english", "academic english"}:
        text = (
            "This section presents the scope and organization of the review. "
            "Its purpose is to guide the reader and establish the transition toward "
            "the evidence-based analysis developed in the following sections."
        )
    else:
        raise ValueError(f"No existe una plantilla organizativa segura para el idioma de salida {output_language!r}.")
    return {
        "section_id": section_id,
        "section_title": section_title,
        "draft_text": text,
        "claims": [],
        "generation_attempt": 0,
        "section_validation": {
            "validation_ok": True,
            "errors": [],
            "citation_errors": [],
            "claim_errors": [],
            "numeric_errors": [],
            "valid_citation_count": 0,
            "substantive_sentence_count": 0,
            "source_free_organizational_section": True,
        },
        "deterministic_normalization": {
            "applied": True,
            "normalization_version": "v3_source_free_organizational_template",
            "source_free_organizational_section": True,
            "reason": "No evidence assigned by outline and section type permits an organizational introduction or conclusion.",
        },
    }


def build_section_prompt_v2(section, evidence, quantitative_context, previous_errors, policy):
    """Prompt del contrato canonical_sentences_v2 (Fase 3, evidence
    handles) -- SEPARADO por completo de ``build_section_prompt``
    (legacy), nunca lo reutiliza ni comparte texto de reglas con él.
    El LLM produce EXCLUSIVAMENTE ``{"section_id": ..., "sentences":
    [...]}`` -- nunca ``draft_text``/``claims`` directamente (eso lo
    deriva el sistema, determinísticamente, en
    ``materialize_initial_section_v2``).

    Evidence handles: el LLM NUNCA escribe ``source_filename``/
    ``chunk_id``/``supporting_citations``/strings ``[source | chunk]``
    -- solo referencia evidencia mediante identificadores opacos
    (``"E1"``, ``"E2"``, ...) que el SISTEMA asignó determinísticamente
    (``build_evidence_handle_map``, ``canonical_sentences.py``, mismo
    orden que ``evidence``) antes de construir este prompt. El LLM ve
    cada evidencia numerada con su ``source_filename``/``chunk_id``/
    ``text`` reales -- pero solo puede DEVOLVER el número, nunca los
    identificadores técnicos en sí. Esto elimina estructuralmente la
    posibilidad de que el LLM invente o combine un ``source_filename``
    con un ``chunk_id`` que no correspondan juntos: no existe ningún
    campo de salida donde pueda escribir esos valores.

    Prohibiciones explícitas en el propio prompt: el LLM NO debe
    producir ``supporting_citations``/``source_filename``/``chunk_id``/
    ``claim``/``claim_id``/``claim_uid``/``sentence_id``/
    ``identity_action``/``parent_claim_uids`` en ningún elemento de
    ``sentences[]`` -- el sistema los asigna/resuelve después
    (``validate_and_parse_sentences_v2`` rechaza explícitamente
    cualquiera de estos si el LLM los envía)."""

    section_id = safe_str(section.get("section_id"))
    evidence_handles = [
        {"handle": f"E{i + 1}", "source_filename": row["source_filename"], "chunk_id": row["chunk_id"], "text": row.get("text", "")}
        for i, row in enumerate(evidence)
    ]
    budgets = policy.get("section_budgets") or assign_section_budgets(
        policy.get("outline_sections") or [section],
        policy["target_total_words"],
    )
    budget = budgets[section_id]
    return f"""
Eres el agente redactor de un sistema multiagente para estados del arte científicos.

CONTRATO DE SALIDA: canonical_sentences_v2 -- produces ÚNICAMENTE una
lista de oraciones estructuradas, NUNCA texto de sección ni claims
directamente. El sistema construye el borrador final a partir de lo
que devuelvas -- tu única responsabilidad es el CONTENIDO de cada
oración y a QUÉ evidencia (por número) corresponde.

REGLAS:
1. Usa exclusivamente la evidencia proporcionada en EVIDENCIA_DISPONIBLE.
2. No uses conocimiento externo ni Ground Truth.
3. No referencies ningún número de evidencia (handle) fuera de los
   listados en EVIDENCIA_DISPONIBLE.
4. No inventes autores, años, datasets, métricas, valores ni resultados.
5. No sustituyas un handle de evidencia por otro.
6. El estilo bibliográfico {policy.get('citation_style', '')} no autoriza inventar autores o años.
7. {language_instruction(policy.get('output_language', 'español académico'))}
8. Modo de escritura: {policy.get('writing_mode', '')}. Enfoque: {policy.get('focus_mode', '')}.
9. Extensión objetivo: {budget['target_words']} palabras;
   rango orientativo: {budget['minimum_words']}-{budget['maximum_words']}.
10. UNA oración por elemento de "sentences" -- nunca combines dos
    oraciones en un solo "text", nunca dejes un "text" vacío.
11. "text" contiene ÚNICAMENTE el texto de la oración -- SIN ningún
    identificador técnico ni número de evidencia dentro. Nunca escribas
    "source_filename", "chunk_id", corchetes de cita, ni el propio
    handle (ej. "E1") dentro de "text".
12. "supporting_evidence_ids" solo puede contener HANDLES (los números
    "E1", "E2", ... tal como aparecen en EVIDENCIA_DISPONIBLE) -- NUNCA
    el source_filename ni el chunk_id en sí. Un handle que no exista en
    EVIDENCIA_DISPONIBLE invalida la respuesta completa.
13. Un valor numérico solo puede escribirse si aparece literalmente en
    el texto de uno de los handles de evidencia citados por esa misma
    oración.
14. Toda oración con contenido factual/científico debe llevar al menos
    un handle en "supporting_evidence_ids". Omite cualquier oración que
    no tenga evidencia documental real.
15. PROHIBIDO incluir en cualquier elemento de "sentences" los campos:
    "supporting_citations", "source_filename", "chunk_id", "claim",
    "claim_id", "claim_uid", "sentence_id", "identity_action",
    "parent_claim_uids". El sistema los asigna/resuelve después -- si
    los incluyes, la respuesta completa será rechazada.
16. Devuelve ÚNICAMENTE JSON válido -- sin fences de Markdown (nunca
    ```json ni ```), sin texto antes ni después del JSON.

FORMATO EXACTO (el único permitido):
{{
  "section_id": "{section_id}",
  "sentences": [
    {{
      "text": "Una sola oración, sin identificadores técnicos dentro.",
      "supporting_evidence_ids": ["E1"]
    }}
  ]
}}

SECCIÓN DEL ESQUEMA:
{json.dumps(section, ensure_ascii=False, indent=2)}

EVIDENCIA_DISPONIBLE (referencia cada una ÚNICAMENTE por su "handle" -- nunca por source_filename/chunk_id):
{json.dumps(evidence_handles, ensure_ascii=False, indent=2)}

CONTEXTO CUANTITATIVO CONFIRMADO:
{json.dumps(quantitative_context, ensure_ascii=False, indent=2)}

ERRORES DE UN INTENTO ANTERIOR:
{json.dumps(previous_errors or [], ensure_ascii=False, indent=2)}
""".strip()
