from __future__ import annotations
import json

PROMPT_VERSION = "v16_thematic_agent_01"


def build_thematic_prompt(
    context, valid_sources, title_map, repair_plan=None, *,
    min_sections=None, max_sections=None, enforce_section_count=False,
):
    repair = ""
    if repair_plan:
        repair = "\nREPARACIÓN DIRIGIDA:\n" + json.dumps(
            repair_plan,
            ensure_ascii=False,
        )
    structure_guidance = ""
    if min_sections is not None or max_sections is not None:
        if enforce_section_count:
            structure_guidance = (
                "\nESTRUCTURA: suggested_state_of_art_structure debe tener "
                f"entre {min_sections} y {max_sections} secciones (límite estricto, "
                "no lo excedas ni quedes por debajo).\n"
            )
        else:
            structure_guidance = (
                f"\nESTRUCTURA: min_sections={min_sections} y max_sections={max_sections} "
                "son ÚNICAMENTE orientativos, no un límite estricto. La granularidad real "
                "debe emerger de los temas identificados en el corpus -- no fuerces el "
                "conteo de secciones a un número fijo, y evita crear secciones "
                "redundantes que separen artificialmente el mismo tema.\n"
            )
    return (
        "Analiza exclusivamente la KB proporcionada. No uses Ground Truth, "
        "bibliografía ni conocimiento externo.\n"
        "Devuelve un objeto JSON con corpus_summary, themes, research_gaps, "
        "suggested_state_of_art_structure y comparative_dimensions.\n"
        "Cada tema debe tener representative_papers con source_filename y title exactos. "
        "Cada gap y dimensión debe tener fuentes válidas.\n"
        f"FUENTES VÁLIDAS: {json.dumps(valid_sources, ensure_ascii=False)}\n"
        f"TÍTULOS: {json.dumps(title_map, ensure_ascii=False)}\n"
        f"CORPUS: {json.dumps(context, ensure_ascii=False)}{structure_guidance}{repair}"
    )
