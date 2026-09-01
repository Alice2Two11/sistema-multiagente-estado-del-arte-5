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
        "El campo del nombre de cada tema se llama EXACTAMENTE \"theme_name\" -- nunca "
        "\"theme_title\" ni \"title\" (esos nombres son para otros campos: \"title\" es "
        "para representative_papers, \"section_title\" es para "
        "suggested_state_of_art_structure, un bloque distinto).\n"
        "En research_gaps, el texto del vacío va en el campo \"description\" -- nunca "
        "\"gap_description\". En comparative_dimensions, el nombre de la dimensión va en "
        "el campo \"dimension\" -- nunca \"dimension_name\".\n"
        "suggested_state_of_art_structure es una lista de OBJETOS (nunca strings sueltos "
        "con solo el título de la sección) -- cada elemento debe tener EXACTAMENTE estas "
        "4 claves: \"section_id\" (string corto, ej. \"S1\"), \"section_title\" (el título "
        "de la sección), \"description\" (1-2 oraciones sobre qué cubre la sección) y "
        "\"recommended_sources\" (lista de source_filename de FUENTES VÁLIDAS relevantes "
        "para esa sección). Un elemento que sea solo el título en texto plano, sin las "
        "otras 3 claves, se descarta entero -- nunca se repara automáticamente.\n"
        f"FUENTES VÁLIDAS: {json.dumps(valid_sources, ensure_ascii=False)}\n"
        f"TÍTULOS: {json.dumps(title_map, ensure_ascii=False)}\n"
        f"CORPUS: {json.dumps(context, ensure_ascii=False)}{structure_guidance}{repair}"
    )
