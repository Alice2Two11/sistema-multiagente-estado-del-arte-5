# ============================================================
# Verificación de key_arguments contra evidencia real (05)
# ------------------------------------------------------------
# Hallazgo que motiva esto (investigación real, experimento_paper_39/41):
# 05 sintetiza sus key_arguments a partir de fichas resumidas (03) y
# análisis temático (04) -- NUNCA contra el texto fuente real. Eso
# permite que aparezca un key_argument sintéticamente plausible que
# ningún paper específico respalda con esa especificidad (caso
# confirmado: "integración cuantitativa de variables sociales con
# modelos espaciales", sección S2 de un experimento de Sostenibilidad,
# jamás verificable en 07 porque nunca existió evidencia real).
#
# Esta función hace lo mismo que ya hace 06 al buscar evidencia para
# escribir una sección (retrieve_section_evidence, con el mismo piso de
# relevancia mínima ya aplicado ahí) pero ANTES, sobre cada
# key_argument individual, para descartar los que no tengan ningún
# respaldo real en los papers ya asignados a esa sección -- así 06
# nunca llega a intentar escribir sobre algo que 05 prometió sin base.
# ============================================================
from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.tools.draft_writing.retrieval import (
    query_chroma_restricted,
    query_csv_restricted,
)


def _section_source_filenames(section: Mapping[str, Any]) -> list[str]:
    papers = section.get("papers_to_use") or []
    out = []
    for paper in papers:
        name = paper.get("source_filename") if isinstance(paper, dict) else paper
        name = str(name or "").strip()
        if name:
            out.append(name)
    return out


def _key_argument_has_evidence(
    key_argument: str,
    collection: Any,
    chunks_df: Any,
    source_filenames: list[str],
    min_relevance_score: float,
) -> bool:
    """True si existe al menos un chunk, dentro de los papers ya
    asignados a la sección, con relevancia semántica o léxica real
    para este key_argument puntual -- misma lógica de dos niveles
    (Chroma restringido + respaldo léxico) que ya usa 06 al redactar."""

    semantic_hits = query_chroma_restricted(
        collection,
        chunks_df,
        key_argument,
        source_filenames,
        top_k=1,
        min_relevance_score=min_relevance_score,
    )
    if semantic_hits:
        return True

    lexical_hits = query_csv_restricted(
        chunks_df,
        key_argument,
        source_filenames,
        top_k=1,
        min_overlap_tokens=2,  # más estricto que el de 06 (1): esto es un
        # chequeo de "existe algo real", no un último recurso para llenar
        # cupo de evidencia -- exigir 2+ palabras compartidas evita que
        # una sola palabra común (ej. un conector) cuente como respaldo.
    )
    return bool(lexical_hits)


def verify_and_prune_unsupported_key_arguments(
    outline: Mapping[str, Any],
    *,
    collection: Any,
    chunks_df: Any,
    min_relevance_score: float,
) -> dict[str, Any]:
    """Recorre cada sección del esquema y descarta los key_arguments
    que no tengan ningún respaldo real recuperable en los papers ya
    asignados a esa sección. Nunca en silencio: cada eliminación queda
    registrada en ``section["key_arguments_removed_unverified"]`` y en
    el resumen devuelto, para que quede trazable en la metodología.

    Secciones sin papers asignados (introducción, cierre, por diseño)
    se dejan intactas -- no hay nada contra qué verificar, y eso ya es
    una regla explícita y legítima del esquema, no un vacío a corregir.
    """

    sections = outline.get("sections")
    if not isinstance(sections, list):
        return {"sections_checked": 0, "key_arguments_removed": 0, "details": []}

    details: list[dict[str, Any]] = []
    total_removed = 0

    for section in sections:
        if not isinstance(section, dict):
            continue

        key_arguments = section.get("key_arguments")
        if not isinstance(key_arguments, list) or not key_arguments:
            continue

        source_filenames = _section_source_filenames(section)
        if not source_filenames:
            # Sección legítimamente sin papers (ej. introducción) -- no se
            # verifica nada, se deja tal cual llegó de 05.
            continue

        kept: list[str] = []
        removed: list[str] = []
        for key_argument in key_arguments:
            text = str(key_argument or "").strip()
            if not text:
                continue
            if _key_argument_has_evidence(
                text, collection, chunks_df, source_filenames, min_relevance_score
            ):
                kept.append(text)
            else:
                removed.append(text)

        if removed:
            section["key_arguments"] = kept
            section["key_arguments_removed_unverified"] = removed
            total_removed += len(removed)
            details.append(
                {
                    "section_id": section.get("section_id"),
                    "removed_count": len(removed),
                    "removed_key_arguments": removed,
                }
            )

    return {
        "sections_checked": sum(
            1 for s in sections if isinstance(s, dict) and _section_source_filenames(s)
        ),
        "key_arguments_removed": total_removed,
        "details": details,
    }
