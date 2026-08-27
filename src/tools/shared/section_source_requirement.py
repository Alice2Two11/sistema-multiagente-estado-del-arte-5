"""Contrato ÚNICO y compartido entre Stage 05 (generación de esquema,
``outline_generation``) y Stage 06 (redacción, ``draft_writing``) de
qué secciones pueden legítimamente carecer de ``papers_to_use``/
evidencia recuperada -- fuente única de verdad, importada por ambas
etapas, para que una sección aprobada por 05 nunca pueda ser
rechazada por 06 por una definición diferente (la inconsistencia real
que este módulo cierra: Stage 05 reconocía introducción/discusión/
gaps-vacíos/cierre/conclusión, Stage 06 solo introducción/conclusión/
cierre).

Sin hardcodear ningún título, dominio o experimento concreto -- la
clasificación depende exclusivamente de ``section_type``/
``section_title`` de la propia sección, nunca de su ``section_id`` ni
de contenido específico."""

from __future__ import annotations

import re
from typing import Any, Mapping

SOURCE_REQUIRED = "SOURCE_REQUIRED"
SOURCE_FREE_ORGANIZATIONAL = "SOURCE_FREE_ORGANIZATIONAL"

# section_type ESTRUCTURADO reconocido como organizativo/sintético --
# mismos valores que ya usaba Stage 05 (SECTION_ALLOW_TYPES), ahora
# fuente única compartida con Stage 06.
ORGANIZATIONAL_SECTION_TYPES = frozenset({
    "introduccion", "introducción", "introduction",
    "discusion", "discusión", "discussion",
    "gaps", "research_gaps", "vacios", "vacíos",
    "cierre", "closing",
    "conclusion", "conclusión", "conclusiones", "conclusions",
})

# Fallback por título -- SOLO se consulta cuando section_type está
# ausente/vacío. Mismos tokens que ya usaba Stage 05
# (SECTION_ALLOW_TITLE_TOKENS), incluidos "perspectivas"/"tendencias"
# (secciones de cierre prospectivo, igual de organizativas).
ORGANIZATIONAL_TITLE_TOKENS = (
    "introducción", "introduccion", "introduction",
    "discusión", "discusion", "discussion",
    "vacíos", "vacios", "gaps",
    "conclusión", "conclusion", "conclusiones", "conclusions",
    "perspectivas", "tendencias", "cierre",
)


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def classify_section_source_requirement(section: Mapping[str, Any]) -> str:
    """Clasifica una sección del esquema en ``SOURCE_REQUIRED`` o
    ``SOURCE_FREE_ORGANIZATIONAL``, de forma puramente estructural y
    determinista, sin LLM.

    Prioridad estricta: si ``section_type`` está presente (no vacío),
    decide por sí solo -- nunca se consulta el título en ese caso. Una
    sección ``linea_tematica``/``fundamentos``/``comparacion``/
    ``metodologia`` (o cualquier ``section_type`` no reconocido como
    organizativo) es SIEMPRE ``SOURCE_REQUIRED``, aunque su título
    contenga por accidente una palabra como "conclusiones" -- el
    ``section_type`` estructurado nunca se sobrescribe por una
    coincidencia de texto libre.

    Solo cuando ``section_type`` está ausente o vacío se recurre al
    título como respaldo (mismo comportamiento histórico de Stage 05
    para ese caso concreto)."""

    section_type = _normalize(section.get("section_type"))
    if section_type:
        return (
            SOURCE_FREE_ORGANIZATIONAL
            if section_type in ORGANIZATIONAL_SECTION_TYPES
            else SOURCE_REQUIRED
        )

    title = _normalize(section.get("section_title"))
    if any(token in title for token in ORGANIZATIONAL_TITLE_TOKENS):
        return SOURCE_FREE_ORGANIZATIONAL
    return SOURCE_REQUIRED


def section_is_source_free_organizational(section: Mapping[str, Any]) -> bool:
    """Azúcar sintáctico booleano sobre ``classify_section_source_
    requirement`` -- para los call sites que solo necesitan la
    decisión binaria (Stage 05's validación/reparación, Stage 06's
    gate de evidencia)."""

    return classify_section_source_requirement(section) == SOURCE_FREE_ORGANIZATIONAL
