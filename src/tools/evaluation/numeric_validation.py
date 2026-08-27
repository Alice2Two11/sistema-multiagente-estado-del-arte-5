"""Bloque 5A de la migración de la etapa 08: validación numérica y factual de números.

Copias LITERALES de ``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb``,
celda 19 (bloque "MÉTRICAS FACTUALES Y DE TRAZABILIDAD FINAL"), únicamente
la parte de comprobación numérica. **No incluye** citas
(``citation_error_rate``/``df_final_citation_check``), cobertura de
evidencia, ``factual_precision``/``hallucination_rate``/
``traceability_text_coverage`` ni LLM Judge — todo eso queda fuera de
alcance de este bloque, por instrucción explícita.

Ciclo de vida del control numérico (punto 1 del pedido, verificado)
-----------------------------------------------------------------------
1. **Llega como artefacto de 07** solo en la ruta histórica 07C
   (``RECHECK_NUMERIC_CSV_PATH = DIR_VERIFY/"post_correction_numeric_check.csv"``,
   confirmado en una ronda anterior — no aplica al orquestador activo).
2. En la ruta activa (07 directo), 08 lo **sintetiza** en la celda 9 desde
   ``upstream.numeric_check_rows`` (nombre de archivo
   ``agent08_upstream_numeric_check.csv``, dentro de
   ``DIR_EVALUATION``) — ya documentado en la ronda anterior.
3. **Se recalcula por completo y se sobrescribe incondicionalmente** en
   esta celda (19): el código de aquí NO lee ni consulta el valor
   sintetizado en la celda 9 en ningún punto — reconstruye
   ``df_numeric_recheck`` desde cero a partir de ``sections`` (el texto
   evaluado, oración por oración) y ``chunks_clean_for_rag.csv``
   (``df_chunks``), y lo escribe de nuevo en la MISMA ruta
   (``RECHECK_NUMERIC_CSV_PATH``). Esto ocurre igual en ambas ramas (07 y
   07C) — no hay ningún ``if source_stage == ...`` en esta celda. **Corrección
   frente a una nota anterior**: no es "solo en la ruta directa" — es
   incondicional, en ambas rutas.
4. **No participa en los 15 outputs obligatorios** de la celda 23
   (verificado por grep exhaustivo, ronda anterior) — es un artefacto de
   trabajo intermedio, no parte de la auditoría final de completitud.

Mapa función → celda original
------------------------------
| Función | Celda | Dependencias | Entradas | Salidas | Excepciones | Archivos |
|---|---|---|---|---|---|---|
| ``to_bool`` | 1 | ninguna (NO estaba en el Bloque 1 — se necesita aquí) | ``value: Any`` | ``bool`` | ninguna | ninguno |
| ``split_section_sentences`` | 19 (nueva, no nombrada en el notebook — ver nota) | ``safe_str`` (Bloque 1) | ``section_text: str`` | ``list[str]`` | ninguna | ninguno |
| ``build_section_text_by_id`` | 19 (código de módulo) | ``safe_str`` | ``sections: list[dict]`` | ``dict[str, str]`` | ninguna | ninguno |
| ``build_chunk_text_by_pair`` | 19 (código de módulo) | ``safe_str`` | ``chunks: list[dict]`` | ``dict[tuple[str,str], str]`` | ninguna | ninguno |
| ``extract_numeric_rows`` | 19 (bucle principal, código de módulo) | ``citation_pattern``, ``strip_internal_citations``, ``numeric_search_variants`` (Bloque 1); ``split_section_sentences`` | ``section_text_by_id``, ``chunk_text_by_pair`` | ``list[dict]`` (equivalente en memoria a ``agent08_upstream_numeric_check.csv`` YA RECALCULADO) | ninguna | ninguno |
| ``aggregate_numeric_metrics`` | 19 (código de módulo) | ``to_bool`` | ``numeric_rows`` | ``dict`` con ``total_numeric_values``/``numeric_failures``/``numeric_error_rate`` | ninguna | ninguno |
| ``build_numeric_metric_rows`` | 19 (bloque ``factual_metric_rows``, código de módulo — solo la fila de ``numeric_error_rate``) | ninguna | ``numeric_error_rate`` | ``list[dict]`` de 1 fila | ninguna | ninguno |

**Reutilización explícita del Bloque 1** (no se duplica nada): este módulo
importa ``safe_str``, ``strip_internal_citations``, ``normalize_numeric_token``,
``numeric_search_variants`` y ``citation_pattern`` directamente de
``src/tools/evaluation/text_normalization.py``. La celda 19 del notebook
REDEFINE estas mismas funciones/constante de forma idéntica dentro de la
propia celda (duplicación real del notebook, no un error de esta
migración) — aquí se importan en vez de reescribirlas.

``split_section_sentences`` NO es ``split_sentences`` (Bloque 1/3)
-----------------------------------------------------------------------
La celda 19 usa un separador de oraciones DISTINTO al de la celda 17:
``re.split(r"(?<=[.!?])\\s+|\\n+", section_text)`` (también corta por
saltos de línea sueltos) y solo aplica ``.strip()` a cada fragmento — NO
colapsa espacios internos repetidos como sí hace ``split_sentences``. Son
dos funciones genuinamente distintas en el notebook real; se preservan
como dos funciones distintas aquí, sin intentar unificarlas.

Fórmulas y reglas preservadas exactamente (celda 19)
-----------------------------------------------------
- ``NUMERIC_PATTERN = re.compile(r"(?<![\\w])[-+]?\\d+(?:[.,]\\d+)?(?:\\s*%)?")``
  — signo opcional, separador decimal `.` o `,`, porcentaje opcional con
  espacio opcional antes del `%`, y un *negative lookbehind* que evita
  matchear un número pegado a una letra/dígito previo (p. ej. no matchea el
  "5" dentro de "abc5").
- Los números se extraen DESPUÉS de quitar las citas internas de la
  oración (``strip_internal_citations``), para que ningún dígito dentro de
  un ``chunk_id`` se cuente como un valor numérico del texto.
- ``matched_pairs``: para cada par citado en la MISMA oración, se compara
  el texto del chunk (normalizado con ``.replace(",", ".").casefold()``,
  **no** con ``normalize_numeric_token`` — es una normalización de texto
  completo, no de un token numérico aislado) contra cada variante del
  número (``numeric_search_variants``) mediante `in` (substring), no
  igualdad exacta.
- ``evaluation_status``: exactamente dos valores posibles —
  ``"CHECKED"`` (la oración tenía al menos una cita) o
  ``"NO_CITATION_IN_SENTENCE"`` (no tenía ninguna). **No existe una
  tercera categoría "ambiguo"** en el código real — se documenta así,
  sin inventar una que no está en el notebook.
- ``found_in_cited_chunks``: booleano, `True` si ``matched_pairs`` no está
  vacío. Es la única señal binaria de "soportado" — no hay una clasificación
  de tres estados.
- ``total_numeric_values``/``numeric_failures`` se calculan SOLO sobre las
  filas con ``evaluation_status == "CHECKED"`` — las filas
  ``NO_CITATION_IN_SENTENCE`` no cuentan ni como éxito ni como fallo, se
  excluyen de la tasa.
- ``numeric_error_rate = numeric_failures / total_numeric_values if
  total_numeric_values else None`` — devuelve **``None``**, no ``0.0``,
  cuando no hubo valores comprobables. Distinto del patrón "else 0.0" de
  ``semantic_f1`` (Bloque 4B) — se preserva cada uno tal cual es en su
  bloque de origen, sin unificarlos.
- Columnas de ``df_numeric_recheck``, en este orden exacto: ``section_id``,
  ``sentence_index``, ``numeric_index``, ``numeric_value``, ``context``,
  ``cited_pair_count``, ``cited_pairs``, ``matched_pairs``,
  ``found_in_cited_chunks``, ``evaluation_status``.
- La fila final de ``numeric_error_rate`` usa la clave ``"description"``
  (no ``"method"`` como las filas de ``automatic_metric_rows`` de los
  Bloques 4A-4C) — son dos estructuras de fila DISTINTAS en el notebook
  real (``automatic_metric_rows`` vs. ``factual_metric_rows``); se
  preserva la clave real, no se unifica con la convención de otro bloque.

No leas archivos dentro de estas funciones: ``sections``/``chunks`` llegan
como ``list[dict]`` ya cargados por el llamador (el cargador de artefactos
queda separado, fuera de este bloque). Importar este módulo no lee
archivos, no llama a OpenAI y no tiene efectos secundarios.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.tools.evaluation.text_normalization import (
    citation_pattern,
    numeric_search_variants,
    safe_str,
    strip_internal_citations,
)

NUMERIC_PATTERN = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?:\s*%)?")

NUMERIC_ROW_COLUMNS = [
    "section_id",
    "sentence_index",
    "numeric_index",
    "numeric_value",
    "context",
    "cited_pair_count",
    "cited_pairs",
    "matched_pairs",
    "found_in_cited_chunks",
    "evaluation_status",
]


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return safe_str(value).casefold() in {"true", "1", "yes", "si", "sí"}


def split_section_sentences(section_text: Any) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", safe_str(section_text))
        if sentence.strip()
    ]


def build_section_text_by_id(sections: list[dict[str, Any]]) -> dict[str, str]:
    section_text_by_id: dict[str, str] = {}
    for section in sections:
        section_id = safe_str(section.get("section_id"))
        text = (
            section.get("verified_text")
            if section.get("verified_text") is not None
            else section.get("draft_text")
        )
        section_text_by_id[section_id] = safe_str(text)
    return section_text_by_id


def build_chunk_text_by_pair(chunks: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    return {
        (safe_str(chunk["source_filename"]), safe_str(chunk["chunk_id"])): safe_str(
            chunk["text"]
        )
        for chunk in chunks
    }


def extract_numeric_rows(
    *,
    section_text_by_id: dict[str, str],
    chunk_text_by_pair: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    numeric_rows: list[dict[str, Any]] = []

    for section_id, section_text in section_text_by_id.items():
        sentences = split_section_sentences(section_text)

        for sentence_index, sentence in enumerate(sentences, start=1):
            cited_pairs = [
                (source_filename.strip(), chunk_id.strip())
                for source_filename, chunk_id in citation_pattern.findall(sentence)
            ]

            sentence_without_citations = strip_internal_citations(sentence)
            numeric_values = NUMERIC_PATTERN.findall(sentence_without_citations)

            for numeric_index, numeric_value in enumerate(numeric_values, start=1):
                variants = numeric_search_variants(numeric_value)
                matched_pairs = []

                for pair in cited_pairs:
                    chunk_text = chunk_text_by_pair.get(pair, "")
                    normalized_chunk = chunk_text.replace(",", ".").casefold()
                    if any(variant in normalized_chunk for variant in variants):
                        matched_pairs.append(pair)

                numeric_rows.append(
                    {
                        "section_id": section_id,
                        "sentence_index": sentence_index,
                        "numeric_index": numeric_index,
                        "numeric_value": numeric_value,
                        "context": sentence_without_citations,
                        "cited_pair_count": len(cited_pairs),
                        "cited_pairs": json.dumps(cited_pairs, ensure_ascii=False),
                        "matched_pairs": json.dumps(matched_pairs, ensure_ascii=False),
                        "found_in_cited_chunks": bool(matched_pairs),
                        "evaluation_status": (
                            "CHECKED" if cited_pairs else "NO_CITATION_IN_SENTENCE"
                        ),
                    }
                )

    return numeric_rows


def aggregate_numeric_metrics(numeric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked_numeric_values = [
        row for row in numeric_rows if row["evaluation_status"] == "CHECKED"
    ]
    total_numeric_values = int(len(checked_numeric_values))

    numeric_failures = (
        int(sum(1 for row in checked_numeric_values if not to_bool(row["found_in_cited_chunks"])))
        if total_numeric_values
        else 0
    )

    numeric_error_rate = (
        numeric_failures / total_numeric_values if total_numeric_values else None
    )

    return {
        "total_numeric_values": total_numeric_values,
        "numeric_failures": numeric_failures,
        "numeric_error_rate": numeric_error_rate,
    }


def build_numeric_metric_rows(numeric_error_rate: float | None) -> list[dict[str, Any]]:
    return [
        {
            "metric": "numeric_error_rate",
            "value": numeric_error_rate,
            "description": (
                "Proporción de valores numéricos que no aparecen "
                "en los chunks citados. Es null si no hubo valores comprobables."
            ),
        }
    ]
