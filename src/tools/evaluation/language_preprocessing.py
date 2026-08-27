"""Bloque 3 (parte a) de la migración de la etapa 08: segmentación de
oraciones y detección de idioma, sin llamadas a LLM.

Copias LITERALES de ``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb``,
celda 17 (bloque "MÉTRICAS AUTOMÁTICAS") para ``split_sentences``/
``chunk_text_by_sentences``, y celda 15 (bloque "Fingerprint, backup,
preprocesamiento") para ``detect_language_code``.

Mapa función → celda original
------------------------------
| Función | Celda | Dependencias | Entradas | Salidas | Excepciones | Archivos |
|---|---|---|---|---|---|---|
| ``split_sentences`` | 17 | ``safe_str`` (Bloque 1) | ``text: Any`` | ``list[str]`` de oraciones | ninguna | ninguno |
| ``chunk_text_by_sentences`` | 17 | ``split_sentences`` | ``text``, ``max_chars: int``, ``overlap_chars: int = 0`` | ``list[str]`` de chunks | ninguna | ninguno |
| ``detect_language_code`` | 15 | ``langdetect.detect`` (+ ``DetectorFactory.seed = 0``, celda 1) | ``text: Any`` | código de idioma (str, p.ej. ``"es"``/``"en"``) | ``ValueError`` si hay menos de 20 palabras en los primeros 20000 caracteres; ``RuntimeError`` si ``langdetect`` falla internamente | ninguno |

``evenly_spaced_items`` (celda 17) también existe en esa celda pero **NO se
incluye aquí**: se usa exclusivamente para muestrear pares de chunks
semánticos/BERTScore (Bloque 4), no para preprocesamiento lingüístico ni
traducción — no hay ninguna llamada a esa función en el camino de
``split_sentences``/``chunk_text_by_sentences``/``translate_text_to_language``.

Hallazgo sobre ``DetectorFactory.seed = 0``
---------------------------------------------
El notebook real fija esta semilla UNA VEZ, a nivel de módulo, en la celda 1
(``from langdetect import detect, DetectorFactory`` seguido de
``DetectorFactory.seed = 0``). Fijarla en el import de este módulo violaría
el requisito de "importar no debe detectar idiomas automáticamente / tener
efectos secundarios". Se fija aquí dentro de ``detect_language_code``, en
cada llamada, en vez de a nivel de módulo — mismo valor, mismo efecto de
determinismo, sin side effect en el import. Es el mismo tipo de ajuste
mecánico ya aplicado en los Bloques 1-2 (parametrizar en vez de leer/fijar
un global al importar), no una relajación del comportamiento real.

Importar este módulo no carga modelos, no llama a OpenAI, no lee archivos,
no detecta idiomas automáticamente y no crea directorios.
"""

from __future__ import annotations

import re
from typing import Any

from src.tools.evaluation.text_normalization import safe_str


def split_sentences(text: Any) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+", safe_str(text))
    return [
        re.sub(r"\s+", " ", candidate).strip()
        for candidate in candidates
        if candidate.strip()
    ]


def chunk_text_by_sentences(
    text: Any, max_chars: int, overlap_chars: int = 0
) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()

        if current and len(candidate) > max_chars:
            chunk = " ".join(current).strip()
            chunks.append(chunk)

            overlap_sentences: list[str] = []
            overlap_length = 0
            for previous_sentence in reversed(current):
                if overlap_length + len(previous_sentence) > overlap_chars:
                    break
                overlap_sentences.insert(0, previous_sentence)
                overlap_length += len(previous_sentence) + 1

            current = overlap_sentences + [sentence]
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def detect_language_code(text: Any) -> str:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0  # ver nota del módulo: fijado por llamada, no al importar

    sample = safe_str(text)[:20000]

    if len(sample.split()) < 20:
        raise ValueError("No hay texto suficiente para detectar el idioma.")

    try:
        return detect(sample)
    except Exception as error:
        raise RuntimeError("No se pudo detectar el idioma del texto.") from error
