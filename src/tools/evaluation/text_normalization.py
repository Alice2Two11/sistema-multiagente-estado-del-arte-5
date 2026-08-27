"""Bloque 1 de la migración de la etapa 08: normalización de texto pura.

Las 6 funciones de aquí abajo son copias LITERALES de
``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb`` (extraído el mismo día
que este archivo). No se agregó, quitó ni simplificó ninguna transformación,
expresión regular, orden de operaciones, ni tratamiento de casos borde
(``None``, NaN, listas/dicts, porcentajes, coma/punto decimal). Cualquier
divergencia frente al notebook real es un defecto de esta migración, no una
decisión de diseño — repórtala.

Mapa función → celda original
------------------------------
| Función                    | Celda | Notas de dependencia interna                          |
|-----------------------------|-------|--------------------------------------------------------|
| ``safe_str``                 | 1     | usada por las otras 5                                   |
| ``strip_internal_citations`` | 15    | usa ``safe_str`` + ``citation_pattern`` (celda 15)      |
| ``normalize_content_text``   | 15    | usa ``safe_str`` + ``strip_internal_citations``         |
| ``normalize_claim_text``     | 19    | usa ``strip_internal_citations``                        |
| ``normalize_numeric_token``  | 19    | usa ``safe_str``                                        |
| ``numeric_search_variants``  | 19    | usa ``normalize_numeric_token``                          |

Ninguna otra función del notebook 08 se ha migrado — ver
``src/adapters/AGENT08_INVENTORY.md`` (plan incremental) para el resto de
bloques (Ground Truth, ROUGE/BERTScore/embeddings, checks factuales, LLM
Judge, runtime transaccional, ``StageSpec``), que siguen sin portar por
decisión explícita.

Importar este módulo no carga modelos, no lee archivos, no requiere
OpenAI y no tiene efectos secundarios — solo define funciones puras sobre
``re``/``json``/``pandas.isna``. ``pandas`` se importa porque
``safe_str`` real usa ``pd.isna(value)`` — se conserva esa dependencia tal
cual, no se reemplaza por una verificación de NaN propia (eso sería una
"mejora funcional" no autorizada en este bloque).
"""

from __future__ import annotations

import json
import re

import pandas as pd

# ---------------------------------------------------------------------------
# safe_str — notebook 08, celda 1
# ---------------------------------------------------------------------------


def safe_str(value):
    if value is None:
        return ""

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


# ---------------------------------------------------------------------------
# strip_internal_citations / normalize_content_text — notebook 08, celda 15
# ---------------------------------------------------------------------------

citation_pattern = re.compile(r"\[([^\[\]\|]+?)\s*\|\s*([^\[\]\|]+?)\]")


def strip_internal_citations(text):
    return citation_pattern.sub("", safe_str(text))


def normalize_content_text(text):
    value = safe_str(text)
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", value)
    value = re.sub(r"^\s*#{1,6}\s*", "", value, flags=re.MULTILINE)
    value = strip_internal_citations(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


# ---------------------------------------------------------------------------
# normalize_claim_text / normalize_numeric_token / numeric_search_variants
# — notebook 08, celda 19
# ---------------------------------------------------------------------------


def normalize_claim_text(text):
    value = strip_internal_citations(text)
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(" .;:!?").casefold()


def normalize_numeric_token(value):
    token = safe_str(value)
    token = re.sub(r"\s+", "", token)
    token = token.replace(",", ".")
    return token.casefold()


def numeric_search_variants(value):
    normalized = normalize_numeric_token(value)
    variants = {
        normalized,
        normalized.replace("%", ""),
    }

    if "." in normalized:
        variants.add(normalized.replace(".", ","))
        variants.add(normalized.replace(".", ",").replace("%", ""))

    return {item for item in variants if item}
