"""Parseo robusto de JSON producido por un LLM -- utilidad neutral, sin
dependencia hacia ningún dominio del pipeline. Antes vivía únicamente
en ``src/tools/evaluation/llm_judge.py`` (celda 21 del notebook 08
original); se extrajo aquí para que otros dominios (ej.
``src/adapters/draft_writing_runtime.py``, Agent06) puedan reutilizarla
sin depender arquitectónicamente de ``src/tools/evaluation``.

``llm_judge.py`` reexporta ``parse_json_safely`` desde este módulo
(``from src.utils.json_parsing import parse_json_safely``) para que
cualquier código existente que la importe como
``from src.tools.evaluation.llm_judge import parse_json_safely`` siga
funcionando sin cambios -- la semántica y el comportamiento son
exactamente los mismos, solo cambió dónde vive la definición."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_safely(text: Any) -> Any:
    """Extrae JSON aunque la respuesta venga envuelta en un bloque Markdown.
    Copia literal de notebook 08, celda 1."""

    if isinstance(text, (dict, list)):
        return text

    value = str(text).strip()
    value = re.sub(r"^\s*```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```\s*$", "", value)
    value = value.strip()

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
            return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("La respuesta del LLM no contiene JSON válido.")
