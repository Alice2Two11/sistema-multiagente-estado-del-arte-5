from __future__ import annotations

import json
import re
from typing import Any


OUTLINE_ROOT_KEYS = frozenset(
    {
        "title",
        "objective",
        "narrative_strategy",
        "sections",
    }
)


def _is_outline_root(value: Any) -> bool:
    """Acepta únicamente el objeto raíz completo del esquema.

    No acepta objetos correspondientes a una sección individual,
    ejemplos JSON parciales ni otros diccionarios válidos que puedan
    aparecer en la respuesta del LLM.
    """
    if not isinstance(value, dict):
        return False

    if not OUTLINE_ROOT_KEYS.issubset(value.keys()):
        return False

    if not isinstance(value.get("sections"), list):
        return False

    return True


def parse_outline_response(value: Any) -> dict[str, Any]:
    """Extrae el objeto JSON raíz completo de una respuesta de 05.

    Fail-closed:
    - el resultado debe ser un objeto;
    - debe contener title/objective/narrative_strategy/sections;
    - sections debe ser una lista;
    - nunca devuelve el primer diccionario JSON arbitrario.
    """

    # Ya viene parseado.
    if isinstance(value, dict):
        if _is_outline_root(value):
            return value
        raise ValueError(
            "INVALID_LLM_OUTPUT: el objeto JSON no contiene "
            "la estructura raíz completa del outline."
        )

    if not isinstance(value, str):
        raise ValueError(
            "INVALID_LLM_OUTPUT: la respuesta del outline "
            "debe ser texto JSON u objeto JSON."
        )

    text = value.strip()

    if not text:
        raise ValueError(
            "INVALID_LLM_OUTPUT: respuesta vacía."
        )

    candidates: list[str] = []

    # Primero bloques fenced completos.
    for match in re.finditer(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        candidates.append(
            match.group(1).strip()
        )

    # Después la respuesta completa.
    candidates.append(text)

    decoder = json.JSONDecoder()

    # Se examinan todos los objetos decodificables y solo se
    # acepta aquel que cumple el contrato raíz.
    for candidate in candidates:
        candidate = candidate.strip()

        # Caso directo.
        try:
            obj = json.loads(candidate)
            if _is_outline_root(obj):
                return obj
        except Exception:
            pass

        # Caso texto alrededor del JSON.
        for match in re.finditer(r"\{", candidate):
            start = match.start()

            try:
                obj, _ = decoder.raw_decode(
                    candidate[start:]
                )
            except json.JSONDecodeError:
                continue

            if _is_outline_root(obj):
                return obj

    raise ValueError(
        "INVALID_LLM_OUTPUT: no se encontró un objeto JSON "
        "con la estructura raíz completa del outline "
        "(title, objective, narrative_strategy, sections)."
    )



def extract_first_valid_json(
    value: Any,
    parse_fallback: Any | None = None,
) -> dict[str, Any]:
    """Compatibilidad con el runtime productivo de 05.

    A diferencia de la implementación anterior, NO acepta el primer
    diccionario JSON arbitrario encontrado dentro de la respuesta.
    Solo devuelve el objeto raíz completo del outline.

    ``parse_fallback`` se conserva por compatibilidad, pero cualquier
    resultado del fallback también debe cumplir el contrato raíz.
    """

    try:
        return parse_outline_response(value)

    except Exception as primary_error:

        if parse_fallback is None:
            raise primary_error

        # El fallback tampoco puede saltarse la validación estructural.
        fallback_result = parse_fallback(value)

        if _is_outline_root(fallback_result):
            return fallback_result

        raise ValueError(
            "INVALID_LLM_OUTPUT: el parser fallback devolvió JSON, "
            "pero no corresponde al objeto raíz completo del outline."
        ) from primary_error
