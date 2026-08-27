"""Corpus eligibility gate de Stage 03 (Agent03), en DOS FASES.

Un documento individual no útil o no validable NUNCA debe detener
todo el corpus -- ni siquiera de forma transitoria vía un RETRY/HALT
del quality gate científico ANTES de que el sistema tenga oportunidad
de clasificarlo documentalmente. Por eso el gate se divide en dos
fases, ejecutadas en momentos distintos del flujo de Stage03:

FASE 1 -- PRE-ELIGIBILIDAD (``classify_pre_eligibility`` /
``apply_pre_eligibility_policy``): se aplica INMEDIATAMENTE DESPUÉS
de la extracción y el repair de título, ANTES de la clasificación de
relevancia (que depende de un LLM y de ``topic_profile``). Solo usa
señales decidibles sin relevancia temática:

- ``EXCLUDE``: review/survey ya confirmado (``review_exclusion.py``).
- ``QUARANTINE``: título irrecuperable tras el repair ya intentado, o
  contenido/evidencia insuficiente.
- ``CANDIDATE``: ninguna de las anteriores -- continúa a
  clasificación de relevancia.

Las fichas EXCLUDE/QUARANTINE de esta fase NUNCA participan en
``build_revision_plan`` -- ni en el intento 1 ni en el intento 2, ni
antes ni después del bloque de relevancia. Esto es lo que corrige la
causa raíz real: antes, una ficha con título irrecuperable llegaba al
plan de revisión científico EN EL INTENTO 1 (antes de que el bloque
compartido, donde antes vivía el gate completo, llegara a ejecutarse),
producía ``MISSING_OR_INVALID_TITLE`` y forzaba RETRY/HALT sin que el
sistema tuviera oportunidad de marcarla QUARANTINE.

FASE 2 -- ELIGIBILIDAD FINAL (``classify_corpus_eligibility`` /
``apply_corpus_eligibility_policy``): se aplica DESPUÉS de la
clasificación de relevancia, sobre las fichas que en fase 1 quedaron
``CANDIDATE``. Construye SIEMPRE sobre fase 1 (nunca duplica su
lógica): si fase 1 ya decidió EXCLUDE/QUARANTINE, esa decisión se
preserva sin volver a evaluarla; solo las ``CANDIDATE`` se reevalúan
con la señal de relevancia:

- ``EXCLUDE``: fuera de scope o dominio excluido
  (``relevance_level == "exclude"``).
- ``QUARANTINE``: relevancia indeterminable.
- ``INCLUDE``: documento pertinente, usable y permitido.

SOLO ``INCLUDE`` (fase 2) entra al quality gate científico estricto
(``build_revision_plan``, con validación de campos como
``target_domain``/``methods_or_models``/``main_results``) y puede
requerir retry por campos faltantes.

Sin lógica nueva duplicada: ambas fases combinan exclusivamente
señales YA EXISTENTES en otros módulos --

- ``is_review_excluded`` (``review_exclusion.py``): review/survey
  confirmado por tipo documental o título.
- ``is_bad_card`` (``card_validation.py``): título irrecuperable.
- ``card.get("evidence")``: contenido/evidencia insuficiente.
- ``relevance_level == "exclude"`` (``relevance_classification.py``,
  producido por el LLM de relevancia a partir de ``topic_profile``/
  ``excluded_domains`` del experimento -- ver ``src/prompts.py``):
  fuera de scope o dominio excluido.
- ``has_valid_classification`` (``relevance_classification.py``):
  relevancia indeterminable (clasificación de relevancia incompleta
  o nunca ejecutada con éxito).

Ninguna de estas señales depende de un dominio, filename ni
experimento concretos -- multidominio y genérico por construcción,
heredado de los módulos que reutiliza.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .card_validation import is_bad_card
from .relevance_classification import has_valid_classification
from .review_exclusion import is_review_excluded

INCLUDE = "INCLUDE"
EXCLUDE = "EXCLUDE"
QUARANTINE = "QUARANTINE"
CANDIDATE = "CANDIDATE"

# Campo canónico único: fuente de verdad de la elegibilidad de una
# ficha para el resto de Stage03 (revision_plan, KB, summary, quality,
# manifest) -- evita que cada consumidor reimplemente su propia
# cascada y diverja de las demás. Toma los 4 valores de arriba a lo
# largo del flujo: ausente -> CANDIDATE/EXCLUDE/QUARANTINE (fase 1) ->
# INCLUDE/EXCLUDE/QUARANTINE (fase 2, valor final).
CORPUS_ELIGIBILITY_FIELD = "corpus_eligibility"

QUARANTINE_AUDIT_COLUMNS = [
    "source_filename",
    "reason",
    "title_irrecoverable",
    "content_insufficient",
    "relevance_indeterminate",
    "created_at",
]


def _normalize(value: Any) -> str:
    return str(value if value is not None else "").strip().casefold()


def classify_pre_eligibility(card: Mapping[str, Any]) -> dict[str, Any]:
    """FASE 1: clasifica UNA ficha en ``EXCLUDE`` | ``QUARANTINE`` |
    ``CANDIDATE``, usando exclusivamente señales decidibles ANTES de
    la clasificación de relevancia (nunca consulta ``relevance_
    level``/``has_valid_classification`` -- esas dependen del LLM de
    relevancia, que todavía no corrió en este punto del flujo).

    1. Ya excluida por policy de reviews (``is_review_excluded``) ->
       ``EXCLUDE``.
    2. Título irrecuperable tras el repair ya intentado
       (``is_bad_card``) -> ``QUARANTINE``.
    3. Sin evidencia recuperada (contenido insuficiente) ->
       ``QUARANTINE``.
    4. Ninguna de las anteriores -> ``CANDIDATE`` (continúa a
       clasificación de relevancia).
    """

    if is_review_excluded(card):
        return {"state": EXCLUDE, "reason": "Excluida por policy de reviews (review_exclusion.py)."}

    title_irrecoverable = is_bad_card(card)
    content_insufficient = not card.get("evidence")

    if title_irrecoverable or content_insufficient:
        reasons = []
        if title_irrecoverable:
            reasons.append("título irrecuperable")
        if content_insufficient:
            reasons.append("contenido/evidencia insuficiente")
        return {
            "state": QUARANTINE, "reason": "; ".join(reasons),
            "title_irrecoverable": title_irrecoverable,
            "content_insufficient": content_insufficient,
            "relevance_indeterminate": False,
        }

    return {"state": CANDIDATE, "reason": "Sin señal de exclusión/cuarentena documental -- candidata a clasificación de relevancia."}


def classify_corpus_eligibility(card: Mapping[str, Any]) -> dict[str, Any]:
    """FASE 2: clasifica UNA ficha en ``INCLUDE`` | ``EXCLUDE`` |
    ``QUARANTINE`` -- construye SIEMPRE sobre ``classify_pre_
    eligibility`` (fuente única, nunca duplica su lógica). Si fase 1
    ya decidió ``EXCLUDE``/``QUARANTINE``, esa decisión se preserva
    tal cual, sin volver a evaluarla; solo una ficha ``CANDIDATE`` se
    reevalúa con la señal de relevancia (ya disponible en este punto
    del flujo, después de que el LLM de relevancia haya corrido):

    - fuera de scope/dominio excluido (``relevance_level ==
      "exclude"``) -> ``EXCLUDE``.
    - relevancia indeterminable (``not has_valid_classification``) ->
      ``QUARANTINE``.
    - ninguna de las anteriores -> ``INCLUDE``.
    """

    pre = classify_pre_eligibility(card)
    if pre["state"] != CANDIDATE:
        return pre

    if _normalize(card.get("relevance_level")) == "exclude":
        return {"state": EXCLUDE, "reason": "Fuera de scope o dominio excluido (relevance_level=exclude)."}

    if not has_valid_classification(card):
        return {
            "state": QUARANTINE, "reason": "relevancia indeterminable",
            "title_irrecoverable": False, "content_insufficient": False,
            "relevance_indeterminate": True,
        }

    return {"state": INCLUDE, "reason": "Documento pertinente, usable y permitido."}


def is_corpus_include(card: Mapping[str, Any]) -> bool:
    """True SOLO si esta ficha ya fue confirmada ``INCLUDE`` (fase 2
    completa). Una ficha ``CANDIDATE`` (fase 1 completa, fase 2
    pendiente), ``EXCLUDE`` o ``QUARANTINE`` NUNCA es ``INCLUDE`` --
    el quality gate científico (``build_revision_plan``) solo debe
    ver fichas con elegibilidad final ya resuelta.

    Si el campo canónico está completamente ausente (ningún gate
    corrió todavía sobre esta ficha -- solo ocurre en flujos/tests que
    construyen cards a mano sin pasar por ``apply_pre_eligibility_
    policy``), cae en la única señal ya disponible en ese caso:
    ``is_review_excluded`` -- nunca trata una review ya excluida como
    ``INCLUDE`` por defecto."""

    value = card.get(CORPUS_ELIGIBILITY_FIELD)
    if value is not None:
        return value == INCLUDE
    return not is_review_excluded(card)


def is_corpus_quarantined(card: Mapping[str, Any]) -> bool:
    return card.get(CORPUS_ELIGIBILITY_FIELD) == QUARANTINE


def _quarantine_audit_row(card: Mapping[str, Any], classification: Mapping[str, Any], created_at: str) -> dict[str, Any]:
    return {
        "source_filename": card.get("source_filename"),
        "reason": classification["reason"],
        "title_irrecoverable": classification.get("title_irrecoverable", False),
        "content_insufficient": classification.get("content_insufficient", False),
        "relevance_indeterminate": classification.get("relevance_indeterminate", False),
        "created_at": created_at,
    }


def apply_pre_eligibility_policy(
    cards: Sequence[Mapping[str, Any]], *, created_at: str,
) -> dict[str, Any]:
    """Aplica ``classify_pre_eligibility`` (FASE 1) a todas las
    fichas, persistiendo el estado en ``card["corpus_eligibility"]``
    (``EXCLUDE``/``QUARANTINE``/``CANDIDATE``). Debe invocarse
    INMEDIATAMENTE después del repair de título de cada intento,
    ANTES de que ``build_revision_plan`` se llame por primera vez en
    ese intento -- así ``build_revision_plan`` nunca recibe como
    bloqueante una ficha que ya sea documentalmente EXCLUDE o
    QUARANTINE.

    Para ``QUARANTINE``: además de marcar el campo canónico, setea
    ``include_in_state_of_art = False`` (mismo contrato que ya
    consume Stage04/``corpus_filtering.py`` -- ningún cambio ahí).

    Devuelve ``{"cards": [...], "quarantine_audit_rows": [...],
    "counts": {"exclude": int, "quarantine": int, "candidate": int}}``.
    """

    updated_cards: list[dict[str, Any]] = []
    quarantine_audit_rows: list[dict[str, Any]] = []
    counts = {"exclude": 0, "quarantine": 0, "candidate": 0}

    for card in cards:
        classification = classify_pre_eligibility(card)
        state = classification["state"]
        new_card = dict(card)
        new_card[CORPUS_ELIGIBILITY_FIELD] = state

        if state == QUARANTINE:
            new_card["include_in_state_of_art"] = False
            counts["quarantine"] += 1
            quarantine_audit_rows.append(_quarantine_audit_row(card, classification, created_at))
        elif state == EXCLUDE:
            counts["exclude"] += 1
        else:
            counts["candidate"] += 1

        updated_cards.append(new_card)

    return {
        "cards": updated_cards,
        "quarantine_audit_rows": quarantine_audit_rows,
        "counts": counts,
    }


def apply_corpus_eligibility_policy(
    cards: Sequence[Mapping[str, Any]], *, created_at: str,
) -> dict[str, Any]:
    """Aplica ``classify_corpus_eligibility`` (FASE 2) a todas las
    fichas, persistiendo el estado FINAL en ``card["corpus_
    eligibility"]`` -- fuente única de verdad para el resto del flujo
    (revision_plan, KB, summary, quality, manifest). Debe invocarse
    DESPUÉS de la clasificación de relevancia, sobre TODAS las cards
    (las que ya son EXCLUDE/QUARANTINE de fase 1 simplemente preservan
    su estado -- reevaluarlas es un no-op seguro, nunca cambia su
    resultado, dado que ``classify_corpus_eligibility`` construye
    sobre ``classify_pre_eligibility``).

    Para ``QUARANTINE`` (nueva, decidida aquí por relevancia
    indeterminable): mismo tratamiento que en fase 1 --
    ``include_in_state_of_art = False`` + fila de auditoría.

    ``EXCLUDE`` no se retoca aquí si ya lo era desde fase 1 o si
    ``review_exclusion.py``/el LLM de relevancia ya dejaron
    ``include_in_state_of_art=False`` puesto.

    Devuelve ``{"cards": [...], "quarantine_audit_rows": [...],
    "counts": {"include": int, "exclude": int, "quarantine": int}}``.
    """

    updated_cards: list[dict[str, Any]] = []
    quarantine_audit_rows: list[dict[str, Any]] = []
    counts = {"include": 0, "exclude": 0, "quarantine": 0}

    for card in cards:
        classification = classify_corpus_eligibility(card)
        state = classification["state"]
        new_card = dict(card)
        new_card[CORPUS_ELIGIBILITY_FIELD] = state

        if state == QUARANTINE:
            new_card["include_in_state_of_art"] = False
            counts["quarantine"] += 1
            quarantine_audit_rows.append(_quarantine_audit_row(card, classification, created_at))
        elif state == EXCLUDE:
            counts["exclude"] += 1
        else:
            counts["include"] += 1

        updated_cards.append(new_card)

    return {
        "cards": updated_cards,
        "quarantine_audit_rows": quarantine_audit_rows,
        "counts": counts,
    }
