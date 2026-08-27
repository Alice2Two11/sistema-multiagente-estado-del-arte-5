"""Exclusión determinista y auditable de reviews en Stage 03 (Agent03).

Sin LLM, sin hardcodear nombres de archivo, dominio o experimento: la
decisión depende exclusivamente del TIPO DOCUMENTAL de la propia
ficha (nunca de la tarea científica que estudia) y de la policy activa
(``extraction_policy.exclude_reviews``). Reutiliza la abstracción YA
EXISTENTE de relevancia (``include_in_state_of_art``/``relevance_
level``/``relevance_reason``, ver ``relevance_classification.py`` y
los valores reales que reconoce ``corpus_filtering.py`` en Stage 04 --
``"exclude"``) en vez de crear un mecanismo paralelo.

Separación explícita de dos dimensiones semánticas distintas, nunca
mezcladas:

- DOCUMENT TYPE (qué ES el documento): review, survey, systematic
  review, meta-analysis, research article, empirical study, etc. --
  señal de ``document_type``/``paper_type`` (campos estructurados) y
  de marcadores bibliográficos en el TÍTULO del paper completo.
- SCIENTIFIC TASK (qué tarea ESTUDIA el documento): classification,
  forecasting, segmentation, detection, regression, generation, etc.
  -- señal de ``task_type``. Un survey puede estudiar perfectamente
  "classification" o cualquier otra tarea; ``task_type`` NUNCA
  participa en esta clasificación, ni como señal positiva de review
  ni como contradicción -- solo se reporta en la auditoría como
  contexto informativo.

Cascada de señales de tipo documental, en orden de prioridad:
1. ``document_type`` (metadata estructurada, si la ficha la tiene --
   no existe hoy en el esquema de extracción, pero se consulta de
   forma segura y retrocompatible por si una fuente de metadata
   previa a la extracción llegara a poblarla en el futuro).
2. ``paper_type`` (campo estructurado ya existente).
3. Marcadores bibliográficos inequívocos en el título del paper
   completo (nunca en su contenido/secciones -- esta función solo
   recibe el título, nunca el texto interno del paper, así que un
   paper primario con una sección "Related Work"/"Literature Review"
   nunca puede activarla).
4. Si el título es irrecuperable ("no especificado"), no aporta
   señal -- la recuperación de título ya se intenta con el mecanismo
   existente (``title_repair.py``, primeros chunks del PDF + LLM)
   antes de que esta función se invoque (orden garantizado en
   ``extraction_agent.py``); esta función nunca reintenta esa
   recuperación por su cuenta.
5. Sin ninguna señal de tipo documental -> ``UNKNOWN`` (nunca se
   asume review). Fail-closed: la ausencia de evidencia nunca se
   interpreta como evidencia de exclusión.

Contradicción: solo un campo de TIPO DOCUMENTAL explícito y distinto
de review (``document_type``/``paper_type``) puede contradecir un
marcador bibliográfico del título, produciendo ``UNCERTAIN`` en vez de
``EXCLUDE`` -- ``task_type`` nunca contradice nada, por diseño.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

REVIEW_TYPE_VALUES = {
    "review", "survey", "systematic review", "literature review",
    "scoping review", "meta-analysis", "meta analysis",
}
_EMPTY_OR_UNSPECIFIED_VALUES = {"", "no especificado", "none", "nan", "n/a"}

EXCLUSION_POLICY_RULE = "exclude_reviews"
EXCLUDED_RELEVANCE_LEVEL = "exclude"

# Marcadores de título inequívocos -- términos bibliográficos estándar
# (nunca vocabulario de dominio científico) usados casi exclusivamente
# en títulos de reviews/surveys. \b garantiza límite de palabra --
# "overview" nunca matchea "review". Ordenados de más a menos
# específico: el primer match ganador determina el marcador reportado
# en auditoría (ej. "systematic review" en vez de solo "review" si el
# título contiene la frase completa).
_TITLE_REVIEW_MARKER_PATTERNS = [
    ("systematic review", re.compile(r"\bsystematic\s+reviews?\b", re.IGNORECASE)),
    ("literature review", re.compile(r"\bliterature\s+reviews?\b", re.IGNORECASE)),
    ("scoping review", re.compile(r"\bscoping\s+reviews?\b", re.IGNORECASE)),
    ("meta-analysis", re.compile(r"\bmeta[-\s]?analys(?:is|es)\b", re.IGNORECASE)),
    ("survey", re.compile(r"\bsurveys?\b", re.IGNORECASE)),
    ("review", re.compile(r"\breviews?\b", re.IGNORECASE)),
]

# Marcador propio, además de include_in_state_of_art/relevance_level:
# distingue "excluida por esta regla determinista de reviews" de
# "excluida por el clasificador LLM de relevancia" (que corre después
# y sobre un criterio distinto) -- necesario para que build_revision_
# plan y el cálculo de cobertura de campos críticos sepan exactamente
# cuáles fichas saltar sin depender de heurísticas sobre el texto de
# relevance_reason.
EXCLUDED_BY_RULE_FIELD = "excluded_by_policy_rule"

# Marcador de fichas UNKNOWN (sin ninguna señal de tipo documental,
# ni estructurada ni de título) -- ver build_revision_plan/revision_
# strategy.py: una ficha UNKNOWN con campos críticos faltantes nunca
# se asume review, pero SIGUE BLOQUEANDO Stage03 exactamente igual que
# una ficha con tipo documental conocido (mismo retry/HALT_STAGE) --
# nunca se excluye ni pasa silenciosamente. Este marcador solo permite
# que build_revision_plan le asigne el reason_code explícito y
# auditable DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID en vez del genérico
# histórico (conservado aparte en underlying_reason_code), para dejar
# claro que la causa de fondo es "tipo documental desconocido", no
# simplemente "campos faltantes".
UNKNOWN_TYPE_FIELD = "review_type_classification"
UNKNOWN_TYPE_VALUE = "UNKNOWN"

EXCLUSION_AUDIT_COLUMNS = [
    "source_filename",
    "detected_document_type",
    "detected_paper_type",
    "detected_task_type",
    "detected_title_marker",
    "action",
    "reason",
    "policy_rule",
    "created_at",
]


def _normalize(value: Any) -> str:
    return str(value if value is not None else "").strip().casefold()


def _title_review_marker(title: Any) -> str | None:
    """Devuelve el marcador bibliográfico más específico encontrado en
    el título (``"systematic review"`` antes que ``"review"`` suelto,
    etc.), o ``None`` si no hay ninguno. Nunca se aplica sobre título
    vacío/"no especificado" -- eso significa "sin señal", no "sin
    review": la recuperación de título ya se intenta antes con el
    mecanismo existente (``title_repair.py``)."""

    text = _normalize(title)
    if not text or text in _EMPTY_OR_UNSPECIFIED_VALUES:
        return None
    for marker, pattern in _TITLE_REVIEW_MARKER_PATTERNS:
        if pattern.search(text):
            return marker
    return None


def classify_review_exclusion(
    card: Mapping[str, Any], *, exclude_reviews: bool,
) -> dict[str, Any]:
    """Clasifica UNA ficha, de forma puramente determinista (sin LLM),
    en ``"EXCLUDE"`` | ``"KEEP"`` | ``"UNCERTAIN"`` (con
    ``classification`` adicional ``"UNKNOWN"`` cuando no hay ninguna
    señal de tipo documental en absoluto), mediante una cascada de
    señales de TIPO DOCUMENTAL -- nunca de tarea científica:

    1. ``exclude_reviews=False`` -> siempre ``"KEEP"``.
    2. ``document_type``/``paper_type`` (tipo documental estructurado)
       indican review de forma confiable (valor en
       ``REVIEW_TYPE_VALUES``) -> ``"EXCLUDE"``.
    3. Si el tipo documental estructurado no contradice explícitamente
       (vacío o coincide), pero el título del paper completo contiene
       un marcador bibliográfico inequívoco -> ``"EXCLUDE"``, salvo
       que el tipo documental estructurado indique explícitamente algo
       DISTINTO de review -> ``"UNCERTAIN"`` (contradicción entre
       título y tipo documental estructurado).
    4. Sin ninguna señal de tipo documental (ni campos, ni título) ->
       ``"KEEP"`` con ``classification="UNKNOWN"`` -- nunca se asume
       review por ausencia de evidencia.

    ``task_type`` (tarea científica: classification, forecasting,
    segmentation, etc.) NUNCA participa en esta decisión -- ni como
    señal positiva ni como contradicción; solo se reporta en la
    auditoría como contexto informativo. Un survey puede estudiar
    cualquier tarea científica sin que eso cambie que es un survey.
    """

    if not exclude_reviews:
        return {
            "action": "KEEP", "classification": None,
            "detected_document_type": card.get("document_type"),
            "detected_paper_type": card.get("paper_type"),
            "detected_task_type": card.get("task_type"), "detected_title_marker": None,
            "reason": "Policy no solicita excluir reviews (exclude_reviews=False).",
        }

    document_type = _normalize(card.get("document_type"))
    paper_type = _normalize(card.get("paper_type"))
    title_marker = _title_review_marker(card.get("title"))

    # Tipo documental estructurado: document_type tiene prioridad SI
    # ES INFORMATIVO (no vacío/"no especificado"/"none"/"nan"/"n/a");
    # si no lo es, se usa paper_type (también solo si es informativo).
    # Nunca se combinan ni se promedian -- es una cascada por
    # INFORMATIVIDAD, no una votación, y nunca un simple chequeo de
    # "string no vacío": "no especificado" es un string truthy pero
    # NO aporta señal, así que nunca puede ganarle a un paper_type
    # real como "survey" solo por venir primero en la cascada.
    document_type_is_informative = document_type not in _EMPTY_OR_UNSPECIFIED_VALUES
    paper_type_is_informative = paper_type not in _EMPTY_OR_UNSPECIFIED_VALUES
    if document_type_is_informative:
        structured_type = document_type
    elif paper_type_is_informative:
        structured_type = paper_type
    else:
        structured_type = ""
    structured_is_review = structured_type in REVIEW_TYPE_VALUES
    structured_is_empty = structured_type in _EMPTY_OR_UNSPECIFIED_VALUES
    structured_contradicts = not structured_is_empty and not structured_is_review

    if structured_is_review:
        return {
            "action": "EXCLUDE", "classification": None,
            "detected_document_type": card.get("document_type"),
            "detected_paper_type": card.get("paper_type"),
            "detected_task_type": card.get("task_type"), "detected_title_marker": title_marker,
            "reason": "Clasificada de forma confiable como review (tipo documental estructurado); excluida por policy.exclude_reviews.",
        }

    if title_marker:
        if structured_contradicts:
            return {
                "action": "UNCERTAIN", "classification": None,
                "detected_document_type": card.get("document_type"),
                "detected_paper_type": card.get("paper_type"),
                "detected_task_type": card.get("task_type"), "detected_title_marker": title_marker,
                "reason": (
                    f"El título contiene el marcador bibliográfico '{title_marker}' pero el tipo "
                    f"documental estructurado ('{structured_type}') indica explícitamente algo "
                    "distinto -- clasificación contradictoria, no se excluye automáticamente. "
                    "task_type nunca participa en esta comparación."
                ),
            }
        return {
            "action": "EXCLUDE", "classification": None,
            "detected_document_type": card.get("document_type"),
            "detected_paper_type": card.get("paper_type"),
            "detected_task_type": card.get("task_type"), "detected_title_marker": title_marker,
            "reason": (
                f"Tipo documental estructurado sin clasificar, pero el título contiene el "
                f"marcador bibliográfico inequívoco '{title_marker}'; excluida por "
                "policy.exclude_reviews."
            ),
        }

    if structured_contradicts:
        return {
            "action": "KEEP", "classification": None,
            "detected_document_type": card.get("document_type"),
            "detected_paper_type": card.get("paper_type"),
            "detected_task_type": card.get("task_type"), "detected_title_marker": None,
            "reason": f"Tipo documental estructurado ('{structured_type}') indica explícitamente que no es review.",
        }

    return {
        "action": "KEEP", "classification": UNKNOWN_TYPE_VALUE,
        "detected_document_type": card.get("document_type"),
        "detected_paper_type": card.get("paper_type"),
        "detected_task_type": card.get("task_type"), "detected_title_marker": None,
        "reason": "Ninguna señal de tipo documental disponible (ni campos estructurados ni título) -- no se asume review.",
    }


def is_review_excluded(card: Mapping[str, Any]) -> bool:
    """True si esta ficha YA fue marcada como excluida por esta regla
    determinista concreta (nunca confunde con exclusión por baja
    relevancia del clasificador LLM, que usa el mismo ``relevance_
    level`` pero no este marcador)."""

    return card.get(EXCLUDED_BY_RULE_FIELD) == EXCLUSION_POLICY_RULE


def is_unknown_document_type(card: Mapping[str, Any]) -> bool:
    """True si esta ficha quedó marcada ``UNKNOWN`` (sin ninguna señal
    de tipo documental) por ``apply_review_exclusion_policy`` -- nunca
    se le asume review. Si además la ficha es inválida (campos
    faltantes, título inválido, etc.), ``build_revision_plan``
    (``revision_strategy.py``) la distingue de una ficha con tipo
    documental conocido usando el reason_code explícito
    ``DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID`` en vez del genérico
    histórico (``MISSING_CRITICAL_FIELDS``, etc., conservado aparte en
    ``underlying_reason_code``) -- pero SIGUE bloqueando exactamente
    igual que antes (mismo retry/HALT_STAGE): esto nunca excluye la
    ficha ni la deja pasar silenciosamente, solo hace auditable que la
    causa de fondo es "tipo documental desconocido", no simplemente
    "campos faltantes"."""

    return card.get(UNKNOWN_TYPE_FIELD) == UNKNOWN_TYPE_VALUE


def apply_review_exclusion_policy(
    cards: Sequence[Mapping[str, Any]], *, exclude_reviews: bool, created_at: str,
) -> dict[str, Any]:
    """Aplica ``classify_review_exclusion`` a todas las fichas.

    Para cada ficha con ``action == "EXCLUDE"``: reutiliza el contrato
    YA EXISTENTE de relevancia -- ``include_in_state_of_art = False``,
    ``relevance_level = "exclude"`` (mismo valor literal que ya
    reconoce ``corpus_filtering.py``), y agrega ``relevance_reason``
    (sin sobrescribir una razón previa no vacía) más el marcador
    ``excluded_by_policy_rule``. NUNCA rellena ni inventa
    ``methods_or_models``/``evaluation_metrics``/``main_results`` --
    esos campos quedan exactamente como la extracción los produjo.

    Para cada ficha con ``classification == "UNKNOWN"`` (sin ninguna
    señal de tipo documental): se marca con ``review_type_
    classification = "UNKNOWN"`` -- nunca se excluye. Si además le
    faltan campos críticos, ``build_revision_plan`` sigue bloqueando
    exactamente igual que a cualquier otra ficha inválida (mismo
    retry/HALT_STAGE), pero le asigna el reason_code explícito
    ``DOCUMENT_TYPE_UNKNOWN_AND_CARD_INVALID`` en vez del genérico
    histórico, para que quede auditable que la causa de fondo es
    "tipo documental desconocido" -- nunca la deja pasar
    silenciosamente ni la excluye por asumir que es review.

    Fichas ``"UNCERTAIN"``/``"KEEP"`` (con señal, pero no-review) no
    se tocan más allá de lo anterior.

    Devuelve ``{"cards": [...], "audit_rows": [...], "num_excluded":
    int, "num_uncertain": int}`` -- ``audit_rows`` sigue el shape de
    ``EXCLUSION_AUDIT_COLUMNS``, con una fila por ficha EXCLUDE o
    UNCERTAIN (nunca por KEEP -- evita inflar el CSV de auditoría con
    fichas sin ninguna señal de review)."""

    updated_cards: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    num_excluded = 0
    num_uncertain = 0

    for card in cards:
        classification = classify_review_exclusion(card, exclude_reviews=exclude_reviews)
        action = classification["action"]
        new_card = dict(card)

        if action == "EXCLUDE":
            new_card["include_in_state_of_art"] = False
            new_card["relevance_level"] = EXCLUDED_RELEVANCE_LEVEL
            existing_reason = str(new_card.get("relevance_reason") or "").strip()
            if not existing_reason or existing_reason.casefold() in _EMPTY_OR_UNSPECIFIED_VALUES:
                new_card["relevance_reason"] = classification["reason"]
            new_card[EXCLUDED_BY_RULE_FIELD] = EXCLUSION_POLICY_RULE
            num_excluded += 1
            audit_rows.append({
                "source_filename": card.get("source_filename"),
                "detected_document_type": classification["detected_document_type"],
                "detected_paper_type": classification["detected_paper_type"],
                "detected_task_type": classification["detected_task_type"],
                "detected_title_marker": classification.get("detected_title_marker"),
                "action": action,
                "reason": classification["reason"],
                "policy_rule": EXCLUSION_POLICY_RULE,
                "created_at": created_at,
            })
        elif action == "UNCERTAIN":
            num_uncertain += 1
            audit_rows.append({
                "source_filename": card.get("source_filename"),
                "detected_document_type": classification["detected_document_type"],
                "detected_paper_type": classification["detected_paper_type"],
                "detected_task_type": classification["detected_task_type"],
                "detected_title_marker": classification.get("detected_title_marker"),
                "action": action,
                "reason": classification["reason"],
                "policy_rule": EXCLUSION_POLICY_RULE,
                "created_at": created_at,
            })
        elif classification.get("classification") == UNKNOWN_TYPE_VALUE:
            new_card[UNKNOWN_TYPE_FIELD] = UNKNOWN_TYPE_VALUE

        updated_cards.append(new_card)

    return {
        "cards": updated_cards,
        "audit_rows": audit_rows,
        "num_excluded": num_excluded,
        "num_uncertain": num_uncertain,
    }
