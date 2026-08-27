"""LLM Judge (celda 21): rúbrica, prompt, validación estricta, reintentos.

Mismo patrón que ``translation.py`` (Bloque 3): recibe ``llm_factory``
(construye un cliente NUEVO por intento, igual que ``get_llm(...)`` dentro
del bucle real) en vez de construir el LLM internamente.

No incluye el cacheo por fingerprint (``judge_cache_valid``,
``LLM_JUDGE_MANIFEST_PATH``) ni la escritura de
``attempt_{n}.txt``/``llm_judge_evaluation.json``/``llm_judge_scores.csv``
— eso es persistencia, migrada aparte.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from langchain_core.messages import HumanMessage

# JSON parsing cleanup: la implementación local duplicada de
# parse_json_safely se retiró -- se delega a la canónica. La
# semántica y el comportamiento son exactamente los mismos (era una
# copia byte a byte); la posibilidad de inyectar
# ``parse_json_safely_fn`` en las funciones existentes se conserva
# intacta más abajo.
from src.utils.json_parsing import parse_json_safely

from src.tools.evaluation.text_normalization import safe_str

PROMPT_VERSION = "v5_rubric_reference_comparison_strict_json"

JUDGE_CRITERIA = {
    "coherence": "Continuidad lógica entre ideas, secciones y transiciones.",
    "organization": "Estructura clara, progresión temática y distribución adecuada.",
    "critical_depth": "Comparación, contraste, limitaciones, vacíos y análisis crítico.",
    "synthesis_quality": "Integración de múltiples trabajos en una narrativa propia.",
    "argumentative_clarity": (
        "Claridad de la tesis, precisión de afirmaciones y legibilidad académica."
    ),
}


def balanced_excerpt(text: Any, maximum_chars: int) -> tuple[str, bool]:
    value = safe_str(text)
    if len(value) <= maximum_chars:
        return value, False

    part = maximum_chars // 3
    middle_start = max(0, len(value) // 2 - part // 2)
    excerpt = (
        value[:part]
        + "\n\n[...]\n\n"
        + value[middle_start : middle_start + part]
        + "\n\n[...]\n\n"
        + value[-part:]
    )
    return excerpt, True


def build_judge_prompt(
    *,
    topic_name: str,
    source_stage: str,
    automatic_metrics: dict[str, float],
    factual_metrics: dict[str, Any],
    generated_judge_text: str,
    ground_truth_judge_text: str,
    previous_errors: list[str] | None = None,
) -> str:
    error_instruction = ""
    if previous_errors:
        error_instruction = (
            "\nLa respuesta anterior fue inválida. Corrige estos errores:\n"
            + json.dumps(previous_errors, ensure_ascii=False, indent=2)
        )

    return f"""
Eres un evaluador académico. Compara un estado del arte generado
contra una revisión de literatura real publicada.

TEMA:
{topic_name}

REGLAS:
1. Evalúa únicamente calidad académica y cobertura comparativa.
2. No sustituyas la auditoría factual del pipeline; esa dimensión procede
   del upstream seleccionado ({source_stage}) y se reporta por separado.
3. Usa la rúbrica 1-5:
   1 = muy deficiente,
   2 = deficiente,
   3 = aceptable,
   4 = buena,
   5 = excelente.
4. Cada puntuación debe incluir una justificación concreta.
5. evidence_from_generated puede incluir hasta tres fragmentos breves,
   cada uno de máximo 20 palabras.
6. missing_topics_or_omissions debe basarse en contenido visible
   en el Ground Truth y ausente o débil en el texto generado.
7. No inventes papers, autores ni resultados.
8. Devuelve únicamente JSON válido.

CRITERIOS:
{json.dumps(JUDGE_CRITERIA, ensure_ascii=False, indent=2)}

MÉTRICAS AUTOMÁTICAS:
{json.dumps(automatic_metrics, ensure_ascii=False, indent=2)}

MÉTRICAS FACTUALES:
{json.dumps(factual_metrics, ensure_ascii=False, indent=2)}

ESTADO DEL ARTE GENERADO:
{generated_judge_text}

GROUND TRUTH — REVISIÓN DE LITERATURA:
{ground_truth_judge_text}

FORMATO:
{{
  "scores": {{
    "coherence": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }},
    "organization": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }},
    "critical_depth": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }},
    "synthesis_quality": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }},
    "argumentative_clarity": {{
      "score": 1,
      "justification": "",
      "evidence_from_generated": []
    }}
  }},
  "strengths": [],
  "organization_differences": [],
  "missing_topics_or_omissions": [
    {{
      "topic": "",
      "ground_truth_basis": "",
      "importance": "",
      "search_keywords": []
    }}
  ],
  "overall_assessment": ""
}}
{error_instruction}
""".strip()


def validate_judge_result(result: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(result, dict):
        return ["judge_result_not_object"]

    scores = result.get("scores")
    if not isinstance(scores, dict):
        errors.append("scores_not_object")
        return errors

    missing_criteria = sorted(set(JUDGE_CRITERIA) - set(scores))
    unknown_criteria = sorted(set(scores) - set(JUDGE_CRITERIA))
    if missing_criteria:
        errors.append("missing_criteria:" + ",".join(missing_criteria))
    if unknown_criteria:
        errors.append("unknown_criteria:" + ",".join(unknown_criteria))

    for criterion in JUDGE_CRITERIA:
        item = scores.get(criterion)
        if not isinstance(item, dict):
            errors.append(f"{criterion}:not_object")
            continue

        try:
            score = float(item.get("score"))
            if score < 1 or score > 5 or score != int(score):
                errors.append(f"{criterion}:score_not_integer_1_to_5")
        except Exception:
            errors.append(f"{criterion}:invalid_score")

        if not safe_str(item.get("justification")):
            errors.append(f"{criterion}:empty_justification")

        evidence = item.get("evidence_from_generated")
        if not isinstance(evidence, list):
            errors.append(f"{criterion}:evidence_not_list")
        else:
            if len(evidence) > 3:
                errors.append(f"{criterion}:too_many_evidence_items")
            for evidence_item in evidence:
                if len(safe_str(evidence_item).split()) > EVIDENCE_ITEM_MAX_WORDS:
                    errors.append(f"{criterion}:evidence_item_over_20_words")

    for list_field in ["strengths", "organization_differences", "missing_topics_or_omissions"]:
        if not isinstance(result.get(list_field), list):
            errors.append(f"{list_field}:not_list")

    omissions = result.get("missing_topics_or_omissions", [])
    if isinstance(omissions, list):
        for index, omission in enumerate(omissions, start=1):
            if not isinstance(omission, dict):
                errors.append(f"omission_{index}:not_object")
                continue
            for key in ["topic", "ground_truth_basis", "importance", "search_keywords"]:
                if key not in omission:
                    errors.append(f"omission_{index}:missing_{key}")
            if "search_keywords" in omission and not isinstance(
                omission["search_keywords"], list
            ):
                errors.append(f"omission_{index}:search_keywords_not_list")

    if not safe_str(result.get("overall_assessment")):
        errors.append("empty_overall_assessment")

    return errors


EVIDENCE_ITEM_MAX_WORDS = 20
"""Mismo límite que ya exige el prompt (regla 5, build_judge_prompt) y
valida validate_judge_result (evidence_item_over_20_words) -- una sola
constante para no duplicar el número mágico 20 en dos lugares."""


def evidence_length_violations(
    parsed: Any, max_words: int = EVIDENCE_ITEM_MAX_WORDS,
) -> list[dict[str, Any]]:
    """Recalcula, a partir del payload ya parseado, cada elemento de
    ``evidence_from_generated`` que excede ``max_words`` -- una entrada
    estructurada por violación, en el mismo orden posicional en que
    ``validate_judge_result`` las detecta (una entrada por elemento
    sobrante, nunca deduplicada por criterio: si dos elementos del
    mismo criterio exceden el límite, produce dos violaciones).

    NUNCA modifica ``parsed`` ni ningún elemento de evidencia -- solo
    lee y reporta. Cada entrada trae ``criterion``,
    ``evidence_item_index`` (posición dentro de la lista de ese
    criterio), ``actual_word_count``, ``max_word_count`` y un
    ``message`` en español listo para inyectar en el prompt de retry,
    pidiendo reformulación (nunca recorte mecánico) para preservar el
    significado."""

    violations: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return violations
    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        return violations

    for criterion in JUDGE_CRITERIA:
        item = scores.get(criterion)
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence_from_generated")
        if not isinstance(evidence, list):
            continue
        for index, evidence_item in enumerate(evidence):
            text = safe_str(evidence_item)
            count = len(text.split())
            if count > max_words:
                violations.append({
                    "criterion": criterion,
                    "evidence_item_index": index,
                    "actual_word_count": count,
                    "max_word_count": max_words,
                    "evidence_item_text": text,
                    "message": (
                        f"{criterion}: evidence_from_generated[{index}] tiene "
                        f"{count} palabras; máximo permitido = {max_words}. "
                        "Reformula ese elemento conservando su significado."
                    ),
                })
    return violations


def build_retry_feedback(
    parsed: Any, errors: list[str], max_words: int = EVIDENCE_ITEM_MAX_WORDS,
) -> list[str]:
    """Construye el feedback que recibe el SIGUIENTE intento --
    reemplaza cada entrada de ``errors`` con forma
    ``"{criterion}:evidence_item_over_20_words"`` por su mensaje
    estructurado y específico (ubicación + conteo exacto, ver
    ``evidence_length_violations``); cualquier OTRO código de error
    (score inválido, justificación vacía, estructura, etc.) se
    conserva TAL CUAL, sin ningún cambio -- ``error_code`` estable,
    mismo comportamiento histórico fuera de este caso concreto.

    Nunca trunca ni reformula nada por sí misma -- solo construye el
    texto que le PIDE al LLM que reformule, con la información exacta
    que antes le faltaba (qué elemento, cuántas palabras, cuál límite).
    Si hay más entradas de error para un criterio que violaciones
    detectadas (no debería ocurrir en la práctica), el código original
    se conserva como respaldo en vez de perder la señal de error."""

    violations = evidence_length_violations(parsed, max_words)
    messages_by_criterion: dict[str, list[str]] = {}
    for violation in violations:
        messages_by_criterion.setdefault(violation["criterion"], []).append(violation["message"])

    consumed = {criterion: 0 for criterion in messages_by_criterion}
    feedback: list[str] = []
    for err in errors:
        if str(err).endswith(":evidence_item_over_20_words"):
            criterion = str(err).split(":", 1)[0]
            available = messages_by_criterion.get(criterion, [])
            position = consumed.get(criterion, 0)
            if position < len(available):
                feedback.append(available[position])
                consumed[criterion] = position + 1
                continue
        feedback.append(err)
    return feedback


def run_llm_judge(
    *,
    topic_name: str,
    source_stage: str,
    automatic_metrics: dict[str, float],
    factual_metrics: dict[str, Any],
    generated_plain_text: str,
    ground_truth_plain_text: str,
    max_generated_chars: int,
    max_ground_truth_chars: int,
    max_attempts: int,
    llm_factory: Callable[[], Any],
    parse_json_safely_fn: Callable[[str], Any] = parse_json_safely,
) -> dict[str, Any]:
    """Reproduce el bucle de reintentos real (celda 21, rama ``else`` —
    sin cacheo). Un intento fallido NO se reintenta con el mismo cliente:
    se construye una instancia NUEVA por intento, igual que
    ``get_llm(...)`` dentro del bucle real. Ningún error se silencia:
    ``json_parse_error``/errores de validación se acumulan en
    ``previous_errors`` y se inyectan en el siguiente prompt, tal cual.

    Devuelve ``{"result": dict, "raw_attempts": list[str], "attempt_errors":
    list[list[str]], "judge_mode": "new"}`` -- ``attempt_errors[i]`` son los
    códigos de error ESTABLES (``validate_judge_result``, sin enriquecer)
    del intento ``i``, para auditoría; el feedback enriquecido (ver
    ``build_retry_feedback``) solo se usa para construir el prompt del
    siguiente intento, nunca sustituye el registro de auditoría.
    (Persistencia de ``attempt_{n}.txt``/JSON/manifest queda fuera de este
    módulo.)
    """

    generated_judge_text, generated_truncated = balanced_excerpt(
        generated_plain_text, max_generated_chars
    )
    ground_truth_judge_text, ground_truth_truncated = balanced_excerpt(
        ground_truth_plain_text, max_ground_truth_chars
    )

    previous_errors: list[str] = []
    llm_judge_result = None
    raw_attempts: list[str] = []
    attempt_errors: list[list[str]] = []

    for _attempt in range(1, max_attempts + 1):
        prompt = build_judge_prompt(
            topic_name=topic_name,
            source_stage=source_stage,
            automatic_metrics=automatic_metrics,
            factual_metrics=factual_metrics,
            generated_judge_text=generated_judge_text,
            ground_truth_judge_text=ground_truth_judge_text,
            previous_errors=previous_errors,
        )

        llm = llm_factory()
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_text = safe_str(response.content)
        raw_attempts.append(raw_text)

        try:
            parsed = parse_json_safely_fn(raw_text)
        except Exception as error:
            previous_errors = [f"json_parse_error:{error}"]
            attempt_errors.append(list(previous_errors))
            continue

        errors = validate_judge_result(parsed)
        attempt_errors.append(list(errors))
        if not errors:
            llm_judge_result = parsed
            break

        # Feedback estructurado, ANTES de reintentar: cada violación de
        # longitud se reemplaza por su ubicación+conteo exactos (ver
        # build_retry_feedback) -- nunca se trunca ni se reformula nada
        # aquí, solo se construye el texto que le pide al LLM que
        # reformule conservando el significado. Cualquier otro tipo de
        # error (mezclado o no con violaciones de longitud) se conserva
        # tal cual, sin ningún atajo de aceptación -- el flujo de retry
        # sigue siendo el único camino, nunca hay repair silencioso.
        previous_errors = build_retry_feedback(parsed, errors)

    if llm_judge_result is None:
        raise ValueError(
            "El LLM Judge no produjo un resultado válido después de "
            f"{max_attempts} intentos. Errores: {previous_errors}"
        )

    return {
        "result": llm_judge_result,
        "raw_attempts": raw_attempts,
        "attempt_errors": attempt_errors,
        "judge_mode": "new",
        "generated_excerpt_truncated": generated_truncated,
        "ground_truth_excerpt_truncated": ground_truth_truncated,
    }


def build_judge_score_rows(llm_judge_result: dict[str, Any]) -> list[dict[str, Any]]:
    judge_score_rows = []
    for criterion in JUDGE_CRITERIA:
        item = llm_judge_result["scores"][criterion]
        judge_score_rows.append(
            {
                "metric": criterion,
                "score_1_to_5": int(item["score"]),
                "justification": safe_str(item["justification"]),
                "evidence_from_generated": json.dumps(
                    item["evidence_from_generated"], ensure_ascii=False
                ),
            }
        )
    return judge_score_rows
