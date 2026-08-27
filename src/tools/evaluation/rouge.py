"""Bloque 4A de la migración de la etapa 08: ROUGE-L.

Copias LITERALES de ``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb``,
celda 17 (bloque "MÉTRICAS AUTOMÁTICAS"). Incluye EXCLUSIVAMENTE la parte
de ROUGE-L: selección del texto ya preparado, configuración de
``RougeScorer``, ``use_stemmer`` según el idioma del Ground Truth, cálculo
de precisión/recall/F1, y las 3 filas correspondientes de
``automatic_metric_rows``. NO incluye similitud semántica ni BERTScore
(están en la misma celda pero son Bloque 4B/4C, no migrados aquí).

Mapa función → celda original
------------------------------
| Función | Celda | Dependencias | Entradas | Salidas | Excepciones | Archivos |
|---|---|---|---|---|---|---|
| ``build_rouge_scorer`` | 17 | ``rouge_score.rouge_scorer.RougeScorer`` (inyectable vía ``scorer_factory``) | ``ground_truth_language: str`` | instancia de ``RougeScorer`` configurada para ``["rougeL"]`` | ninguna | ninguno |
| ``compute_rouge_l`` | 17 | el ``scorer`` ya construido | ``ground_truth_plain_text: str``, ``generated_for_rouge: str``, ``scorer`` | objeto ``Score`` de ``rouge_score`` (``.precision``/``.recall``/``.fmeasure``) | las que ``scorer.score(...)`` propague | ninguno |
| ``build_rouge_metric_rows`` | 17 (bloque ``automatic_metric_rows``, código de módulo — no una función nombrada) | ninguna | el resultado de ``compute_rouge_l`` | ``list[dict]`` de 3 filas (``rougeL_precision``/``rougeL_recall``/``rougeL_fmeasure``) | ninguna | ninguno |

Selección del texto ya preparado (confirmado por lectura exacta de la celda 17)
--------------------------------------------------------------------------------
``rouge_result = rouge.score(ground_truth_plain_text, generated_for_rouge)["rougeL"]``.
**Orden de argumentos real, preservado tal cual**: el PRIMER argumento de
``.score(...)`` en ``rouge_score`` es la referencia; el SEGUNDO es la
hipótesis. El notebook pasa ``ground_truth_plain_text`` primero (referencia)
y ``generated_for_rouge`` segundo (hipótesis) — ``generated_for_rouge`` es
el texto YA resuelto por ``resolve_generated_text_for_rouge`` (Bloque 3):
el generado, traducido o no según corresponda. El Ground Truth NUNCA se
traduce ni se modifica para ROUGE — se usa ``ground_truth_plain_text`` tal
cual. Invertir este orden cambiaría qué métrica es precisión y cuál es
recall (``rouge_score`` no es simétrico) — se preserva exactamente.

``use_stemmer`` (confirmado): ``use_stemmer=(ground_truth_language == "en")``
— el stemmer de ``rouge_score`` (Porter, solo funciona bien en inglés) se
activa únicamente cuando el idioma del GROUND TRUTH (no el generado) es
inglés. No hay ninguna otra condición.

``method`` de las 3 filas (confirmado, literal): las tres llevan el mismo
valor, ``"global_text_after_translation_to_ground_truth_language"`` —
incluso cuando no hubo traducción real (``translation_mode ==
"not_required_same_language"``); el notebook no distingue esos dos casos en
la etiqueta ``method``, se preserva tal cual sin "corregirlo".

Importar este módulo no carga modelos, no llama a OpenAI y no lee
archivos. ``rouge_score`` y ``nltk`` deben estar instalados para construir
el scorer real (``build_rouge_scorer`` sin ``scorer_factory`` inyectada).
"""

from __future__ import annotations

from typing import Any, Callable

METHOD_LABEL = "global_text_after_translation_to_ground_truth_language"


def build_rouge_scorer(
    *, ground_truth_language: str, scorer_factory: Callable[..., Any] | None = None
) -> Any:
    if scorer_factory is None:
        from rouge_score import rouge_scorer as rouge_scorer_module

        scorer_factory = rouge_scorer_module.RougeScorer

    return scorer_factory(["rougeL"], use_stemmer=(ground_truth_language == "en"))


def compute_rouge_l(
    *, ground_truth_plain_text: str, generated_for_rouge: str, scorer: Any
) -> Any:
    return scorer.score(ground_truth_plain_text, generated_for_rouge)["rougeL"]


def build_rouge_metric_rows(rouge_result: Any) -> list[dict[str, Any]]:
    return [
        {
            "metric": "rougeL_precision",
            "value": float(rouge_result.precision),
            "method": METHOD_LABEL,
        },
        {
            "metric": "rougeL_recall",
            "value": float(rouge_result.recall),
            "method": METHOD_LABEL,
        },
        {
            "metric": "rougeL_fmeasure",
            "value": float(rouge_result.fmeasure),
            "method": METHOD_LABEL,
        },
    ]
