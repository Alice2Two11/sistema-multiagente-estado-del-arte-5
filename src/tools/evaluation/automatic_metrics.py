"""Ensamblaje en memoria de las métricas automáticas de la etapa 08.

Conecta, EN EL MISMO ORDEN que la celda 17 del notebook
(``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb``), los cuatro bloques
ya extraídos y aceptados:

```
Bloque 3 (idioma y traducción)
  -> Bloque 4A (ROUGE-L)
  -> Bloque 4B (similitud semántica)
  -> Bloque 4C (BERTScore)
```

No reimplementa ninguna fórmula: cada paso llama directamente a la función
ya extraída y probada en su propio bloque. Sin persistencia (no escribe
CSV/manifiesto) y sin ``StageSpec`` — ambos quedan para más adelante.

Mapa de orden de ejecución (celda 17, confirmado por lectura exacta antes
de escribir este módulo — ``grep`` de las 10 apariciones de ``"metric":``)
--------------------------------------------------------------------------
1.  ``resolve_generated_text_for_rouge`` (Bloque 3) -> texto para ROUGE + `translation_mode`.
2.  ``build_rouge_scorer`` + ``compute_rouge_l`` (Bloque 4A).
3.  ``build_rouge_metric_rows`` (Bloque 4A) -> 3 filas: `rougeL_precision`, `rougeL_recall`, `rougeL_fmeasure`.
4.  ``build_semantic_chunks`` (Bloque 4B) para generado y Ground Truth.
5.  Validación original (celda 17, preservada tal cual, no dentro de
    ``build_semantic_chunks`` — ver docstring del Bloque 4B): si algún lado
    queda vacío, ``ValueError("No se pudieron construir chunks para la
    evaluación semántica.")``.
6.  ``build_embedding_model`` (Bloque 4B).
7.  ``encode_chunks`` (Bloque 4B) para ambos lados.
8.  ``compute_similarity_matrix`` (Bloque 4B) -> ``semantic_matrix``, calculada
    UNA sola vez y reutilizada también por BERTScore (paso 12) — el
    notebook real nunca recalcula una segunda matriz.
9.  ``compute_semantic_precision_recall_f1`` + ``build_document_embedding`` +
    ``compute_global_semantic_similarity`` (Bloque 4B).
10. ``build_semantic_alignment_rows`` (Bloque 4B).
11. ``build_semantic_metric_rows`` (Bloque 4B) -> 4 filas: `semantic_precision`,
    `semantic_recall`, `semantic_f1`, `global_semantic_similarity`.
12. ``select_bertscore_pair_indices`` + ``build_bertscore_pairs`` (Bloque 4C),
    usando el MISMO ``semantic_matrix`` del paso 8.
13. ``run_bertscore`` (Bloque 4C) — valida internamente que haya candidatos
    (``ValueError`` si no) antes de llamar al scorer.
14. ``aggregate_bertscore`` + ``enrich_bertscore_pair_metadata`` (Bloque 4C).
15. ``build_bertscore_metric_rows`` (Bloque 4C) -> 3 filas: `bertscore_precision`,
    `bertscore_recall`, `bertscore_f1`.
16. ``automatic_metric_rows = rouge_rows + semantic_rows + bertscore_rows``
    — mismo orden exacto que la lista literal de la celda 17 (10 filas:
    3+4+3), confirmado antes de implementar.

Nada de esto escribe archivos: la función devuelve una estructura en
memoria (``AutomaticMetricsResult``). Ningún error se captura ni se
silencia — un fallo de traducción, del encoder o de BERTScore se propaga
tal cual, igual que en cada bloque por separado.

``evaluation_policy`` debe llegar YA RESUELTA por el llamador (validada
contra las 24 claves obligatorias de ``EVALUATION_POLICY`` — ver rondas
anteriores) y debe tener, como mínimo, estas claves consumidas aquí:
``translate_for_rouge_if_language_differs``, ``max_translation_chars_per_chunk``,
``semantic_chunk_chars``, ``semantic_chunk_overlap_chars``,
``max_semantic_chunks_per_text``, ``evaluation_embedding_model``,
``bertscore_model``, ``max_bertscore_pairs``. Este módulo NO les pone
ningún default: si falta una, el acceso a diccionario lanza ``KeyError``
de forma natural — no se agrega ninguna validación nueva no pedida.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.tools.evaluation.bertscore import (
    aggregate_bertscore,
    build_bertscore_metric_rows,
    build_bertscore_pairs,
    enrich_bertscore_pair_metadata,
    run_bertscore,
    select_bertscore_pair_indices,
)
from src.tools.evaluation.rouge import (
    build_rouge_metric_rows,
    build_rouge_scorer,
    compute_rouge_l,
)
from src.tools.evaluation.semantic_similarity import (
    build_document_embedding,
    build_embedding_model,
    build_semantic_alignment_rows,
    build_semantic_chunks,
    build_semantic_metric_rows,
    compute_global_semantic_similarity,
    compute_semantic_precision_recall_f1,
    compute_similarity_matrix,
    encode_chunks,
)
from src.tools.evaluation.translation import resolve_generated_text_for_rouge


@dataclass(frozen=True)
class AutomaticMetricsResult:
    generated_for_rouge: str
    translation_mode: str
    generated_chunks: list[str]
    ground_truth_chunks: list[str]
    semantic_matrix: Any
    semantic_alignment_rows: list[dict[str, Any]]
    bertscore_pair_metadata: list[dict[str, Any]]
    automatic_metric_rows: list[dict[str, Any]]


def build_automatic_metrics(
    *,
    generated_plain_text: str,
    ground_truth_plain_text: str,
    generated_language: str,
    ground_truth_language: str,
    evaluation_policy: dict[str, Any],
    translation_llm_factory: Callable[[], Any],
    embedding_model_factory: Callable[[str], Any] | None = None,
    bertscore_score_fn: Callable[..., Any] | None = None,
) -> AutomaticMetricsResult:
    # --- 1-3: Bloque 3 + Bloque 4A (ROUGE-L) -------------------------------
    generated_for_rouge, translation_mode = resolve_generated_text_for_rouge(
        generated_plain_text=generated_plain_text,
        generated_language=generated_language,
        ground_truth_language=ground_truth_language,
        translate_for_rouge=evaluation_policy["translate_for_rouge_if_language_differs"],
        llm_factory=translation_llm_factory,
        max_chars_per_chunk=evaluation_policy["max_translation_chars_per_chunk"],
    )

    rouge_scorer = build_rouge_scorer(ground_truth_language=ground_truth_language)
    rouge_result = compute_rouge_l(
        ground_truth_plain_text=ground_truth_plain_text,
        generated_for_rouge=generated_for_rouge,
        scorer=rouge_scorer,
    )
    rouge_rows = build_rouge_metric_rows(rouge_result)

    # --- 4-11: Bloque 4B (similitud semántica) -----------------------------
    generated_chunks = build_semantic_chunks(
        generated_plain_text,
        semantic_chunk_chars=evaluation_policy["semantic_chunk_chars"],
        semantic_chunk_overlap_chars=evaluation_policy["semantic_chunk_overlap_chars"],
        max_semantic_chunks_per_text=evaluation_policy["max_semantic_chunks_per_text"],
    )
    ground_truth_chunks = build_semantic_chunks(
        ground_truth_plain_text,
        semantic_chunk_chars=evaluation_policy["semantic_chunk_chars"],
        semantic_chunk_overlap_chars=evaluation_policy["semantic_chunk_overlap_chars"],
        max_semantic_chunks_per_text=evaluation_policy["max_semantic_chunks_per_text"],
    )

    if not generated_chunks or not ground_truth_chunks:
        raise ValueError(
            "No se pudieron construir chunks para la evaluación semántica."
        )

    embedding_model = build_embedding_model(
        evaluation_embedding_model=evaluation_policy["evaluation_embedding_model"],
        model_factory=embedding_model_factory,
    )
    generated_embeddings = encode_chunks(embedding_model, generated_chunks)
    ground_truth_embeddings = encode_chunks(embedding_model, ground_truth_chunks)

    semantic_matrix = compute_similarity_matrix(generated_embeddings, ground_truth_embeddings)

    semantic_precision, semantic_recall, semantic_f1 = compute_semantic_precision_recall_f1(
        semantic_matrix
    )
    generated_document_embedding = build_document_embedding(generated_embeddings)
    ground_truth_document_embedding = build_document_embedding(ground_truth_embeddings)
    global_semantic_similarity = compute_global_semantic_similarity(
        generated_document_embedding, ground_truth_document_embedding
    )

    generated_best_scores = semantic_matrix.max(axis=1)
    ground_truth_best_scores = semantic_matrix.max(axis=0)
    semantic_alignment_rows = build_semantic_alignment_rows(
        similarity_matrix=semantic_matrix,
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        generated_best_scores=generated_best_scores,
        ground_truth_best_scores=ground_truth_best_scores,
    )
    semantic_rows = build_semantic_metric_rows(
        semantic_precision=semantic_precision,
        semantic_recall=semantic_recall,
        semantic_f1=semantic_f1,
        global_semantic_similarity=global_semantic_similarity,
    )

    # --- 12-15: Bloque 4C (BERTScore), reutiliza semantic_matrix ----------
    precision_pair_indices, recall_pair_indices = select_bertscore_pair_indices(
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        max_bertscore_pairs=evaluation_policy["max_bertscore_pairs"],
    )
    bertscore_candidates, bertscore_references, bertscore_pair_metadata = build_bertscore_pairs(
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        semantic_matrix=semantic_matrix,  # misma matriz del paso 8, no recalculada
        precision_pair_indices=precision_pair_indices,
        recall_pair_indices=recall_pair_indices,
    )
    bertscore_precision_values, bertscore_recall_values, bertscore_f1_values = run_bertscore(
        candidates=bertscore_candidates,
        references=bertscore_references,
        bertscore_model=evaluation_policy["bertscore_model"],
        bertscore_score_fn=bertscore_score_fn,
    )
    bertscore_precision, bertscore_recall, bertscore_f1 = aggregate_bertscore(
        bertscore_precision_values, bertscore_recall_values, bertscore_f1_values
    )
    bertscore_pair_metadata = enrich_bertscore_pair_metadata(
        pair_metadata=bertscore_pair_metadata,
        candidates=bertscore_candidates,
        references=bertscore_references,
        precision_values=bertscore_precision_values,
        recall_values=bertscore_recall_values,
        f1_values=bertscore_f1_values,
    )
    bertscore_rows = build_bertscore_metric_rows(
        bertscore_precision=bertscore_precision,
        bertscore_recall=bertscore_recall,
        bertscore_f1=bertscore_f1,
    )

    # --- 16: concatenación en el orden exacto del notebook -----------------
    automatic_metric_rows = rouge_rows + semantic_rows + bertscore_rows

    return AutomaticMetricsResult(
        generated_for_rouge=generated_for_rouge,
        translation_mode=translation_mode,
        generated_chunks=generated_chunks,
        ground_truth_chunks=ground_truth_chunks,
        semantic_matrix=semantic_matrix,
        semantic_alignment_rows=semantic_alignment_rows,
        bertscore_pair_metadata=bertscore_pair_metadata,
        automatic_metric_rows=automatic_metric_rows,
    )
