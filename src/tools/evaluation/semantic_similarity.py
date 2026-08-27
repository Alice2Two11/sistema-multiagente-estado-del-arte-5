"""Bloque 4B de la migración de la etapa 08: similitud semántica por embeddings.

Copias LITERALES de ``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb``,
celda 17 (bloque "MÉTRICAS AUTOMÁTICAS"), la parte de similitud semántica
(no BERTScore, no ROUGE — esos son Bloque 4A/4C). Todas las funciones
devuelven estructuras en memoria: no se escribe ningún CSV ni manifest
aquí (eso es persistencia, Bloque 6).

Separación (requisito 2 del pedido)
-------------------------------------
- **Funciones puras de chunking/alineación** (sin modelo, sin red): 
  ``build_semantic_chunks``, ``compute_similarity_matrix``,
  ``compute_semantic_precision_recall_f1``, ``build_document_embedding``,
  ``compute_global_semantic_similarity``, ``build_semantic_alignment_rows``,
  ``build_semantic_metric_rows``.
- **Construcción del modelo** (inyectable, no productivo por defecto):
  ``build_embedding_model``.
- **Codificación de textos** (usa el modelo ya construido):
  ``encode_chunks``.

Mapa función → celda original
------------------------------
| Función | Celda | Dependencias | Entradas | Salidas | Excepciones | Archivos |
|---|---|---|---|---|---|---|
| ``build_semantic_chunks`` | 17 | ``chunk_text_by_sentences`` (Bloque 3), ``evenly_spaced_items`` | ``text``, ``semantic_chunk_chars``, ``semantic_chunk_overlap_chars``, ``max_semantic_chunks_per_text`` | ``list[str]`` de chunks muestreados uniformemente | ninguna (la validación de "no vacío" vive en el llamador — ver nota) | ninguno |
| ``evenly_spaced_items`` | 17 | ``numpy`` | ``items: Sequence``, ``maximum: int`` | ``list`` submuestreado uniformemente vía ``np.linspace`` | ninguna | ninguno |
| ``build_embedding_model`` | 17 | ``sentence_transformers.SentenceTransformer`` (inyectable vía ``model_factory``) | ``evaluation_embedding_model: str`` | instancia del modelo | las que ``SentenceTransformer(...)`` propague | descarga el modelo la primera vez (comportamiento real de la librería, no introducido aquí) |
| ``encode_chunks`` | 17 | el modelo ya construido | ``model``, ``chunks: list[str]`` | matriz de embeddings normalizados (``normalize_embeddings=True``) | las que ``model.encode(...)`` propague | ninguno |
| ``compute_similarity_matrix`` | 17 | ``sklearn.metrics.pairwise.cosine_similarity`` | ``generated_embeddings``, ``ground_truth_embeddings`` | matriz NxM de similitud coseno | ninguna | ninguno |
| ``compute_semantic_precision_recall_f1`` | 17 | ``numpy`` (vía la matriz) | ``similarity_matrix`` | ``(precision, recall, f1)`` | ninguna | ninguno |
| ``build_document_embedding`` | 17 | ``numpy`` | ``chunk_embeddings`` | embedding promedio (``.mean(axis=0)``) | ninguna | ninguno |
| ``compute_global_semantic_similarity`` | 17 | ``cosine_similarity`` | ``generated_document_embedding``, ``ground_truth_document_embedding`` | similitud global (float) | ninguna | ninguno |
| ``build_semantic_alignment_rows`` | 17 (bloque ``semantic_alignment_rows``, código de módulo) | ninguna | matriz + chunks + puntajes | ``list[dict]`` (equivalente en memoria a ``semantic_alignment.csv``) | ninguna | ninguno |
| ``build_semantic_metric_rows`` | 17 (bloque ``automatic_metric_rows``, código de módulo) | ninguna | ``semantic_precision``, ``semantic_recall``, ``semantic_f1``, ``global_semantic_similarity`` | ``list[dict]`` de 4 filas | ninguna | ninguno |

Validación de chunks vacíos (confirmada, celda 17): el notebook real
levanta ``ValueError("No se pudieron construir chunks para la evaluación
semántica.")`` si ``generated_chunks`` o ``ground_truth_chunks`` quedan
vacíos DESPUÉS de ``build_semantic_chunks``. Esa comprobación no vive
dentro de ``build_semantic_chunks`` en el notebook (es un ``if`` aparte,
después de construir ambos), así que aquí tampoco se mete dentro de la
función — se deja como responsabilidad del llamador (el runtime no migrado
en este bloque), igual que en el notebook.

Fórmulas preservadas exactamente (celda 17)
-----------------------------------------------
- ``generated_best_scores = similarity_matrix.max(axis=1)`` (mejor match de
  cada chunk generado hacia el Ground Truth).
- ``ground_truth_best_scores = similarity_matrix.max(axis=0)`` (mejor match
  de cada chunk del Ground Truth hacia el generado).
- ``semantic_precision = float(generated_best_scores.mean())``.
- ``semantic_recall = float(ground_truth_best_scores.mean())``.
- ``semantic_f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0`` — **no** división por
  cero cuando ambos son 0: cae al ``else 0.0`` explícito del notebook, se
  preserva tal cual.
- ``global_semantic_similarity = cosine_similarity([doc_emb_generado],
  [doc_emb_gt])[0][0]`` sobre el promedio (``.mean(axis=0)``) de los
  embeddings de chunk de cada lado.

Manejo de vectores cero (verificado, sin comportamiento especial en el
notebook real): no hay ningún chequeo de norma cero antes de
``cosine_similarity`` en la celda 17 — se apoya únicamente en
``normalize_embeddings=True`` de ``SentenceTransformer.encode``. Si un
chunk produjera un embedding todo-ceros, ``cosine_similarity`` de
``sklearn`` devuelve ``0.0`` para ese par (no lanza, no da ``NaN`` — es el
comportamiento real y documentado de esa función ante un vector cero); este
módulo no agrega ninguna verificación adicional que el notebook no tenga.

Alineación (confirmada, celda 17): dos bloques de filas —
``generated_to_ground_truth`` (para cada chunk generado, su mejor match en
el Ground Truth, vía ``argmax`` por fila) y ``ground_truth_to_generated``
(para cada chunk del Ground Truth, su mejor match en el generado, vía
``argmax`` por columna). ``source_preview``/``matched_preview`` truncan a
300 caracteres (``[:300]``), preservado tal cual.

Importar este módulo no carga modelos, no llama a OpenAI y no lee
archivos. ``sentence-transformers`` y ``scikit-learn`` deben estar
instalados para construir el modelo/calcular similitud reales
(``build_embedding_model``/``compute_similarity_matrix`` sin factories
inyectadas) — igual que ``rouge_score``/``nltk`` en el Bloque 4A.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from src.tools.evaluation.language_preprocessing import chunk_text_by_sentences


def evenly_spaced_items(items: Sequence[Any], maximum: int) -> list[Any]:
    if len(items) <= maximum:
        return list(items)

    indices = np.linspace(0, len(items) - 1, num=maximum, dtype=int)
    return [items[int(index)] for index in indices]


def build_semantic_chunks(
    text: str,
    *,
    semantic_chunk_chars: int,
    semantic_chunk_overlap_chars: int,
    max_semantic_chunks_per_text: int,
) -> list[str]:
    chunks = chunk_text_by_sentences(
        text, semantic_chunk_chars, semantic_chunk_overlap_chars
    )
    return evenly_spaced_items(chunks, max_semantic_chunks_per_text)


def build_embedding_model(
    *, evaluation_embedding_model: str, model_factory: Callable[[str], Any] | None = None
) -> Any:
    if model_factory is None:
        from sentence_transformers import SentenceTransformer

        model_factory = SentenceTransformer

    return model_factory(evaluation_embedding_model)


def encode_chunks(model: Any, chunks: list[str]) -> Any:
    return model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)


def compute_similarity_matrix(generated_embeddings: Any, ground_truth_embeddings: Any) -> Any:
    from sklearn.metrics.pairwise import cosine_similarity

    return cosine_similarity(generated_embeddings, ground_truth_embeddings)


def compute_semantic_precision_recall_f1(similarity_matrix: Any) -> tuple[float, float, float]:
    generated_best_scores = similarity_matrix.max(axis=1)
    ground_truth_best_scores = similarity_matrix.max(axis=0)

    semantic_precision = float(generated_best_scores.mean())
    semantic_recall = float(ground_truth_best_scores.mean())

    semantic_f1 = (
        2 * semantic_precision * semantic_recall / (semantic_precision + semantic_recall)
        if (semantic_precision + semantic_recall) > 0
        else 0.0
    )

    return semantic_precision, semantic_recall, semantic_f1


def build_document_embedding(chunk_embeddings: Any) -> Any:
    return chunk_embeddings.mean(axis=0)


def compute_global_semantic_similarity(
    generated_document_embedding: Any, ground_truth_document_embedding: Any
) -> float:
    from sklearn.metrics.pairwise import cosine_similarity

    return float(
        cosine_similarity(
            [generated_document_embedding], [ground_truth_document_embedding]
        )[0][0]
    )


def build_semantic_alignment_rows(
    *,
    similarity_matrix: Any,
    generated_chunks: list[str],
    ground_truth_chunks: list[str],
    generated_best_scores: Any,
    ground_truth_best_scores: Any,
) -> list[dict[str, Any]]:
    semantic_alignment_rows: list[dict[str, Any]] = []

    for generated_index, score in enumerate(generated_best_scores):
        best_gt_index = int(similarity_matrix[generated_index].argmax())
        semantic_alignment_rows.append(
            {
                "direction": "generated_to_ground_truth",
                "source_chunk_index": generated_index,
                "matched_chunk_index": best_gt_index,
                "similarity": float(score),
                "source_preview": generated_chunks[generated_index][:300],
                "matched_preview": ground_truth_chunks[best_gt_index][:300],
            }
        )

    for gt_index, score in enumerate(ground_truth_best_scores):
        best_generated_index = int(similarity_matrix[:, gt_index].argmax())
        semantic_alignment_rows.append(
            {
                "direction": "ground_truth_to_generated",
                "source_chunk_index": gt_index,
                "matched_chunk_index": best_generated_index,
                "similarity": float(score),
                "source_preview": ground_truth_chunks[gt_index][:300],
                "matched_preview": generated_chunks[best_generated_index][:300],
            }
        )

    return semantic_alignment_rows


def build_semantic_metric_rows(
    *,
    semantic_precision: float,
    semantic_recall: float,
    semantic_f1: float,
    global_semantic_similarity: float,
) -> list[dict[str, Any]]:
    return [
        {
            "metric": "semantic_precision",
            "value": semantic_precision,
            "method": "mean_best_generated_to_ground_truth_chunk_similarity",
        },
        {
            "metric": "semantic_recall",
            "value": semantic_recall,
            "method": "mean_best_ground_truth_to_generated_chunk_similarity",
        },
        {
            "metric": "semantic_f1",
            "value": semantic_f1,
            "method": "harmonic_mean_semantic_precision_recall",
        },
        {
            "metric": "global_semantic_similarity",
            "value": global_semantic_similarity,
            "method": "cosine_similarity_mean_document_embeddings",
        },
    ]
