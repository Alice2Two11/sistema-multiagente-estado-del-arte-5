"""Bloque 4C de la migración de la etapa 08: BERTScore sobre pares alineados.

Copias LITERALES de ``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb``,
celda 17 (bloque "MÉTRICAS AUTOMÁTICAS"), la parte de BERTScore. Recibe
``generated_chunks``, ``ground_truth_chunks`` y ``semantic_matrix`` ya
producidos por el Bloque 4B (``src/tools/evaluation/semantic_similarity.py``)
— no vuelve a calcular embeddings ni similitud coseno.

Mapa función → celda original
------------------------------
| Función | Celda | Dependencias | Entradas | Salidas | Excepciones | Archivos |
|---|---|---|---|---|---|---|
| ``select_bertscore_pair_indices`` | 17 | ``numpy`` | ``generated_chunks``, ``ground_truth_chunks``, ``max_bertscore_pairs`` | ``(precision_pair_indices, recall_pair_indices)`` | ninguna | ninguno |
| ``build_bertscore_pairs`` | 17 | ``semantic_matrix`` ya calculado (Bloque 4B) | los índices anteriores + chunks + matriz | ``(candidates, references, pair_metadata)`` | ninguna | ninguno |
| ``run_bertscore`` | 17 | ``bert_score.score`` (inyectable vía ``bertscore_score_fn``) | ``candidates``, ``references``, ``bertscore_model`` | ``(precision_values, recall_values, f1_values)`` | ``ValueError`` si ``candidates`` está vacío; las que el scorer inyectado o real propague | ninguno (la llamada real a ``bert_score.score`` puede descargar pesos del modelo — comportamiento real de esa librería, no introducido aquí) |
| ``aggregate_bertscore`` | 17 | ninguna | los 3 arrays de valores | ``(bertscore_precision, bertscore_recall, bertscore_f1)`` (medias) | ninguna | ninguno |
| ``enrich_bertscore_pair_metadata`` | 17 (bloque ``metadata.update(...)``, código de módulo) | ninguna | ``pair_metadata``, candidatos/referencias, los 3 arrays | ``list[dict]`` (equivalente en memoria a ``bertscore_alignment.csv``) | ninguna | ninguno |
| ``build_bertscore_metric_rows`` | 17 (bloque ``automatic_metric_rows``, código de módulo) | ninguna | los 3 promedios | ``list[dict]`` de 3 filas | ninguna | ninguno |

Selección de pares — fórmulas preservadas EXACTAS (celda 17)
-----------------------------------------------------------------
```
precision_pair_indices = np.linspace(
    0, len(generated_chunks) - 1,
    num=min(len(generated_chunks), max(1, MAX_BERTSCORE_PAIRS // 2)),
    dtype=int,
)
recall_pair_indices = np.linspace(
    0, len(ground_truth_chunks) - 1,
    num=min(len(ground_truth_chunks), max(1, MAX_BERTSCORE_PAIRS - len(precision_pair_indices))),
    dtype=int,
)
```
El reparto es asimétrico por diseño (`//` es división entera): con
``MAX_BERTSCORE_PAIRS`` impar, ``recall`` recibe un par más que
``precision``. Con ``MAX_BERTSCORE_PAIRS=1``, ambas ramas caen en
``max(1, ...)`` y terminan generando 1 índice cada una — es decir, el
notebook real puede producir **más pares que ``MAX_BERTSCORE_PAIRS``**
cuando ese valor es muy bajo (el ``max(1, ...)`` nunca deja una rama en 0
índices); se preserva tal cual, no es un error de este módulo.

Construcción de pares (confirmado): ``precision_pair_indices`` recorre
``generated_chunks`` y busca su mejor match en el Ground Truth
(``semantic_matrix[generated_index].argmax()``, alineación por FILA —
misma dirección que ``generated_to_ground_truth`` del Bloque 4B).
``recall_pair_indices`` recorre ``ground_truth_chunks`` y busca su mejor
match en lo generado (``semantic_matrix[:, gt_index].argmax()``, por
COLUMNA). En AMBOS bucles el candidato es siempre
``generated_chunks[generated_index]`` y la referencia siempre
``ground_truth_chunks[gt_index]`` — el orden candidato/referencia no se
invierte según la dirección. **No hay deduplicación**: si el mismo par
`(generated_index, gt_index)` surge en ambos bucles (posible si sus
mejores matches mutuos coinciden), aparece dos veces en
``candidates``/``references``/``pair_metadata`` — confirmado que el
notebook no filtra esto, se preserva.

``if not bertscore_candidates: raise ValueError("No se construyeron pares para BERTScore.")``
— preservado tal cual, verificado DESPUÉS de construir ambas listas.

Llamada real a BERTScore (preservada, valores hardcodeados en el notebook,
NO en ``EVALUATION_POLICY``): ``bert_score.score(candidates, references,
model_type=BERTSCORE_MODEL, verbose=True, batch_size=8,
rescale_with_baseline=False)``. ``model_type`` es la única pieza que viene
de la política (``BERTSCORE_MODEL``); ``verbose``, ``batch_size`` y
``rescale_with_baseline`` son literales fijos en el código real — se
preservan como literales fijos aquí también, no se parametrizan.

Importar este módulo no carga el modelo de BERTScore, no descarga pesos,
no lee archivos, no escribe CSV y no llama a OpenAI — ``from bert_score
import score`` ocurre de forma diferida DENTRO de ``run_bertscore`` (solo
cuando no se inyecta ``bertscore_score_fn``), nunca al importar el módulo.
"""

from __future__ import annotations

import os

from typing import Any, Callable

import numpy as np

METHOD_LABEL = "bidirectional_semantically_aligned_chunks"


def select_bertscore_pair_indices(
    *,
    generated_chunks: list[str],
    ground_truth_chunks: list[str],
    max_bertscore_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    precision_pair_indices = np.linspace(
        0,
        len(generated_chunks) - 1,
        num=min(len(generated_chunks), max(1, max_bertscore_pairs // 2)),
        dtype=int,
    )

    recall_pair_indices = np.linspace(
        0,
        len(ground_truth_chunks) - 1,
        num=min(
            len(ground_truth_chunks),
            max(1, max_bertscore_pairs - len(precision_pair_indices)),
        ),
        dtype=int,
    )

    return precision_pair_indices, recall_pair_indices


def build_bertscore_pairs(
    *,
    generated_chunks: list[str],
    ground_truth_chunks: list[str],
    semantic_matrix: Any,
    precision_pair_indices: np.ndarray,
    recall_pair_indices: np.ndarray,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    bertscore_candidates: list[str] = []
    bertscore_references: list[str] = []
    bertscore_pair_metadata: list[dict[str, Any]] = []

    for generated_index in precision_pair_indices:
        generated_index = int(generated_index)
        gt_index = int(semantic_matrix[generated_index].argmax())
        bertscore_candidates.append(generated_chunks[generated_index])
        bertscore_references.append(ground_truth_chunks[gt_index])
        bertscore_pair_metadata.append(
            {
                "direction": "generated_to_ground_truth",
                "generated_chunk_index": generated_index,
                "ground_truth_chunk_index": gt_index,
                "semantic_similarity": float(semantic_matrix[generated_index, gt_index]),
            }
        )

    for gt_index in recall_pair_indices:
        gt_index = int(gt_index)
        generated_index = int(semantic_matrix[:, gt_index].argmax())
        bertscore_candidates.append(generated_chunks[generated_index])
        bertscore_references.append(ground_truth_chunks[gt_index])
        bertscore_pair_metadata.append(
            {
                "direction": "ground_truth_to_generated",
                "generated_chunk_index": generated_index,
                "ground_truth_chunk_index": gt_index,
                "semantic_similarity": float(semantic_matrix[generated_index, gt_index]),
            }
        )

    return bertscore_candidates, bertscore_references, bertscore_pair_metadata


def run_bertscore(
    *,
    candidates: list[str],
    references: list[str],
    bertscore_model: str,
    bertscore_score_fn: Callable[..., Any] | None = None,
) -> tuple[Any, Any, Any]:
    if not candidates:
        raise ValueError("No se construyeron pares para BERTScore.")

    if bertscore_score_fn is None:
        os.environ["MPLBACKEND"] = "Agg"
        from bert_score import score as bertscore_score_fn

    return bertscore_score_fn(
        candidates,
        references,
        model_type=bertscore_model,
        verbose=True,
        batch_size=8,
        rescale_with_baseline=False,
    )


def aggregate_bertscore(
    precision_values: Any, recall_values: Any, f1_values: Any
) -> tuple[float, float, float]:
    return (
        float(precision_values.mean()),
        float(recall_values.mean()),
        float(f1_values.mean()),
    )


def enrich_bertscore_pair_metadata(
    *,
    pair_metadata: list[dict[str, Any]],
    candidates: list[str],
    references: list[str],
    precision_values: Any,
    recall_values: Any,
    f1_values: Any,
) -> list[dict[str, Any]]:
    for index, metadata in enumerate(pair_metadata):
        metadata.update(
            {
                "bertscore_precision": float(precision_values[index]),
                "bertscore_recall": float(recall_values[index]),
                "bertscore_f1": float(f1_values[index]),
                "generated_preview": candidates[index][:300],
                "ground_truth_preview": references[index][:300],
            }
        )
    return pair_metadata


def build_bertscore_metric_rows(
    *, bertscore_precision: float, bertscore_recall: float, bertscore_f1: float
) -> list[dict[str, Any]]:
    return [
        {"metric": "bertscore_precision", "value": bertscore_precision, "method": METHOD_LABEL},
        {"metric": "bertscore_recall", "value": bertscore_recall, "method": METHOD_LABEL},
        {"metric": "bertscore_f1", "value": bertscore_f1, "method": METHOD_LABEL},
    ]
