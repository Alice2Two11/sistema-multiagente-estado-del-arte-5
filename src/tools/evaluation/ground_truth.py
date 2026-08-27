"""Bloque 2 de la migración de la etapa 08: consumo del Ground Truth
ya preparado por Stage 01 (congelado, autoridad canónica).

GROUND TRUTH CLEANUP (Stage 08): Stage 01 es la única autoridad sobre
dónde y cómo se materializan los artefactos de Ground Truth -- Stage
08 consume exactamente lo que 01 produce en ``GROUND_TRUTH_DIR``
(``<experiment_dir>/00_ground_truth/``, confirmado en
``evaluation_stagespec_wiring.py``), sin crear rutas nuevas, sin mover
archivos y sin volver a decidir nada sobre el contenido del Ground
Truth.

El único artefacto que Stage 08 necesita es
``ground_truth_literature_review.txt`` -- el texto de la revisión de
literatura ya aislado por 01. Si existe y contiene texto, se usa tal
cual. Si falta o está vacío, Stage 08 falla cerrado
(``GROUND_TRUTH_PREPARATION_MISSING``) -- nunca:

- abre ``ground_truth.pdf`` ni ningún otro PDF del Ground Truth
  (elimina la dependencia de ``fitz``/PyMuPDF en Stage 08);
- cae a ``ground_truth_full_text.txt`` como sustituto;
- vuelve a detectar dónde empieza/termina la revisión de literatura
  (Literature Review/Related Work) -- esa es una decisión científica
  que solo le corresponde a 01.

Las funciones ``load_ground_truth_full_text``, ``extract_gt_literature_
review``, ``extract_pdf_text``, ``normalize_pdf_text``,
``find_headings``, ``build_heading_pattern`` y los alias
``GT_START_ALIASES``/``GT_END_ALIASES`` que existían aquí (copias
literales del notebook 08 original, celda 13) se retiraron: auditoría
confirmó que su único consumidor era el fallback que este cambio
elimina -- ninguna tenía otro llamador en ``src/``."""

from __future__ import annotations

from pathlib import Path


# Nombre de archivo literal que Stage 01 produce dentro de
# GROUND_TRUTH_DIR -- el único artefacto de Ground Truth que Stage 08
# consume.
PREEXTRACTED_GT_FILENAME = "ground_truth_literature_review.txt"


def resolve_ground_truth_comparable_text(
    *,
    ground_truth_dir: str | Path,
    minimum_words: int,
    require_explicit_end_heading: bool,
) -> tuple[str, dict, Path]:
    """Consume el Ground Truth ya preparado por Stage 01 -- fail-closed.

    ``require_explicit_end_heading`` se conserva en la firma por
    compatibilidad con los llamadores existentes (``evaluation_
    pipeline.py``, ``evaluation_orchestrator_runtime.py`` siguen
    pasándolo desde ``evaluation_policy``), pero ya no tiene efecto:
    la detección de encabezado de cierre era responsabilidad de la
    extracción que este cambio elimina -- 01 ya resolvió esa decisión
    antes de escribir ``ground_truth_literature_review.txt``.
    """
    ground_truth_dir = Path(ground_truth_dir)
    preextracted_path = ground_truth_dir / PREEXTRACTED_GT_FILENAME

    if not preextracted_path.is_file():
        raise ValueError(
            "GROUND_TRUTH_PREPARATION_MISSING: no existe "
            f"{preextracted_path} -- Stage 01 debe producir "
            f"{PREEXTRACTED_GT_FILENAME} antes de ejecutar Stage 08. "
            "Stage 08 no reabre el PDF del Ground Truth ni vuelve a "
            "detectar la revisión de literatura."
        )

    text = preextracted_path.read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        raise ValueError(
            f"GROUND_TRUTH_PREPARATION_MISSING: {preextracted_path} "
            "existe pero está vacío."
        )

    if len(text.split()) < minimum_words:
        raise ValueError(
            "La revisión de literatura del Ground Truth "
            "es demasiado corta para una evaluación válida."
        )

    metadata = {
        "source_mode": "preextracted_literature_review",
        "start_heading": "preextracted_literature_review",
        "start_alias": "literature review",
        "start_position": None,
        "end_heading": "preextracted_end",
        "end_alias": None,
        "end_position": None,
    }

    return text, metadata, preextracted_path
