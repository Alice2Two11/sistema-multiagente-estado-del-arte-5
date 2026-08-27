"""Wiring de orquestador para la etapa 07 (verificación), NUEVO en esta iteración.

Este archivo NO existía en el repositorio. Se creó porque, a diferencia de
02-06, la etapa 07 no tiene un ``build_real_*_execution(project_dir,
attempt_number)`` de una sola llamada — sus notebooks (00 y 07) generan
configuración en módulos Python propios del proyecto (``config.py``,
``rag_policy.py``, ``llm_utils.py``, ``rag_utils.py`` dentro de
``PROJECT_DIR/src/``), no en ``active_experiment.json``.

Dependencias reales identificadas para 07 (inventario pedido)
---------------------------------------------------------------
Del notebook 07 (rama productiva, no FIXTURE_MODE, celdas 4/7/9/11/13/15):

1. ``config.py`` del proyecto (generado por notebook 00): EXPERIMENT_ID,
   EXPERIMENT_DIR, OUTPUTS_DIR, OUTLINE_DIR, ORCHESTRATOR_DIR,
   VERIFICATION_TRACEABILITY_DIR, CHUNKS_DIR, CHROMA_DIR,
   EMBEDDING_MODEL_NAME, OPENAI_MODEL, CHROMA_COLLECTION_NAME,
   VERIFICATION_POLICY, POST_CORRECTION_RECHECK_POLICY.
   → NO existe en ``src/`` del repo. Aquí se reconstruye equivalente desde
   ``active_experiment.json`` (ver ``load_verification_configuration``),
   siguiendo la MISMA convención que ya usan 02-06
   (``active.get("<etapa>_policy", {})`` mezclado sobre los defaults de
   ``src/config/*_policy_config.py``). Esta es una decisión nueva, no una
   verificación contra el ``active_experiment.json`` real del usuario — se
   señala explícitamente para que se confirme antes de confiar en ella.
2. ``rag_policy.py`` del proyecto → ``get_rag_policy()``. NO existe en
   ``src/``. Se usa ``active.get("rag_policy", {})`` (ya presente en
   ``active_experiment.json`` para 02-06) en su lugar.
3. ``llm_utils.py`` del proyecto → ``get_llm()``. NO existe en ``src/`` pero
   es un envoltorio trivial sobre ``ChatOpenAI`` + la credencial resuelta;
   aquí se reemplaza directamente por ``ChatOpenAI(model=..., temperature=0)``
   tras ``resolve_openai_api_key(project_dir=...)``, exactamente el mismo
   patrón que ya usan ``build_openai_outline_runtime`` /
   ``build_real_thematic_dependencies`` en ``src/adapters/``. No es una
   etapa nueva de riesgo: es el patrón ya validado en el repo.
4. ``rag_utils.py`` del proyecto → ``load_chroma_collection()``, y la clase
   ``Agent07ChromaRetriever`` (retriever incremental para claims), ambas
   definidas EN LA CELDA 11 DEL NOTEBOOK, no en ningún módulo.
   → **PORTADA en esta iteración** a
   ``src/adapters/verification_incremental_retriever.py`` — la clase
   ``Agent07ChromaRetriever`` es una copia literal (mismos campos, mismo
   orden, mismo comportamiento ante ausencia de evidencia); solo el wiring
   alrededor (`load_chroma_collection`, `get_rag_policy()`, `_sha256` local)
   se reemplazó por equivalentes ya existentes en ``src/`` — ver el docstring
   de ese módulo para el detalle exacto de cada sustitución.
   ``build_experimental_verification_execution`` ahora construye el
   retriever real y lo pasa a ``build_agent07_runtime_dependencies``; ya NO
   corre con ``incremental_retriever=None``.

Ya existentes y reutilizados sin modificar (en ``src/``):
- ``src/adapters/verification_notebook.py``: ``prepare_agent07_execution``,
  ``execute_prepared_agent07``, ``commit_executed_agent07``,
  ``resume_agent07_execution``, ``AGENT07_STAGE_NAME``.
- ``src/adapters/verification_runtime.py``: ``Agent07RuntimeInput``,
  ``build_agent07_runtime_dependencies``.
- ``src/adapters/agent06_verification_handoff.py``:
  ``build_agent07_input_from_committed_agent06``, ``Agent07RetrieverBinding``
  (se usa esta función directamente — NO ``resolve_committed_agent06_output``,
  que en ``verification_notebook.py`` usa una convención de rutas distinta,
  ``experiment_root/outputs/04_outline/...`` en vez de
  ``experiment_root/05_outputs/04_outline/...``, y que el propio notebook 07
  tampoco usa en su rama productiva).
- ``src/config/verification_policy_config.py``: ``get_verification_input_policy``.
- ``src/tools/verification/resolution.py``: ``RESOLUTION_FP_VERSION``.

Ver ``AGENT07_CONFIG_EQUIVALENCE.md`` (mismo directorio) para el mapa de
equivalencia campo por campo entre la celda 7 del notebook y
``load_verification_configuration`` de aquí abajo — incluye qué está
verificado por identidad de código y qué sigue siendo una inferencia por
convención no confirmada contra un ``active_experiment.json`` real.

Validación de compatibilidad — corregida en esta iteración
-------------------------------------------------------------
``agent06_verification_handoff.validate_agent07_experiment_compatibility``
exige literales de ruta fijos de una sesión de Colab concreta
(``code_root == "/content/tesis_codigo"``, etc. — ver
``AGENT07_CONFIG_EQUIVALENCE.md`` sección 6). Eso NO es una invariante
semántica real, es un accidente de una sesión particular; por eso NO se
llama a esa función aquí (y no se modifica: sigue existiendo tal cual para
quien la use directamente). En su lugar, ``validate_agent07_orchestrator_compatibility``
(más abajo) reimplementa la MISMA comprobación semántica real que esa
función sí tiene (el diccionario ``aliases`` — que el ``agent07_config`` no
haya divergido del config activo global) parametrizada por ``project_dir``/
``experiment_dir`` en vez de literales fijos, y agrega una comprobación de
``experiment_id`` que la función original no tenía explícita. Esto es lo que
efectivamente se llama en ``build_experimental_verification_execution`` en
vez de pasar ``active_experiment_config=None`` a ciegas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class MissingRequiredActiveExperimentKeyError(KeyError):
    """``active_experiment.json`` no tiene una clave que ``config.py`` (real,
    notebook 00) exige como obligatoria — ver ``_load_active_experiment`` en
    la celda 9 (``%%writefile .../src/config.py``): construye
    ``required_keys`` y levanta ``ValueError("...Faltan: [...]")`` si falta
    alguna. Aquí se reproduce esa misma exigencia en vez de rellenar con un
    default silencioso."""


def _require_active_experiment_key(active: Mapping[str, Any], key: str) -> Any:
    if key not in active:
        raise MissingRequiredActiveExperimentKeyError(
            f"active_experiment.json no tiene la clave obligatoria {key!r}. "
            "config.py (generado por el notebook 00 real) exige esta clave "
            "y levantaría ValueError si faltara — no se rellena con un "
            "default silencioso. Ejecuta primero el notebook 00."
        )
    return active[key]


# Copia literal de src/rag_policy.py (notebook 00, celda 11) —
# REVIEW_SECTION_LABELS_ES, REVIEW_SECTION_PATTERNS y
# RAG_ALLOWED_CONTENT_POLICY están hardcodeados en esa celda, NO vienen de
# active_experiment.json; se reproducen aquí verbatim para que
# _derive_rag_policy_like_notebook00 devuelva exactamente lo mismo que
# get_rag_policy() real.
_NOTEBOOK00_REVIEW_SECTION_LABELS_ES = {
    "related_work": "trabajos relacionados",
    "literature_review": "revisión de literatura",
    "state_of_the_art": "estado del arte",
    "background": "antecedentes",
    "theoretical_background": "marco teórico / antecedentes teóricos",
    "previous_work": "trabajo previo",
    "prior_work": "trabajo anterior",
}
_NOTEBOOK00_REVIEW_SECTION_PATTERNS = [
    r"\brelated\s+work\b",
    r"\bliterature\s+review\b",
    r"\bstate\s+of\s+the\s+art\b",
    r"\bbackground\b",
    r"\btheoretical\s+background\b",
    r"\bprevious\s+work\b",
    r"\bprior\s+work\b",
    r"\btrabajos?\s+relacionados?\b",
    r"\brevisión\s+de\s+literatura\b",
    r"\brevision\s+de\s+literatura\b",
    r"\brevisión\s+bibliográfica\b",
    r"\brevision\s+bibliografica\b",
    r"\bestado\s+del\s+arte\b",
    r"\bantecedentes\b",
    r"\bmarco\s+teórico\b",
    r"\bmarco\s+teorico\b",
    r"\btrabajos?\s+previos?\b",
]
_NOTEBOOK00_RAG_ALLOWED_CONTENT_POLICY = (
    "Solo se indexan fragmentos de papers de referencia "
    "que no pertenezcan a secciones de revisión, antecedentes, "
    "trabajos relacionados o bibliografía. "
    "El Ground Truth se reserva exclusivamente para evaluación."
)
_NOTEBOOK00_RAG_POLICY_REQUIRED_KEYS = {
    "exclude_review_sections_from_reference_papers",
    "excluded_reference_section_types",
    "ground_truth_usage",
    "use_ground_truth_for_generation",
    "use_ground_truth_for_rag",
    "use_ground_truth_for_verification",
    "use_ground_truth_for_evaluation",
    "retrieval_profiles",
    "indexing",
    "generation",
}


def _derive_rag_policy_like_notebook00(raw_rag_policy: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce ``get_rag_policy()`` real (``src/rag_policy.py``, notebook 00
    celda 11) a partir de ``active_experiment.json["rag_policy"]`` crudo.

    NO es un passthrough: el notebook real valida y RESHAPEA la política
    (renombra ``excluded_reference_section_types`` a ``review_section_types``
    como lista ordenada, agrega ``ground_truth_policy`` anidado y tres
    campos hardcodeados que no vienen de ``active_experiment.json``, aplana
    ``indexing``/``generation`` a escalares con nombres nuevos). Lanza los
    mismos ``ValueError`` que el módulo real ante una política inválida o
    incompleta — mismos mensajes de validación, mismas condiciones.
    """

    if not isinstance(raw_rag_policy, dict) or not raw_rag_policy:
        raise ValueError("RAG_POLICY debe ser un diccionario no vacío.")

    missing = sorted(_NOTEBOOK00_RAG_POLICY_REQUIRED_KEYS - set(raw_rag_policy))
    if missing:
        raise ValueError(f"RAG_POLICY está incompleta. Faltan: {missing}")

    ground_truth_policy = {
        "use_ground_truth_for_generation": bool(
            raw_rag_policy["use_ground_truth_for_generation"]
        ),
        "use_ground_truth_for_rag": bool(raw_rag_policy["use_ground_truth_for_rag"]),
        "use_ground_truth_for_verification": bool(
            raw_rag_policy["use_ground_truth_for_verification"]
        ),
        "use_ground_truth_for_evaluation": bool(
            raw_rag_policy["use_ground_truth_for_evaluation"]
        ),
    }
    if any(
        [
            ground_truth_policy["use_ground_truth_for_generation"],
            ground_truth_policy["use_ground_truth_for_rag"],
            ground_truth_policy["use_ground_truth_for_verification"],
        ]
    ):
        raise ValueError("El Ground Truth solo puede utilizarse para evaluación.")
    if not ground_truth_policy["use_ground_truth_for_evaluation"]:
        raise ValueError("El Ground Truth debe estar habilitado para evaluación.")

    exclude_review_sections = bool(
        raw_rag_policy["exclude_review_sections_from_reference_papers"]
    )
    if not exclude_review_sections:
        raise ValueError(
            "La política metodológica exige excluir secciones de revisión del RAG."
        )

    review_section_types = set(raw_rag_policy["excluded_reference_section_types"])
    if not review_section_types:
        raise ValueError("excluded_reference_section_types no puede estar vacío.")

    retrieval_profiles = dict(raw_rag_policy["retrieval_profiles"])
    indexing_config = dict(raw_rag_policy["indexing"])
    rag_generation_config = dict(raw_rag_policy["generation"])

    if not isinstance(retrieval_profiles, dict) or not retrieval_profiles:
        raise ValueError("retrieval_profiles debe ser un diccionario no vacío.")
    if not isinstance(indexing_config, dict):
        raise ValueError("indexing debe ser un diccionario.")
    if not isinstance(rag_generation_config, dict):
        raise ValueError("generation debe ser un diccionario.")
    if "batch_size" not in indexing_config:
        raise ValueError("indexing requiere batch_size.")
    for key in ("temperature", "answer_max_words"):
        if key not in rag_generation_config:
            raise ValueError(f"generation requiere {key!r}.")

    index_batch_size = int(indexing_config["batch_size"])
    rag_temperature = float(rag_generation_config["temperature"])
    rag_answer_max_words = int(rag_generation_config["answer_max_words"])

    if index_batch_size <= 0:
        raise ValueError("INDEX_BATCH_SIZE debe ser mayor que cero.")
    if not 0.0 <= rag_temperature <= 2.0:
        raise ValueError("RAG_TEMPERATURE debe estar entre 0 y 2.")
    if rag_answer_max_words <= 0:
        raise ValueError("RAG_ANSWER_MAX_WORDS debe ser mayor que cero.")

    return {
        "ground_truth_policy": ground_truth_policy,
        "exclude_review_sections_from_reference_papers": exclude_review_sections,
        "review_section_types": sorted(review_section_types),
        "review_section_labels_es": dict(_NOTEBOOK00_REVIEW_SECTION_LABELS_ES),
        "review_section_patterns": list(_NOTEBOOK00_REVIEW_SECTION_PATTERNS),
        "rag_allowed_content_policy": _NOTEBOOK00_RAG_ALLOWED_CONTENT_POLICY,
        "retrieval_profiles": retrieval_profiles,
        "indexing": indexing_config,
        "generation": rag_generation_config,
        "index_batch_size": index_batch_size,
        "rag_temperature": rag_temperature,
        "rag_answer_max_words": rag_answer_max_words,
    }


def load_verification_configuration(
    project_dir: str | Path, attempt_number: int = 1
) -> dict[str, Any]:
    from src.config.verification_policy_config import get_verification_input_policy
    from src.tools.verification.resolution import RESOLUTION_FP_VERSION
    from src.adapters.verification_runtime import AGENT07_RUNTIME_METRICS_VERSION
    from src.config.verification_policy_config import (
        PROVISIONAL_BUNDLE_FINGERPRINT_VERSION,
    )

    root = Path(project_dir).resolve()
    active = json.loads((root / "active_experiment.json").read_text(encoding="utf-8"))
    experiment_id = active["active_experiment_id"]
    experiment_dir = root / experiment_id
    outputs = experiment_dir / "05_outputs"
    outline_dir = outputs / "04_outline"
    verification_dir = outputs / "06_verification_traceability"
    state_path = outputs / "00_orchestrator_planner" / "pipeline_state.json"
    repo_root = Path(__file__).resolve().parents[2]

    # Rutas de Chroma/chunks: DERIVADAS de experiment_dir, igual que en
    # config.py real (celda 9 de notebook 00) — no son claves persistidas en
    # active_experiment.json (celda 8 no las escribe), así que no tiene
    # sentido leerlas con active.get(...) como si pudieran venir
    # sobrescritas: en el flujo real nunca lo están. Antes este adaptador
    # tenía un prefijo "05_outputs/01_rag" espurio en chroma_dir/
    # chroma_manifest_path que NO existe en config.py real
    # (CHROMA_DIR = EXPERIMENT_DIR / "04_chroma_index", sin pasar por
    # OUTPUTS_DIR) — corregido aquí. chunks_dir/chunks_manifest_path ya
    # coincidían.
    chroma_dir = experiment_dir / "04_chroma_index"
    chunks_dir = experiment_dir / "03_chunks"
    chroma_manifest_path = chroma_dir / "chroma_index_manifest.json"
    chunks_manifest_path = chunks_dir / "chunks_clean_for_rag.jsonl"

    # Las 4 claves ya cerradas en la ronda anterior + embedding_model y
    # rag_policy (esta ronda): TODAS obligatorias. config.py real (celda 9)
    # exige las 7 en su set `required_keys` y levanta ValueError si faltan
    # — no se rellenan con un default silencioso. Ver
    # AGENT07_CONFIG_EQUIVALENCE.md para la evidencia exacta (cita de la
    # celda) de cada una.
    openai_model = str(_require_active_experiment_key(active, "openai_model")).strip()
    collection_name = str(
        _require_active_experiment_key(active, "chroma_collection_name")
    ).strip()
    fixed_verification_policy = _require_active_experiment_key(active, "verification_policy")
    fixed_post_correction_recheck_policy = _require_active_experiment_key(
        active, "post_correction_recheck_policy"
    )
    if not isinstance(fixed_verification_policy, dict) or not fixed_verification_policy:
        raise ValueError(
            "active_experiment.json['verification_policy'] debe ser un diccionario "
            "no vacío (mismo chequeo que config.py real)."
        )
    if (
        not isinstance(fixed_post_correction_recheck_policy, dict)
        or not fixed_post_correction_recheck_policy
    ):
        raise ValueError(
            "active_experiment.json['post_correction_recheck_policy'] debe ser un "
            "diccionario no vacío (mismo chequeo que config.py real)."
        )
    if not openai_model:
        raise ValueError("openai_model no puede estar vacío (mismo chequeo que config.py real).")
    if not collection_name:
        raise ValueError(
            "chroma_collection_name no puede estar vacío (mismo chequeo que config.py real)."
        )

    # embedding_model: clave real confirmada en config.py (celda 9):
    # EMBEDDING_MODEL_NAME = str(ACTIVE_EXPERIMENT["embedding_model"]).strip().
    # "embedding_model_name" NUNCA es una clave de active_experiment.json —
    # se eliminó la prioridad artificial que la probaba primero.
    embedding_model = str(_require_active_experiment_key(active, "embedding_model")).strip()
    if not embedding_model:
        raise ValueError("embedding_model no puede estar vacío (mismo chequeo que config.py real).")

    # rag_policy: clave obligatoria (celda 9: "RAG_POLICY" está en el set de
    # dicts no vacíos exigidos). El adaptador NO pasa el dict crudo de
    # active_experiment.json["rag_policy"] tal cual: el notebook real lo
    # transforma mediante get_rag_policy() (rag_policy.py, celda 11) antes de
    # que 07 lo consuma — ver _derive_rag_policy_like_notebook00 más abajo,
    # que reproduce esa transformación literal (mismas claves derivadas,
    # mismas validaciones, mismos valores hardcodeados de
    # review_section_labels_es/review_section_patterns/
    # rag_allowed_content_policy).
    raw_rag_policy = _require_active_experiment_key(active, "rag_policy")
    if not isinstance(raw_rag_policy, dict) or not raw_rag_policy:
        raise ValueError(
            "active_experiment.json['rag_policy'] debe ser un diccionario no vacío "
            "(mismo chequeo que config.py real)."
        )
    rag_policy = _derive_rag_policy_like_notebook00(raw_rag_policy)

    complete_policy = get_verification_input_policy()
    # verification_overrides/reverification_overrides son las claves ya
    # validadas arriba (obligatorias, no un .get con default) — dict() aquí
    # solo copia, no relaja la exigencia.
    verification_overrides = dict(fixed_verification_policy)
    reverification_overrides = dict(fixed_post_correction_recheck_policy)

    verification_policy = dict(complete_policy)
    verification_policy.update(verification_overrides)
    correction_policy = dict(complete_policy)
    correction_policy.update(verification_overrides)
    reverification_policy = dict(complete_policy)
    reverification_policy.update(reverification_overrides)

    agent07_config = {
        "runtime_mode": "productive",
        "external_paper_search": False,
        "verification_policy": verification_policy,
        "correction_policy": correction_policy,
        "reverification_policy": reverification_policy,
        "verification_prompt_version": complete_policy["verification_user_prompt_version"],
        "correction_prompt_version": complete_policy["correction_user_prompt_version"],
        "reverification_prompt_version": complete_policy["reverification_user_prompt_version"],
        "verification_budgets": {
            "max_llm_attempts": int(complete_policy["max_llm_attempts_per_claim"]),
            "max_format_repair_attempts": int(complete_policy["max_format_repair_attempts"]),
            "max_additional_retrieval_requests": int(
                complete_policy["max_additional_retrieval_requests"]
            ),
        },
        "correction_budgets": {
            "max_llm_attempts": int(complete_policy["max_correction_llm_attempts"]),
            "max_format_repair_attempts": int(
                complete_policy["max_correction_format_repair_attempts"]
            ),
            "max_proposals_per_claim": int(
                complete_policy["max_correction_proposals_per_claim"]
            ),
        },
        "reverification_budgets": {
            "max_llm_attempts": int(complete_policy["max_reverification_llm_attempts"]),
            "max_format_repair_attempts": int(
                complete_policy["max_reverification_format_repair_attempts"]
            ),
        },
        "verification_model": openai_model,
        "correction_model": openai_model,
        "reverification_model": openai_model,
        "embedding_model": embedding_model,
        "collection_name": collection_name,
        "chroma_collection_name": collection_name,
    }
    policy_versions = {
        "verification": agent07_config["verification_prompt_version"],
        "correction": agent07_config["correction_prompt_version"],
        "reverification": agent07_config["reverification_prompt_version"],
    }
    schema_versions = {
        "provisional_bundle": PROVISIONAL_BUNDLE_FINGERPRINT_VERSION,
        "multi_proposal_resolution": RESOLUTION_FP_VERSION,
        "runtime_metrics": AGENT07_RUNTIME_METRICS_VERSION,
    }
    experiment_paths = {
        "code_root": str(repo_root),
        "project_root": str(root),
        "experiment_root": str(experiment_dir),
        "root": str(experiment_dir),
        "pipeline_state_path": str(state_path),
        "outline_paper_mapping_path": str(outline_dir / "outline_paper_mapping.csv"),
        "agent07_output_dir": str(verification_dir),
        "agent07_staging_dir": str(outputs / ".agent07_staging"),
        "chroma_dir": str(chroma_dir),
        "chunks_dir": str(chunks_dir),
    }
    active_experiment_config = {
        "active_experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "embedding_model": embedding_model,
        "openai_model": openai_model,
        "chroma_collection_name": collection_name,
        "verification_policy": verification_policy,
        "verification_prompt_version": agent07_config["verification_prompt_version"],
        "verification_budgets": agent07_config["verification_budgets"],
        "rag_policy": rag_policy,
    }
    return {
        "project_dir": root,
        "experiment_id": experiment_id,
        "experiment_dir": experiment_dir,
        "attempt_number": int(attempt_number),
        "agent07_config": agent07_config,
        "policy_versions": policy_versions,
        "schema_versions": schema_versions,
        "experiment_paths": experiment_paths,
        "active_experiment_config": active_experiment_config,
        "outline_paper_mapping_path": experiment_paths["outline_paper_mapping_path"],
        "state_path": state_path,
        "chroma_dir": chroma_dir,
        "chroma_manifest_path": chroma_manifest_path,
        "chunks_manifest_path": chunks_manifest_path,
        "rag_policy": rag_policy,
    }


def validate_agent07_orchestrator_compatibility(
    *,
    active_experiment_config: Mapping[str, Any],
    agent07_config: Mapping[str, Any],
    experiment_paths: Mapping[str, str],
    project_dir: str | Path,
    experiment_dir: str | Path,
) -> None:
    """Validación de compatibilidad semántica, parametrizada (no rutas fijas).

    Sustituye, para el orquestador, a
    ``agent06_verification_handoff.validate_agent07_experiment_compatibility``
    (que exige literales de Colab — ver docstring del módulo). Conserva la
    MISMA invariante real de esa función (diccionario ``aliases``: el
    ``agent07_config`` no puede haber divergido del config activo global) y
    agrega una comprobación de identidad de experimento que esa función no
    tenía. Deliberadamente NO valida ``code_root``: qué copia del código se
    usa no determina si los artefactos pertenecen al experimento correcto.

    Lanza ``ValueError`` con los mismos prefijos de código que la función
    original cuando aplica (``AGENT07_GLOBAL_CONFIG_INVALID``,
    ``AGENT07_GLOBAL_CONFIG_MISMATCH:<campo>``), y nuevos códigos propios
    para las comprobaciones de ruta/experimento
    (``AGENT07_EXPERIMENT_PATH_MISMATCH:<campo>``,
    ``AGENT07_EXPERIMENT_ID_MISMATCH``).
    """

    if not isinstance(active_experiment_config, Mapping) or not isinstance(
        agent07_config, Mapping
    ):
        raise ValueError("AGENT07_GLOBAL_CONFIG_INVALID")

    expected_project_root = str(Path(project_dir).resolve())
    expected_experiment_root = str(Path(experiment_dir).resolve())
    actual_project_root = experiment_paths.get("project_root")
    actual_experiment_root = experiment_paths.get("experiment_root")

    if actual_project_root != expected_project_root:
        raise ValueError("AGENT07_EXPERIMENT_PATH_MISMATCH:project_root")
    if actual_experiment_root != expected_experiment_root:
        raise ValueError("AGENT07_EXPERIMENT_PATH_MISMATCH:experiment_root")
    if not actual_experiment_root.startswith(actual_project_root):
        raise ValueError(
            "AGENT07_EXPERIMENT_PATH_MISMATCH:experiment_root_not_under_project_root"
        )

    experiment_id = active_experiment_config.get("active_experiment_id")
    if experiment_id is not None and Path(experiment_dir).name != experiment_id:
        raise ValueError("AGENT07_EXPERIMENT_ID_MISMATCH")

    aliases = {
        "verification_policy": ("verification_policy",),
        "verification_prompt_version": ("verification_prompt_version",),
        "verification_budgets": ("verification_budgets",),
        "verification_model": ("verification_model", "openai_model"),
        "correction_model": ("correction_model", "openai_model"),
    }
    for out_key, candidates in aliases.items():
        expected = next(
            (
                active_experiment_config.get(k)
                for k in candidates
                if active_experiment_config.get(k) is not None
            ),
            None,
        )
        actual = agent07_config.get(out_key)
        if expected is not None and actual != expected:
            raise ValueError(f"AGENT07_GLOBAL_CONFIG_MISMATCH:{out_key}")


def build_experimental_verification_execution(
    project_dir: str | Path,
    attempt_number: int = 1,
    *,
    llm_factory: Any = None,
    chroma_client_factory: Any = None,
    embedding_function_factory: Any = None,
):
    """Devuelve ``(dependencies, runtime_input)`` para la etapa 07.

    Nombre deliberadamente "experimental": la equivalencia con el notebook
    original todavía no está demostrada campo por campo contra un
    ``active_experiment.json`` real (ver ``AGENT07_CONFIG_EQUIVALENCE.md``),
    aunque el retriever incremental ya es una copia literal y las pruebas de
    caracterización (``tests/orchestration/test_verification_characterization.py``)
    atraviesan las funciones reales de 07 con dependencias deterministas.

    ``llm_factory``/``chroma_client_factory``/``embedding_function_factory``
    son puntos de inyección opcionales (mismo patrón que
    ``build_extraction_runtime`` en ``src/adapters/extraction_runtime.py``)
    para poder probar este constructor REAL sin red — ver
    ``tests/orchestration/test_verification_build_execution_real.py``. Si no
    se pasan, se usa el camino productivo real (``ChatOpenAI`` +
    ``resolve_openai_api_key`` + ``chromadb.PersistentClient`` +
    ``SentenceTransformerEmbeddingFunction``).

    ``attempt_number`` se ignora deliberadamente: ``prepare_agent07_execution``
    deriva el intento internamente desde ``pipeline_state.json``
    (``attempts_used + 1``), no acepta uno externo — a diferencia de 02-06.
    Se mantiene el parámetro solo por uniformidad de firma con las demás
    etapas.
    """

    from src.adapters.agent06_verification_handoff import (
        Agent07RetrieverBinding,
        build_agent07_input_from_committed_agent06,
    )
    from src.adapters.verification_incremental_retriever import (
        build_agent07_chroma_retriever,
    )
    from src.adapters.verification_notebook import AGENT07_STAGE_NAME
    from src.adapters.verification_runtime import (
        Agent07RuntimeInput,
        build_agent07_runtime_dependencies,
    )
    from src.orchestration.pipeline_orchestrator import (
        DRAFT_STAGE_NAME,
        ensure_pipeline_state,
    )

    cfg = load_verification_configuration(project_dir, attempt_number)
    store = ensure_pipeline_state(cfg["project_dir"])

    committed_agent06_output = build_agent07_input_from_committed_agent06(
        store=store,
        stage_name=DRAFT_STAGE_NAME,
        agent07_config=cfg["agent07_config"],
        policy_versions=cfg["policy_versions"],
        schema_versions=cfg["schema_versions"],
        experiment_paths=cfg["experiment_paths"],
        outline_paper_mapping_path=cfg["outline_paper_mapping_path"],
    )
    runtime_input = Agent07RuntimeInput(
        committed_agent06_output=committed_agent06_output,
        agent07_config=cfg["agent07_config"],
        policy_versions=cfg["policy_versions"],
        schema_versions=cfg["schema_versions"],
        experiment_paths=cfg["experiment_paths"],
    )

    validate_agent07_orchestrator_compatibility(
        active_experiment_config=cfg["active_experiment_config"],
        agent07_config=cfg["agent07_config"],
        experiment_paths=cfg["experiment_paths"],
        project_dir=cfg["project_dir"],
        experiment_dir=cfg["experiment_dir"],
    )

    if llm_factory is None:
        from src.io.credentials import resolve_openai_api_key

        resolve_openai_api_key(project_dir=cfg["project_dir"], required=True)
        from langchain_openai import ChatOpenAI

        llm_factory = ChatOpenAI

    verification_llm = llm_factory(model=cfg["agent07_config"]["verification_model"], temperature=0.0)
    correction_llm = llm_factory(model=cfg["agent07_config"]["correction_model"], temperature=0.0)
    reverification_llm = llm_factory(
        model=cfg["agent07_config"]["reverification_model"], temperature=0.0
    )

    incremental_retriever, retriever_binding_kwargs = build_agent07_chroma_retriever(
        chroma_dir=cfg["chroma_dir"],
        chroma_collection_name=cfg["agent07_config"]["chroma_collection_name"],
        embedding_model_name=cfg["agent07_config"]["embedding_model"],
        chroma_manifest_path=cfg["chroma_manifest_path"],
        chunks_manifest_path=cfg["chunks_manifest_path"],
        committed_experiment_id=committed_agent06_output["experiment_id"],
        rag_policy=cfg["rag_policy"],
        chroma_client_factory=chroma_client_factory,
        embedding_function_factory=embedding_function_factory,
    )
    retriever_binding = Agent07RetrieverBinding(**retriever_binding_kwargs)

    dependencies = build_agent07_runtime_dependencies(
        config=cfg["agent07_config"],
        experiment_paths=cfg["experiment_paths"],
        verification_llm=verification_llm,
        correction_llm=correction_llm,
        reverification_llm=reverification_llm,
        incremental_retriever=incremental_retriever,
        # None deliberado: la compatibilidad semántica YA se validó arriba
        # con validate_agent07_orchestrator_compatibility (parametrizada).
        # Si aquí se pasara cfg["active_experiment_config"], esta llamada
        # dispararía internamente la validate_agent07_experiment_compatibility
        # ORIGINAL (rutas hardcodeadas de Colab) y fallaría para cualquier
        # project_dir que no sea exactamente esos literales — que es
        # justamente lo que se reemplazó, no algo que se esté evitando sin
        # sustituto. validate_productive_retriever_binding (la otra
        # comprobación real que cuelga de este parámetro) se sigue
        # ejecutando igual, porque no depende de active_experiment_config:
        # usa `active_experiment_config or config`, y aquí cae a `config`
        # (cfg["agent07_config"]), que ya trae collection_name/embedding_model
        # consistentes con el retriever recién construido.
        active_experiment_config=None,
        retriever_binding=retriever_binding,
        chroma_manifest_path=str(cfg["chroma_manifest_path"]),
        chunks_manifest_path=str(cfg["chunks_manifest_path"]),
        committed_experiment_id=committed_agent06_output["experiment_id"],
    )
    assert AGENT07_STAGE_NAME == "07_agente_verificador"
    return dependencies, runtime_input
