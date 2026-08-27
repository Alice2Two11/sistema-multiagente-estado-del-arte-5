"""Wiring de producción para el ``StageSpec`` de 08 — resuelve
``active_experiment.json``, el upstream de 07 (NUNCA 07C: se pasa
``agent07c_directory=None`` explícitamente en la llamada), el draft y los
chunks reales, y arma los kwargs de ``run_evaluation_pipeline``.

Nombre deliberadamente "experimental" (mismo criterio que 07): la
equivalencia de configuración de 08 contra un ``active_experiment.json``
real no se verificó campo por campo como sí se hizo para 07 — aquí se
asume que ``active_experiment.json["evaluation_policy"]`` ya es el dict
resuelto con las 24 claves reales (sin default, `KeyError` si falta
alguna), siguiendo el mismo patrón que 07, pero sin la ronda de
verificación dedicada que 07 sí tuvo.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.adapters.evaluation_orchestrator_runtime import build_experimental_evaluation_execution
from src.tools.evaluation.text_normalization import normalize_content_text


def _load_active_experiment(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "active_experiment.json").read_text(encoding="utf-8"))


def _load_chunks_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_execution_for_stagespec(project_dir: str | Path, attempt_number: int = 1) -> dict[str, Any]:
    project_dir = Path(project_dir)
    active = _load_active_experiment(project_dir)
    experiment_id = active["active_experiment_id"]
    experiment_dir = project_dir / experiment_id
    outputs = experiment_dir / "05_outputs"

    evaluation_policy = active["evaluation_policy"]
    if not isinstance(evaluation_policy, dict) or not evaluation_policy:
        raise ValueError(
            "active_experiment.json['evaluation_policy'] debe ser un diccionario no vacío."
        )

    from src.orchestration.pipeline_orchestrator import ensure_pipeline_state

    store = ensure_pipeline_state(project_dir)

    # pipeline_outcome (SUCCESS | PARTIAL_HALT): resuelto fail-closed
    # ANTES de tocar cualquier artefacto de 07 -- si 07 tuvo un fallo
    # técnico real, o si es un HALT_STAGE científico sin autorización
    # explícita, esto lanza y 08 nunca se ejecuta. "allow_partial_halt_
    # evaluation" es un booleano EXPLÍCITO en evaluation_policy -- nunca
    # se activa por defecto ni se infiere del estado de 07.
    from src.adapters.evaluation_pipeline_outcome import resolve_pipeline_outcome_for_evaluation

    pipeline_outcome_metadata = resolve_pipeline_outcome_for_evaluation(
        store=store,
        allow_partial_halt=bool(evaluation_policy.get("allow_partial_halt_evaluation", False)),
    )

    dir_verify = outputs / "06_verification_traceability"
    dir_evaluation = outputs / "07_evaluation"

    # source_draft_fingerprint (definición canónica única 06->07->08):
    # el draft canónico NO vive necesariamente en la ruta fija
    # "05_draft/state_of_art_draft.json" -- una migración técnica (ej.
    # el bootstrap de identidad, patch19) publica una NUEVA copia
    # comprometida en un directorio propio, sin tocar la original. La
    # ÚNICA fuente de verdad de "qué draft comprometió 06 realmente"
    # es resolve_committed_agent06_artifacts() -- la MISMA función que
    # ya usa 07 (vía build_agent07_input_from_committed_agent06) para
    # resolver su propio source_draft_fingerprint. Usar una ruta fija
    # aquí, en vez de esta resolución causal, es lo que producía
    # AGENT08_SOURCE_DRAFT_FINGERPRINT_MISMATCH tras cualquier
    # publicación que no fuera la escritura original de 06 -- 08 hasheaba
    # un archivo distinto al que 07 realmente verificó.
    from src.adapters.agent06_verification_handoff import resolve_committed_agent06_artifacts

    _, _, agent06_paths, _ = resolve_committed_agent06_artifacts(store=store, stage_name="06_agente_redactor")
    draft_json_path = agent06_paths["state_of_art_draft.json"]
    draft_md_path = agent06_paths["state_of_art_draft.md"]

    from src.adapters.evaluation_upstream import resolve_agent08_upstream_input

    upstream = resolve_agent08_upstream_input(
        agent07_directory=dir_verify,
        draft_json_path=draft_json_path,
        draft_markdown_path=draft_md_path,
        agent07c_directory=None,  # NUNCA 07C: excluido del flujo activo por decisión explícita
    )

    # upstream_fingerprint (punto 4): se lee el fingerprint COMPUESTO real
    # que dejó el COMMIT de 07 en StateStore -- no se crea uno paralelo.
    # Si 07 todavía no se comprometió, queda en None deliberadamente (no
    # es degradación silenciosa: 07 sin comprometer es un estado legítimo
    # antes de la primera ejecución). Pero si 07 SÍ está comprometido y el
    # fingerprint falta o es ilegible, eso es un error real -- no se
    # degrada a None en silencio.
    committed_agent07 = store.load().stages.get("07_agente_verificador")
    if committed_agent07 is None:
        upstream_fingerprint = None
    else:
        upstream_fingerprint = committed_agent07.fingerprints.composite
        if not upstream_fingerprint:
            raise RuntimeError(
                "AGENT08_UPSTREAM_FINGERPRINT_MISSING: 07_agente_verificador está "
                "comprometido en StateStore pero su fingerprint compuesto está "
                "vacío o ilegible -- no se propaga un upstream_fingerprint inválido a 08."
            )

    draft = json.loads(Path(upstream.generated_state_of_art_json_path).read_text(encoding="utf-8"))
    sections = draft["sections"]
    generated_plain_text = normalize_content_text(
        "\n\n".join(section.get("draft_text", "") or "" for section in sections)
    )

    chunks_dir = experiment_dir / "03_chunks"
    chunks = _load_chunks_csv(chunks_dir / "chunks_clean_for_rag.csv")

    ground_truth_dir = experiment_dir / "00_ground_truth"

    openai_model = active.get("openai_model", "gpt-4.1-mini")
    topic_name = active.get("generation_profile", {}).get("topic_name", "")

    from src.io.credentials import resolve_openai_api_key

    resolve_openai_api_key(project_dir=project_dir, required=True)
    from langchain_openai import ChatOpenAI

    def translation_llm_factory():
        return ChatOpenAI(model=openai_model, temperature=evaluation_policy["translation_temperature"])

    def judge_llm_factory():
        return ChatOpenAI(model=openai_model, temperature=evaluation_policy["judge_temperature"])

    return {
        **build_experimental_evaluation_execution(
            generated_plain_text=generated_plain_text,
            sections=sections,
            chunks=chunks,
            traceability_rows=list(upstream.traceability_rows),
            source_stage=upstream.source_stage,
            upstream_runtime_status=upstream.upstream_runtime_status,
            reverification_performed=upstream.reverification_performed,
            reverification_reason=upstream.reverification_reason,
            claims_verified=upstream.claims_verified,
            claims_requiring_manual_review=upstream.claims_requiring_manual_review,
            manual_review_claim_ids=list(upstream.manual_review_claim_ids),
            generated_status=draft.get("status"),
            evaluation_ready_json_path=str(upstream.generated_state_of_art_json_path),
            experiment_id=experiment_id,
            topic_name=topic_name,
            ground_truth_dir=str(ground_truth_dir),
            evaluation_policy=evaluation_policy,
            translation_llm_factory=translation_llm_factory,
            embedding_model_factory=None,  # None -> factory productiva real (SentenceTransformer)
            bertscore_score_fn=None,  # None -> bert_score.score real
            judge_llm_factory=judge_llm_factory,
            upstream_fingerprint=upstream_fingerprint,
        ),
        "output_dir": str(dir_evaluation),
        "numeric_check_output_dir": str(dir_evaluation),
        "backup_root": str(outputs / ".evaluation_backups"),
        "_openai_model": openai_model,
        "_pipeline_outcome_metadata": pipeline_outcome_metadata,
    }
