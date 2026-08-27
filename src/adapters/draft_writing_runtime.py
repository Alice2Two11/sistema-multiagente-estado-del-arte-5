from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.agents.draft_writing_agent import (
    CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT,
    LEGACY_VERSIONS as LEGACY_RUNTIME_VERSIONS,
    DraftWritingAgent,
)
from src.config.draft_writing_policy_config import (
    LEGACY_RETRIEVAL_STRATEGY,
    get_draft_writing_policy,
)
from src.contracts.agent_input import (
    AgentContext,
    AgentInput,
    ArtifactReference,
    ExecutionMode,
    PreviousAttemptSummary,
)
from src.state.fingerprints import fingerprint_mapping, sha256_file
from src.utils.json_parsing import parse_json_safely


# LEGACY_RUNTIME_VERSIONS: fuente canonica unica en
# src/agents/draft_writing_agent.py (LEGACY_VERSIONS), importada
# arriba con alias -- ver ese modulo para los valores y su
# documentacion completa. Antes de esta unificacion, este archivo
# definia su propia copia identica (duplicacion documentada como
# deuda preexistente); ahora agent y runtime consumen exactamente
# el mismo objeto dict.
#
# STAGE06-FINAL-CLEANUP: REQUIRED_DRAFT_ARTIFACTS fue eliminada -- sus
# 3 únicos consumidores vivían en el subgrafo PREPARE/COMMIT/RESUME
# (prepare_draft_execution/execute_prepared_draft/commit_executed_draft/
# resume_draft_execution), también eliminado por completo: el flujo real
# de Stage06 usa exclusivamente pipeline_orchestrator.py::
# _draft_runtime_transaction -> src/runtime/draft_writing_protocol.py
# (execute_draft_transaction/resolve_draft_resume), confirmado sin
# ningún consumidor de este archivo para esa transacción.


@dataclass
class DraftWritingRuntime:
    invoke_fn: object
    collection: object

    def invoke(self, prompt):
        return self.invoke_fn(prompt)

    def parse(self, raw):
        # Extracción robusta del texto real de la respuesta -- reutiliza
        # parse_json_safely (src/tools/evaluation/llm_judge.py, ya
        # probado y usado en producción por 03B/08) en vez de duplicar
        # lógica de extracción de JSON: maneja dict/list ya parseados,
        # fences ```json ... ``` y ``` ... ```, y JSON puro, escaneando
        # con json.JSONDecoder().raw_decode desde cualquier {/[ -- más
        # robusto que buscar el primer '{' y el último '}' a mano (evita
        # capturar basura tras el objeto JSON real).
        content = getattr(raw, "content", raw)
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            # Bloques de contenido estructurado del proveedor (ej. Anthropic/
            # OpenAI "content blocks": [{"type": "text", "text": "..."}]) --
            # se concatena el texto de cada bloque, nunca se serializa la
            # lista completa como texto crudo (eso rompería el parseo).
            parts: list[str] = []
            for block in content:
                if isinstance(block, Mapping):
                    block_text = block.get("text")
                    if isinstance(block_text, str):
                        parts.append(block_text)
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts)
        try:
            parsed = parse_json_safely(content)
        except Exception as exc:
            # Fail-closed: nunca se convierte una respuesta inválida en
            # un dict vacío ni en draft_text="" -- se propaga un error
            # explícito, igual que antes.
            raise ValueError("INVALID_LLM_OUTPUT") from exc
        if not isinstance(parsed, dict):
            raise ValueError("INVALID_LLM_OUTPUT")
        return parsed


def build_openai_draft_runtime(
    model,
    temperature,
    collection,
    *,
    project_dir=None,
    llm_factory=None,
    human_message_factory=None,
):
    from src.io.credentials import load_runtime_credential

    load_runtime_credential("OPENAI_API_KEY", project_dir=project_dir)
    if llm_factory is None:
        from langchain_openai import ChatOpenAI

        llm_factory = ChatOpenAI
    if human_message_factory is None:
        from langchain_core.messages import HumanMessage

        human_message_factory = HumanMessage
    llm = llm_factory(model=model, temperature=float(temperature))
    return DraftWritingRuntime(
        lambda prompt: llm.invoke([human_message_factory(content=prompt)]).content,
        collection,
    )


def _runtime_versions(strategy: str) -> dict[str, str]:
    # Única ruta productiva confirmada empíricamente en las 10 corridas
    # experimentales reales -- fail-closed, sin fallback silencioso.
    if strategy != LEGACY_RETRIEVAL_STRATEGY:
        raise ValueError(f"UNSUPPORTED_DRAFT_RETRIEVAL_STRATEGY:{strategy!r}")
    return dict(LEGACY_RUNTIME_VERSIONS)


def build_runtime_draft_policy(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the runtime policy for the single supported retrieval strategy."""
    requested = dict(overrides or {})
    policy = get_draft_writing_policy(requested)
    strategy = str(policy["retrieval_strategy"])
    policy.update(_runtime_versions(strategy))
    policy.pop("quantitative_selection_version", None)
    policy.pop("budget_version", None)
    return policy


def resolve_pipeline_state_path(project_dir, experiment_id):
    root = Path(project_dir).resolve()
    exp = root / experiment_id
    canonical = exp / "05_outputs" / "00_orchestrator_planner" / "pipeline_state.json"
    if canonical.is_file():
        return canonical
    raise FileNotFoundError(f"pipeline_state.json no encontrado en {canonical}")


def _collection_name(item):
    name = getattr(item, "name", None)
    return str(name if name is not None else item)


def _open_chroma_client(path, client_factory=None):
    if client_factory is None:
        import chromadb

        client_factory = chromadb.PersistentClient
    try:
        return client_factory(path=str(path))
    except TypeError:
        return client_factory(str(path))


def _validate_chroma_manifest(
    chroma_dir: Path,
    *,
    expected_collection: str,
    experiment_id: str | None,
) -> None:
    """Valida ``chroma_index_manifest.json`` de la ruta canónica contra
    lo que Stage 06 espera -- mismo principio que ya aplica Stage 03
    (``extraction_runtime.py``), pero como función propia y pequeña:
    no existe un helper standalone reutilizable en Stage 03 (su
    validación está inline, mezclada con globals de módulo de
    ``common_config`` resueltos al import -- reutilizarla tal cual
    introduciría acoplamiento indebido de Stage 06 a esos globals de
    Stage 03/00). Fail-closed: manifest ausente o inconsistente ->
    ValueError, nunca reparación silenciosa.
    """
    manifest_path = chroma_dir / "chroma_index_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "chroma_index_manifest.json inexistente: "
            f"{manifest_path}. Ejecuta el notebook 02."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "chroma_index_manifest.json contiene JSON inválido."
        ) from error
    if not isinstance(manifest, Mapping):
        raise TypeError("chroma_index_manifest.json debe contener un objeto.")

    manifest_collection = str(manifest.get("collection_name", ""))
    if manifest_collection != expected_collection:
        raise ValueError(
            "Configuración inconsistente: collection_name del "
            "manifiesto Chroma no coincide con el experimento activo "
            f"({manifest_collection!r} != {expected_collection!r})."
        )

    if experiment_id is not None and "experiment_id" in manifest:
        manifest_experiment_id = str(manifest.get("experiment_id", ""))
        if manifest_experiment_id != str(experiment_id):
            raise ValueError(
                "Configuración inconsistente: experiment_id del "
                "manifiesto Chroma no coincide con el experimento activo "
                f"({manifest_experiment_id!r} != {experiment_id!r})."
            )

    if "chroma_dir" in manifest:
        manifest_chroma_dir = Path(str(manifest["chroma_dir"]))
        if manifest_chroma_dir.resolve() != chroma_dir.resolve():
            raise ValueError(
                "Configuración inconsistente: chroma_dir del "
                "manifiesto no coincide con la ruta canónica usada."
            )


def resolve_chroma_dir(
    experiment_dir,
    expected_collection,
    explicit_path=None,
    *,
    client_factory=None,
    experiment_id=None,
):
    """Resuelve el índice Chroma que Stage 02 ya construyó -- Stage 06
    solo consume, nunca busca ni reconstruye.

    Chroma cleanup (Stage 06): se eliminó por completo la búsqueda
    heurística (búsqueda de ``chroma.sqlite3`` con glob recursivo
    sobre todo el árbol
    del experimento) y la ambigüedad que podía producir
    (``CHROMA_DIR_AMBIGUOUS``). La única ruta resuelta es
    ``explicit_path`` (si se provee -- contrato real del flujo, no una
    búsqueda) o, si no, la ruta canónica ``experiment_dir /
    "04_chroma_index"`` -- la misma que ya usan correctamente Stage 03
    y Stage 07. Si esa ruta no existe, no tiene ``chroma.sqlite3``, su
    manifest no valida, o no contiene la colección esperada -> falla
    cerrado. Nunca elige silenciosamente otro índice del disco.
    """
    exp = Path(experiment_dir).resolve()
    chroma_dir = (
        Path(explicit_path).expanduser().resolve()
        if explicit_path
        else exp / "04_chroma_index"
    )

    if not chroma_dir.is_dir() or not (chroma_dir / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            "Directorio real de Chroma inexistente: "
            f"{chroma_dir}. Ejecuta el notebook 02."
        )

    _validate_chroma_manifest(
        chroma_dir, expected_collection=expected_collection, experiment_id=experiment_id
    )

    try:
        client = _open_chroma_client(chroma_dir, client_factory)
        names = {_collection_name(item) for item in client.list_collections()}
    except Exception as exc:
        raise FileNotFoundError(
            f"CHROMA_COLLECTION_NOT_FOUND:{expected_collection}; "
            f"error abriendo el cliente en {chroma_dir}: "
            f"{type(exc).__name__}"
        ) from exc

    if expected_collection not in names:
        raise FileNotFoundError(
            f"CHROMA_COLLECTION_NOT_FOUND:{expected_collection}; "
            f"observed={sorted(names)}"
        )

    return chroma_dir.resolve()


def load_draft_configuration(
    project_dir,
    attempt_number=1,
    *,
    chroma_client_factory=None,
    policy_overrides: Mapping[str, Any] | None = None,
):
    root = Path(project_dir).resolve()
    active = json.loads((root / "active_experiment.json").read_text(encoding="utf-8"))
    experiment_id = active["active_experiment_id"]
    experiment_dir = root / experiment_id
    outputs = experiment_dir / "05_outputs"
    outline = outputs / "04_outline"
    thematic = outputs / "03_thematic_analysis"
    draft = outputs / "05_draft"
    rag = active.get("rag_policy", {})
    active_policy = dict(active.get("draft_generation_policy", {}))
    if policy_overrides:
        active_policy.update(dict(policy_overrides))
    policy = build_runtime_draft_policy(active_policy)
    generation = active.get("generation_profile", {})
    policy.update(
        {
            "experiment_profile": active.get("experiment_profile", {}),
            "topic_profile": active.get("topic_profile", {}),
            "generation_profile": generation,
            "rag_policy": rag,
            "output_language": generation.get("output_language", "español académico"),
            "writing_mode": generation.get("writing_mode", ""),
            "focus_mode": generation.get("focus_mode", ""),
            "citation_style": generation.get("citation_style", ""),
            "target_total_words": int(generation["target_total_words"]),
            "min_total_words": int(generation["min_total_words"]),
            "max_total_words": int(generation["max_total_words"]),
        }
    )
    # CONFIG-E (Stage 06): chroma_collection_name es responsabilidad de
    # 00_setup_config.ipynb -- 07 ya lo exige fail-closed
    # (verification_orchestrator_runtime.py::_require_active_experiment_key);
    # 06 usaba active.get("chroma_collection_name",
    # "reference_papers_chunks"), un fallback hardcodeado silencioso
    # que aquí queda eliminado. La centralización general de la
    # resolución de Chroma (incluida resolve_chroma_dir/rglob) NO se
    # toca en este bloque -- solo esta configuración puntual.
    collection_name = active.get("chroma_collection_name")
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError(
            "active_experiment.json['chroma_collection_name'] debe ser "
            "un string no vacío (00_setup_config.ipynb es su autoridad)."
        )
    chroma_dir = resolve_chroma_dir(
        experiment_dir,
        collection_name,
        active.get("chroma_dir"),
        client_factory=chroma_client_factory,
        experiment_id=experiment_id,
    )
    paths = {
        "outline_json": outline / "state_of_art_outline.json",
        "outline_mapping": outline / "outline_paper_mapping.csv",
        "outline_validation": outline / "outline_validation_report.json",
        "outline_manifest": outline / "outline_generation_manifest.json",
        "kb_final": thematic / "kb_final_for_thematic_analysis.csv",
        "thematic_manifest": thematic / "thematic_analysis_manifest.json",
        "thematic_validation": thematic / "thematic_validation_report.json",
        "chunks_clean": Path(
            active.get(
                "chunks_clean_path",
                experiment_dir / "03_chunks" / "chunks_clean_for_rag.csv",
            )
        ),
        "chroma_manifest": Path(
            active.get(
                "chroma_manifest_path",
                outputs / "01_rag" / "chroma_index_manifest.json",
            )
        ),
        "quantitative_table": outputs
        / "02_scientific_knowledge_base"
        / "quantitative_comparative_table.csv",
        "dataset_summary": outputs
        / "02_scientific_knowledge_base"
        / "dataset_technique_summary.csv",
        "quantitative_manifest": outputs
        / "02_scientific_knowledge_base"
        / "quantitative_extraction_manifest.json",
    }
    # CONFIG-E (Stage 06): openai_model es responsabilidad de
    # 00_setup_config.ipynb -- 04/05/07 ya lo exigen fail-closed; 06
    # usaba active.get("openai_model", "gpt-4o-mini"), un fallback
    # hardcodeado silencioso que aquí queda eliminado.
    openai_model = active.get("openai_model")
    if not isinstance(openai_model, str) or not openai_model.strip():
        raise ValueError(
            "active_experiment.json['openai_model'] debe ser un string "
            "no vacío (00_setup_config.ipynb es su autoridad)."
        )
    return {
        "project_dir": root,
        "experiment_id": experiment_id,
        "run_id": active.get("run_id", experiment_id),
        "attempt_number": int(attempt_number),
        "model": openai_model,
        "embedding_model_name": active.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "chroma_collection_name": collection_name,
        "chroma_dir": chroma_dir,
        "policy": policy,
        "output_dir": draft,
        "state_path": resolve_pipeline_state_path(root, experiment_id),
        "paths": paths,
        "experiment_dir": experiment_dir,
    }


def _previous_draft_attempt(cfg):
    if cfg["attempt_number"] != 2:
        return None
    payload = json.loads(Path(cfg["state_path"]).read_text(encoding="utf-8"))
    stage = payload.get("stages", {}).get("06_agente_redactor", {})
    if stage.get("requested_transition", {}).get("action") != "RETRY":
        raise RuntimeError("El intento 2 requiere una transición RETRY persistida.")
    return PreviousAttemptSummary(
        quality_status=stage.get("quality_status", "NEEDS_REVISION"),
        failure_reason_codes=tuple(stage.get("failure_reason_codes", [])),
        blocking_warnings=tuple(
            str(item.get("code", ""))
            for item in stage.get("warnings", [])
            if item.get("blocking") and item.get("code")
        ),
        previous_artifacts={},
    )


def _dependency_references(cfg) -> dict[str, ArtifactReference]:
    required = (
        "outline_json",
        "outline_mapping",
        "outline_validation",
        "outline_manifest",
        "kb_final",
        "thematic_manifest",
        "thematic_validation",
        "chunks_clean",
    )
    dependencies = {}
    for name in required:
        path = Path(cfg["paths"][name])
        if not path.is_file():
            raise FileNotFoundError(f"{name} no existe: {path}")
        dependencies[name] = ArtifactReference(str(path), sha256_file(path))
    chroma_manifest = Path(cfg["paths"]["chroma_manifest"])
    if chroma_manifest.is_file():
        dependencies["chroma_manifest"] = ArtifactReference(
            str(chroma_manifest), sha256_file(chroma_manifest)
        )
    optional = ("quantitative_table", "dataset_summary", "quantitative_manifest")
    present = [name for name in optional if Path(cfg["paths"][name]).is_file()]
    if present and len(present) != len(optional):
        raise FileNotFoundError("INVALID_QUANTITATIVE_CONTEXT")
    for name in present:
        path = Path(cfg["paths"][name])
        dependencies[name] = ArtifactReference(str(path), sha256_file(path))
    return dependencies


def _draft_signature(cfg, dependencies) -> dict[str, Any]:
    policy = {key: value for key, value in cfg["policy"].items() if key != "current_fingerprint"}
    # Única ruta productiva confirmada empíricamente en las 10 corridas
    # experimentales reales: draft_representation_contract siempre
    # canonical_sentences_v2. Ausencia o cualquier otro valor (incluido
    # "legacy" explícito) falla fail-closed, antes de construir firma
    # alguna -- sin fallback silencioso.
    contract = policy.get("draft_representation_contract")
    if contract != CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT:
        raise ValueError(f"UNKNOWN_DRAFT_REPRESENTATION_CONTRACT:{contract!r}")
    signature = {
        "stage": "06_agente_redactor",
        "stage_version": policy["stage_version"],
        "experiment_id": cfg["experiment_id"],
        "experiment_dir": str(cfg["experiment_dir"]),
        "openai_model": cfg["model"],
        "embedding_model_name": cfg["embedding_model_name"],
        "chroma_collection_name": cfg["chroma_collection_name"],
        "topic_profile": policy.get("topic_profile", {}),
        "experiment_profile": policy.get("experiment_profile", {}),
        "generation_profile": policy.get("generation_profile", {}),
        "rag_policy": policy.get("rag_policy", {}),
        "draft_generation_policy": policy,
        "paths": {key: value.path for key, value in dependencies.items()},
        "hashes": {key: value.hash for key, value in dependencies.items()},
        "prompt_version": policy["prompt_version"],
        "rag_version": policy["rag_version"],
        "validation_version": policy["validation_version"],
        "draft_representation_contract": CANONICAL_SENTENCES_DRAFT_REPRESENTATION_CONTRACT,
    }
    return signature


def build_draft_agent_input(cfg):
    dependencies = _dependency_references(cfg)
    signature = _draft_signature(cfg, dependencies)
    policy = dict(cfg["policy"])
    policy["current_fingerprint"] = fingerprint_mapping(signature)
    cfg["policy"] = policy
    return AgentInput(
        experiment_id=cfg["experiment_id"],
        run_id=cfg["run_id"],
        stage_name="06_agente_redactor",
        attempt_number=cfg["attempt_number"],
        mode=ExecutionMode.FULL_RUN,
        agent_context=AgentContext(
            allowed_tools=(
                "llm",
                "chroma",
                "csv_retrieval",
                "atomic_write",
                "draft_validation",
            ),
            output_directory=str(cfg["output_dir"]),
            runtime_resources={
                "model": cfg["model"],
                "chroma_collection_name": cfg["chroma_collection_name"],
                "embedding_model_name": cfg["embedding_model_name"],
                "chroma_dir": str(cfg["chroma_dir"]),
            },
        ),
        dependencies=dependencies,
        policy=policy,
        previous_attempt=_previous_draft_attempt(cfg),
    )


def build_chroma_collection(cfg):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(cfg["chroma_dir"]))
    embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg["embedding_model_name"]
    )
    return client.get_collection(
        name=cfg["chroma_collection_name"], embedding_function=embedding
    )


def build_real_draft_execution(
    project_dir,
    attempt_number=1,
    *,
    collection_factory=None,
    runtime_factory=None,
    chroma_client_factory=None,
    policy_overrides: Mapping[str, Any] | None = None,
):
    cfg = load_draft_configuration(
        project_dir,
        attempt_number,
        chroma_client_factory=chroma_client_factory,
        policy_overrides=policy_overrides,
    )
    collection = (collection_factory or build_chroma_collection)(cfg)
    runtime_builder = runtime_factory or build_openai_draft_runtime
    runtime = runtime_builder(
        cfg["model"],
        cfg["policy"]["temperature"],
        collection,
        project_dir=cfg["project_dir"],
    )
    return DraftWritingAgent(runtime), build_draft_agent_input(cfg), cfg

