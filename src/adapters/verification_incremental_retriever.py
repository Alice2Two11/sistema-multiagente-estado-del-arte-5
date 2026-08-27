"""Retriever incremental de la etapa 07, portado desde el notebook 07.

``Agent07ChromaRetriever`` de aquí abajo es una copia LITERAL de la clase
definida inline en la celda 11 (rama productiva, no FIXTURE_MODE) de
``07_agente_verificador_trazabilidad_LIMPIO.ipynb``. No se ha cambiado
ninguna línea de su lógica: mismos nombres de campos, mismo orden de
construcción de ``selected``, mismos criterios de corte (``top_k``/
``fetch_k``), mismo cálculo de ``fused_rrf_score``, mismo comportamiento
ante ausencia de evidencia (``selected=()`` y
``stop_reason="NO_NEW_EVIDENCE"`` cuando no hay candidatos autorizados).
Verificar con un diff contra la celda original si se sospecha cualquier
divergencia — ver ``tests/orchestration/test_verification_characterization.py``,
que compara ambas fuentes.

Lo que SÍ cambia respecto al notebook es el código *alrededor* de la clase
(la función que la instancia), porque esa parte no es lógica del retriever
sino wiring de notebook que dependía de módulos generados por el proyecto
(``rag_utils.load_chroma_collection``, ``get_rag_policy()``) que no existen
en ``src/``:

- ``load_chroma_collection(CHROMA_DIR, collection_name=..., model_name=...)``
  se reemplaza por la misma operación ya implementada en
  ``src/adapters/draft_writing_runtime.build_chroma_collection`` (mismo
  ``chromadb.PersistentClient`` + ``SentenceTransformerEmbeddingFunction`` +
  ``get_collection``) — no es una implementación nueva ni reducida, es la
  misma operación que el repo ya tenía escrita para otra etapa.
- El ``_sha256`` local del notebook se reemplaza por
  ``src.state.fingerprints.sha256_file``, que hace exactamente lo mismo
  (SHA-256 por bloques de 1 MiB) y ya está probado en el repo.
- ``get_rag_policy()`` (generado por notebook 00) se reemplaza por leer
  ``active_experiment.json["rag_policy"]`` directamente — la misma fuente
  que ya usan 02-06 para ``rag_policy`` — en vez de inventar un valor.

Nada de esto toca el comportamiento del retriever en sí: son sustituciones
de "cómo se obtiene un dato" por una fuente ya existente y equivalente en
``src/``, no cambios al algoritmo de recuperación.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Agent07ChromaRetriever:
    """Retriever productivo restringido a fuentes autorizadas.

    Copia literal de la clase homónima definida en la celda 11 de
    07_agente_verificador_trazabilidad_LIMPIO.ipynb (rama productiva),
    EXTENDIDA de forma retrocompatible en AGENTIC-RETRIEVAL-BLOQUE-4
    para soportar ``query_override``/``top_k_override`` opcionales en
    ``request`` -- ausentes ambos, el comportamiento es idéntico al
    original (mismo ``query_texts=[claim_text]``, mismo ``self.top_k``).
    No se cambió: collection, embeddings, fetch_k, scoring, dedupe,
    allowed_source_filenames, source filtering,
    native_scores_by_retriever, fused_rrf_score. ``self.top_k`` nunca
    se muta -- ``top_k_override`` solo afecta el corte de esa llamada
    concreta (variable local ``effective_top_k``).
    """

    def __init__(
        self,
        *,
        collection,
        experiment_id,
        collection_name,
        embedding_model,
        chroma_manifest_fingerprint,
        chunks_manifest_fingerprint,
        top_k=8,
        fetch_k=35,
    ):
        self.collection = collection
        self.experiment_id = experiment_id
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chroma_manifest_fingerprint = (
            chroma_manifest_fingerprint
        )
        self.chunks_manifest_fingerprint = (
            chunks_manifest_fingerprint
        )
        self.top_k = int(top_k)
        self.fetch_k = int(fetch_k)

    def retrieve_more(self, request):
        claim_id = str(request["claim_id"])
        claim_context = request["claim_context"]
        claim_text = str(
            claim_context.get("claim_text")
            or claim_context.get("original_claim_text")
            or ""
        ).strip()

        allowed_sources = tuple(
            request["allowed_source_filenames"]
        )
        allowed_set = set(allowed_sources)

        if not claim_text:
            raise ValueError(
                "AGENT07_RETRIEVER_CLAIM_TEXT_MISSING"
            )

        # AGENTIC-RETRIEVAL-BLOQUE-4: overrides opcionales,
        # retrocompatibles. Ausencia de ambos -> comportamiento
        # exactamente anterior. Fail-closed si están presentes pero
        # mal formados -- nunca coacción vía str(...)/int(...).
        query_override = request.get("query_override")
        if query_override is not None:
            if not isinstance(query_override, str) or not query_override.strip():
                raise ValueError("AGENT07_RETRIEVER_QUERY_OVERRIDE_INVALID")
            effective_query = query_override
        else:
            effective_query = claim_text

        top_k_override = request.get("top_k_override")
        if top_k_override is not None:
            if (
                not isinstance(top_k_override, int)
                or isinstance(top_k_override, bool)
                or top_k_override <= 0
                or top_k_override > self.fetch_k
            ):
                raise ValueError("AGENT07_RETRIEVER_TOP_K_OVERRIDE_INVALID")
            effective_top_k = top_k_override
        else:
            effective_top_k = self.top_k

        result = self.collection.query(
            query_texts=[effective_query],
            n_results=self.fetch_k,
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        selected = []
        seen_pairs = set()

        for native_rank, (text, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1,
        ):
            metadata = metadata or {}
            source = str(
                metadata.get("source_filename") or ""
            ).strip()
            chunk_id = str(
                metadata.get("chunk_id") or ""
            ).strip()
            text = str(text or "").strip()

            if (
                not source
                or not chunk_id
                or not text
                or source not in allowed_set
            ):
                continue

            pair = (source, chunk_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            score = 1.0 - float(distance)
            selected.append(
                {
                    "source_filename": source,
                    "chunk_id": chunk_id,
                    "text": text,
                    "retrieval_sources": ("chroma",),
                    "query_ids": (claim_id,),
                    "all_native_ranks": (native_rank,),
                    "native_ranks_by_retriever": {
                        "chroma": native_rank
                    },
                    "native_scores_by_retriever": {
                        "chroma": score
                    },
                    "native_score_types_by_retriever": {
                        "chroma": "cosine_similarity"
                    },
                    "first_seen_round": 1,
                    "last_seen_round": 1,
                    "fused_rrf_score": (
                        1.0 / (60.0 + native_rank)
                    ),
                    "text_variants": (),
                    "contradiction_signals": (),
                }
            )

            if len(selected) >= effective_top_k:
                break

        return {
            "selected_candidates": tuple(selected),
            "rounds_executed": 1,
            "total_candidates_seen": len(documents),
            "total_unique_candidates_seen": len(seen_pairs),
            "queries_executed_total": 1,
            "new_unique_pairs_seen": len(seen_pairs),
            "queries": (
                {
                    "query_id": claim_id,
                    "query_text": effective_query,
                },
            ),
            "discarded_candidates": (),
            "retrieval_trace": (
                {
                    "claim_id": claim_id,
                    "allowed_sources": allowed_sources,
                    "selected_count": len(selected),
                },
            ),
            "contradiction_signals": (),
            "technical_issue_codes": (),
            "technical_status": "COMPLETED",
            "stop_reason": (
                "STRUCTURAL_COVERAGE_SATISFIED"
                if selected
                else "NO_NEW_EVIDENCE"
            ),
            "queries_remaining": 0,
            "total_unique_candidates_retained": len(selected),
            "new_unique_pairs_selected": len(selected),
            "structural_coverage_improved": bool(selected),
            "structural_coverage_improved_this_delta": (
                bool(selected)
            ),
        }


def build_agent07_chroma_retriever(
    *,
    chroma_dir: str | Path,
    chroma_collection_name: str,
    embedding_model_name: str,
    chroma_manifest_path: str | Path,
    chunks_manifest_path: str | Path,
    committed_experiment_id: str,
    rag_policy: dict[str, Any],
    chroma_client_factory: Any = None,
    embedding_function_factory: Any = None,
):
    """Reproduce el wiring de la celda 11 (no FIXTURE_MODE) fuera del notebook.

    Devuelve ``(retriever, retriever_binding_kwargs)`` donde
    ``retriever_binding_kwargs`` es el mismo diccionario de campos que la
    celda 11 pasa a ``Agent07RetrieverBinding`` — se devuelve por separado
    para que el llamador construya el ``Agent07RetrieverBinding`` real de
    ``src/adapters/agent06_verification_handoff.py`` sin duplicar esa clase
    aquí.

    Lanza ``ValueError("El manifest de Chroma pertenece a otro
    experimento.")`` en el mismo caso y con el mismo mensaje que la celda
    11, si ``chroma_manifest["experiment_id"] != committed_experiment_id``.
    """

    from src.state.fingerprints import sha256_file

    chroma_manifest_path = Path(chroma_manifest_path)
    chunks_manifest_path = Path(chunks_manifest_path)

    chroma_manifest = json.loads(chroma_manifest_path.read_text(encoding="utf-8"))
    if chroma_manifest.get("experiment_id") != committed_experiment_id:
        raise ValueError("El manifest de Chroma pertenece a otro experimento.")

    if chroma_client_factory is None or embedding_function_factory is None:
        import chromadb
        from chromadb.utils import embedding_functions

        chroma_client_factory = chroma_client_factory or chromadb.PersistentClient
        embedding_function_factory = (
            embedding_function_factory
            or embedding_functions.SentenceTransformerEmbeddingFunction
        )

    client = chroma_client_factory(path=str(Path(chroma_dir).resolve()))
    embedding_function = embedding_function_factory(model_name=embedding_model_name)
    collection = client.get_collection(
        name=chroma_collection_name, embedding_function=embedding_function
    )

    retriever_binding_kwargs = {
        "experiment_id": committed_experiment_id,
        "collection_name": chroma_collection_name,
        "embedding_model": embedding_model_name,
        "chroma_manifest_fingerprint": sha256_file(chroma_manifest_path),
        "chunks_manifest_fingerprint": sha256_file(chunks_manifest_path),
    }

    default_profile = (
        rag_policy.get("retrieval_profiles", {}).get("default", {})
        if isinstance(rag_policy, dict)
        else {}
    )
    top_k = int(default_profile.get("top_k", 8))
    fetch_k = int(default_profile.get("fetch_k", 35))

    retriever = Agent07ChromaRetriever(
        collection=collection,
        top_k=top_k,
        fetch_k=fetch_k,
        **retriever_binding_kwargs,
    )
    return retriever, retriever_binding_kwargs
