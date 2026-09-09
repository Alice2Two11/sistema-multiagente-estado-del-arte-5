# LÉEME PRIMERO — guía corta y operativa

Esta es una guía rápida de orientación. Para la documentación técnica
completa (arquitectura, flujo, formatos, outputs, reproducibilidad) ver
**`README.md`**. Para el procedimiento de smoke test real en Colab ver
**`COLAB_SMOKE_TEST.md`**.

## ¿Qué es esto?

El código fuente reutilizable y transaccional del pipeline multiagente de
la tesis (etapas 00-08), migrado desde los notebooks operativos originales
en `notebooks_2/`. **Las 9 etapas están migradas y ejecutables desde el
orquestador**, incluida la 08 (evaluación) y el ciclo correctivo real
`06 ↔ 07`.

## Estado real (verificado en esta entrega)

- Pipeline completo ejecutable: `00 → 01 → 02 → 03 → 03B → 04 → 05 → 06 →
  07 ↔ 06 → 07 → 08`.
- Ciclo correctivo `06 ↔ 07` probado de punta a punta **con dobles
  deterministas** (código productivo real, LLM/Chroma/retriever
  simulados): 07 detecta un claim corregible, emite `RETURN`, 06 corrige
  solo la sección afectada en modo `REVISION`, 07 reverifica y emite
  `ADVANCE` hacia 08.
- **Actualización tras la fase de integración real (OpenAI/Chroma)**: se
  corrieron numerosos experimentos reales y se encontraron y corrigieron
  15+ bugs de contrato (prompt del corrector, validadores de
  `validation.py`, la fórmula de `hallucination_rate` de 08, y un bug de
  no-determinismo en el retriever RAG). Registro completo en
  `README.md` sección 27. **Actualización posterior (sección 28): el
  ciclo `06 ↔ 07` ya se confirmó completo de punta a punta con datos
  reales**, en múltiples experimentos y dominios, incluida al menos una
  corrida con dos rondas consecutivas.
- Etapa 08 migrada completa: métricas automáticas (ROUGE-L, BERTScore,
  similitud semántica), LLM Judge, métricas factuales/trazabilidad
  (incluye `hallucination_rate`, `hallucination_rate_broad` y
  `unverified_rate` — ver README sección 27), persistencia de los 15
  outputs, fingerprints, contrato transaccional y `StageSpec` real
  conectado al orquestador.
- **468/468** escenarios pasando en 37 suites de `tests/orchestration/`
  (correr con `python3 archivo.py`, no con `pytest` — ver advertencia
  abajo).

## Código real por etapa (módulos reales, verificados en disco)

```text
00  src/bootstrap/, src/state/, src/orchestration/, src/contracts/, src/io/
01  infraestructura común (bootstrap/config/contracts/io/state)
02  src/agents/extraction_agent.py, src/adapters/extraction_runtime.py,
    src/runtime/extraction_protocol.py, src/tools/extraction/
03B src/capabilities/quantitative_extraction.py,
    src/adapters/quantitative_extraction_runtime.py,
    src/runtime/quantitative_extraction_protocol.py,
    src/tools/quantitative_extraction/
04  src/agents/thematic_analysis_agent.py,
    src/adapters/thematic_analysis_runtime.py,
    src/runtime/thematic_analysis_protocol.py, src/tools/thematic_analysis/
05  src/agents/outline_generation_agent.py,
    src/adapters/outline_generation_runtime.py,
    src/runtime/outline_generation_protocol.py, src/tools/outline_generation/
06  src/agents/draft_writing_agent.py, src/adapters/draft_writing_runtime.py,
    src/adapters/draft_writing_notebook.py,
    src/runtime/draft_writing_protocol.py, src/tools/draft_writing/
07  src/agents/verification_agent.py, src/adapters/verification_runtime.py,
    src/adapters/verification_notebook.py,
    src/adapters/claim_verification_context.py,
    src/adapters/verification_claim_canonicalization.py,
    src/adapters/verification_incremental_retriever.py,
    src/adapters/agent06_verification_handoff.py,
    src/tools/verification/, src/tools/verification/cycle_round_persistence.py
08  src/tools/evaluation/evaluation_pipeline.py (ensamblador integral),
    src/tools/evaluation/{ground_truth,language_preprocessing,translation,
    rouge,semantic_similarity,bertscore,automatic_metrics,numeric_validation,
    claim_citation_audit,factual_assembly,llm_judge,final_validation,
    final_report}.py,
    src/adapters/{evaluation_upstream,evaluation_fingerprint,
    evaluation_persistence,evaluation_orchestrator_runtime,
    evaluation_stagespec_wiring}.py
```

**Corrección frente a versiones anteriores de este archivo**: las carpetas
de herramientas usan el nombre completo de cada etapa
(`src/tools/outline_generation/`, `src/tools/quantitative_extraction/`,
`src/tools/thematic_analysis/`), no las abreviaturas `outline/`,
`quantitative/`, `thematic/` que aparecían antes. `src/orchestration/`
(el orquestador real: `run_stage`, `run_pipeline`, `decision_engine`) y
`src/tools/evaluation/` (toda la etapa 08) tampoco figuraban en la versión
previa de este archivo.

## El ciclo `06 ↔ 07`

```text
06 redacta -> 07 verifica -> claim corregible con evidencia real
  -> RETURN a 06 (writer_revision_request en writer_verifier_cycle/round_NN/)
  -> 06 corrige SOLO la sección afectada (modo REVISION)
  -> 07 reverifica -> ADVANCE -> 08
```

Cableado y probado con dobles deterministas (ver arriba), y **confirmado
completo con datos reales** en la segunda fase de integración —
ver `README.md` sección 28 (y sección 27 para el patrón que se observaba
antes de esos arreglos: `DEFER_TO_MANUAL_REVIEW` por evidencia débil, o
`PROPOSE_CHANGE` rechazado en el validador semántico, ambos siguen siendo
desenlaces normales y esperables). Detalle completo en `README.md`,
secciones 6, 17, 27 y 28.

## Ejecutar el pipeline

```bash
python3 -m src.orchestration_langgraph.pipeline_graph --project-dir /ruta/a/PROJECT_DIR --until 08_evaluacion_experimental
```

Hasta una etapa específica: agregar `--until 07_agente_verificador`.
Reinicio limpio (cambiaste código, o la corrida anterior se cortó mal):
agregar `--fresh-start` — ver `README.md` sección 14.
Reejecutar aunque ya esté `COMPLETED`: agregar `--force-rerun`. Detalle
completo, incluidos requisitos previos (`active_experiment.json`,
`OPENAI_API_KEY`), en `README.md` secciones 11-14.

## Ejecutar pruebas

```bash
python3 smoke_test.py
python3 smoke_test_draft.py
for f in tests/orchestration/test_*.py; do python3 "$f" || echo "FALLÓ: $f"; done
```

**No uses `pytest` para `tests/orchestration/`** — verificado
empíricamente que marca "passed" cualquier escenario interno fallido (el
decorador `@scenario` captura la excepción y no la relanza). Detalle en
`README.md` sección 13.

## Notebooks operativos

```text
GitHub  (este repositorio)   -> código reutilizable y pruebas
Google Drive / notebooks_2/  -> notebooks operativos 00-08 (fuente científica original)
/content en Colab            -> clon temporal, efímero, para ejecutar
```

## Reglas de mantenimiento

No subir: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ipynb_checkpoints/`,
`.env`.

Cualquier cambio de comportamiento observable frente a los notebooks
originales requiere una decisión independiente, documentada, y pruebas de
regresión — no se hacen mejoras silenciosas.
