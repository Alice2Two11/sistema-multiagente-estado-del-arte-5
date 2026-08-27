# Sistema multiagente para generación y evaluación automatizada de estados del arte

## 1. Nombre y propósito del proyecto

Este repositorio contiene el código fuente reutilizable y las pruebas
automatizadas de un **sistema multiagente basado en modelos de lenguaje
grandes (LLM)** que genera, verifica y evalúa automáticamente **estados del
arte** (revisiones de literatura científica) a partir de un corpus de
artículos proporcionado por el usuario.

Es el soporte de código de una tesis de maestría. Este `README.md` describe
el **estado real y verificado del código** al momento de esta entrega — no
un diseño aspiracional.

## 2. Problema de investigación

Redactar un estado del arte exige leer, comparar y sintetizar decenas de
artículos, verificar que cada afirmación tenga respaldo textual explícito, y
mantener trazabilidad completa entre cada oración generada y la evidencia
que la sostiene. Hacerlo manualmente es lento y propenso a errores de cita,
extrapolación o alucinación factual. Los LLM pueden generar texto fluido con
rapidez, pero sin control adicional no garantizan que cada afirmación esté
efectivamente respaldada por el corpus.

## 3. Objetivo general del sistema

Diseñar e implementar un pipeline multiagente que:

1. procese un corpus de papers científicos y construya una base de
   conocimiento estructurada y trazable;
2. redacte un estado del arte por secciones, citando explícitamente la
   fuente y el fragmento (`chunk`) de cada afirmación;
3. verifique automáticamente cada afirmación contra el corpus, clasifique su
   nivel de soporte y, cuando sea posible, proponga una corrección
   localizada respaldada por evidencia real;
4. cierre un ciclo correctivo real entre el redactor y el verificador antes
   de dar por definitivo el borrador;
5. evalúe el resultado final con métricas automáticas, un LLM Judge y
   métricas factuales y de trazabilidad, comparándolo contra un Ground
   Truth cuando existe.

## 4. Arquitectura multiagente actual

El sistema distingue tres capas:

### Agentes (lógica científica de cada etapa)

Viven en `src/agents/`. Cada agente encapsula la decisión científica de su
etapa (qué extraer, cómo redactar, cómo verificar un claim) y es agnóstico
de persistencia, red o LLM concretos — recibe sus dependencias inyectadas.

```text
src/agents/extraction_agent.py
src/agents/thematic_analysis_agent.py
src/agents/outline_generation_agent.py
src/agents/draft_writing_agent.py
src/agents/verification_agent.py
```

### Infraestructura (orquestación, estado, contratos)

No contiene lógica científica. Sostiene el pipeline como sistema
transaccional:

```text
src/orchestration/   StageSpec, run_stage(), run_pipeline(), decision_engine
src/state/           StateStore, PipelineState, fingerprints
src/contracts/       AgentInput, AgentResult (contrato uniforme de I/O de cada agente)
src/bootstrap/       preparación del proyecto y del experimento
src/io/              escritura atómica, credenciales
src/adapters/        conecta agentes + runtime real (LLM, Chroma, disco) con el contrato de StageSpec
src/runtime/         protocolos de ejecución (build_execution/runtime_transaction/resolve_resume) por etapa
src/config/          políticas y parámetros por etapa, sin valores por defecto silenciosos
```

### Módulos de evaluación (etapa 08, sin agente propio)

`src/tools/evaluation/` no define un "agente" en el sentido de
`src/agents/` — es un conjunto de módulos puros (normalización, ROUGE-L,
similitud semántica, BERTScore, auditoría numérica, auditoría de claims y
citas, LLM Judge) ensamblados por `src/tools/evaluation/evaluation_pipeline.py`
y conectados al contrato transaccional mediante
`src/adapters/evaluation_orchestrator_runtime.py`. La diferencia con los
agentes 02-07 es deliberada: 08 no toma una decisión científica única y
reintentable como los agentes — compone y persiste métricas.

## 5. Flujo completo

```text
00 → 01 → 02 → 03 → 03B → 04 → 05 → 06 → 07 ↔ 06 → 07 → 08
```

| Etapa | Nombre | Qué hace |
|---|---|---|
| 00 | Orquestación y planificación | Prepara el experimento, resuelve configuración común, inicializa `pipeline_state.json`. |
| 01 | Ingesta y preparación documental | Organiza los PDF de entrada, produce chunks limpios para RAG. |
| 02 | Extracción de información científica | Construye fichas por paper (problema, métodos, datasets, resultados, limitaciones). |
| 03 | Extracción de KB | Consolida las fichas en una base de conocimiento estructurada. |
| 03B | Extracción cuantitativa | Identifica y normaliza métricas y resultados numéricos, vinculados a paper y chunk. |
| 04 | Análisis temático | Agrupa métodos/datasets/resultados en temas y comparaciones. |
| 05 | Generación del esquema | Convierte el análisis temático en la estructura de secciones del estado del arte. |
| 06 | Redacción del borrador | Redacta cada sección citando `[fuente.pdf \| chunk_id]`, con RAG real. |
| 07 | Verificación factual y trazabilidad | Descompone el borrador en claims, verifica cada uno contra el corpus, clasifica veredicto y elegibilidad de corrección. |
| 08 | Evaluación experimental | Compara contra Ground Truth con métricas automáticas, LLM Judge y métricas factuales; persiste los 15 outputs finales. |

## 6. El ciclo correctivo 06 ↔ 07

07 no es un simple validador de paso único. Cuando encuentra un claim
`PARTIALLY_SUPPORTED` con evidencia real disponible, construye una
propuesta de corrección localizada (`propose_correction`) y, si la acepta,
emite una transición **`RETURN`** hacia 06 en vez de `ADVANCE`:

```text
06 redacta
   ↓
07 verifica -> claim corregible con evidencia real
   ↓
RETURN a 06, con una "writer_revision_request" trazable
   ↓
06 entra en modo REVISION: corrige SOLO la sección/claim señalado,
   el resto del borrador queda intacto
   ↓
07 reverifica el borrador corregido
   ↓
si el claim ya es SUPPORTED -> ADVANCE hacia 08
```

Este ciclo se persiste en disco de forma transaccional en
`writer_verifier_cycle/round_NN/` (ver sección 17) — 07 crea la ronda en
`AWAITING_REVISION`, 06 la completa (nunca la crea) a `REVISION_COMPLETED`,
y un segundo intento de completar la misma ronda se rechaza explícitamente
(no hay doble escritura silenciosa).

### `07C`: excluido del flujo activo

`07C` (reverificación de una corrección ya aplicada automáticamente) **no
forma parte del flujo activo**. La ruta real es `07 → 06 (RETURN) → 07`
directamente, sin pasar por 07C. El código conserva compatibilidad
histórica con 07C únicamente en `src/adapters/agent07c_handoff.py` y
mensajes/nombres de archivo heredados donde era inevitable — su presencia
en el código no significa que participe del registro de etapas activo
(`STAGE_ORDER` en `src/orchestration/pipeline_orchestrator.py` no lo
incluye).

### Identidad estable de claims (`claim_uid`)

`claim_id` (ej. `S5_C2`) es **posicional**: 06 lo recalcula cada vez que
regenera una sección completa en modo `REVISION` (regenera la sección
entera si tiene algún issue, no solo el claim señalado), así que la misma
etiqueta puede referirse a afirmaciones distintas entre rondas. Para
rastrear un claim de forma confiable entre rondas, cada claim tiene además:

- `claim_uid`: UUID4 opaco, minteado una sola vez, nunca recalculado por
  contenido ni posición.
- `claim_version`: entero monótono, se incrementa solo cuando ese claim
  específico se reescribe.
- `parent_claim_uids`: linaje explícito — vacío al mintear, un padre en
  una continuación o *split*, dos o más en un *merge*.

La identidad se declara explícitamente por 06 (vía el contrato de
respuesta del LLM: `identity_action` = `CONTINUE` / `NEW` / `SPLIT_CHILD`
/ `MERGE`) — nunca se infiere después por similitud de texto. Cuando un
`writer_revision_request` señala un `claim_uid` concreto, 06 debe
preservarlo determinísticamente; si el LLM declara una identidad distinta,
la respuesta se rechaza (fail-closed), nunca se sobreescribe en silencio.
Ver `src/tools/draft_writing/claim_identity.py`.

**Contrato de identidad por experimento** (`CycleState.claim_identity_
contract_version`, en `pipeline_state.json`): `"LEGACY"` (comportamiento
posicional, experimentos anteriores a este mecanismo) o `"STABLE_UID_V1"`.
El contrato se resuelve de forma híbrida — si ya hay uno declarado para el
ciclo, es normativo (nunca se re-infiere ni se permite un downgrade
silencioso si algún `claim_uid` se pierde); solo en la primera ronda, sin
nada declarado todavía, se infiere una vez de los propios claims.

**Migración de un experimento `LEGACY`** ocurre de dos formas, nunca de
forma implícita:
1. *Dentro de una ronda real de revisión*: 06 declara una señal explícita
   (`claim_identity_migration_signal=true` en su `committed_agent06_
   output`, emitida automáticamente por `agent06_verification_handoff.py`
   solo cuando el ciclo está en `LEGACY` y **todos** los claims publicados
   ya tienen `claim_uid`/`claim_version` completos) — cuenta como una
   ronda normal del ciclo `06 ↔ 07`.
2. *Bootstrap administrativo* (`src/tools/draft_writing/claim_identity_
   bootstrap.py::bootstrap_legacy_claim_identity_for_exhausted_cycle`),
   para un ciclo que **ya agotó sus rondas científicas**
   (`rounds_used >= max_rounds`) y no puede consumir otra solo para
   adquirir identidad. Publica una copia nueva del draft (nunca sobrescribe
   el histórico) con `claim_uid` recién minteado para cada claim —
   `parent_claim_uids=[]` siempre, nunca se reconstruye una identidad
   histórica por similitud — y compromete un nuevo resultado de 06 con
   `ADVANCE → 07` real vía `StateStore`, sin tocar `rounds_used`,
   `max_rounds`, ni ninguna decisión o claim históricos. Es idempotente:
   si el ciclo ya es `STABLE_UID_V1`, no hace nada.

### Reconstrucción causal del `decision_log`

`decision_log` es *append-only* y puede acumular ejecuciones espurias
(ej. reintentos tras un bug de control de flujo ya corregido) — ni
`decision_log[-1]` ni `state.stages[etapa]` (el estado vigente, que una
ejecución espuria posterior puede sobrescribir) son fuentes confiables por
sí solas de "qué fue lo último que realmente pasó". `src/orchestration/
decision_log_frontier.py` reconstruye la cadena causal real a partir de
las transiciones que cada decisión solicitó, con dos funciones para dos
preguntas distintas:

- `authoritative_decision_log_entry_for_stage`: "¿cuál es el estado
  terminal **vigente** de esta etapa?" — usada para reconocer un restart.
- `committed_predecessor_for_stage`: "¿cuál fue el **último commit que
  realmente habilitó** la etapa siguiente?" (`COMPLETED` + `ADVANCE` hacia
  el target exacto) — usada para resolver una dependencia upstream (ej.
  qué draft de 06 debe leer 07). Son preguntas distintas a propósito: la
  primera puede apuntar a una ejecución espuria; la segunda nunca.

`src/adapters/agent06_verification_handoff.py::resolve_committed_agent06_
artifacts` usa siempre la segunda — es también la función que resuelve el
draft canónico que lee 08 (ver más abajo), garantizando una única
definición de "qué comprometió 06 realmente" en toda la cadena 06→07→08.


## 7. RAG, memoria documental, KB científica y trazabilidad

- **Memoria documental (etapa 01)**: los PDF de entrada se convierten en
  chunks limpios, identificados por `(source_filename, chunk_id)` — la
  unidad atómica de cita en todo el pipeline.
- **RAG**: 06 recupera evidencia por sección al redactar; 07 puede además
  hacer una recuperación **independiente** por claim (no solo heredar la
  evidencia que 06 citó), etiquetando esa evidencia con
  `usage_role="SUPPORT"` — la única vía productiva real que asigna ese rol
  concreto (ver `src/adapters/verification_incremental_retriever.py`).
- **KB científica (etapas 02/03/03B/04)**: fichas por paper, extracción
  cuantitativa y análisis temático, consolidadas antes de generar el
  esquema.
- **Trazabilidad**: toda afirmación del borrador lleva citas internas
  `[archivo.pdf | chunk_id]`; 07 construye una matriz de trazabilidad
  completa (`claim_traceability_rows` + `claim_evidence_traceability_rows`
  + `correction_traceability_rows`) que 08 audita.

## 8. Estructura real de carpetas

```text
tesis-sistema-multiagente-main/
├── README.md                    (este archivo)
├── LEEME_PRIMERO.md              (guía corta operativa)
├── COLAB_SMOKE_TEST.md           (procedimiento de smoke test real)
├── requirements.txt
├── smoke_test.py                 (contrato transaccional genérico, con dobles)
├── smoke_test_draft.py           (fallo de build_execution -> FAILED, con dobles)
├── src/
│   ├── adapters/                 conecta agentes/runtime real con StageSpec
│   ├── agents/                   lógica científica de cada agente (02-07)
│   ├── bootstrap/                preparación de proyecto/experimento
│   ├── capabilities/              capacidades reutilizables (extracción cuantitativa)
│   ├── config/                   políticas por etapa
│   ├── contracts/                AgentInput / AgentResult
│   ├── io/                       escritura atómica, credenciales
│   ├── orchestration/            StageSpec, run_stage, run_pipeline, decision_engine
│   ├── runtime/                  protocolos build_execution/runtime_transaction/resolve_resume
│   ├── state/                    StateStore, PipelineState, fingerprints
│   └── tools/
│       ├── draft_writing/        herramientas de la etapa 06
│       ├── evaluation/           módulos puros de la etapa 08 (ROUGE-L, BERTScore, LLM Judge, ...)
│       ├── extraction/           herramientas de la etapa 02
│       ├── outline_generation/   herramientas de la etapa 05
│       ├── quantitative_extraction/  herramientas de la etapa 03B
│       ├── thematic_analysis/    herramientas de la etapa 04
│       └── verification/         herramientas de la etapa 07
└── tests/
    ├── orchestration/            suites principales (auto-contenidas, ver sección 13)
    ├── evaluation/
    ├── verification/
    ├── fixtures/
    ├── integration/
    ├── v16/
    └── v17/
```

**Nota sobre nombres de carpeta**: las carpetas de herramientas usan el
nombre completo de cada etapa (`outline_generation`, `quantitative_extraction`,
`thematic_analysis`), no abreviaturas — es el nombre real en disco, verificado
en esta entrega.

## 9. Requisitos

- Python **3.11** (versión exacta validada en el smoke test real de
  Colab; el código en sí solo requiere `>= 3.10`).
- Entorno **aislado** vía `virtualenv` — nunca el Python global de Colab.
  El índice persistente real de Chroma no podía abrirse con `chromadb`
  `0.5.x` (`KeyError('_type')`); `requirements.txt`/`constraints-colab.txt`
  fijan `chromadb==1.5.9`, la única versión confirmada funcional.
- Versiones exactas confirmadas (`pip check` sin dependencias rotas):
  `langchain-core==0.3.86`, `langchain-openai==0.3.35`, `openai==1.109.1`,
  `chromadb==1.5.9`, `transformers==4.46.3`, `tokenizers==0.20.3`,
  `huggingface-hub==0.36.2`, `sentence-transformers==5.7.0`,
  `bert-score==0.3.13`, `numpy==2.4.6`, `pandas==3.0.5`,
  `scikit-learn==1.9.0`, `matplotlib==3.11.1`, `PyMuPDF==1.28.2`,
  `rouge-score==0.1.2`, `tabulate==0.10.0`, `cryptography==46.0.7`,
  `torch==2.13.0`. Ver `requirements.txt` (pines exactos) y
  `constraints-colab.txt` (mismas versiones, como restricción explícita
  adicional).

## 10. Instalación

Entorno aislado real (recomendado, el que efectivamente se validó):

```bash
git clone <repo> /content/proyecto_estado_arte
bash /content/proyecto_estado_arte/scripts/setup_colab.sh
```

`scripts/setup_colab.sh` crea el virtualenv en `/content/venv_estado_arte`
(configurable con `VENV_DIR`), instala `requirements.txt` con
`constraints-colab.txt` como restricción explícita (`-c`), corre
`pip check`, y verifica en caliente que `chromadb`/`transformers`/
`tokenizers` queden en las versiones exactas confirmadas. A partir de ahí,
todo se ejecuta con el Python del venv, nunca el global:

```bash
/content/venv_estado_arte/bin/python -m src.orchestration.pipeline_orchestrator --project-dir /content/proyecto_estado_arte
# o, equivalente, vía el wrapper que ya fija MPLBACKEND=Agg, el venv y
# --project-dir (usa THESIS_PROJECT_DIR/ESTADO_ARTE_PYTHON si se
# necesita otra ruta) -- cualquier flag adicional, incluido --start-stage,
# se pasa transparentemente al orquestador real:
python3 scripts/run_pipeline.py --start-stage 07_agente_verificador
```

Instalación mínima sin venv (solo para desarrollo/tests fuera de Colab,
sin garantía de las versiones exactas de Chroma/torch):

```bash
python3 -m pip install -r requirements.txt
```

Para el procedimiento completo de instalación + smoke test en Colab, ver
`COLAB_SMOKE_TEST.md`.

## 11. Configuración de `OPENAI_API_KEY`

`src/io/credentials.py` resuelve credenciales en este orden:

1. Variable de entorno `OPENAI_API_KEY`, si está definida.
2. Dentro de Google Colab, `google.colab.userdata` (el "Secrets" del
   notebook) — este camino se importa de forma diferida y solo se activa si
   el módulo `google.colab` existe.
3. Un archivo cifrado local (vía `cryptography.fernet.Fernet`), para uso
   fuera de Colab sin exponer la clave en texto plano.

Fuera de Colab, la vía más simple es:

```bash
export OPENAI_API_KEY="sk-..."
```

## 12. Formato esperado de `active_experiment.json`

Debe existir en la raíz de `PROJECT_DIR` (fuera de la carpeta del
experimento), con al menos:

```json
{
  "active_experiment_id": "exp_paper_02",
  "run_id": "run_2026_08_07",
  "evaluation_policy": { "...": "..." }
}
```

- `active_experiment_id` es **obligatorio** — sin él, `run_pipeline()`
  lanza `FileNotFoundError` indicando que falta correr la etapa 00.
- `run_id` es opcional; si falta, se usa el mismo valor que
  `active_experiment_id`.
- `evaluation_policy` es **obligatorio para llegar a la etapa 08** — debe
  ser un diccionario no vacío (`build_execution_for_stagespec` lo valida
  explícitamente y lanza `ValueError` si falta o está vacío). No tiene
  valores por defecto silenciosos.
- El directorio del experimento se resuelve como
  `PROJECT_DIR/{active_experiment_id}/`.

## 13. Cómo ejecutar pruebas

**No uses `pytest` para `tests/orchestration/`.** Verificado empíricamente:
esas suites usan un decorador `@scenario` que captura toda excepción
internamente y la registra en una lista `RESULTS` sin relanzarla — bajo
`pytest`, cada función `test_*` termina sin excepción y se marca `passed`
**aunque la aserción interna haya fallado realmente**. El corredor
confiable es ejecutar cada archivo directamente:

```bash
python3 tests/orchestration/test_writer_revision_cycle_core.py
python3 tests/orchestration/test_writer_verifier_cycle_e2e.py
python3 tests/orchestration/test_writer_verifier_round_conflict_integration.py
python3 tests/orchestration/test_agent07_manifest_conditional.py
python3 tests/orchestration/test_verification_claim_canonicalization.py
python3 tests/orchestration/test_verification_numeric_risk_characterization.py
python3 tests/orchestration/test_qualitative_correction_keyerror_and_return.py
```

Cada archivo imprime `N/N escenarios OK` y termina con código de salida 0
si todo pasó. Para correr **todas** las suites de `tests/orchestration/`:

```bash
for f in tests/orchestration/test_*.py; do python3 "$f" || echo "FALLÓ: $f"; done
```

Estado verificado en esta entrega: **468/468** en 37 suites.

## 14. Cómo ejecutar el pipeline

```bash
python3 -m src.orchestration.pipeline_orchestrator --project-dir /ruta/a/PROJECT_DIR
```

`run_pipeline()` no es un `for` fijo sobre las etapas: interpreta la
`RequestedTransition` real que devuelve cada etapa (`ADVANCE`, `RETRY`,
`RETURN`, `HALT_STAGE`, `STOP_PIPELINE`). Por eso el mismo comando cubre el
ciclo `06 ↔ 07` sin ningún flag especial: si 07 emite `RETURN`, el bucle
vuelve a 06 automáticamente; cuando 07 finalmente emite `ADVANCE`, sigue
hacia 08.

### Ejecutar hasta una etapa específica

```bash
python3 -m src.orchestration.pipeline_orchestrator --project-dir /ruta/a/PROJECT_DIR --until 07_agente_verificador
```

`--until` acepta cualquier clave de `STAGE_ORDER` y detiene el bucle apenas
esa etapa produce un resultado (incluso si pedía `ADVANCE`).

### Usar `--force-rerun`

```bash
python3 -m src.orchestration.pipeline_orchestrator --project-dir /ruta/a/PROJECT_DIR --force-rerun
```

Reejecuta la etapa inicial (`--until` o la primera) aunque ya esté
`COMPLETED` y vigente según fingerprints en `pipeline_state.json`. Las
etapas alcanzadas después por `ADVANCE`/`RETRY`/`RETURN` siguen evaluando
sus propios fingerprints normalmente — `--force-rerun` no las fuerza a
todas.

### Retomar el pipeline en una etapa específica: `--start-stage`

```bash
python3 -m src.orchestration.pipeline_orchestrator --project-dir /ruta/a/PROJECT_DIR --start-stage 07_agente_verificador
```

Arranca el recorrido directamente en la etapa indicada, en vez de la
primera de `STAGE_ORDER` — ejecuta **únicamente** esa etapa y las que
resulten de sus transiciones reales, nunca las anteriores. A diferencia
de `--force-rerun`, no ignora fingerprints por defecto: si la etapa ya
está `COMPLETED` y vigente, se reconoce `SKIPPED_FRESH` con normalidad; si
su último commit fue `FAILED` (ej. tras un `HALT_STAGE`), esto la
reintenta con un `decision_id` nuevo, sin tocar ninguna etapa previa. Es
la vía oficial para reintentar una sola etapa desde un estado terminal
(ver `scripts/run_pipeline.py`, que pasa cualquier flag adicional
transparentemente al orquestador real, incluido este).

## 15. Outputs principales

### Etapa 06 (borrador)

- `state_of_art_draft.json` — borrador con secciones, citas y claims.
- `draft_generation_manifest.json`, `draft_validation_report.json`,
  `draft_length_check.csv`, `draft_quality_check.csv`.

### Etapa 07 (verificación)

Cuatro artefactos científicos incondicionales, más uno condicional, en
`PROJECT_DIR/{active_experiment_id}/05_outputs/06_verification_traceability/`
(nombre de carpeta heredado del notebook original, no de la clave del
`StageSpec`):

```text
provisional_verification_traceability_bundle.json
multi_proposal_resolution_result.json
agent07_runtime_report.json
agent07_artifact_manifest.json
writer_revision_request.json      (SOLO si la transición es RETURN)
```

### Etapa 08 — los 15 outputs obligatorios de evaluación

En `PROJECT_DIR/{active_experiment_id}/05_outputs/07_evaluation/` (mismo
motivo de numeración heredada):

```text
 1. automatic_metrics.csv
 2. semantic_chunk_alignment.csv
 3. bertscore_chunk_alignment.csv
 4. factual_metrics.csv
 5. final_citation_check.csv
 6. final_claim_audit.csv
 7. llm_judge_evaluation.json
 8. llm_judge_scores.csv
 9. corpus_gap_suggestions.csv
10. corpus_gap_suggestions.md
11. final_selected_metrics.csv
12. evaluation_summary.json
13. final_evaluation_report.md
14. evaluation_validation_report.json
15. evaluation_manifest.json
```

`agent08_upstream_numeric_check.csv` **no** es uno de los 15 — es un
artefacto intermedio que 08 sobrescribe internamente, no un output final
auditado (`src/adapters/evaluation_persistence.py`,
`INTERMEDIATE_NUMERIC_CHECK_FILENAME`).

## 16. Ubicación de `pipeline_state.json`

```text
PROJECT_DIR/{active_experiment_id}/05_outputs/00_orchestrator_planner/pipeline_state.json
```

Es el estado transaccional canónico de **todo** el pipeline (todas las
etapas comprometidas, sus fingerprints, su `AgentResult` persistido).

## 17. Ubicación de `writer_verifier_cycle/round_NN`

```text
PROJECT_DIR/{active_experiment_id}/05_outputs/writer_verifier_cycle/round_01/
```

Cada ronda del ciclo 06↔07 vive en su propia carpeta numerada, con al
menos:

```text
writer_revision_request.json     (de 07, al crear la ronda)
input_draft_reference.json       (de 07)
agent07_result.json              (de 07)
transition.json                  (de 07)
fingerprints.json                (de 07)
revised_draft.json               (de 06, al completar la ronda)
revision_changelog.json          (de 06)
revision_resolution_matrix.json  (de 06)
unresolved_issues.json           (de 06)
fingerprint.json                 (de 06)
_round_status.json               (estado interno: AWAITING_REVISION -> REVISION_COMPLETED)
```

## 18. PREPARE / EXECUTE / COMMIT / RESUME

Cada etapa se ejecuta con el mismo protocolo transaccional
(`src/state/state_store.py`, `run_stage()` en
`src/orchestration/pipeline_orchestrator.py`):

1. **PREPARE**: `store.prepare_execution(...)` registra la intención de
   ejecutar, generando un `decision_id`. Si ya hay una ejecución pendiente
   sin comprometer, lanza `RuntimeError` — nunca hay dos PREPARE
   simultáneos sin resolver.
2. **EXECUTE**: se corre `build_execution` + la lógica real del agente
   (fuera del `StateStore`).
3. **Persistencia del resultado**: `store.persist_agent_result(...)` guarda
   el `AgentResult` en disco antes de comprometerlo — permite recuperarlo
   si el proceso muere entre EXECUTE y COMMIT.
4. **COMMIT**: `store.commit_execution(...)` es la única escritura que
   marca la etapa como `COMPLETED`/`FAILED` en `pipeline_state.json`,
   junto con sus fingerprints.
5. **RESUME**: si se vuelve a llamar `run_stage()` sobre una etapa con una
   ejecución pendiente sin comprometer, `resolve_resume` decide si el
   resultado ya persistido puede comprometerse directamente (sin
   reinvocar al agente/LLM) o si hace falta reejecutar.

## 19. Fingerprints y reconstrucción

`src/state/fingerprints.py` calcula un fingerprint compuesto por etapa a
partir de tres partes independientes: `input_fingerprint`,
`config_fingerprint`, `dependencies_fingerprint` (`build_stage_fingerprints`).
Si cualquiera de las tres cambia, el fingerprint compuesto cambia, y
`run_stage()` decide reconstruir esa etapa en vez de reutilizar el
resultado `COMPLETED` anterior (`SKIPPED_FRESH` solo ocurre cuando el
fingerprint compuesto coincide exactamente). El fingerprint aprobado de 07
se propaga a 08 como `upstream_fingerprint` en la firma de evaluación
(`src/adapters/evaluation_fingerprint.py`) — si 07 está comprometido pero
su fingerprint no existe, falla explícitamente en vez de degradarse a
`None` en silencio.

## 20. Limitaciones actuales

- La ruta real de 07/08 (con LLM y Chroma reales) **no se ha ejecutado
  todavía** en este entorno de desarrollo — toda la migración se validó con
  dobles deterministas (LLM, retriever, embeddings, BERTScore). El smoke
  test real en Colab es el primer punto donde se ejercita la integración
  con red real.
- El gate determinista de precheck para claims **cuantitativos**
  (`numeric_risk`) está caracterizado (ver
  `tests/orchestration/test_verification_numeric_risk_characterization.py`)
  pero no se investigó su relación completa con el notebook 03B ni se
  intentó resolver — queda como tarea independiente.
- 07C permanece en el código por compatibilidad histórica pero no forma
  parte del registro de etapas activo; su eliminación física es una
  migración separada.

## 21. Estado de validación

- **468/468** escenarios en 37 suites de `tests/orchestration/`, ejecutados
  con `python3 archivo.py` (no `pytest`), verificados dos veces: en el
  entorno de desarrollo y descomprimiendo el ZIP de entrega en un
  directorio nuevo.
- El ciclo cualitativo completo `06 → 07 (RETURN) → 06 (REVISION) → 07
  (ADVANCE)` está probado de punta a punta con código productivo real
  (`VerificationAgent`, `propose_correction`,
  `build_provisional_verification_traceability_bundle`,
  `DraftWritingAgent` en modo `REVISION`), sustituyendo únicamente LLM,
  Chroma y retriever por dobles deterministas.
- La etapa 08 completa (métricas automáticas, factuales, LLM Judge,
  persistencia de los 15 outputs, fingerprints, contrato transaccional,
  `StageSpec`) está migrada y probada con las mismas sustituciones.

## 22. Advertencia: smoke test real con Chroma y OpenAI

**Nada de lo anterior sustituye una corrida real.** Todas las pruebas de
este repositorio usan dobles deterministas para LLM, Chroma y retriever —
ninguna ejercita la integración real de red, autenticación, límites de tasa
o comportamiento no determinista de un LLM real. El primer punto donde eso
se valida es el smoke test real descrito en `COLAB_SMOKE_TEST.md`, que debe
ejecutarse en un entorno con credenciales de OpenAI y una colección Chroma
real.

## 23. Reproducibilidad

- Todas las funciones puras de `src/tools/` no leen ni escriben archivos ni
  llaman a red — reciben sus datos de entrada y devuelven estructuras en
  memoria, lo que las hace deterministas y testeables sin dobles.
- `langdetect` fija `DetectorFactory.seed = 0` para detección de idioma
  determinista (ver `src/tools/evaluation/language_preprocessing.py`).
- Los fingerprints (sección 19) permiten verificar si una corrida
  reproduce exactamente las mismas entradas/configuración/dependencias que
  una corrida previa.
- El componente **no determinista real** es el propio LLM (OpenAI): el
  pipeline no fija `temperature=0` de forma global ni cachea respuestas de
  producción; la reproducibilidad exacta del texto generado no está
  garantizada entre corridas con LLM real.

## 24. Relación con la tesis

Este repositorio es el soporte de implementación de la tesis. El diseño
multiagente, el ciclo correctivo 06↔07, y las métricas de evaluación de la
etapa 08 corresponden directamente a los objetivos específicos y la
metodología descritos en el documento de tesis. Los notebooks operativos
originales (00-08) son la fuente de referencia científica original de este
proyecto y no forman parte de este repositorio; este repositorio contiene
la migración a un sistema transaccional productivo (`StateStore`,
`StageSpec`, contratos `AgentInput`/`AgentResult`) sobre esa misma base
científica, sin alterar las decisiones científicas de los notebooks
originales salvo donde se documenta explícitamente lo contrario (ver
secciones 6 y 20).

## 25. Evaluación 08: `SUCCESS` / `PARTIAL_HALT`

08 acepta exactamente dos tipos de entrada científica desde 07 — nunca un
fallo técnico real (`execution_status=FAILED`, errores de runtime/
contrato/artefactos, que jamás son evaluables):

1. 07 `COMPLETED` + `APPROVED`/`APPROVED_WITH_WARNINGS` + `ADVANCE → 08` →
   `pipeline_outcome = "SUCCESS"` (`verification_approved=true`,
   `autonomous_convergence=true`, `human_review_required=false`).
2. 07 `COMPLETED` + `NEEDS_REVISION` + `HALT_STAGE` por agotamiento
   científico (ej. `WRITER_VERIFIER_MAX_ROUNDS_EXHAUSTED`, 3/3 rondas
   usadas sin aprobación completa) → `pipeline_outcome = "PARTIAL_HALT"`
   (`verification_approved=false`, `autonomous_convergence=false`,
   `human_review_required=true`, con `rounds_used`/`max_rounds` y el
   `reason_code` real de 07 registrados).

El segundo camino **nunca se activa automáticamente**: requiere
`"allow_partial_halt_evaluation": true` explícito en
`active_experiment.json["evaluation_policy"]`. El `decision_log` de 07
nunca se reescribe — su entrada sigue diciendo `HALT_STAGE` igual que
antes; 08 solo decide, como una decisión propia y separada, que ese
`HALT_STAGE` específico es evaluable bajo el flag explícito. Ver
`src/adapters/evaluation_pipeline_outcome.py`.

08 lee el draft canónico y sus artefactos usando la misma resolución
causal que 07 (`resolve_committed_agent06_artifacts`, ver sección 6) —
nunca una ruta fija — y valida el `source_draft_fingerprint` contra lo que
07 realmente comprometió, para cualquier draft, incluidos los publicados
por el bootstrap de identidad. Ground Truth se resuelve **exclusivamente**
en esta etapa (`src/tools/evaluation/ground_truth.py`); ninguna otra etapa
(02-07) lee ni recibe contenido de Ground Truth — reforzado con listas de
rechazo explícitas en varios módulos de `src/tools/` además de la
validación de política en `src/adapters/verification_orchestrator_runtime.py`.
07C sigue excluido también aquí (nunca se pasa `agent07c_directory`).

## 26. Módulos de configuración planos en `src/` (duplicación conocida)

`src/config.py`, `src/experiment_config.py`, `src/generation_config.py`,
`src/rag_policy.py`, `src/rag_utils.py`, `src/llm_utils.py`,
`src/prompts.py`, `src/pdf_utils.py`, `src/io_utils.py` son módulos
**planos**, en la raíz de `src/`, que las celdas de los notebooks
operativos originales (`00_setup_config` y `07_agente_verificador`)
escriben/leen directamente — **no** forman parte del sistema del
orquestador (`src/orchestration/`, `src/adapters/`, `src/agents/`,
`src/tools/`) y ningún módulo de ese sistema los importa. Son dos capas
de configuración independientes que hoy coinciden (ambas aplican la misma
política de aislamiento de Ground Truth, por ejemplo) pero que **podrían
desincronizarse** si una se edita sin la otra. Se conservan porque
notebooks reales todavía los usan; la fuente de verdad para todo lo que
ejecuta el orquestador (03→08 vía `StateStore`/`StageSpec`) es siempre
`active_experiment.json`, nunca estos módulos.
