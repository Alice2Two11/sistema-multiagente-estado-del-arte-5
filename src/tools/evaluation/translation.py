"""Bloque 3 (parte b) de la migración de la etapa 08: traducción por chunks
para ROUGE, con el LLM inyectable.

Copias LITERALES de la celda 17 del notebook 08
(``08_evaluacion_12_CITAS_Y_NUMEROS_CORREGIDOS.ipynb``). Mismo patrón de
separación que ya usa el repo en ``src/tools/verification/prompting.py``:
las funciones de aquí construyen el prompt y procesan la respuesta, pero
NO construyen el cliente LLM ellas mismas.

**Corrección de esta ronda — ciclo de vida del LLM**: el notebook real
llama a ``get_llm(model=OPENAI_MODEL, temperature=TRANSLATION_TEMPERATURE)``
DENTRO del bucle, una vez por cada chunk — no reutiliza un único cliente
para todos los fragmentos. La versión anterior de este módulo recibía un
único objeto ``llm`` inyectado y lo reutilizaba en todas las iteraciones,
lo cual no reproducía ese ciclo de vida literalmente (aunque un
``ChatOpenAI`` real no mantiene estado conversacional entre llamadas, así
que el resultado observable no cambiaba — pero el pedido es correcto en
que no era una extracción literal). Corregido: ahora se recibe
``llm_factory: Callable[[], Any]`` y se llama ``llm_factory()`` en cada
iteración del bucle, construyendo una instancia nueva por chunk — igual
que el notebook real construye un ``ChatOpenAI`` nuevo por chunk vía
``get_llm(...)``.

Mapa función → celda original
------------------------------
| Función | Celda | Dependencias | Entradas | Salidas | Excepciones | Archivos |
|---|---|---|---|---|---|---|
| ``build_translation_prompt`` | 17 | ninguna (nueva: el notebook arma el prompt inline dentro de ``translate_text_to_language``, no como función separada — se extrajo aquí para poder probarlo y reutilizarlo sin duplicar el f-string) | ``chunk: str``, ``target_language_code: str`` | prompt (str), idéntico byte a byte al que arma el notebook | ninguna | ninguno |
| ``translate_text_to_language`` | 17 | ``chunk_text_by_sentences``, ``build_translation_prompt``, ``safe_str``, un ``llm_factory`` inyectado que devuelve un objeto con ``.invoke()`` | ``text``, ``target_language_code``, ``llm_factory``, ``max_chars_per_chunk: int`` (parametrizado; el notebook usa el global ``MAX_TRANSLATION_CHARS_PER_CHUNK``) | texto traducido completo (str) | ``ValueError`` si algún chunk traduce a vacío; ``ValueError`` si la proporción de palabras traducidas/originales queda fuera de ``[0.35, 2.75]``; cualquier excepción que ``llm_factory()`` o ``.invoke()`` propaguen (el notebook no captura ni reintenta ninguna de las dos) | ninguno (la llamada real a OpenAI la hace el cliente construido por ``llm_factory``, no esta función) |
| ``resolve_generated_text_for_rouge`` | 17 (bloque de decisión, código de módulo — no una función nombrada en el notebook, ver nota abajo) | ``translate_text_to_language`` | ver docstring | ``(texto_para_rouge, translation_mode)`` | las de ``translate_text_to_language`` | ninguno — el caching en disco (``TRANSLATED_GENERATED_TEXT_PATH``/``TRANSLATION_MANIFEST_PATH``) es persistencia y queda fuera de este bloque (Bloque 6) |

La factory productiva (no incluida aquí — vive donde se ensamblen las
dependencias reales, junto al resto de construcción de ``ChatOpenAI`` de
07/LLM Judge) debe construir el cliente con exactamente ``OPENAI_MODEL`` y
``TRANSLATION_TEMPERATURE`` resueltos por el llamador, igual que
``get_llm(model=OPENAI_MODEL, temperature=TRANSLATION_TEMPERATURE)`` real —
esta función no fija ningún valor de modelo/temperatura por sí misma, los
recibe ya resueltos en cada llamada a la factory que le pasan.

No se agregan reintentos: el notebook real no reintenta ni la construcción
del cliente ni la llamada ``.invoke()`` — cualquier fallo de cualquiera de
las dos se propaga tal cual, sin capturarlo.

Nota sobre ``resolve_generated_text_for_rouge``: en el notebook real es un
bloque de código a nivel de módulo (``if TRANSLATE_FOR_ROUGE and
generated_language != ground_truth_language: ... else: ...``), no una
función. Se envolvió aquí por la misma razón que
``resolve_ground_truth_comparable_text`` en el Bloque 2. **Se reproduce
solo la decisión y la traducción en sí — NO el cacheo en disco
(``translation_cache_valid``/``TRANSLATED_GENERATED_TEXT_PATH``/
``TRANSLATION_MANIFEST_PATH``/``FORCE_REBUILD_EVALUATION``), que es
persistencia y pertenece al Bloque 6 (runtime transaccional).** Por eso
esta función siempre traduce quando corresponde, sin consultar ni escribir
ningún caché — el llamador (todavía no migrado) es responsable de decidir
si reutiliza una traducción cacheada antes de invocar esto.

**Dirección de traducción confirmada por lectura exacta de la celda 17**:
el notebook real SOLO traduce el texto GENERADO, nunca el Ground Truth
(``generated_for_rouge = translate_text_to_language(generated_plain_text,
ground_truth_language)`` — el idioma objetivo siempre es el del Ground
Truth). No existe ninguna rama que traduzca el Ground Truth.

Importar este módulo no carga modelos, no llama a OpenAI (solo importa
``HumanMessage``, una clase de datos sin efectos secundarios, no un
cliente) y no lee archivos. La llamada real a OpenAI ocurre únicamente
cuando el llamador invoca ``translate_text_to_language`` con un
``llm_factory`` real ya construido — nunca por importar este módulo.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage

from src.tools.evaluation.language_preprocessing import chunk_text_by_sentences
from src.tools.evaluation.text_normalization import safe_str


def build_translation_prompt(chunk: str, target_language_code: str) -> str:
    return f"""
Translate the following academic text into language code
"{target_language_code}".

Rules:
1. Preserve the scientific meaning and technical terminology.
2. Do not summarize, expand, explain, or add facts.
3. Preserve paragraph and sentence boundaries when possible.
4. Return only the translated text.

TEXT:
{chunk}
""".strip()


def translate_text_to_language(
    text: str,
    target_language_code: str,
    *,
    llm_factory: Callable[[], Any],
    max_chars_per_chunk: int,
) -> str:
    translation_chunks = chunk_text_by_sentences(
        text, max_chars_per_chunk, overlap_chars=0
    )

    translated_parts = []
    for chunk_index, chunk in enumerate(translation_chunks, start=1):
        prompt = build_translation_prompt(chunk, target_language_code)

        llm = llm_factory()  # una instancia nueva por chunk, igual que get_llm(...) real
        response = llm.invoke([HumanMessage(content=prompt)])
        translated = safe_str(response.content)

        if not translated:
            raise ValueError(
                f"La traducción devolvió un fragmento vacío: {chunk_index}."
            )

        translated_parts.append(translated)

    translated_text = "\n\n".join(translated_parts).strip()

    source_word_count = max(len(text.split()), 1)
    translated_word_count = len(translated_text.split())
    ratio = translated_word_count / source_word_count

    if not 0.35 <= ratio <= 2.75:
        raise ValueError(
            f"La traducción tiene una proporción de longitud anómala: {ratio:.3f}."
        )

    return translated_text


def resolve_generated_text_for_rouge(
    *,
    generated_plain_text: str,
    generated_language: str,
    ground_truth_language: str,
    translate_for_rouge: bool,
    llm_factory: Callable[[], Any],
    max_chars_per_chunk: int,
) -> tuple[str, str]:
    """Reproduce la decisión real de la celda 17 (sin el cacheo en disco —
    ver nota del módulo). Devuelve ``(texto_para_rouge, translation_mode)``,
    con ``translation_mode`` en
    ``{"new_translation", "not_required_same_language"}`` — el notebook
    real también puede producir ``"cached_translation"``, pero esa rama es
    puramente de caché en disco y no aplica aquí.
    """

    if translate_for_rouge and generated_language != ground_truth_language:
        generated_for_rouge = translate_text_to_language(
            generated_plain_text,
            ground_truth_language,
            llm_factory=llm_factory,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        translation_mode = "new_translation"
    else:
        generated_for_rouge = generated_plain_text
        translation_mode = "not_required_same_language"

    return generated_for_rouge, translation_mode
