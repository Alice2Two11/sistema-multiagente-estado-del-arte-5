# ============================================================
# 04 - AGENTE DE ANÁLISIS TEMÁTICO
# ============================================================

# Analiza los papers seleccionados para identificar temas,
# comparaciones, brechas de investigación y estructura temática.

from __future__ import annotations

from datetime import datetime, timezone
import json

from src.contracts.agent_result import *
from src.tools.thematic_analysis import *


class ThematicAnalysisAgent:

    def __init__(self, dependencies):
        self.dependencies = dependencies

    def execute(self, agent_input):
        start = datetime.now(timezone.utc).isoformat()
        llm_calls = 0 #Inicializa en cero el contador de llamadas al LLM.

        try:
            #Comprueba que realmente se esté ejecutando la etapa 04. Si no, detiene la ejecución por configuración incorrecta.
            if agent_input.stage_name != "04_agente_analisis_tematico": 
                raise ValueError("INVALID_CONFIGURATION")

            #Permite solamente el intento 1 o 2
            if agent_input.attempt_number not in (1, 2):
                raise ValueError("INVALID_CONFIGURATION")

            bundle, df, m03 = validate_dependencies(agent_input) #Valida y carga las dependencias que necesita el agente.

            final, excluded = filter_corpus(df) #Filtra el corpus. final → papers que sí se usarán; excluded → papers que fueron descartados.
            
            #Añade al corpus la información cuantitativa proveniente del 03B. final → corpus enriquecido con información cuantitativa; qmeta → metadatos sobre esa integración; qwarn → advertencias encontradas durante la integración.
            final, qmeta, qwarn = integrate_quantitative_context(final,bundle)
            # Reduce el contenido de los papers a la información más relevante para que el análisis temático no reciba textos demasiado largos.
            context = compact_for_thematic_analysis(final,int(agent_input.policy.get("max_field_chars", 3500)))
            # Crea un conjunto con los nombres de archivo válidos de los papers que quedaron en el corpus final.
            valid = set(final.source_filename.astype(str))
            # Crea un diccionario que relaciona cada archivo válido con el título correspondiente de su paper.
            title_map = dict(zip(final.source_filename.astype(str),final.title.astype(str)))
            
            # Si este es el segundo intento, crea un plan de reparación usando los errores detectados en el intento anterior.
            repair_plan = (
                build_repair_plan(
                    agent_input.previous_attempt.failure_reason_codes #Toma los códigos de error del intento anterior.
                    if agent_input.previous_attempt #Si sí existe un intento anterior, usa sus errores. Si no existe, usa una lista vacía.
                    else ()
                )
                if agent_input.attempt_number == 2
                else None
            )

            # Obtiene de la política el número mínimo y máximO de secciones sugeridas para la estructura del estado del arte.
            min_s = agent_input.policy.get("min_sections")
            max_s = agent_input.policy.get("max_sections")

            # Indica si el número mínimo y máximo de secciones debe cumplirse obligatoriamente o solo tomarse como referencia.
            enforce_section_count = bool(
                agent_input.policy
                .get("structure_policy", {})
                .get("enforce_section_count", False)
            )

            # Construye el prompt que se enviará al LLM con el contexto las referencias válidas, posibles reparaciones y reglas de estructura.
            prompt = self.dependencies.build_prompt(
                context,
                list(valid),
                title_map,
                repair_plan,
                min_sections=min_s,
                max_sections=max_s,
                enforce_section_count=enforce_section_count
            )

            raw = self.dependencies.invoke(prompt)
            llm_calls = 1

            # Convierte la respuesta del LLM a datos que Python pueda usar y cuenta lo que generó antes de hacer correcciones.
            payload = self.dependencies.parse(raw)
            raw_counts = inspect_thematic_payload(payload)
            # Ordena la respuesta del LLM según la estructura esperada y registra los problemas o correcciones que encuentre.
            data, schema_issues, alias_repairs = normalize_thematic_output(payload,return_repairs=True)
            # Aplica correcciones automáticas y reproducibles a la salida temática, usando los títulos y la lista de papers válidos.
            data, deterministic_repairs = apply_deterministic_repairs(data,title_map,valid)
            # Une en una sola lista todas las correcciones realizadas.
            repairs = alias_repairs + deterministic_repairs
            # Verifica que las referencias usadas en el análisis temático correspondan realmente a papers válidos del corpus.
            ref_codes, counts, _ = validate_references(data,final)
            # Cuenta cuántos elementos quedaron en cada tabla después de organizar el análisis temático.
            table_counts = thematic_table_counts(data)
            # Comprueba que la cantidad de elementos del JSON original coincida con la cantidad que terminó en las tablas.
            flattening_codes, consistency = validate_json_to_tables(raw_counts,table_counts)
            # Junta en una sola lista todos los errores detectados durante la validación del análisis temático.
            codes = ([x["code"] for x in schema_issues]+ ref_codes+ flattening_codes)

            # Detecta referencias inválidas y verifica que el análisis temático
            # tenga temas, brechas de investigación y dimensiones comparativas.
            if any(
                r.get("type") == "INVALID_REFERENCE_REMOVED"
                for r in repairs
            ):
                codes.append("INVALID_REPRESENTATIVE_SOURCE")
                
            if not data["themes"]:
                codes.append("EMPTY_THEMATIC_OUTPUT")

            if not data["research_gaps"]:
                codes.append("EMPTY_THEMATIC_OUTPUT")

            if not data["comparative_dimensions"]:
                codes.append("EMPTY_THEMATIC_OUTPUT")

            sc = len(data["suggested_state_of_art_structure"])

            # Comprueba si la estructura propuesta tiene menos o más
            # secciones de las indicadas en la política.
            structure_too_short = (min_s is not None and sc < int(min_s))
            structure_too_long = (max_s is not None and sc > int(max_s))

            # Si la cantidad de secciones es obligatoria, registra un error cuando queda por debajo o por encima del límite.
            if enforce_section_count:
                if structure_too_short:
                    codes.append("STRUCTURE_TOO_SHORT")
                if structure_too_long:
                    codes.append("STRUCTURE_TOO_LONG")

            # Calcula métricas para evaluar la calidad del análisis temático y comprobar si cumple con la estructura esperada.
            metrics = calculate_diagnostic_metrics(
                data,
                final,
                counts,
                min_sections=min_s,
                max_sections=max_s,
                enforce_section_count=enforce_section_count
            )

            # Comprueba si el plan de reparación realmente se aplicó, elimina errores duplicados y decide la calidad y la acción siguiente.
            if repair_plan and not repairs:
                codes.append("REPAIR_PLAN_NOT_APPLIED")
            codes = tuple(
                dict.fromkeys(codes)
            )
            quality, action = classify_quality(
                codes,
                agent_input.attempt_number,
                bool(
                    agent_input.policy
                    .get("manual_review_policy", {})
                    .get("allowed", True)
                )
            )

            # Guarda en una sola estructura el resultado de todas las validaciones realizadas sobre el análisis temático.
            validation = {
                "validation_ok": not codes, #si la validación salió bien.
                "failure_reason_codes": list(codes), #Guarda la lista de problemas encontrados.
                "metrics": metrics, #Guarda las métricas calculadas para evaluar el análisis temático.
                "repairs": repairs, #Guarda todas las correcciones que se hicieron.
                "repair_plan": repair_plan or [], #Guarda el plan de reparación del segundo intento. Si no existe, guarda una lista vacía.
                "json_to_tables_consistency": consistency,
                **qmeta,
            }

            ## Guarda la información necesaria para identificar, auditar y reproducir esta ejecución del análisis temático.
            manifest = {
                "stage": agent_input.stage_name, #Guarda qué etapa produjo el resultado.
                "experiment_id": agent_input.experiment_id, #Guarda a qué experimento pertenece.
                "run_id": agent_input.run_id, #Guarda a qué ejecución específica pertenece.
                "attempt_number": agent_input.attempt_number, # Registra si fue el intento 1 o el intento 2.
                "quality_status": quality.value, #Guarda el estado de calidad final del análisis temático.
                "safety_policy": {
                    "uses_ground_truth": False, # No utilizó el estado del arte real o Ground Truth.
                    "uses_external_knowledge": False, # No utilizó conocimiento externo al corpus.
                    "uses_review_sections": False, 
                    "uses_bibliography": False, # No utilizó la bibliografía como fuente adicional de conocimiento.
                },
                "diagnostic_metrics": metrics,
                "quantitative_context": qmeta,
            }

            # Guarda en archivos los resultados finales del análisis temático,
            # junto con la validación y la información de la ejecución.
            artifacts = write_thematic_artifacts(
                agent_input.agent_context.output_directory,
                final,
                excluded,
                raw
                if isinstance(raw, str)
                else json.dumps(raw, ensure_ascii=False),
                data,
                validation,
                manifest
            )

            warnings = tuple(
                AgentWarning(
                    code=c,
                    severity=WarningSeverity.WARNING,
                    blocking=codes != (),
                    message=c
                )
                for c in tuple(qwarn) + codes
            )

            # Convierte los problemas detectados en advertencias
            # que el sistema puede registrar y mostrar.
            transition = RequestedTransition(
                action=TransitionAction(action),
                target_stage=None,
                reason_code=quality.value,
                requires_human_confirmation=(quality== QualityStatus.APPROVED_PENDING_MANUAL_REVIEW)
            )

            # Devuelve al orquestador el resultado final del agente 04 con su calidad, métricas, advertencias, artefactos y transición.
            return AgentResult(
                execution_status=ExecutionStatus.COMPLETED,
                quality_status=quality,
                decision=DecisionInfo(
                    code="THEMATIC_ANALYSIS_EVALUATED",
                    rationale="Análisis temático validado contra corpus cerrado."
                ),
                quality_metrics={
                    "scientific": metrics,
                    "technical": qmeta
                },
                warnings=warnings,
                failure_reason_codes=codes,
                requested_transition=transition,
                output_artifacts=artifacts,
                tool_usage=ToolUsage(
                    llm_calls=llm_calls,
                    validation_calls=1
                ),
                attempt_number=agent_input.attempt_number,
                started_at=start,
                completed_at=datetime.now(
                    timezone.utc
                ).isoformat()
            )

        # Captura cualquier error de la etapa 04
        # y revisa si corresponde a alguno de los errores conocidos.
        except Exception as exc:
            msg = str(exc)

            known = [
                c
                for c in [
                    "KB_NOT_FOUND",
                    "KB_JSONL_NOT_FOUND",
                    "SCIENTIFIC_EXTRACTION_MANIFEST_NOT_FOUND",
                    "SCIENTIFIC_EXTRACTION_MANIFEST_MISMATCH",
                    "DEPENDENCY_HASH_MISMATCH",
                    "INVALID_KB_SCHEMA",
                    "EMPTY_THEMATIC_CORPUS",
                    "QUANTITATIVE_MANIFEST_NOT_FOUND",
                    "QUANTITATIVE_MANIFEST_MISMATCH",
                    "QUANTITATIVE_ARTIFACT_NOT_FOUND",
                    "INVALID_QUANTITATIVE_CONTEXT",
                    "GROUND_TRUTH_POLICY_VIOLATION",
                    "CREDENTIAL_NOT_FOUND",
                    "INVALID_CONFIGURATION",
                ]
                if c in msg
            ]

            code = (
                known[0]
                if known
                else (
                    "DEPENDENCY_NOT_FOUND"
                    if isinstance(exc, FileNotFoundError)
                    else "RUNTIME_DEPENDENCY_FAILED"
                )
            )

            # Si la etapa 04 falla, devuelve un resultado de error,
            # registra la causa y solicita detener esta etapa.
            return AgentResult(
                execution_status=ExecutionStatus.FAILED,
                quality_status=QualityStatus.REJECTED,
                decision=DecisionInfo(
                    code="THEMATIC_ANALYSIS_FAILED",
                    rationale="Falló la ejecución de 04."
                ),
                quality_metrics={
                    "scientific": {},
                    "technical": {}
                },
                warnings=(
                    AgentWarning(
                        code=code,
                        severity=WarningSeverity.ERROR,
                        blocking=True,
                        message=msg
                    ),
                ),
                failure_reason_codes=(code,),
                requested_transition=RequestedTransition(
                    action=TransitionAction.HALT_STAGE,
                    target_stage=None,
                    reason_code=code,
                    requires_human_confirmation=False
                ),
                output_artifacts={},
                tool_usage=ToolUsage(
                    llm_calls=llm_calls
                ),
                attempt_number=agent_input.attempt_number,
                started_at=start,
                completed_at=datetime.now(
                    timezone.utc
                ).isoformat(),
                error={
                    "type": type(exc).__name__,
                    "message": msg,
                    "stage": agent_input.stage_name
                }
            )
