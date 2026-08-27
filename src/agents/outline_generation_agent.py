# ============================================================
# 05 - AGENTE GENERADOR DE ESQUEMA
# Genera, valida y guarda la estructura del estado del arte
# ============================================================

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from src.contracts.agent_result import *
from src.tools.outline_generation import *


class OutlineGenerationAgent:
    def __init__(self, runtime):
        self.runtime = runtime
    def execute(self, agent_input):
        start = datetime.now(timezone.utc).isoformat()
        llm_calls = 0
        try:
            bundle = validate_outline_dependencies(agent_input) #valida y carga lo que necesita la etapa 05
            ctx = build_outline_context(bundle, agent_input) #construye el contexto que usará el generador de esquema
            valid = set(ctx["valid_source_filenames"]) #guarda los nombres de los papers válidos
            tm = dict(zip(bundle["kb"].source_filename.astype(str),bundle["kb"].title.astype(str))) #crea un diccionario archivo → título del paper
            title_to_source = {v: k for k, v in tm.items()} #crea el diccionario inverso: título → archivo
            out = Path(agent_input.agent_context.output_directory) #obtiene la carpeta donde se guardarán los resultados
            manifest_path = out / "outline_generation_manifest.json" #define dónde está el manifiesto del agente 05
            reuse = False #inicialmente asume que no reutilizará una ejecución anterior
            raw = ""

            if manifest_path.exists() and not agent_input.policy["force_rebuild"]:
                try:
                    pm = json.loads(manifest_path.read_text())
                    reuse = (
                        pm.get("fingerprint") == agent_input.policy.get("current_fingerprint") #verifica que las entradas no hayan cambiado
                        and all((out / n).exists() for n in NAMES) #comprueba que todos los archivos esperados sigan existiendo
                        and bool(pm.get("validation_report", {}).get("validation_ok")) #verifica que el esquema anterior haya sido validado correctamente
                    )

                except Exception:
                    reuse = False

            if reuse:
                outline = json.loads((out / "state_of_art_outline.json").read_text()) #carga el esquema ya generado desde state_of_art_outline.json
                raw = (out / "state_of_art_outline_raw.txt").read_text() #carga también la respuesta original del LLM guardada en state_of_art_outline_raw.txt

                if not isinstance(outline, dict):
                    reuse = False

            # Si no puede reutilizar un esquema anterior, genera uno nuevo con el LLM
            # y convierte la respuesta a la estructura que usará el agente.
            if not reuse:
                raw = self.runtime.invoke(build_outline_generation_prompt(ctx))
                llm_calls = 1
                outline = self.runtime.parse(raw)

            if not isinstance(outline, dict):
                raise ValueError("INVALID_LLM_OUTPUT")

            # Revisa y corrige las fuentes utilizadas en el esquema,
            # asegurando que correspondan a papers válidos del corpus.
            sr, us = repair_outline_sources(
                outline, #esquema generado
                valid, #lista de papers permitidos
                tm, #relación archivo → título
                title_to_source, #relación título → archivo
                float(agent_input.policy.get("title_match_cutoff", 0.55))# qué tan parecido debe ser un título para considerarlo una coincidencia; por defecto usa 0.55
            )

            # Revisa y corrige el resumen de cobertura del esquema,
            # asegurando que los papers mencionados correspondan al corpus válido.
            cr, uc = repair_coverage_summary(
                outline, #el esquema generado.
                valid, #papers permitidos
                tm, #relación archivo → título
                title_to_source, #relación título → archivo
                float(agent_input.policy.get("title_match_cutoff", 0.55)) #nivel mínimo de similitud aceptado para reconocer un título
            )

            # Obtiene la lista de temas generados por el agente 04.
            # Si no existe una estructura temática válida, usa una lista vacía.
            themes = (
                bundle["thematic"].get("themes", [])
                if isinstance(bundle.get("thematic"), dict)
                else []
            )

            # Intenta completar las secciones que quedaron sin papers
            # usando únicamente los temas válidos provenientes del agente 04.
            section_papers_repairs, sections_without_upstream_evidence = (
                repair_empty_section_papers(outline, themes, valid)
            )

            # Valida el esquema generado usando los papers permitidos,
            # los límites de secciones y los resultados de las reparaciones previas.
            validation = validate_outline(
                outline,
                valid,
                int(agent_input.policy.get("min_sections", 4)),
                int(agent_input.policy.get("max_sections", 5)),
                sr,us,cr,uc
            )

            # Añade a la validación información de trazabilidad sobre el experimento,
            # la versión usada y las reparaciones realizadas sobre las secciones.
            validation.update({
                "experiment_id": agent_input.experiment_id,
                "validation_version": agent_input.policy.get("validation_version"),
                "section_papers_repairs": section_papers_repairs,
                "sections_without_upstream_evidence": sections_without_upstream_evidence
            })

            codes = reason_codes(validation)


            # Clasifica la calidad del esquema según el resultado de la validación:
            # aprobado si todo está correcto; si no, necesita revisión.
            quality = (
                QualityStatus.APPROVED
                if validation["validation_ok"]
                else QualityStatus.NEEDS_REVISION
            )

            # Decide qué debe hacer el agente después de validar el esquema:
            # avanzar, reintentar una vez o detener la etapa.
            if quality is QualityStatus.APPROVED:
                action = TransitionAction.ADVANCE
            elif agent_input.attempt_number == 1:
                action = TransitionAction.RETRY
            else:
                action = TransitionAction.HALT_STAGE

            # Crea el manifiesto de la etapa 05 con información de trazabilidad,
            # reutilización de resultados y estado de la ejecución.
            manifest = {
                "stage": agent_input.stage_name,
                "experiment_id": agent_input.experiment_id,
                "run_id": agent_input.run_id,
                "attempt_number": agent_input.attempt_number,
                "fingerprint": agent_input.policy.get("current_fingerprint"),
                "automatic_decision": {
                    "status": "outputs_are_current" if reuse else "rebuild_executed",
                    "rebuild_executed": not reuse
                },

                # Guarda en el manifiesto las restricciones de generación
                # configuradas para esta ejecución del esquema.
                "generation_constraints": {
                    k: agent_input.policy.get(k)
                    for k in [
                        "length_profile",
                        "min_sections",
                        "max_sections",
                        "output_language",
                        "writing_mode",
                        "focus_mode",
                        "citation_style"
                    ]
                },

                # Registra las reglas de seguridad aplicadas durante la generación del esquema,
                # dejando claro qué fuentes puede y no puede utilizar el agente 05.
                "safety_policy": {
                    "uses_kb_final_for_thematic_analysis": True,
                    "uses_full_kb_for_outline_generation": False,
                    "uses_ground_truth": False,
                    "uses_raw_pdfs": False, #no vuelve a leer directamente los PDFs originales.
                    "external_knowledge_allowed": False,
                    "source_filenames_validated_and_repaired": True
                },

                "validation_report": validation,

                # Guarda un resumen numérico del esquema generado: cuántas secciones tiene y cuántos papers están disponibles y usados.
                "counts": {
                    "sections": len(outline.get("sections", [])),
                    "papers_available": len(valid),
                    "papers_used": validation.get("papers_used_count", 0)
                }
            }

            # Guarda los resultados generados por la etapa 05
            artifacts = write_outline_artifacts(out,outline,raw,validation,manifest)
            # convierte los códigos de validación en advertencias del agente.
            warn = tuple(
                AgentWarning(code=c,severity=WarningSeverity.WARNING,blocking=True,message=c)
                for c in codes
            )

            # Devuelve al orquestador el resultado final de la etapa 05,
            # incluyendo calidad, validación, advertencias, artefactos y transición.
            return AgentResult(
                execution_status=ExecutionStatus.COMPLETED,
                quality_status=quality, #indica si el esquema quedó APPROVED o NEEDS_REVISION.
                decision=DecisionInfo(
                    code="OUTLINE_GENERATION_EVALUATED",
                    rationale="Esquema generado y validado preservando comportamiento del notebook 05."
                ), #registra formalmente que el esquema fue generado y evaluado
                quality_metrics={"scientific": {},"technical": {"validation_ok": validation["validation_ok"],"reused": reuse}}, #guarda si pasó la validación y si el esquema fue reutilizado
                warnings=warn, #entrega las advertencias encontradas.
                failure_reason_codes=codes, #entrega los códigos que explican los problemas detectados.
                requested_transition=RequestedTransition(
                    action=action,
                    target_stage=None,
                    reason_code=quality.value,
                    requires_human_confirmation=False
                ), #le dice al orquestador si debe ADVANCE, RETRY o HALT_STAGE.
                output_artifacts=artifacts,
                tool_usage=ToolUsage(llm_calls=llm_calls,validation_calls=1), #registra cuántas llamadas al LLM y validaciones se hicieron.
                attempt_number=agent_input.attempt_number, #guarda si fue intento 1 o 2.
                started_at=start,
                completed_at=datetime.now(timezone.utc).isoformat()
            )

        # Si ocurre un error durante la etapa 05, identifica su causa,
        # registra el fallo y solicita detener la ejecución de esta etapa.
        except Exception as exc:
            msg = str(exc)
            known = [
                "OUTLINE_INPUT_NOT_FOUND",
                "INVALID_THEMATIC_ANALYSIS_INPUT",
                "THEMATIC_MANIFEST_MISMATCH",
                "DEPENDENCY_HASH_MISMATCH",
                "EMPTY_OUTLINE_KB",
                "INVALID_OUTLINE_KB_SCHEMA",
                "INVALID_SOURCE_TITLE",
                "GROUND_TRUTH_POLICY_VIOLATION",
                "INVALID_CONFIGURATION",
                "INVALID_LLM_OUTPUT"
            ]
            code = next((x for x in known if x in msg),"RUNTIME_DEPENDENCY_FAILED")
            return AgentResult(
                execution_status=ExecutionStatus.FAILED,
                quality_status=QualityStatus.REJECTED,
                decision=DecisionInfo(code="OUTLINE_GENERATION_FAILED",rationale="Falló la ejecución del generador de esquema."),
                quality_metrics={"scientific": {},"technical": {}},
                warnings=(
                    AgentWarning(code=code,severity=WarningSeverity.ERROR,blocking=True,message=msg),
                ),
                failure_reason_codes=(code,),
                requested_transition=RequestedTransition(
                    action=TransitionAction.HALT_STAGE,
                    target_stage=None,
                    reason_code=code,
                    requires_human_confirmation=False
                ),
                output_artifacts={},
                tool_usage=ToolUsage(llm_calls=llm_calls),
                attempt_number=agent_input.attempt_number,
                started_at=start,
                completed_at=datetime.now(timezone.utc).isoformat(),
                error={
                    "type": type(exc).__name__,
                    "message": msg,
                    "stage": agent_input.stage_name
                }
            )
