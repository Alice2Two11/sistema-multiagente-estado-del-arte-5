from __future__ import annotations
OUTLINE_STAGE_NAME='05_generador_esquema'
OUTLINE_STAGE_VERSION='05_CORREGIDO_v1_manifest_auto_generation_profile'
OUTLINE_PROMPT_VERSION='v2_spanish_generation_profile_valid_sources'
OUTLINE_SCHEMA_VERSION='v2_sections_paper_mapping_traceability'
OUTLINE_VALIDATION_VERSION='v2_repair_then_validate'
DEFAULT_POLICY={
 'force_rebuild':False,
 'max_attempts':2,
 'max_field_chars':1800,
 'title_match_cutoff':0.55,
 'stage_version':OUTLINE_STAGE_VERSION,
 'prompt_version':OUTLINE_PROMPT_VERSION,
 'schema_version':OUTLINE_SCHEMA_VERSION,
 'validation_version':OUTLINE_VALIDATION_VERSION,
}

# CONFIG-D (Stage 05): campos con consumidor real confirmado y que son
# responsabilidad de 00_setup_config.ipynb. Auditoría puntual (ver
# outline_generation_agent.py y src/tools/outline_generation/*):
# - "temperature": hoy hardcodeada a 0 en build_openai_outline_runtime,
#   se conecta en este bloque -- ya no hay hardcode.
# - "force_rebuild": consumido en outline_generation_agent.py (decide
#   si reconstruir aunque ya exista manifest).
# - "max_field_chars": consumido en context_builder.py::build_outline_
#   context (trunca campos del KB al construir el contexto).
#
# CONFIG-F fase 1/2 (auditoría y limpieza de configuración stale/
# unwired): los otros 6 campos que 00 producía nunca fueron aceptados
# por _REQUIRED_FROM_00 ni por DEFAULT_POLICY -- este módulo nunca los
# validó ni consumió; si algún llamador todavía los pasa como
# ``overrides``, simplemente se ignoran (compatibilidad hacia atrás,
# sin romper nada). Clasificación final:
# - minimum_paper_coverage_rate/context_warning_tokens/
#   context_max_tokens: TRUE_STALE_CONFIG -- rastreado explícitamente
#   en validate_outline()/reason_codes()/context_builder.py, sin
#   ningún mecanismo equivalente. Candidatos a retirar de 00.
# - allow_source_repair: ABSORBED_BY_OTHER_MECHANISM -- Stage 05 SÍ
#   repara/valida fuentes por título siempre (source_repair.py:
#   repair_outline_sources/repair_coverage_summary), controlado por
#   title_match_cutoff (ya interno). El flag nunca gobernó nada.
# - validate_titles: ABSORBED_BY_OTHER_MECHANISM -- mismo mecanismo
#   que allow_source_repair; sus unresolved_sources/unresolved_
#   coverage SÍ afectan validation_ok y generan reason codes reales
#   (UNRESOLVED_SECTION_SOURCE/UNRESOLVED_COVERAGE_SOURCE).
# - fail_on_invalid_outline: HARD_CODED_EQUIVALENT -- validate_outline()
#   corre siempre; validation_ok=False conduce a NEEDS_REVISION ->
#   RETRY/HALT_STAGE sin ningún condicional por este flag. Decisión de
#   tesis: rechazo de outlines inválidos permanece obligatorio, sin
#   opción de desactivarlo.
_REQUIRED_FROM_00 = {
    "temperature",
    "force_rebuild",
    "max_field_chars",
}

def get_outline_generation_policy(overrides=None):
    missing_from_00 = sorted(_REQUIRED_FROM_00 - set(overrides or {}))
    if missing_from_00:
        raise ValueError(
            "outline_generation_policy: faltan campos obligatorios que "
            "00_setup_config.ipynb debe proporcionar (sin default "
            f"interno): {missing_from_00}"
        )
    p=dict(DEFAULT_POLICY); p.update(dict(overrides or {})); return p
