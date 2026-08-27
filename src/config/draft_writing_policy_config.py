from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LEGACY_RETRIEVAL_STRATEGY = "legacy_chroma_then_csv_restricted"


DEFAULT_DRAFT_WRITING_POLICY: dict[str, Any] = {
    "stage_version": "06_AGENTIC_V16_BEHAVIOR_PRESERVING",
    "prompt_version": "legacy_notebook06_section_prompt_v1",
    "rag_version": "legacy_chroma_then_csv_restricted_v1",
    "validation_version": "legacy_notebook06_validation_v1",
    "temperature": 0.0,
    "force_rebuild": False,
    "max_section_revision_attempts": 2,
    # retrieval_strategy ya NO tiene default interno: es obligatoria
    # desde 00_setup_config.ipynb / Corrida_03_a_08.ipynb (celda de
    # materialización explícita, mismo patrón que draft_representation_
    # contract) -- ver _REQUIRED_FROM_00 más abajo. Su ausencia falla
    # fail-closed en get_draft_writing_policy(), nunca se resuelve en
    # silencio contra este dict.
    "top_k_evidence_per_section": 8,
    "max_evidence_chars": 18000,
    "max_quantitative_rows_per_section": 12,
}

_ALLOWED_RETRIEVAL_STRATEGIES = {
    LEGACY_RETRIEVAL_STRATEGY,
}

# CONFIG-E (Stage 06): campos con consumidor real confirmado y que son
# responsabilidad de 00_setup_config.ipynb (coinciden con
# FIXED_DRAFT_GENERATION_POLICY del notebook). Auditoría puntual --
# regla aplicada: campo de 00 -> consumidor final real confirmado ->
# entonces obligatorio. NO: 00 lo produce -> obligatorio aunque 06 lo
# ignore.
# - temperature: coincide hoy en valor (00=0.0, interno=0.0);
#   consumidor real confirmado (cfg["policy"]["temperature"] en
#   draft_writing_runtime.py, ya acceso directo).
# - force_rebuild: consumidor real confirmado en draft_writing_agent.py
#   (chequeo de reutilización de manifest).
# - max_section_revision_attempts: consumidor real confirmado en
#   draft_writing_agent.py (rango de reintentos de sección).
# - top_k_evidence_per_section: 00=8, interno=8 (coinciden hoy, pero
#   sin este chequeo el campo podía caer en silencio al default);
#   consumidor real confirmado en draft_writing_agent.py.
# - max_quantitative_rows_per_section: 00=8, interno=12 --
#   DISCREPANCIA REAL confirmada; consumidor real confirmado (dos
#   ocurrencias en draft_writing_agent.py).
# - max_evidence_chars: 00=1400, interno=18000 -- DISCREPANCIA REAL
#   adicional; consumidor real extendido confirmado
#   (draft_writing_agent.py, tools/draft_writing/validation.py, y
#   recibido como parámetro requerido -- sin default -- en
#   hybrid_retrieval.py/quantitative_augmentation.py). Los defaults de
#   firma "max_evidence_chars=18000" en tools/draft_writing/
#   retrieval.py quedan intactos: su único camino productivo pasa el
#   valor validado explícitamente, confirmado y aprobado sin tocar.
#
# CONFIG-F fase 2 (limpieza de configuración stale/unwired, auditoría
# CONFIG-F fase 1): los siguientes 5 campos que 00 producía en
# FIXED_DRAFT_GENERATION_POLICY fueron RETIRADOS de este contrato --
# ya NUNCA se aceptan como override ni se ofrecen como configurables.
# El comportamiento real que cada uno pretendía controlar permanece
# INTACTO y sigue siempre activo en el código, verificado con tests
# dedicados (ver tests/config/test_config_f_stale_removal.py):
# - auto_rebuild: TRUE_STALE_CONFIG -- cero consumidores confirmados,
#   ningún mecanismo equivalente. Simplemente retirado.
# - allow_open_search_outside_outline_sources: HARD_CODED_EQUIVALENT
#   -- draft_writing_agent.py::_section_sources() restringe SIEMPRE la
#   evidencia a section.get("papers_to_use") (las fuentes que 05
#   asignó); nunca hubo una rama de "búsqueda abierta". Retirado el
#   flag, preservada la restricción incondicional.
# - validate_citations_against_section_evidence: HARD_CODED_EQUIVALENT
#   -- validate_generated_section() (tools/draft_writing/validation.py)
#   valida citas SIEMPRE, sin condicional, y sus citation_errors ya
#   controlan reintentos/aceptación de sección en execute(). Retirado
#   el flag, preservada la validación incondicional.
# - validate_numeric_values_against_source_chunks:
#   HARD_CODED_EQUIVALENT -- misma función, numeric_errors calculado
#   SIEMPRE. Retirado el flag, preservada la validación incondicional.
# - fail_on_invalid_draft: HARD_CODED_EQUIVALENT -- las transiciones
#   NEEDS_REVISION/REJECTED en execute() ya se disparan siempre que
#   hay errores de validación, sin condicional por este flag. Retirado
#   el flag, preservado el rechazo incondicional de borradores
#   inválidos -- decisión de tesis: en trazabilidad científica no se
#   permite desactivar estas garantías.
#
# STAGE06-STALE-CONFIG-CLEANUP: candidate_multiplier/chroma_quota/
# csv_quota/rrf_quota/rrf_k/max_candidates_per_source/
# quantitative_evidence_quota/organizational_*/substantive_* fueron
# eliminadas por completo del contrato (no solo excluidas de este
# conjunto) -- eran exclusivas de la rama híbrida V17, ya removida, con
# cero consumidores confirmados en src/tests/notebooks. retrieval_
# strategy SÍ es obligatoria desde 00/Corrida_03_a_08.ipynb
# (materialización explícita, mismo patrón que draft_representation_
# contract): a diferencia de esos campos ya eliminados, gobierna qué
# ruta de Stage06 se ejecuta, y su ausencia no puede resolverse en
# silencio contra ningún default interno.
_REQUIRED_FROM_00 = {
    "temperature",
    "force_rebuild",
    "max_section_revision_attempts",
    "top_k_evidence_per_section",
    "max_evidence_chars",
    "max_quantitative_rows_per_section",
    "retrieval_strategy",
}


def _require_integer(policy: Mapping[str, Any], key: str) -> int:
    value = policy.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"DRAFT_POLICY_INVALID_TYPE:{key}:expected_integer")
    return value


def _require_nonempty_string(policy: Mapping[str, Any], key: str) -> str:
    value = policy.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"DRAFT_POLICY_INVALID_TYPE:{key}:expected_nonempty_string")
    return value.strip()


def validate_draft_writing_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate Agent 06 policy with deterministic, explicit errors."""
    if not isinstance(policy, Mapping):
        raise ValueError("DRAFT_POLICY_INVALID_TYPE:policy:expected_mapping")

    validated = dict(policy)
    retrieval_strategy = _require_nonempty_string(validated, "retrieval_strategy")
    if retrieval_strategy not in _ALLOWED_RETRIEVAL_STRATEGIES:
        raise ValueError(
            "DRAFT_POLICY_INVALID:retrieval_strategy:unsupported_strategy"
        )

    top_k = _require_integer(validated, "top_k_evidence_per_section")
    if top_k <= 0:
        raise ValueError(
            "DRAFT_POLICY_INVALID:top_k_evidence_per_section:must_be_greater_than_0"
        )

    for key in ("max_evidence_chars", "max_quantitative_rows_per_section"):
        value = _require_integer(validated, key)
        if value < 0:
            raise ValueError(
                f"DRAFT_POLICY_INVALID:{key}:must_be_greater_than_or_equal_to_0"
            )

    return validated


def get_draft_writing_policy(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    # "temperature"/"max_evidence_chars"/"max_quantitative_rows_per_
    # section" (entre otros) tienen hoy valores distintos o coincidentes
    # entre 00 y el default interno -- sin este chequeo, una policy
    # ausente o incompleta habría hecho que 06 corriera silenciosamente
    # con los valores internos en vez de los de 00.
    missing_from_00 = sorted(_REQUIRED_FROM_00 - set(overrides or {}))
    if missing_from_00:
        raise ValueError(
            "draft_generation_policy: faltan campos obligatorios que "
            "00_setup_config.ipynb debe proporcionar (sin default "
            f"interno): {missing_from_00}"
        )
    policy = dict(DEFAULT_DRAFT_WRITING_POLICY)
    if overrides is None:
        return validate_draft_writing_policy(policy)
    if not isinstance(overrides, Mapping):
        raise ValueError("DRAFT_POLICY_INVALID_TYPE:overrides:expected_mapping")

    override_values = dict(overrides)
    policy.update(override_values)
    return validate_draft_writing_policy(policy)

