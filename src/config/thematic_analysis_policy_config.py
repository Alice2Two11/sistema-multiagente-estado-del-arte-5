"""Políticas exclusivas del Agente 04. Umbrales iniciales diagnósticos."""
from __future__ import annotations
POLICY_STATUS = "PROVISIONAL_NOT_SCIENTIFICALLY_VALIDATED"
DEFAULT_THEMATIC_ANALYSIS_POLICY = {
    "temperature": 0.1,
    "max_field_chars": 3500,
    "max_attempts": 2,
    "allow_quantitative_context": True,
    "require_quantitative_manifest_if_files_exist": True,
    "validate_titles": True,
    "require_gap_supporting_sources": True,
    "require_comparative_sources": True,
    "auto_rebuild": True,
    "force_rebuild": False,
    "thresholds_status": POLICY_STATUS,
    "structure_policy": {"enforce_section_count": False},
    "diagnostic_thresholds": {
        "paper_coverage": None,
        "supported_theme_rate": None,
        "supported_gap_rate": None,
        "comparative_dimension_support_rate": None,
        "valid_reference_rate": None,
    },
    "manual_review_policy": {"allowed": True},
}

# CONFIG-C (Stage 04): campos que son responsabilidad de
# 00_setup_config.ipynb (coinciden con FIXED_THEMATIC_ANALYSIS_POLICY
# del notebook). max_attempts/thresholds_status/structure_policy/
# diagnostic_thresholds/manual_review_policy quedan fuera
# deliberadamente: son contrato interno de 04, nunca responsabilidad
# de 00.
#
# CONFIG-F (auditoría y limpieza de configuración stale/unwired):
# fail_on_invalid_sources -- 00 lo produce, pero nunca estuvo en
# DEFAULT_THEMATIC_ANALYSIS_POLICY ni en _REQUIRED_FROM_00; auditoría
# confirmó 0 consumidores en todo Stage 04, sin ningún mecanismo
# equivalente que lo absorba -- TRUE_STALE_CONFIG. Este módulo nunca
# lo validó ni consumió; si algún llamador todavía lo pasa como
# override, simplemente se ignora (compatibilidad hacia atrás).
_REQUIRED_FROM_00 = {
    "temperature",
    "max_field_chars",
    "allow_quantitative_context",
    "require_quantitative_manifest_if_files_exist",
    "validate_titles",
    "require_gap_supporting_sources",
    "require_comparative_sources",
    "auto_rebuild",
    "force_rebuild",
}

def get_thematic_analysis_policy(overrides=None):
    # "temperature" tiene hoy valores distintos en 00 (0.0) y en el
    # default interno (0.1); "max_field_chars" también difiere (00=
    # 2200, interno=3500). Sin este chequeo, una policy ausente o
    # incompleta habría hecho que 04 corriera silenciosamente con los
    # valores internos en vez de los de 00.
    missing_from_00 = sorted(_REQUIRED_FROM_00 - set(overrides or {}))
    if missing_from_00:
        raise ValueError(
            "thematic_analysis_policy: faltan campos obligatorios que "
            "00_setup_config.ipynb debe proporcionar (sin default "
            f"interno): {missing_from_00}"
        )
    policy={**DEFAULT_THEMATIC_ANALYSIS_POLICY}
    policy["diagnostic_thresholds"]=dict(DEFAULT_THEMATIC_ANALYSIS_POLICY["diagnostic_thresholds"])
    policy["manual_review_policy"]=dict(DEFAULT_THEMATIC_ANALYSIS_POLICY["manual_review_policy"])
    policy["structure_policy"]=dict(DEFAULT_THEMATIC_ANALYSIS_POLICY["structure_policy"])
    if overrides:
        overrides=dict(overrides)
        supplied_structure_policy=overrides.pop("structure_policy",None)
        policy.update(overrides)
        if isinstance(supplied_structure_policy,dict):
            policy["structure_policy"]=dict(policy["structure_policy"])
            policy["structure_policy"].update(supplied_structure_policy)
    if int(policy.get("max_attempts",2)) != 2: raise ValueError("04 admite exactamente dos intentos contractuales.")
    if not isinstance(policy["structure_policy"].get("enforce_section_count"),bool):
        raise TypeError("thematic_analysis_policy.structure_policy.enforce_section_count must be bool")
    return policy
