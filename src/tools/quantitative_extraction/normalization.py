from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

from .input_validation import safe_str


MODEL_PARAMETER_MARKERS = (
    "neuron", "hidden_layer", "layer", "epoch", "batch", "learning_rate",
    "dropout", "depth", "tree", "estimator", "parameter", "hyperparameter",
)
EVIDENCE_KEYS = ("source_text_evidence", "evidence", "supporting_quote")
CANONICAL_QUANT_KEYS = {
    "model_or_method", "metric", "value", "unit", "dataset_or_case",
    "evaluation_scope", "data_resolution", "condition", "source_text_evidence",
    "evidence",
}


@dataclass(frozen=True)
class FlatteningResult:
    techniques: list[dict[str, Any]]
    datasets: list[dict[str, Any]]
    quantitative: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    raw_summary: dict[str, int]
    flattened_summary: dict[str, int]


def normalize_metric_name(value: Any) -> str:
    text = safe_str(value).strip()
    key = re.sub(r"[^a-z0-9]+", "", text.casefold())
    aliases = {
        "r2": "R²", "rsquared": "R²", "rmse": "RMSE", "mae": "MAE",
        "mape": "MAPE", "mbe": "MBE", "accuracy": "Accuracy",
    }
    return aliases.get(key, text)


def extract_numeric_tokens(value: Any) -> list[str]:
    return re.findall(r"[-+]?\d+(?:[.,]\d+)?%?", safe_str(value))


def parse_float_from_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    tokens = extract_numeric_tokens(value)
    if not tokens:
        return None
    try:
        return float(tokens[0].replace("%", "").replace(",", "."))
    except ValueError:
        return None


def detect_unit(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    token_match = re.search(r"[-+]?\d+(?:[.,]\d+)?%?", text)
    if not token_match:
        return ""
    token = token_match.group(0)
    if token.endswith("%"):
        return "%"
    remainder = (text[: token_match.start()] + text[token_match.end() :]).strip()
    remainder = re.sub(r"^[=:;,\s]+|[=:;,\s]+$", "", remainder)
    return remainder


def aggregate_unique(values: Iterable[Any], max_chars: int = 1000) -> str:
    seen: list[str] = []
    for value in values:
        text = safe_str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return "; ".join(seen)[:max_chars]


def _serialize_raw(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return safe_str(value)


def _evidence_from_mapping(value: dict[str, Any], inherited: str = "") -> str:
    for key in EVIDENCE_KEYS:
        text = safe_str(value.get(key)).strip()
        if text:
            return text
    return inherited


def _issue(*, source: str, category: str, code: str, raw_path: str, raw_value: Any, message: str, discarded: bool = False) -> dict[str, Any]:
    return {
        "source_filename": source,
        "error_type": category,
        "error_code": code,
        "error_message": message,
        "raw_path": raw_path,
        "raw_value": _serialize_raw(raw_value),
        "discarded": bool(discarded),
    }


def _is_model_parameter(name: str) -> bool:
    key = re.sub(r"[^a-z0-9_]+", "_", name.casefold())
    return any(marker in key for marker in MODEL_PARAMETER_MARKERS)


def _base_row(source: str, title: str) -> dict[str, Any]:
    return {"source_filename": source, "paper_title": title}


def _normalize_techniques(payload: Any, *, source: str, title: str, path: str, inherited_evidence: str, issues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    candidates = 0
    values = payload if isinstance(payload, list) else ([] if payload is None else [payload])
    for index, value in enumerate(values):
        raw_path = f"{path}[{index}]" if isinstance(payload, list) else path
        if isinstance(value, str):
            candidates += 1
            rows.append({**_base_row(source, title), "technique_name": value, "technique_family": "", "role": "", "source_text_evidence": inherited_evidence, "raw_path": raw_path, "raw_value": value})
        elif isinstance(value, dict):
            candidates += 1
            evidence = _evidence_from_mapping(value, inherited_evidence)
            name = safe_str(value.get("technique_name") or value.get("name")).strip()
            if not name:
                issues.append(_issue(source=source, category="NORMALIZATION_ERROR", code="INVALID_TECHNIQUE_SCHEMA", raw_path=raw_path, raw_value=value, message="La técnica no contiene technique_name/name.", discarded=True))
                continue
            rows.append({**_base_row(source, title), "technique_name": name, "technique_family": safe_str(value.get("technique_family") or value.get("family")), "role": safe_str(value.get("role")), "source_text_evidence": evidence, "raw_path": raw_path, "raw_value": _serialize_raw(value)})
        else:
            candidates += 1
            issues.append(_issue(source=source, category="UNSUPPORTED_SCHEMA", code="INVALID_TECHNIQUE_SCHEMA", raw_path=raw_path, raw_value=value, message=f"Tipo de técnica no soportado: {type(value).__name__}.", discarded=True))
    return rows, candidates


def _stringify_dataset_field(value: Any) -> str:
    """Preserve structured dataset metadata without inventing semantics."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return safe_str(value)
    return _serialize_raw(value)


def _first_descriptive_dataset_text(value: dict[str, Any]) -> str:
    for key in (
        "dataset_name", "name", "case_study", "description", "title",
        "location", "site", "region", "source", "data_source",
    ):
        text = _stringify_dataset_field(value.get(key)).strip()
        if text:
            return text
    return ""


def _normalize_datasets(payload: Any, *, source: str, title: str, path: str, inherited_evidence: str, issues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    candidates = 0
    values = payload if isinstance(payload, list) else ([] if payload is None else [payload])
    for index, value in enumerate(values):
        raw_path = f"{path}[{index}]" if isinstance(payload, list) else path
        if isinstance(value, str):
            candidates += 1
            rows.append({
                **_base_row(source, title),
                "dataset_name": value,
                "description": value,
                "case_study": "",
                "data_type": "",
                "temporal_resolution": "",
                "spatial_resolution": "",
                "analysis_scope": "",
                "coordinates": "",
                "altitude_masl": "",
                "data_split": "",
                "source_text_evidence": inherited_evidence,
                "raw_path": raw_path,
                "raw_value": value,
            })
        elif isinstance(value, dict):
            candidates += 1
            evidence = _evidence_from_mapping(value, inherited_evidence)
            description = _stringify_dataset_field(value.get("description"))
            name = _first_descriptive_dataset_text(value)
            if not name:
                name = "Dataset no nombrado"
                issues.append(_issue(
                    source=source,
                    category="NORMALIZATION_ERROR",
                    code="INVALID_DATASET_SCHEMA",
                    raw_path=raw_path,
                    raw_value=value,
                    message=(
                        "El dataset no contiene texto descriptivo; se conserva como "
                        "'Dataset no nombrado' sin descartar sus metadatos."
                    ),
                    discarded=False,
                ))
            elif not safe_str(value.get("dataset_name") or value.get("name") or value.get("case_study")).strip():
                issues.append(_issue(
                    source=source,
                    category="NORMALIZATION_ERROR",
                    code="INVALID_DATASET_SCHEMA",
                    raw_path=raw_path,
                    raw_value=value,
                    message=(
                        "El dataset no contiene dataset_name/name/case_study; "
                        "se normalizó usando texto descriptivo disponible."
                    ),
                    discarded=False,
                ))
            rows.append({
                **_base_row(source, title),
                "dataset_name": name,
                "description": description,
                "case_study": _stringify_dataset_field(value.get("case_study")),
                "data_type": _stringify_dataset_field(value.get("data_type")),
                "temporal_resolution": _stringify_dataset_field(value.get("temporal_resolution")),
                "spatial_resolution": _stringify_dataset_field(value.get("spatial_resolution")),
                "analysis_scope": _stringify_dataset_field(value.get("analysis_scope")),
                "coordinates": _stringify_dataset_field(value.get("coordinates")),
                "altitude_masl": _stringify_dataset_field(value.get("altitude_masl")),
                "data_split": _stringify_dataset_field(value.get("data_split")),
                "source_text_evidence": evidence,
                "raw_path": raw_path,
                "raw_value": _serialize_raw(value),
            })
        else:
            candidates += 1
            issues.append(_issue(
                source=source,
                category="UNSUPPORTED_SCHEMA",
                code="INVALID_DATASET_SCHEMA",
                raw_path=raw_path,
                raw_value=value,
                message=f"Tipo de dataset no soportado: {type(value).__name__}.",
                discarded=True,
            ))
    return rows, candidates


def _quant_row(*, source: str, title: str, model: str, metric: str, value: Any, unit: str = "", condition: str = "", evidence: str = "", raw_path: str, raw_value: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    extra = extra or {}
    return {
        **_base_row(source, title),
        "model_or_method": model,
        "metric": normalize_metric_name(metric),
        "value": safe_str(value),
        "numeric_value": parse_float_from_value(value),
        "unit": unit or detect_unit(value),
        "dataset_or_case": safe_str(extra.get("dataset_or_case")),
        "evaluation_scope": safe_str(extra.get("evaluation_scope")),
        "data_resolution": safe_str(extra.get("data_resolution")),
        "condition": condition or safe_str(extra.get("condition")),
        "source_text_evidence": evidence,
        "raw_path": raw_path,
        "raw_value": _serialize_raw(raw_value),
    }


def _walk_quantitative(value: Any, *, source: str, title: str, path: str, model: str, inherited_evidence: str, rows: list[dict[str, Any]], issues: list[dict[str, Any]]) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        count = 0
        for index, item in enumerate(value):
            count += _walk_quantitative(item, source=source, title=title, path=f"{path}[{index}]", model=model, inherited_evidence=inherited_evidence, rows=rows, issues=issues)
        return count
    if isinstance(value, dict):
        evidence = _evidence_from_mapping(value, inherited_evidence)
        if "metric" in value and "value" in value:
            rows.append(_quant_row(source=source, title=title, model=safe_str(value.get("model_or_method") or model), metric=safe_str(value.get("metric")), value=value.get("value"), unit=safe_str(value.get("unit")), condition=safe_str(value.get("condition")), evidence=evidence, raw_path=path, raw_value=value, extra=value))
            return 1
        count = 0
        substantive = [(key, nested) for key, nested in value.items() if key not in EVIDENCE_KEYS]
        for key, nested in substantive:
            child_path = f"{path}.{key}"
            if isinstance(nested, dict):
                next_model = model or safe_str(key)
                count += _walk_quantitative(nested, source=source, title=title, path=child_path, model=next_model, inherited_evidence=evidence, rows=rows, issues=issues)
            elif isinstance(nested, list):
                count += _walk_quantitative(nested, source=source, title=title, path=child_path, model=model or safe_str(key), inherited_evidence=evidence, rows=rows, issues=issues)
            elif isinstance(nested, (str, int, float)) and not isinstance(nested, bool):
                condition = "model_parameter" if _is_model_parameter(safe_str(key)) else ""
                rows.append(_quant_row(source=source, title=title, model=model, metric=safe_str(key), value=nested, condition=condition, evidence=evidence, raw_path=child_path, raw_value=nested))
                count += 1
            elif nested is not None:
                count += 1
                issues.append(_issue(source=source, category="UNSUPPORTED_SCHEMA", code="INVALID_QUANTITATIVE_SCHEMA", raw_path=child_path, raw_value=nested, message=f"Tipo cuantitativo no soportado: {type(nested).__name__}.", discarded=True))
        return count
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        metric = path.rsplit(".", 1)[-1]
        rows.append(_quant_row(source=source, title=title, model=model, metric=metric, value=value, evidence=inherited_evidence, raw_path=path, raw_value=value))
        return 1
    issues.append(_issue(source=source, category="UNSUPPORTED_SCHEMA", code="INVALID_QUANTITATIVE_SCHEMA", raw_path=path, raw_value=value, message=f"Tipo cuantitativo no soportado: {type(value).__name__}.", discarded=True))
    return 1


def flatten_results(all_results: list[dict[str, Any]]) -> FlatteningResult:
    techniques: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    quantitative: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    raw_quant_count = raw_tech_count = raw_dataset_count = 0
    papers_quant: set[str] = set(); papers_tech: set[str] = set(); papers_dataset: set[str] = set()

    for item in all_results:
        source = safe_str(item.get("source_filename"))
        title = safe_str(item.get("paper_title"))
        paper_evidence = _evidence_from_mapping(item)
        tech_rows, tech_count = _normalize_techniques(item.get("techniques"), source=source, title=title, path="techniques", inherited_evidence=paper_evidence, issues=issues)
        dataset_rows, dataset_count = _normalize_datasets(item.get("datasets"), source=source, title=title, path="datasets", inherited_evidence=paper_evidence, issues=issues)
        before = len(quantitative)
        quant_count = _walk_quantitative(item.get("quantitative_results"), source=source, title=title, path="quantitative_results", model="", inherited_evidence=paper_evidence, rows=quantitative, issues=issues)
        techniques.extend(tech_rows); datasets.extend(dataset_rows)
        raw_tech_count += tech_count; raw_dataset_count += dataset_count; raw_quant_count += quant_count
        if tech_count: papers_tech.add(source)
        if dataset_count: papers_dataset.add(source)
        if quant_count: papers_quant.add(source)
        if quant_count and len(quantitative) == before:
            issues.append(_issue(source=source, category="FLATTENING_ERROR", code="FLATTENING_FAILED", raw_path="quantitative_results", raw_value=item.get("quantitative_results"), message="Había candidatos cuantitativos pero no se generaron filas.", discarded=True))

    discarded = sum(bool(item.get("discarded")) for item in issues)
    warnings = len(issues)
    return FlatteningResult(
        techniques=techniques,
        datasets=datasets,
        quantitative=quantitative,
        issues=issues,
        raw_summary={
            "papers_with_raw_quantitative_payload": len(papers_quant),
            "raw_quantitative_candidate_count": raw_quant_count,
            "papers_with_raw_techniques": len(papers_tech),
            "raw_technique_candidate_count": raw_tech_count,
            "papers_with_raw_datasets": len(papers_dataset),
            "raw_dataset_candidate_count": raw_dataset_count,
        },
        flattened_summary={
            "flattened_quantitative_rows": len(quantitative),
            "flattened_technique_rows": len(techniques),
            "flattened_dataset_rows": len(datasets),
            "discarded_record_count": discarded,
            "normalization_warning_count": warnings,
        },
    )
