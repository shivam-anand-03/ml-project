"""Prediction, domain validation, uncertainty, local explanations, batch and what-if simulation."""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.features import STANDARD_FEATURES, TARGET_INFO
from app.ml.explain import ModelExplainer
from app.models import ModelVersion, Prediction
from app.services.app_settings import get_app_settings


class PredictionError(ValueError):
    """Client error: invalid or unsupported inputs (HTTP 422)."""


class ModelUnavailable(RuntimeError):
    """No trained model / missing artifact (HTTP 409)."""


class _Store:
    def __init__(self) -> None:
        self._bundles: dict[int, dict[str, Any]] = {}
        self._explainers: dict[int, ModelExplainer] = {}
        self._lock = threading.Lock()

    def bundle(self, mv: ModelVersion) -> dict[str, Any]:
        with self._lock:
            b = self._bundles.get(mv.id)
        if b is None:
            path = Path(mv.artifact_path or "")
            if not path.exists():
                raise ModelUnavailable(f"Model artifact for version {mv.id} is missing on disk ({path}); retrain the model.")
            b = joblib.load(path)
            with self._lock:
                self._bundles[mv.id] = b
        return b

    def explainer(self, mv: ModelVersion) -> ModelExplainer:
        with self._lock:
            e = self._explainers.get(mv.id)
        if e is None:
            b = self.bundle(mv)
            e = ModelExplainer(b["pipeline"], b["features"], b["background"], b["family"])
            with self._lock:
                self._explainers[mv.id] = e
        return e

    def clear(self) -> None:
        with self._lock:
            self._bundles.clear()
            self._explainers.clear()


STORE = _Store()


def resolve_model(db: Session, target: str, model_id: int | None = None, dataset_version_id: int | None = None) -> ModelVersion | None:
    """Production model first; otherwise the best candidate of the most recent completed run."""
    if model_id is not None:
        mv = db.get(ModelVersion, model_id)
        if mv is None:
            raise ModelUnavailable(f"Model version {model_id} does not exist")
        if mv.target != target:
            raise PredictionError(f"Model version {model_id} predicts '{mv.target}', not '{target}'")
        return mv
    stmt = select(ModelVersion).where(ModelVersion.target == target)
    if dataset_version_id is not None:
        stmt = stmt.where(ModelVersion.dataset_version_id == dataset_version_id)
    prod = db.scalars(stmt.where(ModelVersion.status == "production").order_by(ModelVersion.promoted_at.desc())).first()
    if prod:
        return prod
    return db.scalars(stmt.where(ModelVersion.status != "archived", ModelVersion.is_best_of_run.is_(True)).order_by(ModelVersion.id.desc())).first()


# ----------------------------------------------------------------------------------------------
# Input validation against the training domain
# ----------------------------------------------------------------------------------------------

def _coerce_number(name: str, value: Any) -> float:
    if value is None or value == "":
        raise PredictionError(f"'{name}' is required")
    if isinstance(value, bool):
        raise PredictionError(f"'{name}' must be a number")
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionError(f"'{name}' must be a number, got {value!r}") from exc
    if not math.isfinite(v):
        raise PredictionError(f"'{name}' must be finite")
    spec = STANDARD_FEATURES[name]
    if spec.valid_min is not None and v < spec.valid_min:
        raise PredictionError(f"'{spec.label}' must be ≥ {spec.valid_min} {spec.unit}")
    if spec.valid_max is not None and v > spec.valid_max:
        raise PredictionError(f"'{spec.label}' must be ≤ {spec.valid_max} {spec.unit}")
    return v


def validate_inputs(bundle: dict[str, Any], inputs: dict[str, Any], app_settings: dict[str, Any]) -> dict[str, Any]:
    """Return {'row': {...}, 'warnings': [...], 'in_domain': bool, 'extrapolation': 'none|mild|far', 'refuse': str|None}."""
    domain = bundle["training_domain"]
    margin = float(app_settings.get("extrapolation_margin", 0.1))
    tol = float(app_settings.get("fixed_condition_tolerance", 0.05))
    allow_far = bool(app_settings.get("allow_far_extrapolation", False))
    warnings: list[str] = []
    severity = 0  # 0 in-domain, 1 mild, 2 far
    row: dict[str, Any] = {}
    for f in bundle["features"]:
        spec = STANDARD_FEATURES[f]
        d = domain["features"][f]
        if f not in inputs:
            raise PredictionError(f"Missing required input '{spec.label}' ({f})")
        if spec.dtype == "number":
            v = _coerce_number(f, inputs[f])
            lo, hi = d["min"], d["max"]
            rng = max(hi - lo, 1e-12)
            if v < lo or v > hi:
                dist = (lo - v) if v < lo else (v - hi)
                if dist <= margin * rng:
                    severity = max(severity, 1)
                    warnings.append(f"{spec.label} = {v:g} {spec.unit} is slightly outside the experimental range {lo:g}–{hi:g} {spec.unit} (mild extrapolation).")
                else:
                    severity = 2
                    warnings.append(f"{spec.label} = {v:g} {spec.unit} is far outside the experimental range {lo:g}–{hi:g} {spec.unit}.")
            row[f] = v
        else:
            v = str(inputs[f]).strip()
            levels = list(d["levels"].keys())
            match = next((lv for lv in levels if lv.lower() == v.lower()), None)
            if match is None:
                raise PredictionError(f"'{v}' is not a supported {spec.label.lower()}. Supported: {', '.join(levels)}")
            row[f] = match
    for f, cond in domain.get("fixed_conditions", {}).items():
        if f not in inputs or inputs[f] in (None, ""):
            continue  # fixed condition assumed; UI shows it
        spec = STANDARD_FEATURES[f]
        if cond.get("value") is None:
            warnings.append(f"{spec.label} varies in the dataset but is not a model input; the value entered is ignored.")
            continue
        if spec.dtype == "number":
            v = _coerce_number(f, inputs[f])
            ref = float(cond["value"])
            if abs(v - ref) > tol * max(abs(ref), 1e-12):
                severity = 2
                warnings.append(f"All experiments were run at {spec.label.lower()} = {ref:g} {spec.unit}; the model has no evidence for {v:g} {spec.unit}.")
        else:
            if str(inputs[f]).strip().lower() != str(cond["value"]).lower():
                severity = 2
                warnings.append(f"All experiments used {spec.label.lower()} = '{cond['value']}'; '{inputs[f]}' is outside the experimental domain.")
    extrap = {0: "none", 1: "mild", 2: "far"}[severity]
    refuse = None
    if severity == 2 and not allow_far:
        refuse = ("Requested conditions are far outside the experimental domain of the training data. "
                  "Prediction refused to avoid silent extrapolation (enable 'allow far extrapolation' in Settings to override). "
                  + " ".join(warnings))
    elif severity == 2:
        warnings.append("Far extrapolation allowed by settings: treat this prediction as unsupported by data.")
    return {"row": row, "warnings": warnings, "in_domain": severity == 0, "extrapolation": extrap, "refuse": refuse}


# ----------------------------------------------------------------------------------------------
# Core prediction
# ----------------------------------------------------------------------------------------------

def _predict_frame(mv: ModelVersion, rows: list[dict[str, Any]]) -> np.ndarray:
    b = STORE.bundle(mv)
    X = pd.DataFrame(rows)[b["features"]]
    return np.asarray(b["pipeline"].predict(X), dtype=float)


def predict_target(db: Session, target: str, inputs: dict[str, Any], model_id: int | None, explain: bool, app_settings: dict[str, Any]) -> dict[str, Any]:
    mv = resolve_model(db, target, model_id)
    info = TARGET_INFO[target]
    if mv is None:
        return {"target": target, "available": False, "reason": f"No trained {info['short']} model. Train one on a dataset that contains {info['label'].lower()} measurements.",
                "value": None, "unit": info["unit"], "model": None, "warnings": [], "in_domain": None, "explanation": None}
    b = STORE.bundle(mv)
    v = validate_inputs(b, inputs, app_settings)
    if v["refuse"]:
        raise PredictionError(v["refuse"])
    pred = float(_predict_frame(mv, [v["row"]])[0])
    interval = b.get("interval") or {}
    hw = interval.get("half_width")
    out = {
        "target": target, "available": True, "value": pred, "unit": b.get("target_unit") or info["unit"],
        "interval": ({"low": pred - hw, "high": pred + hw, "level": interval.get("level"), "method": interval.get("method"),
                      "test_coverage": interval.get("test_coverage")} if hw is not None else None),
        "model": {"id": mv.id, "algorithm": mv.algorithm, "version_label": mv.version_label, "status": mv.status,
                  "dataset_version_id": mv.dataset_version_id},
        "inputs_used": v["row"], "warnings": v["warnings"], "in_domain": v["in_domain"], "extrapolation": v["extrapolation"],
        "explanation": None,
    }
    if explain:
        try:
            out["explanation"] = STORE.explainer(mv).local(pd.DataFrame([v["row"]]))[0]
        except Exception as exc:
            out["explanation"] = {"error": f"Local explanation unavailable: {type(exc).__name__}: {exc}"}
    return out


def predict_record(db: Session, inputs: dict[str, Any], cof_model_id: int | None = None, wear_model_id: int | None = None,
                   explain: bool = True, save: bool = True, source: str = "single") -> dict[str, Any]:
    app_settings = get_app_settings(db)
    cof = predict_target(db, "cof", inputs, cof_model_id, explain, app_settings)
    wear = predict_target(db, "wear_rate", inputs, wear_model_id, explain, app_settings)
    if not cof["available"] and not wear["available"]:
        raise ModelUnavailable("No trained model is available for any target. Train models first.")
    warnings = list(dict.fromkeys(cof["warnings"] + wear["warnings"]))
    result = {"inputs": inputs, "cof": cof, "wear_rate": wear, "warnings": warnings,
              "in_domain": all(x["in_domain"] for x in (cof, wear) if x["available"]),
              "note": "Model prediction — not an experimentally measured value."}
    if save:
        p = Prediction(
            cof_model_version_id=cof["model"]["id"] if cof["available"] else None,
            wear_model_version_id=wear["model"]["id"] if wear["available"] else None,
            dataset_version_id=(cof["model"] or wear["model"] or {}).get("dataset_version_id"),
            inputs=inputs, predicted_cof=cof["value"], cof_interval=cof.get("interval"), predicted_wear_rate=wear["value"],
            wear_rate_unit=wear.get("unit"), in_domain=result["in_domain"], warnings=warnings,
            explanation={"cof": cof.get("explanation"), "wear_rate": wear.get("explanation")}, source=source,
        )
        db.add(p)
        db.commit()
        result["prediction_id"] = p.id
        result["created_at"] = p.created_at.isoformat()
    return result


def prediction_schema(db: Session, cof_model_id: int | None = None, wear_model_id: int | None = None) -> dict[str, Any]:
    """Describe the inputs the current models accept, so the UI can build a dataset-driven form."""
    out: dict[str, Any] = {"targets": {}, "inputs": [], "fixed_conditions": {}, "not_recorded": [], "app_settings": get_app_settings(db)}
    seen: dict[str, dict] = {}
    for target, mid in (("cof", cof_model_id), ("wear_rate", wear_model_id)):
        mv = resolve_model(db, target, mid)
        if mv is None:
            out["targets"][target] = {"available": False, "reason": f"No trained model for {TARGET_INFO[target]['label'].lower()}"}
            continue
        dom = mv.training_domain or {}
        out["targets"][target] = {"available": True, "model_id": mv.id, "algorithm": mv.algorithm, "status": mv.status,
                                  "version_label": mv.version_label, "unit": (mv.feature_schema or {}).get("target_unit"),
                                  "dataset_version_id": mv.dataset_version_id}
        for f, d in dom.get("features", {}).items():
            entry = seen.setdefault(f, {"name": f, **d, "targets": []})
            entry["targets"].append(target)
        for f, c in dom.get("fixed_conditions", {}).items():
            out["fixed_conditions"].setdefault(f, {"name": f, **c})
        for f in dom.get("not_recorded", []):
            if f not in out["not_recorded"]:
                out["not_recorded"].append(f)
    out["inputs"] = list(seen.values())
    out["not_recorded"] = [{"name": f, "label": STANDARD_FEATURES[f].label, "unit": STANDARD_FEATURES[f].unit} for f in out["not_recorded"] if f not in seen and f not in out["fixed_conditions"]]
    return out


# ----------------------------------------------------------------------------------------------
# Batch & simulation
# ----------------------------------------------------------------------------------------------

def predict_batch(db: Session, rows: list[dict[str, Any]], cof_model_id: int | None, wear_model_id: int | None, save: bool) -> list[dict[str, Any]]:
    out = []
    for i, inputs in enumerate(rows):
        try:
            r = predict_record(db, inputs, cof_model_id, wear_model_id, explain=False, save=save, source="batch")
            out.append({"index": i, "ok": True, **r})
        except (PredictionError, ModelUnavailable) as exc:
            out.append({"index": i, "ok": False, "inputs": inputs, "error": str(exc)})
    return out


def simulate(db: Session, target: str, base_inputs: dict[str, Any], sweep: dict[str, Any] | None, compare: dict[str, list] | None,
             model_id: int | None = None) -> dict[str, Any]:
    app_settings = get_app_settings(db)
    mv = resolve_model(db, target, model_id)
    if mv is None:
        raise ModelUnavailable(f"No trained model for {TARGET_INFO[target]['label'].lower()}")
    b = STORE.bundle(mv)
    features = b["features"]
    dom = b["training_domain"]["features"]
    base = validate_inputs(b, base_inputs, app_settings)
    if base["refuse"]:
        raise PredictionError(base["refuse"])
    baseline = float(_predict_frame(mv, [base["row"]])[0])
    result: dict[str, Any] = {"target": target, "unit": b.get("target_unit"), "model": {"id": mv.id, "algorithm": mv.algorithm, "version_label": mv.version_label},
                              "baseline": {"inputs": base["row"], "value": baseline, "warnings": base["warnings"]}, "domain": dom,
                              "note": "All values are model predictions, not measurements."}

    # sensitivity: ±10 % of the training range for each numeric feature
    sens = []
    for f in features:
        if dom[f]["dtype"] != "number":
            continue
        rng = dom[f]["max"] - dom[f]["min"]
        if rng <= 0:
            continue
        step = 0.1 * rng
        lo = max(dom[f]["min"], base["row"][f] - step)
        hi = min(dom[f]["max"], base["row"][f] + step)
        r_lo, r_hi = dict(base["row"], **{f: lo}), dict(base["row"], **{f: hi})
        p_lo, p_hi = _predict_frame(mv, [r_lo, r_hi])
        sens.append({"feature": f, "label": dom[f]["label"], "unit": dom[f]["unit"], "low_input": lo, "high_input": hi,
                     "low_value": float(p_lo), "high_value": float(p_hi), "delta": float(p_hi - p_lo),
                     "slope_per_unit": float((p_hi - p_lo) / (hi - lo)) if hi > lo else 0.0})
    result["sensitivity"] = sens

    # category comparison at the baseline conditions (e.g. every material)
    comparisons = {}
    for f in features:
        if dom[f]["dtype"] != "category":
            continue
        levels = (compare or {}).get(f) or list(dom[f]["levels"].keys())
        levels = [lv for lv in levels if lv in dom[f]["levels"]]
        rows = [dict(base["row"], **{f: lv}) for lv in levels]
        preds = _predict_frame(mv, rows) if rows else np.array([])
        comparisons[f] = [{"level": lv, "value": float(p), "delta_vs_baseline": float(p - baseline)} for lv, p in zip(levels, preds)]
    result["comparisons"] = comparisons

    # sweep one numeric feature across a range for each level of a categorical feature (or just baseline)
    if sweep:
        sf = sweep.get("feature")
        if sf not in features or dom[sf]["dtype"] != "number":
            raise PredictionError(f"Sweep feature must be one of the model's numeric inputs: {[f for f in features if dom[f]['dtype'] == 'number']}")
        margin = float(app_settings.get("extrapolation_margin", 0.1))
        rng = dom[sf]["max"] - dom[sf]["min"]
        lo = float(sweep.get("min", dom[sf]["min"]))
        hi = float(sweep.get("max", dom[sf]["max"]))
        lo = max(lo, dom[sf]["min"] - margin * rng)
        hi = min(hi, dom[sf]["max"] + margin * rng)
        steps = int(min(max(int(sweep.get("steps", 50)), 2), 500))
        xs = np.linspace(lo, hi, steps)
        by = sweep.get("by")
        if by and (by not in features or dom[by]["dtype"] != "category"):
            raise PredictionError("'by' must be a categorical model input")
        levels = ((compare or {}).get(by) or list(dom[by]["levels"].keys())) if by else [None]
        series = []
        for lv in levels:
            rows = [dict(base["row"], **({by: lv} if by else {}), **{sf: float(x)}) for x in xs]
            preds = _predict_frame(mv, rows)
            series.append({"level": lv, "points": [{"x": float(x), "y": float(p), "extrapolated": bool(x < dom[sf]["min"] or x > dom[sf]["max"])} for x, p in zip(xs, preds)]})
        result["sweep"] = {"feature": sf, "label": dom[sf]["label"], "unit": dom[sf]["unit"], "by": by, "series": series,
                           "experimental_range": [dom[sf]["min"], dom[sf]["max"]]}
    return result


def compare_scenarios(db: Session, target: str, scenarios: list[dict[str, Any]], model_id: int | None = None) -> dict[str, Any]:
    app_settings = get_app_settings(db)
    mv = resolve_model(db, target, model_id)
    if mv is None:
        raise ModelUnavailable(f"No trained model for {TARGET_INFO[target]['label'].lower()}")
    b = STORE.bundle(mv)
    out = []
    baseline_value: float | None = None
    for i, sc in enumerate(scenarios):
        try:
            v = validate_inputs(b, sc, app_settings)
            if v["refuse"]:
                raise PredictionError(v["refuse"])
            val = float(_predict_frame(mv, [v["row"]])[0])
            if baseline_value is None:
                baseline_value = val
            out.append({"index": i, "ok": True, "inputs": v["row"], "value": val, "delta_vs_first": val - baseline_value,
                        "pct_vs_first": (100.0 * (val - baseline_value) / abs(baseline_value)) if baseline_value else None,
                        "warnings": v["warnings"], "in_domain": v["in_domain"]})
        except PredictionError as exc:
            out.append({"index": i, "ok": False, "inputs": sc, "error": str(exc)})
    return {"target": target, "unit": b.get("target_unit"), "model": {"id": mv.id, "algorithm": mv.algorithm}, "scenarios": out,
            "note": "All values are model predictions, not measurements."}
