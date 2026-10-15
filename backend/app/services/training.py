"""Model training: reproducible splits, tuning, evaluation, persistence and SHAP importance."""
from __future__ import annotations

import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.core.features import STANDARD_FEATURES, TARGET_INFO
from app.ml.explain import ModelExplainer
from app.ml.pipeline import (auto_select_features, build_pipeline, candidate_models, make_split,
                             numeric_and_categorical, regression_metrics)
from app.models import DatasetVersion, ModelMetric, ModelVersion, TrainingRun
from app.services.datasets import version_dataframe
from app.services.ingestion import to_jsonable
from app.services.preprocessing import is_varying


@dataclass
class TrainingConfig:
    target: str = "cof"
    dataset_version_id: int = 0
    models: list[str] = field(default_factory=lambda: ["linear", "polynomial", "random_forest", "svr", "xgboost", "mlp"])
    features: list[str] | None = None  # None = automatic selection
    split_strategy: str = "auto"  # auto | grouped | random | chronological
    test_fraction: float = 0.2
    n_folds: int = 5
    search_mode: str = "standard"  # standard | quick
    seed: int = 42
    interval_level: float = 0.9
    notes: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        allowed = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**allowed)
        if cfg.target not in TARGET_INFO:
            raise ValueError(f"Unknown target '{cfg.target}'")
        if not cfg.models:
            raise ValueError("Select at least one model")
        unknown = [m for m in cfg.models if m not in candidate_models(0)]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}")
        if not 0.05 <= cfg.test_fraction <= 0.5:
            raise ValueError("test_fraction must be between 0.05 and 0.5")
        if cfg.n_folds < 2 or cfg.n_folds > 10:
            raise ValueError("n_folds must be between 2 and 10")
        if cfg.split_strategy not in {"auto", "grouped", "random", "chronological"}:
            raise ValueError("split_strategy must be auto, grouped, random or chronological")
        if cfg.search_mode not in {"standard", "quick"}:
            raise ValueError("search_mode must be standard or quick")
        return cfg


class TrainingError(RuntimeError):
    pass


def training_domain(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """Ranges / levels of the model features plus the dataset's fixed and unrecorded conditions."""
    dom: dict[str, Any] = {"features": {}, "fixed_conditions": {}, "not_recorded": []}
    for f in features:
        spec = STANDARD_FEATURES[f]
        s = df[f]
        if spec.dtype == "number":
            v = pd.to_numeric(s, errors="coerce").dropna()
            dom["features"][f] = {"dtype": "number", "unit": spec.unit, "label": spec.label, "min": float(v.min()), "max": float(v.max()),
                                  "p05": float(v.quantile(0.05)), "p95": float(v.quantile(0.95)), "mean": float(v.mean())}
        else:
            counts = s.dropna().astype(str).value_counts()
            dom["features"][f] = {"dtype": "category", "unit": spec.unit, "label": spec.label,
                                  "levels": {str(k): int(c) for k, c in counts.items()}}
    for name, spec in STANDARD_FEATURES.items():
        if spec.role != "input" or name in features:
            continue
        s = df[name] if name in df.columns else pd.Series(dtype=object)
        s = s.dropna()
        if len(s) == 0:
            dom["not_recorded"].append(name)
        elif spec.dtype == "number":
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v) and not is_varying(v):
                dom["fixed_conditions"][name] = {"value": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "unit": spec.unit, "label": spec.label}
            elif len(v):
                dom["fixed_conditions"][name] = {"value": None, "min": float(v.min()), "max": float(v.max()), "unit": spec.unit, "label": spec.label,
                                                 "note": "varies in the data but was not used as a model input"}
        else:
            levels = sorted(set(s.astype(str)))
            if len(levels) == 1:
                dom["fixed_conditions"][name] = {"value": levels[0], "unit": spec.unit, "label": spec.label}
            else:
                dom["fixed_conditions"][name] = {"value": None, "levels": levels, "unit": spec.unit, "label": spec.label,
                                                 "note": "varies in the data but was not used as a model input"}
    return dom


def _fold_scores(gs: GridSearchCV) -> list[float]:
    res = gs.cv_results_
    i = gs.best_index_
    scores = []
    k = 0
    while f"split{k}_test_score" in res:
        scores.append(float(-res[f"split{k}_test_score"][i]))
        k += 1
    return scores


def train_models(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    settings: Settings,
    log: Callable[[str], None] = print,
    on_model_done: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train and evaluate every requested model on identical splits. Returns a result dict with fitted pipelines."""
    target = cfg.target
    if target not in df.columns or df[target].notna().sum() == 0:
        raise TrainingError(f"Target '{target}' is not available in this dataset version — no measurements to learn from.")
    data = df[df["is_usable"]] if "is_usable" in df.columns else df
    data = data[data[target].notna()].reset_index(drop=True)
    if len(data) < 20:
        raise TrainingError(f"Only {len(data)} usable rows with a '{target}' value; at least 20 are required to train and validate a model.")

    notes: list[str] = []
    if cfg.features:
        features = list(dict.fromkeys(cfg.features))
        bad = [f for f in features if f not in STANDARD_FEATURES or STANDARD_FEATURES[f].role not in ("input", "metadata")]
        if bad:
            raise TrainingError(f"Invalid model feature(s): {bad}")
        empty = [f for f in features if data[f].isna().all()]
        if empty:
            raise TrainingError(f"Selected feature(s) have no values in this dataset version: {empty}")
    else:
        features, notes = auto_select_features(data, target)
    if not features:
        raise TrainingError("No varying input feature is available; a model cannot be trained on constants only.")
    numeric, categorical = numeric_and_categorical(features)
    for n in notes:
        log(n)
    log(f"Features: {features} | target: {target} | usable rows: {len(data)}")

    split = make_split(data, cfg.split_strategy, cfg.seed, cfg.test_fraction, n_folds=cfg.n_folds)
    for w in split.warnings:
        log("Split warning: " + w)
    log(f"Split ({split.strategy}): {split.description} Train rows: {len(split.train_idx)}, test rows: {len(split.test_idx)}")
    if len(split.test_idx) < 5:
        raise TrainingError("The held-out test set has fewer than 5 rows; the dataset is too small for this split strategy.")
    X = data[features]
    y = data[target].astype(float).to_numpy()
    X_tr, X_te = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_tr, y_te = y[split.train_idx], y[split.test_idx]
    groups = split.groups_train
    small = len(X_tr) < 200
    if small:
        log(f"Only {len(X_tr)} training rows: the held-out test estimate is unreliable; cross-validation metrics are the primary evidence.")

    candidates = candidate_models(cfg.seed, settings.n_jobs)
    results: list[dict[str, Any]] = []
    for key in cfg.models:
        cand = candidates[key]
        grid = cand.grid if cfg.search_mode == "standard" else cand.quick_grid
        t0 = time.perf_counter()
        try:
            pipe = build_pipeline(cand, numeric, categorical)
            gs = GridSearchCV(pipe, grid or {}, scoring="neg_root_mean_squared_error", cv=split.cv, n_jobs=settings.n_jobs, refit=True)
            gs.fit(X_tr, y_tr, groups=groups)
            best = gs.best_estimator_
            oof = cross_val_predict(clone(best), X_tr, y_tr, cv=split.cv, groups=groups, n_jobs=settings.n_jobs)
            val_metrics = regression_metrics(y_tr, oof)
            fold_rmse = _fold_scores(gs)
            train_metrics = regression_metrics(y_tr, best.predict(X_tr))
            y_pred_te = best.predict(X_te)
            test_metrics = regression_metrics(y_te, y_pred_te)
            duration = time.perf_counter() - t0
            resid = np.abs(y_tr - oof)
            q = float(np.quantile(resid, cfg.interval_level))
            coverage = float(np.mean(np.abs(y_te - y_pred_te) <= q)) if len(y_te) else None
            params = {k.replace("model__", ""): (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in gs.best_params_.items()}
            res = {
                "key": key, "label": cand.label, "family": cand.family, "pipeline": best, "best_params": params,
                "grid": {k.replace("model__", ""): [str(x) if not isinstance(x, (int, float, str, bool)) else x for x in v] for k, v in grid.items()},
                "metrics": {"train": train_metrics, "validation": val_metrics, "test": test_metrics},
                "cv_fold_rmse": fold_rmse, "cv_rmse_mean": float(np.mean(fold_rmse)) if fold_rmse else val_metrics["rmse"],
                "cv_rmse_std": float(np.std(fold_rmse)) if fold_rmse else None,
                "duration_s": duration, "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                "interval": {"method": f"split-conformal: {cfg.interval_level:.0%} quantile of absolute out-of-fold cross-validation residuals",
                             "level": cfg.interval_level, "half_width": q, "test_coverage": coverage,
                             "q50": float(np.quantile(resid, 0.5)), "q95": float(np.quantile(resid, 0.95))},
                "oof_predictions": {"actual": y_tr.tolist(), "predicted": oof.tolist()},
                "test_predictions": {"actual": y_te.tolist(), "predicted": y_pred_te.tolist(),
                                     "rows": X_te.reset_index(drop=True).to_dict(orient="records")},
                "error": None,
            }
            log(f"{cand.label}: CV RMSE {res['cv_rmse_mean']:.5f} | validation R² {val_metrics['r2']:.4f} | test R² {test_metrics['r2']:.4f} | {duration:.1f}s | params {params}")
        except Exception as exc:  # one failing model must not abort the comparison
            res = {"key": key, "label": cand.label, "family": cand.family, "pipeline": None, "error": f"{type(exc).__name__}: {exc}",
                   "duration_s": time.perf_counter() - t0, "metrics": {}}
            log(f"{cand.label} FAILED: {exc}")
        results.append(res)
        if on_model_done:
            on_model_done(res)

    ok = [r for r in results if r["pipeline"] is not None]
    if not ok:
        raise TrainingError("Every candidate model failed to train: " + "; ".join(f"{r['label']}: {r['error']}" for r in results))
    ok.sort(key=lambda r: (r["metrics"]["validation"]["rmse"], -(r["metrics"]["validation"]["r2"] or -1e9)))
    best_key = ok[0]["key"]
    log(f"Best validated model: {ok[0]['label']} (lowest out-of-fold cross-validation RMSE)")
    return {
        "features": features, "numeric": numeric, "categorical": categorical, "target": target, "notes": notes,
        "split": {"strategy": split.strategy, "description": split.description, "warnings": split.warnings,
                  "test_groups": split.test_groups, "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                  "cv_splits": int(split.cv.get_n_splits(X_tr, y_tr, groups)), "small_dataset": small},
        "results": results, "best_key": best_key, "domain": training_domain(data, features), "data": data,
    }


# ----------------------------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------------------------

def _downsample_pairs(actual: list[float], predicted: list[float], rows: list[dict] | None, k: int = 1500, seed: int = 0) -> list[dict]:
    n = len(actual)
    idx = np.arange(n)
    if n > k:
        idx = np.sort(np.random.RandomState(seed).choice(idx, size=k, replace=False))
    out = []
    for i in idx:
        d = {"actual": float(actual[i]), "predicted": float(predicted[i]), "residual": float(actual[i] - predicted[i])}
        if rows is not None:
            d.update({kk: to_jsonable(vv) for kk, vv in rows[i].items()})
        out.append(d)
    return out


def persist_results(db: Session, run: TrainingRun, outcome: dict[str, Any], cfg: TrainingConfig, settings: Settings) -> list[ModelVersion]:
    settings.ensure_dirs()
    created: list[ModelVersion] = []
    data: pd.DataFrame = outcome["data"]
    features = outcome["features"]
    unit = TARGET_INFO[cfg.target]["unit"]
    for r in outcome["results"]:
        if r["pipeline"] is None:
            continue
        is_best = r["key"] == outcome["best_key"]
        # explainability on training data
        try:
            explainer = ModelExplainer(r["pipeline"], features, data.iloc[: min(len(data), 2000)], r["family"])
            importance = explainer.global_importance(data)
        except Exception as exc:  # explanation must never block model persistence
            importance = {"method": None, "error": f"SHAP explanation unavailable: {type(exc).__name__}: {exc}", "importance": []}
        mv = ModelVersion(
            training_run_id=run.id, dataset_version_id=run.dataset_version_id, target=cfg.target, algorithm=r["key"],
            version_label="", artifact_path="", feature_schema={
                "target": cfg.target, "target_unit": unit, "features": [
                    {"name": f, "label": STANDARD_FEATURES[f].label, "unit": STANDARD_FEATURES[f].unit, "dtype": STANDARD_FEATURES[f].dtype} for f in features
                ], "numeric": outcome["numeric"], "categorical": outcome["categorical"]},
            hyperparameters={"best_params": r["best_params"], "search_grid": r["grid"], "search_mode": cfg.search_mode},
            training_domain=to_jsonable(outcome["domain"]),
            cv_results=to_jsonable({
                "fold_rmse": r["cv_fold_rmse"], "cv_rmse_mean": r["cv_rmse_mean"], "cv_rmse_std": r["cv_rmse_std"],
                "split": outcome["split"],
                "test_predictions": _downsample_pairs(r["test_predictions"]["actual"], r["test_predictions"]["predicted"], r["test_predictions"]["rows"]),
                "oof_predictions": _downsample_pairs(r["oof_predictions"]["actual"], r["oof_predictions"]["predicted"], None, k=1000),
            }),
            global_importance=to_jsonable(importance), interval=to_jsonable(r["interval"]), status="candidate",
            is_best_of_run=is_best, training_duration_s=float(r["duration_s"]), n_train=r["n_train"], n_test=r["n_test"],
        )
        db.add(mv)
        db.flush()
        mv.version_label = f"{cfg.target}-{r['key']}-v{mv.id}"
        path = settings.models_dir / f"model_{mv.id}.joblib"
        bundle = {
            "model_version_id": mv.id, "pipeline": r["pipeline"], "features": features, "numeric": outcome["numeric"],
            "categorical": outcome["categorical"], "target": cfg.target, "target_unit": unit, "algorithm": r["key"],
            "family": r["family"], "label": r["label"], "config": asdict(cfg), "dataset_version_id": run.dataset_version_id,
            "training_domain": outcome["domain"], "interval": r["interval"], "metrics": r["metrics"],
            "background": data[features].sample(n=min(200, len(data)), random_state=cfg.seed).reset_index(drop=True),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        }
        joblib.dump(bundle, path)
        mv.artifact_path = str(path)
        for split_name, m in r["metrics"].items():
            for metric in ("r2", "mae", "rmse", "mape"):
                db.add(ModelMetric(model_version_id=mv.id, split=split_name, metric=metric, value=m.get(metric), n_samples=m.get("n", 0),
                                   detail={"mape_coverage": m.get("mape_coverage"), "max_abs_error": m.get("max_abs_error"), "bias": m.get("bias")}))
        created.append(mv)
    db.commit()
    return created


def execute_training_run(session_factory, run_id: int, settings: Settings | None = None) -> None:
    """Entry point for the background worker. Opens its own session; records status, log and errors."""
    settings = settings or get_settings()
    db: Session = session_factory()
    try:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        run.log = []
        db.commit()
        cfg = TrainingConfig.from_dict(run.config)
        version = db.get(DatasetVersion, run.dataset_version_id)
        if version is None:
            raise TrainingError("Dataset version not found")

        def log(msg: str) -> None:
            run.log = [*run.log, {"t": datetime.now(timezone.utc).isoformat(), "msg": msg}]
            db.commit()

        def on_model(res: dict[str, Any]) -> None:
            done = (run.summary or {}).get("models_done", [])
            run.summary = {**(run.summary or {}), "models_done": [*done, res["key"]], "n_models": len(cfg.models)}
            db.commit()

        df = version_dataframe(db, version.id)
        log(f"Loaded dataset version {version.version_number} ({len(df)} rows, {int(df['is_usable'].sum())} usable)")
        outcome = train_models(df, cfg, settings, log, on_model)
        models = persist_results(db, run, outcome, cfg, settings)
        best = next((m for m in models if m.is_best_of_run), models[0])
        leaderboard = []
        for r in outcome["results"]:
            entry = {"algorithm": r["key"], "label": r["label"], "error": r["error"], "duration_s": r["duration_s"]}
            if r["pipeline"] is not None:
                entry.update({"validation_rmse": r["metrics"]["validation"]["rmse"], "validation_r2": r["metrics"]["validation"]["r2"],
                              "test_rmse": r["metrics"]["test"]["rmse"], "test_r2": r["metrics"]["test"]["r2"]})
            leaderboard.append(entry)
        run.summary = {
            **(run.summary or {}), "features": outcome["features"], "notes": outcome["notes"], "split": outcome["split"],
            "best_model_version_id": best.id, "best_algorithm": best.algorithm, "leaderboard": leaderboard,
            "model_version_ids": [m.id for m in models], "domain": outcome["domain"],
        }
        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        db.rollback()
        run = db.get(TrainingRun, run_id)
        if run is not None:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            run.log = [*(run.log or []), {"t": datetime.now(timezone.utc).isoformat(), "msg": "Traceback: " + traceback.format_exc()[-2000:]}]
            run.finished_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def model_metrics_dict(mv: ModelVersion) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in mv.metrics:
        out.setdefault(m.split, {})[m.metric] = m.value
        out[m.split]["n"] = m.n_samples
        if m.metric == "mape":
            out[m.split]["mape_coverage"] = (m.detail or {}).get("mape_coverage")
    return out


def model_summary(mv: ModelVersion) -> dict[str, Any]:
    return {
        "id": mv.id, "training_run_id": mv.training_run_id, "dataset_version_id": mv.dataset_version_id, "target": mv.target,
        "target_unit": (mv.feature_schema or {}).get("target_unit"), "algorithm": mv.algorithm,
        "label": candidate_models(0)[mv.algorithm].label if mv.algorithm in candidate_models(0) else mv.algorithm,
        "version_label": mv.version_label, "status": mv.status, "is_best_of_run": mv.is_best_of_run,
        "features": [f["name"] for f in (mv.feature_schema or {}).get("features", [])],
        "hyperparameters": (mv.hyperparameters or {}).get("best_params", {}),
        "metrics": model_metrics_dict(mv), "training_duration_s": mv.training_duration_s, "n_train": mv.n_train, "n_test": mv.n_test,
        "trained_at": mv.trained_at.isoformat() if mv.trained_at else None, "promoted_at": mv.promoted_at.isoformat() if mv.promoted_at else None,
        "interval": mv.interval, "split": (mv.cv_results or {}).get("split", {}),
        "cv_rmse_mean": (mv.cv_results or {}).get("cv_rmse_mean"), "cv_rmse_std": (mv.cv_results or {}).get("cv_rmse_std"),
    }


def artifact_exists(mv: ModelVersion) -> bool:
    return bool(mv.artifact_path) and Path(mv.artifact_path).exists()
