"""Training runs, model registry, metrics, promotion and explainability."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import DB, Auth
from app.api.schemas import LocalExplainRequest, TrainRequest
from app.config import get_settings
from app.core.features import TARGET_INFO
from app.ml.pipeline import candidate_models
from app.models import DatasetVersion, ModelVersion, TrainingRun
from app.services.app_settings import get_app_settings
from app.services.jobs import get_runner
from app.services.prediction import STORE, ModelUnavailable, PredictionError, validate_inputs
from app.services.training import TrainingConfig, artifact_exists, model_summary

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Auth])

METRIC_EXPLANATIONS = {
    "r2": "R² (coefficient of determination): share of the variance in the measured target that the model reproduces. 1.0 is perfect; 0 means no better than predicting the mean; negative is worse than the mean.",
    "mae": "MAE (mean absolute error): average absolute difference between predicted and measured values, in the target's own unit. Easy to read as a typical error.",
    "rmse": "RMSE (root mean squared error): like MAE but penalises large errors more strongly; in the target's unit. Used to rank models.",
    "mape": "MAPE (mean absolute percentage error): average relative error in %. Rows whose measured value is near zero (e.g. run-in COF ≈ 0) are excluded, so the reported coverage shows the share of rows used.",
    "validation": "Validation metrics come from out-of-fold cross-validation predictions on the training runs — the model never saw the row (or, for grouped splits, the run) it is scored on.",
    "test": "Test metrics come from the untouched held-out set that was never used for tuning or model selection.",
}


def _run_view(run: TrainingRun) -> dict[str, Any]:
    return {"id": run.id, "dataset_version_id": run.dataset_version_id, "target": run.target, "status": run.status, "config": run.config,
            "summary": run.summary, "error": run.error, "log": run.log, "created_at": run.created_at.isoformat() if run.created_at else None,
            "started_at": run.started_at.isoformat() if run.started_at else None, "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "is_active": get_runner().is_active(run.id)}


@router.get("/candidates")
def candidates() -> list[dict[str, Any]]:
    return [{"key": c.key, "label": c.label, "family": c.family, "grid": {k.replace("model__", ""): [str(x) for x in v] for k, v in c.grid.items()},
             "quick_grid": {k.replace("model__", ""): [str(x) for x in v] for k, v in c.quick_grid.items()}, "notes": c.notes}
            for c in candidate_models(0).values()]


@router.get("/metric-explanations")
def metric_explanations() -> dict[str, str]:
    return METRIC_EXPLANATIONS


@router.post("/train", status_code=202)
def train(body: TrainRequest, db: Session = DB) -> dict[str, Any]:
    v = db.get(DatasetVersion, body.dataset_version_id)
    if v is None:
        raise HTTPException(404, "Dataset version not found")
    fs = v.feature_summary or {}
    if not fs.get("targets", {}).get(body.target, {}).get("available"):
        raise HTTPException(400, f"Target '{body.target}' ({TARGET_INFO[body.target]['label']}) has no measurements in dataset version {v.version_number}; "
                                 "a model for it cannot be trained. Upload a dataset that contains this measurement.")
    app = get_app_settings(db)
    cfg = body.model_dump()
    cfg["seed"] = body.seed if body.seed is not None else int(app.get("random_seed", get_settings().random_seed))
    try:
        TrainingConfig.from_dict(cfg)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    run = TrainingRun(dataset_version_id=v.id, target=body.target, status="queued", config=cfg)
    db.add(run)
    db.commit()
    db.refresh(run)
    get_runner().submit(run.id)
    db.refresh(run)
    return _run_view(run)


@router.get("/runs")
def list_runs(limit: int = Query(20, ge=1, le=200), db: Session = DB) -> list[dict[str, Any]]:
    return [_run_view(r) for r in db.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).limit(limit)).all()]


@router.get("/train/{run_id}")
def get_run(run_id: int, db: Session = DB) -> dict[str, Any]:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    view = _run_view(run)
    view["models"] = [model_summary(m) for m in run.models]
    return view


@router.get("")
def list_models(target: str | None = None, status: str | None = None, dataset_version_id: int | None = None, db: Session = DB) -> list[dict[str, Any]]:
    stmt = select(ModelVersion)
    if target:
        stmt = stmt.where(ModelVersion.target == target)
    if status:
        stmt = stmt.where(ModelVersion.status == status)
    if dataset_version_id:
        stmt = stmt.where(ModelVersion.dataset_version_id == dataset_version_id)
    return [model_summary(m) for m in db.scalars(stmt.order_by(ModelVersion.id.desc())).all()]


@router.get("/production")
def production_models(db: Session = DB) -> dict[str, Any]:
    out = {}
    for t in TARGET_INFO:
        prod = db.scalars(select(ModelVersion).where(ModelVersion.target == t, ModelVersion.status == "production").order_by(ModelVersion.promoted_at.desc())).first()
        best = db.scalars(select(ModelVersion).where(ModelVersion.target == t, ModelVersion.status != "archived", ModelVersion.is_best_of_run.is_(True)).order_by(ModelVersion.id.desc())).first()
        out[t] = {"production": model_summary(prod) if prod else None, "latest_best": model_summary(best) if best else None,
                  "active": model_summary(prod or best) if (prod or best) else None}
    return out


def _model_or_404(db: Session, model_id: int) -> ModelVersion:
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(404, "Model version not found")
    return mv


@router.get("/{model_id}")
def get_model(model_id: int, db: Session = DB) -> dict[str, Any]:
    mv = _model_or_404(db, model_id)
    return {**model_summary(mv), "feature_schema": mv.feature_schema, "hyperparameters_full": mv.hyperparameters, "training_domain": mv.training_domain,
            "artifact_available": artifact_exists(mv)}


@router.get("/{model_id}/metrics")
def get_metrics(model_id: int, db: Session = DB) -> dict[str, Any]:
    mv = _model_or_404(db, model_id)
    cv = mv.cv_results or {}
    return {**model_summary(mv), "fold_rmse": cv.get("fold_rmse", []), "test_predictions": cv.get("test_predictions", []),
            "oof_predictions": cv.get("oof_predictions", []), "explanations": METRIC_EXPLANATIONS}


@router.post("/{model_id}/promote")
def promote(model_id: int, db: Session = DB) -> dict[str, Any]:
    """Explicitly promote a model version to production (previous production model of the same target is archived)."""
    mv = _model_or_404(db, model_id)
    if not artifact_exists(mv):
        raise HTTPException(409, "Model artifact missing on disk; retrain before promoting")
    for other in db.scalars(select(ModelVersion).where(ModelVersion.target == mv.target, ModelVersion.status == "production", ModelVersion.id != mv.id)).all():
        other.status = "archived"
    mv.status = "production"
    mv.promoted_at = datetime.now(timezone.utc)
    db.commit()
    return model_summary(mv)


@router.post("/{model_id}/archive")
def archive(model_id: int, db: Session = DB) -> dict[str, Any]:
    mv = _model_or_404(db, model_id)
    mv.status = "archived"
    db.commit()
    STORE.clear()
    return model_summary(mv)


@router.get("/{model_id}/explain/global")
def explain_global(model_id: int, db: Session = DB) -> dict[str, Any]:
    mv = _model_or_404(db, model_id)
    gi = mv.global_importance or {}
    return {"model": model_summary(mv), **gi,
            "disclaimer": "SHAP values describe how the fitted model uses its inputs. They do not by themselves establish physical causation."}


@router.post("/{model_id}/explain/local")
def explain_local(model_id: int, body: LocalExplainRequest, db: Session = DB) -> dict[str, Any]:
    mv = _model_or_404(db, model_id)
    try:
        bundle = STORE.bundle(mv)
        v = validate_inputs(bundle, body.inputs, get_app_settings(db))
        if v["refuse"]:
            raise PredictionError(v["refuse"])
        exp = STORE.explainer(mv).local(pd.DataFrame([v["row"]]))[0]
    except PredictionError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"model": model_summary(mv), "inputs": v["row"], "warnings": v["warnings"], **exp,
            "unit": (mv.feature_schema or {}).get("target_unit"),
            "disclaimer": "Contributions explain this single model prediction relative to the model's average output; they are not measured effects."}


@router.get("/compare/explainability")
def compare_explainability(db: Session = DB) -> dict[str, Any]:
    """Side-by-side global importance of the active COF and wear-rate models."""
    out = {}
    for t in TARGET_INFO:
        prod = db.scalars(select(ModelVersion).where(ModelVersion.target == t, ModelVersion.status == "production").order_by(ModelVersion.promoted_at.desc())).first()
        best = prod or db.scalars(select(ModelVersion).where(ModelVersion.target == t, ModelVersion.status != "archived", ModelVersion.is_best_of_run.is_(True)).order_by(ModelVersion.id.desc())).first()
        out[t] = {"available": best is not None, "model": model_summary(best) if best else None,
                  "importance": (best.global_importance or {}).get("importance", []) if best else [],
                  "method": (best.global_importance or {}).get("method") if best else None,
                  "reason": None if best else f"No trained model for {TARGET_INFO[t]['label'].lower()}"}
    return out
