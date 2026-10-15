"""Overview dashboard statistics (all computed from stored data; nothing hard-coded)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import DB, Auth
from app.core.features import TARGET_INFO
from app.models import Dataset, DatasetVersion, ExperimentalRecord, ModelVersion, Prediction, TrainingRun, ValidationRecord
from app.services.analysis import overview as analysis_overview
from app.services.datasets import version_dataframe, version_summary
from app.services.training import model_summary

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"], dependencies=[Auth])


@router.get("/overview")
def overview(dataset_version_id: int | None = None, material: str | None = None, lubricant: str | None = None,
             load_min: float | None = None, load_max: float | None = None, speed_min: float | None = None, speed_max: float | None = None,
             temp_min: float | None = None, temp_max: float | None = None, db: Session = DB) -> dict[str, Any]:
    version = db.get(DatasetVersion, dataset_version_id) if dataset_version_id else db.scalars(
        select(DatasetVersion).where(DatasetVersion.kind != "raw").order_by(DatasetVersion.id.desc())).first() or db.scalars(
        select(DatasetVersion).order_by(DatasetVersion.id.desc())).first()
    out: dict[str, Any] = {
        "n_datasets": int(db.scalar(select(func.count()).select_from(Dataset)) or 0),
        "n_versions": int(db.scalar(select(func.count()).select_from(DatasetVersion)) or 0),
        "n_models": int(db.scalar(select(func.count()).select_from(ModelVersion)) or 0),
        "n_training_runs": int(db.scalar(select(func.count()).select_from(TrainingRun)) or 0),
        "n_predictions": int(db.scalar(select(func.count()).select_from(Prediction)) or 0),
        "n_validations": int(db.scalar(select(func.count()).select_from(ValidationRecord)) or 0),
        "dataset": None, "version": None, "data": None, "filtered": None, "models": {}, "recent_predictions": [], "recent_validations": [],
        "recent_runs": [],
    }
    if version is not None:
        ds = db.get(Dataset, version.dataset_id)
        out["dataset"] = {"id": ds.id, "name": ds.name, "source_type": ds.source_type}
        out["version"] = version_summary(version)
        out["quality"] = {"score": (version.quality_report or {}).get("score"), "summary": (version.quality_report or {}).get("summary"),
                          "flags": (version.quality_report or {}).get("flags")}
        df = version_dataframe(db, version.id)
        out["data"] = analysis_overview(df) if not df.empty else None
        f = df[df["is_usable"]] if "is_usable" in df.columns else df
        if material:
            f = f[f["material"] == material]
        if lubricant:
            f = f[f["lubricant"] == lubricant]
        for col, lo, hi in (("normal_load_n", load_min, load_max), ("sliding_speed_m_s", speed_min, speed_max), ("temperature_c", temp_min, temp_max)):
            if lo is not None:
                f = f[f[col] >= lo]
            if hi is not None:
                f = f[f[col] <= hi]
        out["filtered"] = {"n_rows": int(len(f)), "n_runs": int(f["test_run_id"].nunique()) if len(f) else 0,
                           "cof_mean": float(f["cof"].mean()) if len(f) and f["cof"].notna().any() else None,
                           "cof_min": float(f["cof"].min()) if len(f) and f["cof"].notna().any() else None,
                           "cof_max": float(f["cof"].max()) if len(f) and f["cof"].notna().any() else None,
                           "wear_rate_mean": float(f["wear_rate"].mean()) if len(f) and f["wear_rate"].notna().any() else None,
                           "materials": sorted(map(str, f["material"].dropna().unique())) if len(f) else []}
    for t in TARGET_INFO:
        prod = db.scalars(select(ModelVersion).where(ModelVersion.target == t, ModelVersion.status == "production").order_by(ModelVersion.promoted_at.desc())).first()
        best = db.scalars(select(ModelVersion).where(ModelVersion.target == t, ModelVersion.status != "archived", ModelVersion.is_best_of_run.is_(True)).order_by(ModelVersion.id.desc())).first()
        active = prod or best
        out["models"][t] = {"label": TARGET_INFO[t]["label"], "active": model_summary(active) if active else None, "is_production": bool(prod),
                            "reason": None if active else ("No wear measurements in the dataset — wear-rate model cannot be trained" if t == "wear_rate" and version is not None and not (version.feature_summary or {}).get("targets", {}).get("wear_rate", {}).get("available") else "Model not trained")}
    out["recent_predictions"] = [{"id": p.id, "inputs": p.inputs, "predicted_cof": p.predicted_cof, "predicted_wear_rate": p.predicted_wear_rate, "in_domain": p.in_domain,
                                  "source": p.source, "created_at": p.created_at.isoformat() if p.created_at else None}
                                 for p in db.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(8)).all()]
    out["recent_validations"] = [{"id": v.id, "experiment_id": v.experiment_id, "measured_cof": v.measured_cof, "predicted_cof": v.predicted_cof,
                                  "cof_abs_error": v.cof_abs_error, "status": v.status, "created_at": v.created_at.isoformat() if v.created_at else None}
                                 for v in db.scalars(select(ValidationRecord).order_by(ValidationRecord.id.desc()).limit(5)).all()]
    out["recent_runs"] = [{"id": r.id, "target": r.target, "status": r.status, "best_algorithm": (r.summary or {}).get("best_algorithm"),
                           "created_at": r.created_at.isoformat() if r.created_at else None}
                          for r in db.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).limit(5)).all()]
    out["n_records_total"] = int(db.scalar(select(func.count()).select_from(ExperimentalRecord)) or 0)
    return out
