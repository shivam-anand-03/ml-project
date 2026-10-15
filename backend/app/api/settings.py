"""Application settings, data dictionary and health."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import DB, Auth
from app.api.schemas import SettingsPatch
from app.config import get_settings
from app.core.features import MATERIAL_CATALOGUE, feature_dictionary
from app.ml.pipeline import candidate_models
from app.services.app_settings import DEFAULT_SETTINGS, get_app_settings, update_app_settings
from app.services.jobs import get_runner

router = APIRouter(prefix="/api/settings", tags=["settings"], dependencies=[Auth])


@router.get("")
def read_settings(db: Session = DB) -> dict[str, Any]:
    s = get_settings()
    return {"settings": get_app_settings(db), "defaults": DEFAULT_SETTINGS, "available_models": list(candidate_models(0).keys()),
            "environment": {"app_version": s.app_version, "environment": s.environment, "database": s.database_url.split("://", 1)[0],
                            "storage_dir": str(s.storage_dir), "api_key_required": bool(s.api_key), "training_workers": s.training_workers,
                            "active_training_jobs": get_runner().active_count()}}


@router.put("")
def write_settings(body: SettingsPatch, db: Session = DB) -> dict[str, Any]:
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    if "default_models" in patch:
        bad = [m for m in patch["default_models"] if m not in candidate_models(0)]
        if bad:
            raise HTTPException(400, f"Unknown model(s): {bad}")
    try:
        return {"settings": update_app_settings(db, patch)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/data-dictionary")
def data_dictionary() -> dict[str, Any]:
    return {"features": feature_dictionary(), "material_catalogue": MATERIAL_CATALOGUE,
            "equations": [
                {"name": "Coefficient of friction", "formula": "COF = F_friction / F_normal", "unit": "–",
                 "note": "Computed per sample from the measured friction force and the applied normal load."},
                {"name": "Specific wear rate", "formula": "k = V / (F × s)", "unit": "mm³/(N·m) when V in mm³, F in N, s in m (equivalently m³/(N·m) = m²/N in SI)",
                 "note": "Requires wear volume V, normal load F and sliding distance s. Never derived from wear mass without a documented density."},
                {"name": "Sliding distance", "formula": "s = v × t", "unit": "m", "note": "v = sliding speed (m/s), t = time (s)."},
                {"name": "Sliding speed from rotation", "formula": "v = π × D × n / 60", "unit": "m/s", "note": "D = track diameter (m), n = rpm."},
            ]}


health_router = APIRouter(tags=["health"])


@health_router.get("/api/health")
def health(db: Session = DB) -> dict[str, Any]:
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    s = get_settings()
    return {"status": "ok" if db_ok else "degraded", "database": db_ok, "app": s.app_name, "version": s.app_version}
