"""User-adjustable application settings persisted in the database."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models import AppSetting

DEFAULT_SETTINGS: dict[str, Any] = {
    "extrapolation_margin": 0.10,  # fraction of the training range tolerated outside [min, max] with a warning
    "allow_far_extrapolation": False,  # when True, far-out-of-domain requests are predicted with a strong warning
    "fixed_condition_tolerance": 0.05,  # relative tolerance for fixed conditions such as load = 50 N
    "interval_level": 0.90,  # conformal prediction-interval level
    "default_split_strategy": "auto",
    "default_search_mode": "standard",  # standard | quick
    "default_models": ["linear", "polynomial", "random_forest", "svr", "xgboost", "mlp"],
    "default_test_fraction": 0.2,
    "default_n_folds": 5,
    "random_seed": 42,
    "project": {
        "title": "Data-Driven Prediction of Lubricant Friction and Wear Using Machine Learning",
        "subtitle": "An Experimental and Machine Learning Approach to Data-Enabled Tribology",
        "organisation": "Mechanical Engineering Major Project",
        "author": "",
    },
}

SETTINGS_KEY = "app"


def get_app_settings(db: Session) -> dict[str, Any]:
    row = db.get(AppSetting, SETTINGS_KEY)
    merged = {**DEFAULT_SETTINGS}
    if row and isinstance(row.value, dict):
        for k, v in row.value.items():
            if k == "project" and isinstance(v, dict):
                merged["project"] = {**DEFAULT_SETTINGS["project"], **v}
            else:
                merged[k] = v
    return merged


def update_app_settings(db: Session, patch: dict[str, Any]) -> dict[str, Any]:
    unknown = [k for k in patch if k not in DEFAULT_SETTINGS]
    if unknown:
        raise ValueError(f"Unknown setting(s): {unknown}")
    row = db.get(AppSetting, SETTINGS_KEY)
    current = dict(row.value) if row and isinstance(row.value, dict) else {}
    for k, v in patch.items():
        if k == "project" and isinstance(v, dict):
            current["project"] = {**current.get("project", {}), **v}
        else:
            current[k] = v
    if row is None:
        row = AppSetting(key=SETTINGS_KEY, value=current)
        db.add(row)
    else:
        row.value = current
    db.commit()
    return get_app_settings(db)
