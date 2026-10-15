"""Experimental-analysis chart data (measured observations only)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import DB, Auth
from app.models import DatasetVersion
from app.services import analysis as svc
from app.services.datasets import version_dataframe

router = APIRouter(prefix="/api/analysis", tags=["analysis"], dependencies=[Auth])


def _df(db: Session, version_id: int):
    v = db.get(DatasetVersion, version_id)
    if v is None:
        raise HTTPException(404, "Dataset version not found")
    df = version_dataframe(db, version_id)
    if df.empty:
        raise HTTPException(404, "Dataset version has no records")
    return df


def _target_ok(df, target: str) -> None:
    if target not in df.columns or df[target].notna().sum() == 0:
        raise HTTPException(404, f"Target '{target}' has no measurements in this dataset version")


@router.get("/{version_id}/overview")
def overview(version_id: int, db: Session = DB) -> dict[str, Any]:
    return svc.overview(_df(db, version_id))


@router.get("/{version_id}/target-vs-time")
def target_vs_time(version_id: int, target: str = Query("cof", pattern="^(cof|wear_rate)$"), group_by: str = "material", bins: int = Query(80, ge=10, le=400), db: Session = DB) -> dict[str, Any]:
    df = _df(db, version_id)
    _target_ok(df, target)
    return svc.target_vs_time(df, target, group_by, bins)


@router.get("/{version_id}/target-vs-feature")
def target_vs_feature(version_id: int, feature: str, target: str = Query("cof", pattern="^(cof|wear_rate)$"), group_by: str = "material", db: Session = DB) -> dict[str, Any]:
    df = _df(db, version_id)
    _target_ok(df, target)
    try:
        return svc.target_vs_feature(df, target, feature, group_by)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{version_id}/material-comparison")
def material_comparison(version_id: int, target: str = Query("cof", pattern="^(cof|wear_rate)$"), group_by: str = "material", db: Session = DB) -> dict[str, Any]:
    df = _df(db, version_id)
    _target_ok(df, target)
    return svc.material_comparison(df, target, group_by)


@router.get("/{version_id}/correlation")
def correlation(version_id: int, target: str = Query("cof", pattern="^(cof|wear_rate)$"), db: Session = DB) -> dict[str, Any]:
    df = _df(db, version_id)
    _target_ok(df, target)
    return svc.correlation(df, target)


@router.get("/{version_id}/distributions")
def distributions(version_id: int, target: str = Query("cof", pattern="^(cof|wear_rate)$"), db: Session = DB) -> dict[str, Any]:
    df = _df(db, version_id)
    _target_ok(df, target)
    return svc.distributions(df, target)


@router.get("/{version_id}/trends")
def trends(version_id: int, target: str = Query("cof", pattern="^(cof|wear_rate)$"), db: Session = DB) -> dict[str, Any]:
    df = _df(db, version_id)
    _target_ok(df, target)
    return svc.trends(df, target)
