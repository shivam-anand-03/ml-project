"""Experimental validation loop: record measured results, compare with predictions, augment datasets."""
from __future__ import annotations

from typing import Any

import numpy as np
from fastapi.responses import Response
from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import DB, Auth
from app.api.schemas import AugmentRequest, ValidationCreate, ValidationUpdate
from app.models import DatasetVersion, ModelVersion, Prediction, ValidationRecord
from app.services import prediction as psvc
from app.services.datasets import augment_version, version_summary

router = APIRouter(prefix="/api/validation", tags=["validation"], dependencies=[Auth])


def _view(v: ValidationRecord) -> dict[str, Any]:
    return {"id": v.id, "experiment_id": v.experiment_id, "prediction_id": v.prediction_id, "model_version_id": v.model_version_id, "inputs": v.inputs,
            "measured_cof": v.measured_cof, "measured_wear_rate": v.measured_wear_rate, "predicted_cof": v.predicted_cof,
            "predicted_wear_rate": v.predicted_wear_rate, "cof_abs_error": v.cof_abs_error, "cof_pct_error": v.cof_pct_error,
            "wear_abs_error": v.wear_abs_error, "wear_pct_error": v.wear_pct_error, "status": v.status, "notes": v.notes,
            "added_to_version_id": v.added_to_version_id, "created_at": v.created_at.isoformat() if v.created_at else None}


def _pct(measured: float | None, predicted: float | None) -> float | None:
    if measured is None or predicted is None or abs(measured) < 1e-9:
        return None
    return 100.0 * abs(measured - predicted) / abs(measured)


@router.post("", status_code=201)
def create(body: ValidationCreate, db: Session = DB) -> dict[str, Any]:
    if body.measured_cof is None and body.measured_wear_rate is None:
        raise HTTPException(400, "Enter at least one measured result (COF or wear rate)")
    predicted_cof = predicted_wear = None
    model_id = None
    inputs = dict(body.inputs)
    if body.prediction_id is not None:
        p = db.get(Prediction, body.prediction_id)
        if p is None:
            raise HTTPException(404, "Prediction not found")
        predicted_cof, predicted_wear, model_id = p.predicted_cof, p.predicted_wear_rate, p.cof_model_version_id
        inputs = {**(p.inputs or {}), **inputs}
    else:
        try:
            r = psvc.predict_record(db, inputs, body.cof_model_id, body.wear_model_id, explain=False, save=True, source="validation")
        except psvc.PredictionError as exc:
            raise HTTPException(422, f"Cannot compare: {exc}") from exc
        except psvc.ModelUnavailable as exc:
            raise HTTPException(409, str(exc)) from exc
        predicted_cof = r["cof"]["value"]
        predicted_wear = r["wear_rate"]["value"]
        model_id = r["cof"]["model"]["id"] if r["cof"]["available"] else (r["wear_rate"]["model"]["id"] if r["wear_rate"]["available"] else None)
        body.prediction_id = r.get("prediction_id")
    rec = ValidationRecord(
        prediction_id=body.prediction_id, model_version_id=model_id, experiment_id=body.experiment_id, inputs=inputs,
        measured_cof=body.measured_cof, measured_wear_rate=body.measured_wear_rate, predicted_cof=predicted_cof, predicted_wear_rate=predicted_wear,
        cof_abs_error=(abs(body.measured_cof - predicted_cof) if body.measured_cof is not None and predicted_cof is not None else None),
        cof_pct_error=_pct(body.measured_cof, predicted_cof),
        wear_abs_error=(abs(body.measured_wear_rate - predicted_wear) if body.measured_wear_rate is not None and predicted_wear is not None else None),
        wear_pct_error=_pct(body.measured_wear_rate, predicted_wear), notes=body.notes,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return _view(rec)


@router.get("")
def list_records(status: str | None = None, db: Session = DB) -> list[dict[str, Any]]:
    stmt = select(ValidationRecord)
    if status:
        stmt = stmt.where(ValidationRecord.status == status)
    return [_view(v) for v in db.scalars(stmt.order_by(ValidationRecord.id.desc())).all()]


@router.get("/summary")
def summary(db: Session = DB) -> dict[str, Any]:
    recs = db.scalars(select(ValidationRecord)).all()
    cof_err = [r.cof_abs_error for r in recs if r.cof_abs_error is not None]
    cof_pct = [r.cof_pct_error for r in recs if r.cof_pct_error is not None]
    wear_err = [r.wear_abs_error for r in recs if r.wear_abs_error is not None]
    by_model: dict[int, list[float]] = {}
    for r in recs:
        if r.cof_abs_error is not None and r.model_version_id:
            by_model.setdefault(r.model_version_id, []).append(r.cof_abs_error)
    models = {}
    for mid, errs in by_model.items():
        mv = db.get(ModelVersion, mid)
        models[mid] = {"algorithm": mv.algorithm if mv else None, "version_label": mv.version_label if mv else None, "n": len(errs),
                       "mae": float(np.mean(errs)), "rmse": float(np.sqrt(np.mean(np.square(errs))))}
    return {"n_records": len(recs), "n_pending": sum(1 for r in recs if r.status == "pending"), "n_approved": sum(1 for r in recs if r.status == "approved"),
            "n_rejected": sum(1 for r in recs if r.status == "rejected"),
            "cof": {"n": len(cof_err), "mae": float(np.mean(cof_err)) if cof_err else None, "rmse": float(np.sqrt(np.mean(np.square(cof_err)))) if cof_err else None,
                    "mape": float(np.mean(cof_pct)) if cof_pct else None, "max_abs_error": float(np.max(cof_err)) if cof_err else None},
            "wear_rate": {"n": len(wear_err), "mae": float(np.mean(wear_err)) if wear_err else None},
            "by_model": models}


@router.patch("/{record_id}")
def update(record_id: int, body: ValidationUpdate, db: Session = DB) -> dict[str, Any]:
    rec = db.get(ValidationRecord, record_id)
    if rec is None:
        raise HTTPException(404, "Validation record not found")
    if body.status is not None:
        if rec.added_to_version_id and body.status != "approved":
            raise HTTPException(409, "This record was already added to a dataset version and cannot be un-approved")
        rec.status = body.status
    if body.notes is not None:
        rec.notes = body.notes
    db.commit()
    return _view(rec)


@router.delete("/{record_id}", status_code=204, response_class=Response, response_model=None)
def delete(record_id: int, db: Session = DB) -> None:
    rec = db.get(ValidationRecord, record_id)
    if rec is None:
        raise HTTPException(404, "Validation record not found")
    if rec.added_to_version_id:
        raise HTTPException(409, "Record already belongs to a dataset version")
    db.delete(rec)
    db.commit()


@router.post("/augment", status_code=201)
def augment(body: AugmentRequest, db: Session = DB) -> dict[str, Any]:
    """Create a new dataset version = parent + the approved validation records (never overwrites the parent)."""
    parent = db.get(DatasetVersion, body.dataset_version_id)
    if parent is None:
        raise HTTPException(404, "Dataset version not found")
    recs = db.scalars(select(ValidationRecord).where(ValidationRecord.id.in_(body.validation_ids))).all()
    missing = set(body.validation_ids) - {r.id for r in recs}
    if missing:
        raise HTTPException(404, f"Validation record(s) not found: {sorted(missing)}")
    not_approved = [r.id for r in recs if r.status != "approved"]
    if not_approved:
        raise HTTPException(409, f"Only approved records can be added; not approved: {not_approved}")
    already = [r.id for r in recs if r.added_to_version_id]
    if already:
        raise HTTPException(409, f"Record(s) already added to a dataset version: {already}")
    v = augment_version(db, parent, recs, body.notes or f"Augmented with {len(recs)} experimental validation record(s)")
    return version_summary(v)
