"""Prediction, batch prediction, history and what-if simulation endpoints."""
from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import DB, Auth
from app.api.schemas import BatchPredictRequest, CompareScenariosRequest, PredictRequest, SimulateRequest
from app.models import ModelVersion, Prediction
from app.services import prediction as svc
from app.services.ingestion import to_jsonable

router = APIRouter(prefix="/api", tags=["predictions"], dependencies=[Auth])


def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, svc.PredictionError):
        return HTTPException(422, str(exc))
    if isinstance(exc, svc.ModelUnavailable):
        return HTTPException(409, str(exc))
    raise exc


@router.get("/predict/schema")
def schema(cof_model_id: int | None = None, wear_model_id: int | None = None, db: Session = DB) -> dict[str, Any]:
    try:
        return svc.prediction_schema(db, cof_model_id, wear_model_id)
    except (svc.PredictionError, svc.ModelUnavailable) as exc:
        raise _handle(exc) from exc


@router.post("/predict")
def predict(body: PredictRequest, db: Session = DB) -> dict[str, Any]:
    try:
        return svc.predict_record(db, body.inputs, body.cof_model_id, body.wear_model_id, body.explain, body.save)
    except (svc.PredictionError, svc.ModelUnavailable) as exc:
        raise _handle(exc) from exc


@router.post("/predict/batch")
def predict_batch(body: BatchPredictRequest, db: Session = DB) -> dict[str, Any]:
    try:
        rows = svc.predict_batch(db, body.rows, body.cof_model_id, body.wear_model_id, body.save)
    except svc.ModelUnavailable as exc:
        raise _handle(exc) from exc
    return {"n": len(rows), "n_ok": sum(1 for r in rows if r["ok"]), "results": rows, "note": "Model predictions — not measured values."}


@router.post("/predict/batch/upload")
async def predict_batch_upload(file: UploadFile = File(...), cof_model_id: int | None = None, wear_model_id: int | None = None, db: Session = DB) -> dict[str, Any]:
    content = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"Could not read file: {exc}") from exc
    if df.empty:
        raise HTTPException(400, "The file has no rows")
    if len(df) > 5000:
        raise HTTPException(400, "At most 5000 rows per batch")
    rows = [{k: to_jsonable(v) for k, v in r.items() if pd.notna(v)} for r in df.to_dict(orient="records")]
    try:
        res = svc.predict_batch(db, rows, cof_model_id, wear_model_id, save=False)
    except svc.ModelUnavailable as exc:
        raise _handle(exc) from exc
    return {"n": len(res), "n_ok": sum(1 for r in res if r["ok"]), "columns": list(df.columns), "results": res}


@router.get("/predictions/history")
def history(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), source: str | None = None, db: Session = DB) -> dict[str, Any]:
    stmt = select(Prediction)
    if source:
        stmt = stmt.where(Prediction.source == source)
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = db.scalars(stmt.order_by(Prediction.id.desc()).offset(offset).limit(limit)).all()
    return {"total": int(total or 0), "items": [_pred_view(p) for p in rows]}


def _pred_view(p: Prediction, full: bool = False) -> dict[str, Any]:
    d = {"id": p.id, "inputs": p.inputs, "predicted_cof": p.predicted_cof, "cof_interval": p.cof_interval, "predicted_wear_rate": p.predicted_wear_rate,
         "wear_rate_unit": p.wear_rate_unit, "in_domain": p.in_domain, "warnings": p.warnings, "source": p.source,
         "cof_model_version_id": p.cof_model_version_id, "wear_model_version_id": p.wear_model_version_id, "dataset_version_id": p.dataset_version_id,
         "created_at": p.created_at.isoformat() if p.created_at else None}
    if full:
        d["explanation"] = p.explanation
    return d


@router.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int, db: Session = DB) -> dict[str, Any]:
    p = db.get(Prediction, prediction_id)
    if p is None:
        raise HTTPException(404, "Prediction not found")
    view = _pred_view(p, full=True)
    mv = db.get(ModelVersion, p.cof_model_version_id) if p.cof_model_version_id else None
    view["cof_model"] = {"id": mv.id, "algorithm": mv.algorithm, "version_label": mv.version_label, "status": mv.status} if mv else None
    return view


@router.get("/predictions/{prediction_id}/export")
def export_prediction(prediction_id: int, format: str = Query("json", pattern="^(json|csv)$"), db: Session = DB) -> Response:
    p = db.get(Prediction, prediction_id)
    if p is None:
        raise HTTPException(404, "Prediction not found")
    view = _pred_view(p, full=True)
    view["note"] = "Model prediction — not an experimentally measured value."
    if format == "json":
        return Response(json.dumps(view, indent=2), media_type="application/json", headers={"Content-Disposition": f"attachment; filename=prediction_{p.id}.json"})
    flat = {**{f"input_{k}": v for k, v in (p.inputs or {}).items()}, "predicted_cof": p.predicted_cof,
            "cof_low": (p.cof_interval or {}).get("low"), "cof_high": (p.cof_interval or {}).get("high"), "predicted_wear_rate": p.predicted_wear_rate,
            "in_domain": p.in_domain, "warnings": " | ".join(p.warnings or []), "cof_model_version_id": p.cof_model_version_id, "created_at": view["created_at"]}
    csv = pd.DataFrame([flat]).to_csv(index=False)
    return Response(csv, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=prediction_{p.id}.csv"})


@router.post("/simulate")
def simulate(body: SimulateRequest, db: Session = DB) -> dict[str, Any]:
    try:
        return svc.simulate(db, body.target, body.base_inputs, body.sweep.model_dump() if body.sweep else None, body.compare, body.model_id)
    except (svc.PredictionError, svc.ModelUnavailable) as exc:
        raise _handle(exc) from exc


@router.post("/simulate/compare")
def simulate_compare(body: CompareScenariosRequest, db: Session = DB) -> dict[str, Any]:
    try:
        return svc.compare_scenarios(db, body.target, body.scenarios, body.model_id)
    except (svc.PredictionError, svc.ModelUnavailable) as exc:
        raise _handle(exc) from exc
