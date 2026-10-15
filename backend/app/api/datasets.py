"""Dataset management endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import DB, Auth
from app.api.schemas import IngestRequest, PreprocessRequest
from app.config import get_settings
from app.core.features import feature_dictionary
from app.models import Dataset, DatasetFile, DatasetVersion, ExperimentalRecord, ModelVersion
from app.services import datasets as svc
from app.services.ingestion import IngestionError, to_jsonable
from app.services.preprocessing import PreprocessOptions

router = APIRouter(prefix="/api/datasets", tags=["datasets"], dependencies=[Auth])


def _dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    return ds


def _version_or_404(db: Session, dataset: Dataset, version_id: int | None) -> DatasetVersion:
    if version_id is None:
        if not dataset.versions:
            raise HTTPException(404, "Dataset has no versions yet — ingest a worksheet first")
        return sorted(dataset.versions, key=lambda v: v.version_number)[-1]
    v = db.get(DatasetVersion, version_id)
    if v is None or v.dataset_id != dataset.id:
        raise HTTPException(404, f"Version {version_id} not found in dataset {dataset.id}")
    return v


def _file_view(f: DatasetFile, with_preview: bool = False) -> dict[str, Any]:
    insp = dict(f.inspection or {})
    sheets = []
    for s in insp.get("sheets", []):
        s2 = dict(s)
        if not with_preview:
            s2.pop("preview", None)
        sheets.append(s2)
    insp["sheets"] = sheets
    return {"id": f.id, "filename": f.filename, "file_type": f.file_type, "size_bytes": f.size_bytes,
            "uploaded_at": f.uploaded_at.isoformat() if f.uploaded_at else None, "inspection": insp}


@router.get("")
def list_datasets(db: Session = DB) -> list[dict[str, Any]]:
    return [svc.dataset_summary(db, d) for d in db.scalars(select(Dataset).order_by(Dataset.id.desc())).all()]


@router.post("/upload", status_code=201)
async def upload_dataset(
    files: list[UploadFile] = File(..., description="One or more .xlsx / .xls / .csv / .dwf files"),
    name: str = Form(...), description: str | None = Form(None), db: Session = DB,
) -> dict[str, Any]:
    if not files:
        raise HTTPException(400, "No files uploaded")
    ds = svc.create_dataset(db, name, description)
    stored, errors = [], []
    for up in files:
        content = await up.read()
        try:
            stored.append(svc.store_upload(db, ds, up.filename or "upload", content))
        except IngestionError as exc:
            errors.append({"filename": up.filename, "error": str(exc)})
    if not stored:
        db.delete(ds)
        db.commit()
        raise HTTPException(400, {"message": "No file could be read", "errors": errors})
    db.refresh(ds)
    return {"dataset": svc.dataset_summary(db, ds), "files": [_file_view(f) for f in stored], "errors": errors}


@router.post("/{dataset_id}/files", status_code=201)
async def add_files(dataset_id: int, files: list[UploadFile] = File(...), db: Session = DB) -> dict[str, Any]:
    ds = _dataset_or_404(db, dataset_id)
    stored, errors = [], []
    for up in files:
        try:
            stored.append(svc.store_upload(db, ds, up.filename or "upload", await up.read()))
        except IngestionError as exc:
            errors.append({"filename": up.filename, "error": str(exc)})
    return {"files": [_file_view(f) for f in stored], "errors": errors}


@router.post("/import-bundled", status_code=201)
def import_bundled(db: Session = DB) -> dict[str, Any]:
    """Import the bundled Kolev 2023 pin-on-disk workbooks (raw + default processed version)."""
    try:
        ds, raw, processed = svc.import_bundled_dataset(db, get_settings())
    except IngestionError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"dataset": svc.dataset_summary(db, ds), "raw_version": svc.version_summary(raw), "processed_version": svc.version_summary(processed)}


@router.get("/feature-dictionary")
def get_feature_dictionary() -> list[dict[str, Any]]:
    return feature_dictionary()


@router.get("/preprocess-options")
def preprocess_options() -> dict[str, Any]:
    d = PreprocessOptions()
    return {k: v for k, v in d.__dict__.items() if k != "material_catalogue"}


@router.get("/versions/{version_id}")
def get_version(version_id: int, db: Session = DB) -> dict[str, Any]:
    v = db.get(DatasetVersion, version_id)
    if v is None:
        raise HTTPException(404, "Version not found")
    models = db.scalars(select(ModelVersion).where(ModelVersion.dataset_version_id == v.id)).all()
    return {**svc.version_summary(v), "column_mapping": v.column_mapping, "cleaning_log": v.cleaning_log, "quality_report": v.quality_report,
            "models_trained": [{"id": m.id, "algorithm": m.algorithm, "target": m.target, "status": m.status, "n_train": m.n_train, "n_test": m.n_test} for m in models]}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = DB) -> dict[str, Any]:
    ds = _dataset_or_404(db, dataset_id)
    return {**svc.dataset_summary(db, ds), "files": [_file_view(f) for f in ds.files],
            "versions": [svc.version_summary(v) for v in sorted(ds.versions, key=lambda v: v.version_number)]}


@router.delete("/{dataset_id}", status_code=204, response_class=Response, response_model=None)
def delete_dataset(dataset_id: int, db: Session = DB) -> None:
    ds = _dataset_or_404(db, dataset_id)
    n_models = db.scalar(select(func.count()).select_from(ModelVersion).join(DatasetVersion, ModelVersion.dataset_version_id == DatasetVersion.id).where(DatasetVersion.dataset_id == ds.id))
    if n_models:
        raise HTTPException(409, f"Dataset has {n_models} trained model version(s); archive/delete them first")
    svc.delete_dataset(db, ds)


@router.get("/{dataset_id}/files/{file_id}/inspection")
def file_inspection(dataset_id: int, file_id: int, db: Session = DB) -> dict[str, Any]:
    ds = _dataset_or_404(db, dataset_id)
    f = db.get(DatasetFile, file_id)
    if f is None or f.dataset_id != ds.id:
        raise HTTPException(404, "File not found")
    return _file_view(f, with_preview=True)


@router.post("/{dataset_id}/ingest", status_code=201)
def ingest(dataset_id: int, body: IngestRequest, db: Session = DB) -> dict[str, Any]:
    ds = _dataset_or_404(db, dataset_id)
    specs = [svc.SheetSpec(s.file_id, s.sheet_name, s.mapping, s.constants, s.units) for s in body.sheets]
    try:
        v = svc.build_raw_version(db, ds, specs, body.notes)
    except IngestionError as exc:
        db.rollback()
        raise HTTPException(400, str(exc)) from exc
    return svc.version_summary(v)


@router.post("/{dataset_id}/preprocess", status_code=201)
def preprocess(dataset_id: int, body: PreprocessRequest, db: Session = DB) -> dict[str, Any]:
    ds = _dataset_or_404(db, dataset_id)
    parent = _version_or_404(db, ds, body.version_id)
    try:
        v = svc.preprocess_version(db, parent, body.options, body.notes)
    except (IngestionError, ValueError) as exc:
        db.rollback()
        raise HTTPException(400, str(exc)) from exc
    return svc.version_summary(v)


@router.get("/{dataset_id}/versions")
def list_versions(dataset_id: int, db: Session = DB) -> list[dict[str, Any]]:
    ds = _dataset_or_404(db, dataset_id)
    return [svc.version_summary(v) for v in sorted(ds.versions, key=lambda v: v.version_number)]


@router.get("/{dataset_id}/preview")
def preview(dataset_id: int, version_id: int | None = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
            material: str | None = None, test_run_id: str | None = None, usable: bool | None = None, db: Session = DB) -> dict[str, Any]:
    ds = _dataset_or_404(db, dataset_id)
    v = _version_or_404(db, ds, version_id)
    stmt = select(ExperimentalRecord).where(ExperimentalRecord.dataset_version_id == v.id)
    if material:
        stmt = stmt.where(ExperimentalRecord.material == material)
    if test_run_id:
        stmt = stmt.where(ExperimentalRecord.test_run_id == test_run_id)
    if usable is not None:
        stmt = stmt.where(ExperimentalRecord.is_usable.is_(usable))
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    recs = db.scalars(stmt.order_by(ExperimentalRecord.id).offset(offset).limit(limit)).all()
    cols = ["id", "test_run_id", "material", "composition", "pore_size_class", "material_pair", "lubricant", "normal_load_n", "sliding_speed_m_s",
            "temperature_c", "time_s", "sliding_distance_m", "rpm", "track_diameter_mm", "friction_force_n", "cof", "wear_volume_mm3", "wear_mass_mg",
            "wear_rate", "is_usable", "flags", "source"]
    return {"version": svc.version_summary(v), "total": int(total or 0), "offset": offset, "limit": limit, "columns": cols,
            "rows": [{c: to_jsonable(getattr(r, c)) for c in cols} for r in recs]}


@router.get("/{dataset_id}/quality")
def quality(dataset_id: int, version_id: int | None = None, db: Session = DB) -> dict[str, Any]:
    ds = _dataset_or_404(db, dataset_id)
    v = _version_or_404(db, ds, version_id)
    return {"version": svc.version_summary(v), "quality_report": v.quality_report, "cleaning_log": v.cleaning_log}


@router.get("/{dataset_id}/quality/report.md", response_class=PlainTextResponse)
def quality_markdown(dataset_id: int, version_id: int | None = None, db: Session = DB) -> str:
    ds = _dataset_or_404(db, dataset_id)
    v = _version_or_404(db, ds, version_id)
    return svc.quality_report_markdown(ds, v)


@router.get("/{dataset_id}/export")
def export(dataset_id: int, version_id: int | None = None, format: str = Query("csv", pattern="^(csv|xlsx)$"), db: Session = DB) -> FileResponse:
    ds = _dataset_or_404(db, dataset_id)
    v = _version_or_404(db, ds, version_id)
    path = svc.export_version(db, v, format, get_settings())
    media = "text/csv" if format == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(path, media_type=media, filename=path.name)
