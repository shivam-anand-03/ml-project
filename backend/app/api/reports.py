"""Report generation and download."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import DB, Auth
from app.api.schemas import ReportRequest
from app.config import get_settings
from app.models import Report
from app.services.reports import ReportError, generate_report, report_view

router = APIRouter(prefix="/api/reports", tags=["reports"], dependencies=[Auth])


@router.post("/generate", status_code=201)
def generate(body: ReportRequest, db: Session = DB) -> dict[str, Any]:
    try:
        rep = generate_report(db, get_settings(), body.model_dump())
    except ReportError as exc:
        raise HTTPException(400, str(exc)) from exc
    return report_view(rep)


@router.get("")
def list_reports(db: Session = DB) -> list[dict[str, Any]]:
    return [report_view(r) for r in db.scalars(select(Report).order_by(Report.id.desc())).all()]


@router.get("/{report_id}/download")
def download(report_id: int, db: Session = DB) -> FileResponse:
    r = db.get(Report, report_id)
    if r is None:
        raise HTTPException(404, "Report not found")
    p = Path(r.file_path)
    if not p.exists():
        raise HTTPException(410, "Report file no longer exists")
    media = "application/pdf" if r.kind == "pdf" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(p, media_type=media, filename=p.name)


@router.delete("/{report_id}", status_code=204, response_class=Response, response_model=None)
def delete(report_id: int, db: Session = DB) -> None:
    r = db.get(Report, report_id)
    if r is None:
        raise HTTPException(404, "Report not found")
    Path(r.file_path).unlink(missing_ok=True)
    db.delete(r)
    db.commit()
