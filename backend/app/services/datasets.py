"""Dataset lifecycle: uploads, inspection, raw/processed/augmented versions, exports, bundled import."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.core.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RECORD_COLUMNS
from app.models import Dataset, DatasetFile, DatasetVersion, ExperimentalRecord, ValidationRecord
from app.services.ingestion import IngestionError, file_type_of, inspect_file, read_table, to_jsonable
from app.services.mapping import MappingError, apply_mapping
from app.services.preprocessing import PreprocessOptions, feature_summary, preprocess
from app.services.quality import quality_report


@dataclass
class SheetSpec:
    file_id: int
    sheet_name: str | None
    mapping: dict[str, str | None]
    constants: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:150]


# ----------------------------------------------------------------------------------------------
# Datasets and files
# ----------------------------------------------------------------------------------------------

def create_dataset(db: Session, name: str, description: str | None = None, source_type: str = "upload") -> Dataset:
    ds = Dataset(name=name.strip() or "Untitled dataset", description=description, source_type=source_type)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def store_upload(db: Session, dataset: Dataset, filename: str, content: bytes, settings: Settings | None = None) -> DatasetFile:
    settings = settings or get_settings()
    ftype = file_type_of(filename)  # raises on unsupported extension
    if len(content) == 0:
        raise IngestionError(f"'{filename}' is empty")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise IngestionError(f"'{filename}' exceeds the {settings.max_upload_mb} MB upload limit")
    folder = settings.uploads_dir / str(dataset.id)
    folder.mkdir(parents=True, exist_ok=True)
    rec = DatasetFile(dataset_id=dataset.id, filename=filename, stored_path="", file_type=ftype, size_bytes=len(content))
    db.add(rec)
    db.flush()
    path = folder / f"{rec.id}_{_safe_name(filename)}"
    path.write_bytes(content)
    rec.stored_path = str(path)
    try:
        rec.inspection = to_jsonable(inspect_file(path, filename))
    except IngestionError:
        path.unlink(missing_ok=True)
        db.rollback()
        raise
    db.commit()
    db.refresh(rec)
    return rec


def file_dataframe(file: DatasetFile, sheet_name: str | None) -> pd.DataFrame:
    return read_table(Path(file.stored_path), sheet_name)


# ----------------------------------------------------------------------------------------------
# Versions
# ----------------------------------------------------------------------------------------------

def _next_version_number(db: Session, dataset_id: int) -> int:
    n = db.scalar(select(func.max(DatasetVersion.version_number)).where(DatasetVersion.dataset_id == dataset_id))
    return int(n or 0) + 1


def _records_from_frame(df: pd.DataFrame, version_id: int, source_default: str = "measured") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cols = [c for c in RECORD_COLUMNS if c in df.columns]
    has_flags = "flags" in df.columns
    has_usable = "is_usable" in df.columns
    has_extra = "extra" in df.columns
    has_source = "source" in df.columns
    has_idx = "row_index" in df.columns
    for i, row in enumerate(df[cols + [c for c in ("flags", "is_usable", "extra", "source", "row_index") if c in df.columns]].itertuples(index=False, name=None)):
        rec: dict[str, Any] = {"dataset_version_id": version_id}
        for c, v in zip(cols, row[: len(cols)]):
            if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
                rec[c] = None
            elif c in NUMERIC_FEATURES:
                rec[c] = float(v)
            else:
                rec[c] = str(v)
        rest = dict(zip([c for c in ("flags", "is_usable", "extra", "source", "row_index") if c in df.columns], row[len(cols):]))
        rec["flags"] = list(rest.get("flags") or []) if has_flags else []
        rec["is_usable"] = bool(rest.get("is_usable", True)) if has_usable else True
        rec["extra"] = to_jsonable(rest.get("extra") or {}) if has_extra else {}
        rec["source"] = str(rest.get("source") or source_default) if has_source else source_default
        rec["row_index"] = int(rest["row_index"]) if has_idx and rest.get("row_index") is not None and not pd.isna(rest["row_index"]) else i
        rows.append(rec)
    return rows


def _store_version(
    db: Session, dataset: Dataset, df: pd.DataFrame, kind: str, parent: DatasetVersion | None,
    column_mapping: dict, cleaning_log: list, notes: str | None,
) -> DatasetVersion:
    if "is_usable" not in df.columns:
        df = df.copy()
        df["is_usable"] = True
    summary = feature_summary(df)
    report = quality_report(df)
    version = DatasetVersion(
        dataset_id=dataset.id, version_number=_next_version_number(db, dataset.id), kind=kind,
        parent_version_id=parent.id if parent else None, notes=notes, record_count=int(len(df)),
        usable_record_count=int(df["is_usable"].sum()), column_mapping=to_jsonable(column_mapping),
        cleaning_log=to_jsonable(cleaning_log), quality_report=report, feature_summary=to_jsonable(summary),
    )
    db.add(version)
    db.flush()
    rows = _records_from_frame(df, version.id)
    if rows:
        db.bulk_insert_mappings(ExperimentalRecord, rows)
    db.commit()
    db.refresh(version)
    return version


def build_raw_version(db: Session, dataset: Dataset, specs: list[SheetSpec], notes: str | None = None) -> DatasetVersion:
    """Create the raw version from mapped sheets. Fails loudly when no usable target can be formed."""
    if not specs:
        raise IngestionError("No worksheets selected for ingestion")
    frames: list[pd.DataFrame] = []
    mapping_doc: list[dict[str, Any]] = []
    files = {f.id: f for f in dataset.files}
    for spec in specs:
        file = files.get(spec.file_id)
        if file is None:
            raise IngestionError(f"File {spec.file_id} does not belong to dataset {dataset.id}")
        df = file_dataframe(file, spec.sheet_name)
        if df.empty:
            raise IngestionError(f"'{file.filename}' / '{spec.sheet_name}' contains no data rows")
        mapping = {k: (v or None) for k, v in spec.mapping.items()}
        try:
            std = apply_mapping(df, mapping, spec.constants, spec.units)
        except MappingError as exc:
            raise IngestionError(f"{file.filename} / {spec.sheet_name}: {exc}") from exc
        std["source_file"] = file.filename
        std["source_sheet"] = spec.sheet_name or ""
        frames.append(std)
        mapping_doc.append({"file_id": file.id, "filename": file.filename, "sheet_name": spec.sheet_name,
                            "mapping": mapping, "constants": spec.constants, "units": spec.units, "rows": int(len(std))})
    raw = pd.concat(frames, ignore_index=True)
    raw["extra"] = [dict(e, _source_file=f, _source_sheet=s) for e, f, s in zip(raw["extra"], raw["source_file"], raw["source_sheet"])]
    raw = raw.drop(columns=["source_file", "source_sheet"])

    has_cof = raw["cof"].notna().any()
    can_derive_cof = raw["friction_force_n"].notna().any() and raw["normal_load_n"].notna().any()
    has_wear = raw["wear_rate"].notna().any()
    if not (has_cof or can_derive_cof or has_wear):
        raise IngestionError(
            "No target can be formed from the mapped columns. Map a 'cof' column, or both 'friction_force_n' and "
            "'normal_load_n' (as a column or a constant), or a 'wear_rate' column."
        )
    raw["is_usable"] = True
    raw["flags"] = [[] for _ in range(len(raw))]
    return _store_version(db, dataset, raw, "raw", None, {"sheets": mapping_doc}, [], notes)


def version_dataframe(db: Session, version_id: int, usable_only: bool = False) -> pd.DataFrame:
    stmt = select(ExperimentalRecord).where(ExperimentalRecord.dataset_version_id == version_id)
    if usable_only:
        stmt = stmt.where(ExperimentalRecord.is_usable.is_(True))
    stmt = stmt.order_by(ExperimentalRecord.id)
    recs = db.scalars(stmt).all()
    cols = RECORD_COLUMNS + ["extra", "flags", "is_usable", "source", "row_index", "id"]
    data = {c: [getattr(r, c) for r in recs] for c in cols}
    df = pd.DataFrame(data)
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def preprocess_version(db: Session, parent: DatasetVersion, options: dict[str, Any] | None, notes: str | None = None) -> DatasetVersion:
    opt = PreprocessOptions.from_dict(options)
    raw = version_dataframe(db, parent.id)
    if raw.empty:
        raise IngestionError("Parent version has no records")
    raw = raw.drop(columns=["id"])
    processed, log = preprocess(raw, opt)
    dataset = db.get(Dataset, parent.dataset_id)
    opts_doc = {k: v for k, v in opt.__dict__.items() if k != "material_catalogue"}
    return _store_version(db, dataset, processed, "processed", parent, {"inherited_from": parent.id, "options": opts_doc}, log, notes)


def augment_version(db: Session, parent: DatasetVersion, validations: list[ValidationRecord], notes: str | None = None) -> DatasetVersion:
    """New version = parent records + approved experimental validation measurements (source='validation')."""
    base = version_dataframe(db, parent.id).drop(columns=["id"])
    new_rows = []
    for v in validations:
        row = {c: None for c in RECORD_COLUMNS}
        for k, val in (v.inputs or {}).items():
            if k in RECORD_COLUMNS:
                row[k] = val
        row["cof"] = v.measured_cof
        row["wear_rate"] = v.measured_wear_rate
        row["test_run_id"] = row.get("test_run_id") or f"VAL_{v.experiment_id}"
        row["extra"] = {"experiment_id": v.experiment_id, "validation_record_id": v.id}
        row["flags"] = ["validation"]
        row["is_usable"] = True
        row["source"] = "validation"
        row["row_index"] = None
        new_rows.append(row)
    add = pd.DataFrame(new_rows)
    combined = pd.concat([base, add], ignore_index=True) if len(add) else base
    for c in NUMERIC_FEATURES:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
    dataset = db.get(Dataset, parent.dataset_id)
    log = [{"step": "augment", "description": "Appended approved experimental validation records to the parent version",
            "rows_affected": int(len(add))}]
    version = _store_version(db, dataset, combined, "augmented", parent, {"inherited_from": parent.id}, log, notes)
    for v in validations:
        v.added_to_version_id = version.id
    db.commit()
    return version


def export_version(db: Session, version: DatasetVersion, fmt: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    df = version_dataframe(db, version.id).drop(columns=["id"])
    df["flags"] = df["flags"].apply(lambda f: ";".join(f or []))
    df["extra"] = df["extra"].apply(lambda e: json.dumps(e or {}))
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dataset_{version.dataset_id}_v{version.version_number}_{version.kind}"
    if fmt == "xlsx":
        path = settings.exports_dir / f"{stem}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="records", index=False)
            pd.DataFrame(version.cleaning_log or []).to_excel(xw, sheet_name="cleaning_log", index=False)
            q = version.quality_report or {}
            pd.DataFrame(q.get("columns", [])).to_excel(xw, sheet_name="quality_columns", index=False)
            pd.DataFrame(q.get("per_run", [])).to_excel(xw, sheet_name="quality_runs", index=False)
    else:
        path = settings.exports_dir / f"{stem}.csv"
        df.to_csv(path, index=False)
    return path


def quality_report_markdown(dataset: Dataset, version: DatasetVersion) -> str:
    q = version.quality_report or {}
    lines = [f"# Data-quality report — {dataset.name} (version {version.version_number}, {version.kind})", "",
             f"Rows: {q.get('n_rows')}  |  Usable: {q.get('usable_rows')}  |  Duplicates: {q.get('duplicate_rows')}  |  Score: {q.get('score')}", ""]
    s = q.get("summary", {})
    lines += [f"- Completeness: {s.get('completeness_pct')} %", f"- Usable share: {s.get('usable_pct')} %",
              f"- COF available: {s.get('has_cof')}", f"- Wear rate available: {s.get('has_wear_rate')}",
              f"- Materials: {s.get('n_materials')}  |  Runs: {s.get('n_runs')}", "", "## Columns", "",
              "| Feature | Unit | Non-null | Missing % | Min | Max | Mean | Constant |", "|---|---|---|---|---|---|---|---|"]
    for c in q.get("columns", []):
        lines.append(f"| {c['label']} | {c['unit']} | {c['non_null']} | {c['missing_pct']:.1f} | {c.get('min', '')} | {c.get('max', '')} | "
                     f"{'' if c.get('mean') is None else round(c['mean'], 4)} | {c.get('constant', '')} |")
    lines += ["", "## Flags", ""] + [f"- {k}: {v}" for k, v in (q.get("flags") or {}).items()]
    lines += ["", "## Cleaning log", ""] + [f"{i + 1}. {e['step']}: {e['description']} (rows: {e['rows_affected']})" for i, e in enumerate(version.cleaning_log or [])]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------
# Bundled dataset (Kolev 2023)
# ----------------------------------------------------------------------------------------------

BUNDLED_NAME = "Kolev 2023 — open-cell AlSi10Mg / AlSi10Mg-Al2O3 pin-on-disk"
BUNDLED_DESCRIPTION = (
    "Friction coefficient data of open-cell AlSi10Mg and AlSi10Mg-Al2O3 materials with different pore sizes by "
    "pin-on-disk test (Mendeley Data, DOI 10.17632/2356m76ktj.1, CC BY 4.0). Dry sliding, 50 N, 0.5 m/s, 420 s; "
    "12 runs = 4 materials × 3 replicates; 462 samples per run. Contains friction force vs time only — no wear data."
)


def import_bundled_dataset(db: Session, settings: Settings | None = None, preprocess_options: dict | None = None) -> tuple[Dataset, DatasetVersion, DatasetVersion]:
    settings = settings or get_settings()
    folder = Path(settings.bundled_dataset_dir) / "Pin-on-disk_data" / "Processed_files"
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise IngestionError(f"Bundled dataset not found at {folder}")
    dataset = create_dataset(db, BUNDLED_NAME, BUNDLED_DESCRIPTION, source_type="bundled")
    specs: list[SheetSpec] = []
    for f in files:
        rec = store_upload(db, dataset, f.name, f.read_bytes(), settings)
        sheet = rec.inspection["sheets"][0]
        mapping = {c: v["feature"] for c, v in sheet["suggested_mapping"].items()}
        meta = sheet["parsed_run"]
        constants = {k: meta[k] for k in ("material", "test_run_id", "normal_load_n", "rpm", "track_diameter_mm", "sliding_speed_m_s") if k in meta}
        constants["lubricant"] = "dry"
        specs.append(SheetSpec(rec.id, sheet["sheet_name"], mapping, constants, {}))
    raw = build_raw_version(db, dataset, specs, notes="Imported from the bundled Kolev 2023 processed workbooks; run metadata parsed from worksheet names; lubricant = dry and load = 50 N from the dataset description.")
    processed = preprocess_version(db, raw, preprocess_options, notes="Default preprocessing (COF = FF / 50 N, sliding distance = 0.5 m/s × t, run-in and outlier flags).")
    return dataset, raw, processed


def delete_dataset(db: Session, dataset: Dataset) -> None:
    """Delete a dataset, its files and versions (children before parents), detaching references that must survive."""
    from app.models import Prediction, Report  # local import to avoid cycles at module load

    version_ids = [v.id for v in dataset.versions]
    if version_ids:
        for p in db.scalars(select(Prediction).where(Prediction.dataset_version_id.in_(version_ids))).all():
            p.dataset_version_id = None
        for r in db.scalars(select(Report).where(Report.dataset_version_id.in_(version_ids))).all():
            r.dataset_version_id = None
        for vr in db.scalars(select(ValidationRecord).where(ValidationRecord.added_to_version_id.in_(version_ids))).all():
            vr.added_to_version_id = None
        db.flush()
    for v in sorted(dataset.versions, key=lambda v: v.version_number, reverse=True):
        db.execute(ExperimentalRecord.__table__.delete().where(ExperimentalRecord.dataset_version_id == v.id))
        db.delete(v)
        db.flush()
    for f in list(dataset.files):
        Path(f.stored_path).unlink(missing_ok=True)
        db.delete(f)
    db.delete(dataset)
    db.commit()


def dataset_summary(db: Session, dataset: Dataset) -> dict[str, Any]:
    versions = sorted(dataset.versions, key=lambda v: v.version_number)
    latest = versions[-1] if versions else None
    return {
        "id": dataset.id, "name": dataset.name, "description": dataset.description, "source_type": dataset.source_type,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
        "n_files": len(dataset.files), "n_versions": len(versions),
        "latest_version_id": latest.id if latest else None,
        "latest_version_number": latest.version_number if latest else None,
        "latest_kind": latest.kind if latest else None,
        "record_count": latest.record_count if latest else 0,
        "usable_record_count": latest.usable_record_count if latest else 0,
        "targets": (latest.feature_summary or {}).get("targets", {}) if latest else {},
    }


def version_summary(v: DatasetVersion) -> dict[str, Any]:
    return {
        "id": v.id, "dataset_id": v.dataset_id, "version_number": v.version_number, "kind": v.kind,
        "parent_version_id": v.parent_version_id, "notes": v.notes, "record_count": v.record_count,
        "usable_record_count": v.usable_record_count, "created_at": v.created_at.isoformat() if v.created_at else None,
        "feature_summary": v.feature_summary, "quality_score": (v.quality_report or {}).get("score"),
        "cleaning_steps": len(v.cleaning_log or []),
    }
