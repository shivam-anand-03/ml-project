"""File ingestion: read Excel / CSV / tribometer .dwf exports and inspect their structure."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.mapping import parse_run_name, suggest_mapping

SUPPORTED_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".csv", ".txt", ".dwf"}
PREVIEW_ROWS = 100


class IngestionError(ValueError):
    pass


def file_type_of(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return ext.lstrip(".")


def _clean_json_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, (np.ndarray, list, tuple)):
        return [_clean_json_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _clean_json_value(x) for k, x in v.items()}
    if isinstance(v, str):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def to_jsonable(obj: Any) -> Any:
    return _clean_json_value(obj)


def read_dwf(path: Path) -> pd.DataFrame:
    """Parse the tribometer `.dwf` text export: ';'-separated, decimal comma, quoted lines, CR endings."""
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    header: list[str] | None = None
    rows: list[list[float | None]] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip().strip('"').strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(";")]
        parts = [p for p in parts if p != ""]
        if not parts:
            continue
        if header is None and any(re.search(r"[A-Za-z]", p) for p in parts):
            header = parts
            continue
        vals: list[float | None] = []
        for p in parts:
            try:
                vals.append(float(p.replace(",", ".")))
            except ValueError:
                vals.append(None)
        rows.append(vals)
    if header is None:
        if not rows:
            raise IngestionError("The .dwf file contains no data")
        header = [f"col_{i + 1}" for i in range(len(rows[0]))]
    width = len(header)
    rows = [r[:width] + [None] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=header)


def read_table(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    ftype = file_type_of(path.name)
    if ftype in {"xlsx", "xlsm", "xls"}:
        xl = pd.ExcelFile(path)
        if not xl.sheet_names:
            raise IngestionError("Workbook has no worksheets")
        sn = sheet_name if sheet_name is not None else xl.sheet_names[0]
        if sn not in xl.sheet_names:
            raise IngestionError(f"Worksheet '{sn}' not found. Available: {xl.sheet_names}")
        df = xl.parse(sn)
    elif ftype == "dwf":
        df = read_dwf(path)
    else:
        df = _read_csv_flexible(path)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df.reset_index(drop=True)


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    sample = path.read_text(encoding="utf-8", errors="replace")[:5000]
    if not sample.strip():
        return pd.DataFrame()
    sep = ","
    if sample.count(";") > sample.count(","):
        sep = ";"
    elif sample.count("\t") > sample.count(","):
        sep = "\t"
    decimal = "," if (sep == ";" and re.search(r"\d,\d", sample)) else "."
    return pd.read_csv(path, sep=sep, decimal=decimal, engine="python")


def list_sheets(path: Path) -> list[str]:
    ftype = file_type_of(path.name)
    if ftype in {"xlsx", "xlsm", "xls"}:
        return pd.ExcelFile(path).sheet_names
    return ["data"]


def _column_profile(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    numeric = pd.to_numeric(non_null, errors="coerce")
    is_numeric = len(non_null) > 0 and numeric.notna().mean() >= 0.95
    profile: dict[str, Any] = {
        "name": str(series.name),
        "dtype": "number" if is_numeric else "category",
        "pandas_dtype": str(series.dtype),
        "non_null": int(non_null.shape[0]),
        "null_count": int(series.isna().sum()),
        "n_unique": int(non_null.nunique()),
        "sample_values": [to_jsonable(v) for v in non_null.head(5).tolist()],
    }
    if is_numeric and numeric.notna().any():
        profile.update(
            {
                "min": to_jsonable(numeric.min()),
                "max": to_jsonable(numeric.max()),
                "mean": to_jsonable(numeric.mean()),
                "std": to_jsonable(numeric.std()),
                "non_numeric_count": int(numeric.isna().sum()),
            }
        )
    return profile


def inspect_dataframe(df: pd.DataFrame, sheet_name: str, filename: str) -> dict[str, Any]:
    columns = [_column_profile(df[c]) for c in df.columns]
    run_meta = parse_run_name(sheet_name) or parse_run_name(Path(filename).stem) or {}
    suggestions = suggest_mapping(list(df.columns))
    preview = df.head(PREVIEW_ROWS)
    return {
        "sheet_name": sheet_name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": columns,
        "duplicate_rows": int(df.duplicated().sum()) if not df.empty else 0,
        "preview_columns": [str(c) for c in preview.columns],
        "preview": [[to_jsonable(v) for v in row] for row in preview.itertuples(index=False, name=None)],
        "parsed_run": run_meta,
        "suggested_mapping": suggestions,
        "is_empty": bool(df.empty),
    }


def inspect_file(path: Path, filename: str | None = None) -> dict[str, Any]:
    """Inspect every worksheet of a file. Never raises on an empty sheet; reports it instead."""
    filename = filename or path.name
    ftype = file_type_of(filename)
    sheets: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        names = list_sheets(path)
    except Exception as exc:  # corrupted workbook
        raise IngestionError(f"Cannot open '{filename}': {exc}") from exc
    for sn in names:
        try:
            df = read_table(path, None if ftype not in {"xlsx", "xlsm", "xls"} else sn)
            sheets.append(inspect_dataframe(df, sn, filename))
        except Exception as exc:
            errors.append(f"{sn}: {exc}")
            sheets.append(
                {"sheet_name": sn, "n_rows": 0, "n_cols": 0, "columns": [], "duplicate_rows": 0,
                 "preview_columns": [], "preview": [], "parsed_run": {}, "suggested_mapping": {},
                 "is_empty": True, "error": str(exc)}
            )
    return {"filename": filename, "file_type": ftype, "sheets": sheets, "errors": errors}
