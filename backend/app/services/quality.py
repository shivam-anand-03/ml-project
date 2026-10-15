"""Data-quality report for a dataset version."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from app.core.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, STANDARD_FEATURES
from app.services.ingestion import to_jsonable


def quality_report(df: pd.DataFrame) -> dict[str, Any]:
    n = int(len(df))
    report: dict[str, Any] = {"n_rows": n, "columns": [], "flags": {}, "per_material": [], "per_run": [], "correlations": {}}
    if n == 0:
        report["usable_rows"] = 0
        report["score"] = 0.0
        return report
    usable = df["is_usable"] if "is_usable" in df.columns else pd.Series(True, index=df.index)
    report["usable_rows"] = int(usable.sum())
    report["unusable_rows"] = int((~usable).sum())
    key_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    report["duplicate_rows"] = int(df.duplicated(subset=key_cols).sum())

    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        s = df[col]
        present = s.notna()
        if not present.any():
            continue
        spec = STANDARD_FEATURES[col]
        entry: dict[str, Any] = {
            "name": col, "label": spec.label, "unit": spec.unit, "dtype": spec.dtype, "role": spec.role,
            "non_null": int(present.sum()), "missing": int((~present).sum()), "missing_pct": float(100.0 * (~present).mean()),
        }
        if spec.dtype == "number":
            v = s.dropna().astype(float)
            q = v.quantile([0.25, 0.5, 0.75]) if len(v) else pd.Series([np.nan] * 3)
            entry.update({
                "min": to_jsonable(v.min()), "max": to_jsonable(v.max()), "mean": to_jsonable(v.mean()),
                "std": to_jsonable(v.std()), "q25": to_jsonable(q.iloc[0]), "median": to_jsonable(q.iloc[1]),
                "q75": to_jsonable(q.iloc[2]), "n_unique": int(v.nunique()), "constant": bool(v.nunique() <= 1),
                "invalid": int(((v < spec.valid_min).sum() if spec.valid_min is not None else 0)
                               + ((v > spec.valid_max).sum() if spec.valid_max is not None else 0)) if len(v) else 0,
            })
        else:
            counts = s.dropna().astype(str).value_counts()
            entry.update({"n_unique": int(counts.shape[0]), "levels": {str(k): int(c) for k, c in counts.head(50).items()}})
        report["columns"].append(entry)

    if "flags" in df.columns:
        cnt: Counter[str] = Counter()
        for fl in df["flags"]:
            for f in (fl or []):
                cnt[str(f)] += 1
        report["flags"] = dict(cnt)

    if "material" in df.columns and df["material"].notna().any():
        g = df.groupby("material", dropna=True)
        for mat, sub in g:
            su = sub[sub["is_usable"]] if "is_usable" in sub.columns else sub
            report["per_material"].append({
                "material": str(mat), "rows": int(len(sub)), "usable": int(len(su)),
                "runs": int(sub["test_run_id"].nunique()),
                "cof_mean": to_jsonable(su["cof"].mean()) if "cof" in su else None,
                "cof_std": to_jsonable(su["cof"].std()) if "cof" in su else None,
                "wear_rate_mean": to_jsonable(su["wear_rate"].mean()) if "wear_rate" in su else None,
            })
    if "test_run_id" in df.columns and df["test_run_id"].notna().any():
        for run, sub in df.groupby("test_run_id", dropna=True):
            report["per_run"].append({
                "test_run_id": str(run), "material": to_jsonable(sub["material"].iloc[0]), "rows": int(len(sub)),
                "usable": int(sub["is_usable"].sum()) if "is_usable" in sub.columns else int(len(sub)),
                "time_min": to_jsonable(sub["time_s"].min()), "time_max": to_jsonable(sub["time_s"].max()),
                "cof_mean": to_jsonable(sub["cof"].mean()), "cof_max": to_jsonable(sub["cof"].max()),
                "normal_load_n": to_jsonable(sub["normal_load_n"].iloc[0]),
                "sliding_speed_m_s": to_jsonable(sub["sliding_speed_m_s"].iloc[0]),
            })

    num = df[[c for c in NUMERIC_FEATURES if c in df.columns and df[c].notna().any() and df[c].nunique() > 1]]
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        report["correlations"] = {"features": list(corr.columns), "matrix": [[to_jsonable(v) for v in row] for row in corr.to_numpy()]}

    # A simple transparent quality score: completeness of inputs/targets and usable share
    completeness = float(np.mean([c["non_null"] / n for c in report["columns"]])) if report["columns"] else 0.0
    usable_share = report["usable_rows"] / n
    report["score"] = round(100.0 * (0.5 * completeness + 0.5 * usable_share), 1)
    report["summary"] = {
        "completeness_pct": round(100 * completeness, 1),
        "usable_pct": round(100 * usable_share, 1),
        "has_cof": bool("cof" in df.columns and df["cof"].notna().any()),
        "has_wear_rate": bool("wear_rate" in df.columns and df["wear_rate"].notna().any()),
        "n_materials": int(df["material"].nunique()) if "material" in df.columns else 0,
        "n_runs": int(df["test_run_id"].nunique()) if "test_run_id" in df.columns else 0,
    }
    return to_jsonable(report)
