"""Documented, reproducible preprocessing of standardized experimental records.

Every operation appends an entry to the cleaning log so the processed dataset version can be audited.
The original (raw) version is never modified: preprocessing always produces a new DataFrame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.features import CATEGORICAL_FEATURES, MATERIAL_CATALOGUE, NUMERIC_FEATURES, STANDARD_FEATURES

REQUIRED_FOR_COF = ("friction_force_n", "normal_load_n")
RELATIVE_CONSTANT_TOLERANCE = 0.01  # a numeric input whose range is < 1 % of its magnitude is treated as fixed


def is_varying(s: pd.Series, tol: float = RELATIVE_CONSTANT_TOLERANCE) -> bool:
    """True when a numeric series spans more than `tol` of its magnitude (e.g. 0.4995-0.5027 m/s is 'fixed')."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2 or s.nunique() <= 1:
        return False
    rng = float(s.max() - s.min())
    scale = max(abs(float(s.abs().max())), 1e-12)
    return rng > tol * scale


@dataclass
class PreprocessOptions:
    drop_duplicates: bool = True
    negative_friction: str = "abs"  # abs | drop | keep   (sensor noise around zero during run-in)
    derive_cof: bool = True
    derive_sliding_distance: bool = True
    derive_speed_from_rpm: bool = True
    derive_material_attributes: bool = True
    flag_run_in: bool = True
    run_in_fraction: float = 0.05  # friction force below this fraction of the run's steady-state median
    exclude_run_in: bool = False
    outlier_zscore: float = 4.0  # per-run robust z-score on the target; 0 disables
    exclude_outliers: bool = False
    drop_rows_missing_target: bool = True
    material_catalogue: dict[str, dict[str, str]] = field(default_factory=lambda: dict(MATERIAL_CATALOGUE))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PreprocessOptions":
        d = dict(d or {})
        allowed = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        opt = cls(**allowed)
        if opt.negative_friction not in {"abs", "drop", "keep"}:
            raise ValueError("negative_friction must be one of abs, drop, keep")
        return opt


def _log(log: list[dict], step: str, description: str, rows: int, **detail: Any) -> None:
    log.append({"step": step, "description": description, "rows_affected": int(rows), **detail})


def _add_flag(flags: list[list[str]], mask: pd.Series, flag: str) -> None:
    for i in np.flatnonzero(mask.to_numpy()):
        flags[i].append(flag)


def preprocess(raw: pd.DataFrame, options: PreprocessOptions | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Return (processed frame, cleaning log). The input frame is not modified."""
    opt = options or PreprocessOptions()
    df = raw.copy().reset_index(drop=True)
    log: list[dict] = []
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), index=df.index, dtype=object)
        else:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    if "extra" not in df.columns:
        df["extra"] = [{} for _ in range(len(df))]
    if "source" not in df.columns:
        df["source"] = "measured"
    n0 = len(df)
    _log(log, "load", "Loaded standardized records from the parent dataset version", n0)

    # 1. Exact duplicate rows ------------------------------------------------------------------
    if opt.drop_duplicates and n0:
        key_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
        dup = df.duplicated(subset=key_cols, keep="first")
        if dup.any():
            df = df.loc[~dup].reset_index(drop=True)
        _log(log, "drop_duplicates", "Removed exact duplicate measurements (all standard columns identical)", int(dup.sum()))

    flags: list[list[str]] = [[] for _ in range(len(df))]
    usable = pd.Series(True, index=df.index)

    # 2. Non-finite / invalid numeric values ----------------------------------------------------
    for col in NUMERIC_FEATURES:
        spec = STANDARD_FEATURES[col]
        s = df[col]
        inf = s.notna() & ~np.isfinite(s.astype(float))
        if inf.any():
            df.loc[inf, col] = np.nan
            _add_flag(flags, inf, f"non_finite:{col}")
            _log(log, "non_finite", f"Replaced non-finite values in {col} with missing", int(inf.sum()))
        if col == "friction_force_n":
            continue  # negative friction handled explicitly below
        lo, hi = spec.valid_min, spec.valid_max
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= s.notna() & (s < lo)
        if hi is not None:
            bad |= s.notna() & (s > hi)
        if bad.any():
            _add_flag(flags, bad, f"out_of_valid_range:{col}")
            usable &= ~bad
            _log(log, "valid_range", f"Flagged {col} outside physical validity range [{lo}, {hi}] as unusable", int(bad.sum()))

    # 3. Negative friction force ----------------------------------------------------------------
    ff = df["friction_force_n"]
    neg = ff.notna() & (ff < 0)
    if neg.any():
        if opt.negative_friction == "abs":
            df.loc[neg, "friction_force_n"] = ff[neg].abs()
            _add_flag(flags, neg, "negative_friction_abs")
            _log(log, "negative_friction", "Negative friction-force readings replaced by their absolute value (sensor sign noise near zero load transfer)", int(neg.sum()))
        elif opt.negative_friction == "drop":
            _add_flag(flags, neg, "negative_friction")
            usable &= ~neg
            _log(log, "negative_friction", "Negative friction-force readings marked unusable", int(neg.sum()))
        else:
            _add_flag(flags, neg, "negative_friction")
            _log(log, "negative_friction", "Negative friction-force readings kept unchanged", int(neg.sum()))

    # 4. Derived material attributes -----------------------------------------------------------
    if opt.derive_material_attributes:
        derived_rows = pd.Series(False, index=df.index)
        for code, attrs in opt.material_catalogue.items():
            m = df["material"].astype("string").str.upper() == code.upper()
            if m.any():
                for k in ("composition", "pore_size_class"):
                    fill = m & df[k].isna()
                    df.loc[fill, k] = attrs.get(k)
                    derived_rows |= fill
        n_derived = int(derived_rows.sum())
        _log(log, "material_attributes", "Filled composition / pore size class from the material catalogue where missing", n_derived)

    # 5. Sliding speed from rpm and track diameter -----------------------------------------------
    if opt.derive_speed_from_rpm:
        m = df["sliding_speed_m_s"].isna() & df["rpm"].notna() & df["track_diameter_mm"].notna()
        if m.any():
            df.loc[m, "sliding_speed_m_s"] = math.pi * df.loc[m, "track_diameter_mm"] / 1000.0 * df.loc[m, "rpm"] / 60.0
            _add_flag(flags, m, "derived:sliding_speed")
        _log(log, "derive_speed", "Sliding speed derived as π · track diameter · rpm / 60 where not provided", int(m.sum()))

    # 6. COF = friction force / normal load -------------------------------------------------------
    if opt.derive_cof:
        zero_load = df["normal_load_n"].notna() & (df["normal_load_n"] <= 0) & df["cof"].isna()
        if zero_load.any():
            _add_flag(flags, zero_load, "zero_normal_load")
            usable &= ~zero_load
            _log(log, "zero_load", "Rows with zero or negative normal load cannot yield a COF and are unusable", int(zero_load.sum()))
        m = df["cof"].isna() & df["friction_force_n"].notna() & df["normal_load_n"].notna() & (df["normal_load_n"] > 0)
        if m.any():
            df.loc[m, "cof"] = df.loc[m, "friction_force_n"] / df.loc[m, "normal_load_n"]
            _add_flag(flags, m, "derived:cof")
        _log(log, "derive_cof", "COF computed as friction force / normal load where not provided", int(m.sum()))

    # 7. Sliding distance = speed × time ----------------------------------------------------------
    if opt.derive_sliding_distance:
        m = df["sliding_distance_m"].isna() & df["sliding_speed_m_s"].notna() & df["time_s"].notna()
        if m.any():
            df.loc[m, "sliding_distance_m"] = df.loc[m, "sliding_speed_m_s"] * df.loc[m, "time_s"]
            _add_flag(flags, m, "derived:sliding_distance")
        _log(log, "derive_distance", "Sliding distance computed as sliding speed × time where not provided", int(m.sum()))

    # 8. Wear-rate definition is never derived silently ------------------------------------------
    wv = df["wear_volume_mm3"].notna() | df["wear_mass_mg"].notna()
    if wv.any() and df["wear_rate"].isna().all():
        _log(log, "wear_rate_not_derived",
             "Wear volume/mass present but wear rate NOT derived automatically: the wear-rate definition, "
             "sliding distance and (for mass) material density must be confirmed by the user", int(wv.sum()))

    # 9. Rows without any target ---------------------------------------------------------------
    no_target = df["cof"].isna() & df["wear_rate"].isna()
    if no_target.any():
        _add_flag(flags, no_target, "missing_target")
        if opt.drop_rows_missing_target:
            usable &= ~no_target
        _log(log, "missing_target", "Rows without COF and without wear rate marked unusable", int(no_target.sum()))

    # 10. Run-in flag ---------------------------------------------------------------------------
    if opt.flag_run_in and df["test_run_id"].notna().any() and df["friction_force_n"].notna().any():
        run_in = pd.Series(False, index=df.index)
        for _, idx in df.groupby("test_run_id").groups.items():
            sub = df.loc[idx]
            if sub["time_s"].notna().any():
                tmax = sub["time_s"].max()
                steady = sub.loc[sub["time_s"] > 0.5 * tmax, "friction_force_n"].median()
            else:
                steady = sub["friction_force_n"].median()
            if pd.notna(steady) and steady > 0:
                run_in.loc[idx] = sub["friction_force_n"].notna() & (sub["friction_force_n"] < opt.run_in_fraction * steady)
        if run_in.any():
            _add_flag(flags, run_in, "run_in")
            if opt.exclude_run_in:
                usable &= ~run_in
        _log(log, "run_in", f"Flagged run-in samples (friction force < {opt.run_in_fraction:.0%} of the run's steady-state median)"
             + (" and excluded them" if opt.exclude_run_in else " (kept; they are genuine measurements)"), int(run_in.sum()))

    # 11. Outliers (per run, robust z-score on the target) --------------------------------------
    if opt.outlier_zscore and opt.outlier_zscore > 0:
        target = "cof" if df["cof"].notna().any() else "wear_rate"
        outl = pd.Series(False, index=df.index)
        grp = df["test_run_id"].fillna("__all__")
        for _, idx in df.groupby(grp).groups.items():
            s = df.loc[idx, target]
            med = s.median()
            mad = (s - med).abs().median()
            if pd.notna(mad) and mad > 0:
                z = 0.6745 * (s - med).abs() / mad
                outl.loc[idx] = z > opt.outlier_zscore
        if outl.any():
            _add_flag(flags, outl, f"outlier:{target}")
            if opt.exclude_outliers:
                usable &= ~outl
        _log(log, "outliers", f"Flagged {target} values with robust z-score > {opt.outlier_zscore} within their run"
             + (" and excluded them" if opt.exclude_outliers else " (kept; flagged only)"), int(outl.sum()))

    df["flags"] = flags
    df["is_usable"] = usable.to_numpy()
    _log(log, "result", "Processed version produced", int(usable.sum()), total_rows=int(len(df)), unusable_rows=int((~usable).sum()))
    return df, log


def feature_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Which targets exist, which inputs vary, and their ranges — used by training and the UI."""
    usable = df[df["is_usable"]] if "is_usable" in df.columns else df
    summary: dict[str, Any] = {"targets": {}, "inputs": {}, "n_rows": int(len(df)), "n_usable": int(len(usable))}
    for t in ("cof", "wear_rate"):
        s = usable[t].dropna() if t in usable.columns else pd.Series(dtype=float)
        summary["targets"][t] = {
            "available": bool(len(s) > 0),
            "n": int(len(s)),
            "min": float(s.min()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
            "mean": float(s.mean()) if len(s) else None,
        }
    for name, spec in STANDARD_FEATURES.items():
        if spec.role not in ("input", "metadata", "identifier"):
            continue
        if name not in usable.columns:
            continue
        s = usable[name].dropna()
        entry: dict[str, Any] = {"present": bool(len(s) > 0), "n": int(len(s)), "dtype": spec.dtype, "unit": spec.unit,
                                 "role": spec.role}
        if spec.dtype == "number":
            entry.update({
                "min": float(s.min()) if len(s) else None,
                "max": float(s.max()) if len(s) else None,
                "n_unique": int(s.nunique()),
                "varies": bool(len(s) and s.nunique() > 1 and is_varying(s)),
            })
        else:
            levels = sorted(map(str, s.unique().tolist()))
            entry.update({"levels": levels[:100], "n_unique": len(levels), "varies": len(levels) > 1})
        summary["inputs"][name] = entry
    summary["n_runs"] = int(usable["test_run_id"].nunique()) if "test_run_id" in usable.columns else 0
    summary["n_materials"] = int(usable["material"].nunique()) if "material" in usable.columns else 0
    return summary
