"""Aggregations for the experimental-analysis charts (measured data only; predictions are never mixed in)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.features import STANDARD_FEATURES, TARGET_INFO
from app.services.ingestion import to_jsonable
from app.services.preprocessing import is_varying

GROUPABLE = ("material", "composition", "pore_size_class", "lubricant", "material_pair", "test_run_id")


def _usable(df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = df[df["is_usable"]] if "is_usable" in df.columns else df
    return d[d[target].notna()]


def _downsample(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(sub) <= n:
        return sub
    idx = np.linspace(0, len(sub) - 1, n).round().astype(int)
    return sub.iloc[idx]


def target_vs_time(df: pd.DataFrame, target: str = "cof", group_by: str = "material", bins: int = 80, run_points: int = 160) -> dict[str, Any]:
    d = _usable(df, target)
    if group_by not in GROUPABLE or group_by not in d.columns or d[group_by].isna().all():
        group_by = "material" if "material" in d.columns else "test_run_id"
    if d["time_s"].isna().all():
        return {"available": False, "reason": "No time column in this dataset"}
    edges = np.linspace(float(d["time_s"].min()), float(d["time_s"].max()), bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    d = d.assign(_bin=pd.cut(d["time_s"], edges, labels=False, include_lowest=True))
    groups = []
    for g, sub in d.groupby(group_by, dropna=True):
        agg = sub.groupby("_bin")[target].agg(["mean", "std", "min", "max", "count"]).reindex(range(bins))
        groups.append({
            "group": str(g), "n_runs": int(sub["test_run_id"].nunique()), "n_rows": int(len(sub)),
            "points": [{"t": float(centers[i]), "mean": to_jsonable(r["mean"]), "std": to_jsonable(r["std"]), "min": to_jsonable(r["min"]),
                        "max": to_jsonable(r["max"]), "n": int(r["count"]) if not np.isnan(r["count"]) else 0}
                       for i, r in agg.iterrows() if not np.isnan(r["mean"])],
        })
    runs = []
    for run, sub in d.sort_values("time_s").groupby("test_run_id", dropna=True):
        s = _downsample(sub, run_points)
        runs.append({"test_run_id": str(run), "group": to_jsonable(sub[group_by].iloc[0]),
                     "points": [{"t": float(t), "y": float(y)} for t, y in zip(s["time_s"], s[target])]})
    return {"available": True, "target": target, "unit": TARGET_INFO[target]["unit"], "group_by": group_by, "groups": groups, "runs": runs,
            "x": {"name": "time_s", "label": "Sliding time", "unit": "s"}}


def target_vs_feature(df: pd.DataFrame, target: str, feature: str, group_by: str = "material", max_points: int = 2500) -> dict[str, Any]:
    if feature not in STANDARD_FEATURES:
        raise ValueError(f"Unknown feature '{feature}'")
    spec = STANDARD_FEATURES[feature]
    d = _usable(df, target)
    base = {"target": target, "unit": TARGET_INFO[target]["unit"], "feature": feature, "label": spec.label, "feature_unit": spec.unit}
    if feature not in d.columns or d[feature].isna().all():
        return {**base, "available": False, "varied": False, "reason": f"{spec.label} was not recorded in this dataset."}
    if spec.dtype == "number":
        s = pd.to_numeric(d[feature], errors="coerce")
        if not is_varying(s):
            return {**base, "available": True, "varied": False, "fixed_value": float(s.mean()),
                    "reason": f"{spec.label} was held constant at about {s.mean():.4g} {spec.unit} in every experiment, so no trend versus {spec.label.lower()} can be observed or claimed from this dataset."}
        gcol = group_by if group_by in d.columns and d[group_by].notna().any() else None
        sample = d.sample(n=min(max_points, len(d)), random_state=0)
        points = [{"x": float(x), "y": float(y), "group": to_jsonable(g), "run": to_jsonable(r)}
                  for x, y, g, r in zip(sample[feature], sample[target], sample[gcol] if gcol else [None] * len(sample), sample["test_run_id"])]
        edges = np.linspace(float(s.min()), float(s.max()), 31)
        d = d.assign(_bin=pd.cut(s, edges, labels=False, include_lowest=True))
        trend = []
        for g, sub in (d.groupby(gcol, dropna=True) if gcol else [("all", d)]):
            agg = sub.groupby("_bin")[target].agg(["mean", "std", "count"])
            trend.append({"group": str(g), "points": [{"x": float((edges[i] + edges[i + 1]) / 2), "mean": float(r["mean"]), "std": to_jsonable(r["std"]), "n": int(r["count"])}
                                                       for i, r in agg.iterrows()]})
        rho, p = stats.spearmanr(s, d[target])
        return {**base, "available": True, "varied": True, "points": points, "trend": trend, "group_by": gcol,
                "spearman": {"rho": to_jsonable(rho), "p_value": to_jsonable(p), "n": int(len(d)),
                             "caveat": "Rows within a run are autocorrelated; p-values treat them as independent and are optimistic."}}
    # categorical feature → distribution per level
    levels = []
    for lv, sub in d.groupby(feature, dropna=True):
        v = sub[target].astype(float)
        q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        levels.append({"level": str(lv), "n": int(len(v)), "n_runs": int(sub["test_run_id"].nunique()), "mean": float(v.mean()), "std": float(v.std()),
                       "min": float(v.min()), "max": float(v.max()), "p05": float(q.iloc[0]), "q1": float(q.iloc[1]), "median": float(q.iloc[2]),
                       "q3": float(q.iloc[3]), "p95": float(q.iloc[4])})
    if len(levels) < 2:
        return {**base, "available": True, "varied": False, "levels": levels, "reason": f"Only one {spec.label.lower()} level is present; no comparison possible."}
    return {**base, "available": True, "varied": True, "levels": levels}


def material_comparison(df: pd.DataFrame, target: str = "cof", group_by: str = "material") -> dict[str, Any]:
    d = _usable(df, target)
    if group_by not in d.columns or d[group_by].isna().all():
        return {"available": False, "reason": f"No '{group_by}' column in this dataset"}
    rows = []
    for g, sub in d.groupby(group_by, dropna=True):
        per_run = []
        for run, rs in sub.groupby("test_run_id", dropna=True):
            if rs["time_s"].notna().any():
                tmax = rs["time_s"].max()
                steady = rs.loc[rs["time_s"] > 0.5 * tmax, target]
            else:
                steady = rs[target]
            per_run.append({"test_run_id": str(run), "mean": float(rs[target].mean()), "steady_mean": float(steady.mean()) if len(steady) else None,
                            "max": float(rs[target].max()), "n": int(len(rs))})
        steady_means = [r["steady_mean"] for r in per_run if r["steady_mean"] is not None]
        rows.append({
            "group": str(g), "n_rows": int(len(sub)), "n_runs": len(per_run), "mean": float(sub[target].mean()), "std": float(sub[target].std()),
            "min": float(sub[target].min()), "max": float(sub[target].max()),
            "steady_mean": float(np.mean(steady_means)) if steady_means else None,
            "steady_std_between_runs": float(np.std(steady_means, ddof=1)) if len(steady_means) > 1 else None,
            "runs": per_run,
            "attributes": {k: to_jsonable(sub[k].iloc[0]) for k in ("composition", "pore_size_class", "lubricant", "normal_load_n", "sliding_speed_m_s") if k in sub.columns},
        })
    test: dict[str, Any] | None = None
    groups_steady = [[r["steady_mean"] for r in row["runs"] if r["steady_mean"] is not None] for row in rows]
    groups_steady = [g for g in groups_steady if len(g) >= 2]
    if len(groups_steady) >= 2:
        h, p = stats.kruskal(*groups_steady)
        test = {"name": "Kruskal–Wallis H test on per-run steady-state means", "statistic": float(h), "p_value": float(p),
                "n_groups": len(groups_steady), "n_runs": int(sum(len(g) for g in groups_steady)),
                "interpretation": ("Differences between groups are unlikely to be due to run-to-run variation alone (p < 0.05)." if p < 0.05 else
                                   "With this number of replicate runs, the between-group differences cannot be distinguished from run-to-run variation (p ≥ 0.05)."),
                "caveat": "Each run contributes one steady-state mean; with few replicates the test has low power."}
    return {"available": True, "target": target, "unit": TARGET_INFO[target]["unit"], "group_by": group_by, "groups": rows, "test": test,
            "steady_state_definition": "mean of the target over the second half of each run's sliding time"}


def correlation(df: pd.DataFrame, target: str = "cof") -> dict[str, Any]:
    d = _usable(df, target)
    cols = [c for c in ("normal_load_n", "sliding_speed_m_s", "temperature_c", "time_s", "sliding_distance_m", "rpm", "track_diameter_mm", "friction_force_n", "cof", "wear_rate")
            if c in d.columns and d[c].notna().sum() > 2 and is_varying(d[c])]
    constant = [c for c in ("normal_load_n", "sliding_speed_m_s", "temperature_c", "lubricant") if c in d.columns and d[c].notna().any() and c not in cols]
    if len(cols) < 2:
        return {"available": False, "reason": "Fewer than two varying numeric columns", "constant": constant}
    corr = d[cols].astype(float).corr(method="spearman")
    return {"available": True, "method": "spearman", "features": cols,
            "labels": [STANDARD_FEATURES[c].label for c in cols], "matrix": [[to_jsonable(v) for v in row] for row in corr.to_numpy()],
            "constant": constant, "caveat": "Spearman rank correlation over all rows; several columns are derived from each other (e.g. COF = FF / load, distance = speed × time)."}


def distributions(df: pd.DataFrame, target: str = "cof", bins: int = 30) -> dict[str, Any]:
    d = _usable(df, target)
    out: dict[str, Any] = {"features": [], "target": target, "by_material": []}
    for c in ("cof", "wear_rate", "friction_force_n", "time_s", "sliding_distance_m", "normal_load_n", "sliding_speed_m_s", "temperature_c"):
        if c not in d.columns or d[c].notna().sum() == 0:
            continue
        v = d[c].astype(float).dropna()
        if v.nunique() <= 1:
            out["features"].append({"feature": c, "label": STANDARD_FEATURES[c].label, "unit": STANDARD_FEATURES[c].unit, "constant": True, "value": float(v.iloc[0]), "n": int(len(v))})
            continue
        counts, edges = np.histogram(v, bins=bins)
        out["features"].append({"feature": c, "label": STANDARD_FEATURES[c].label, "unit": STANDARD_FEATURES[c].unit, "constant": False, "n": int(len(v)),
                                "bins": [{"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])} for i in range(len(counts))],
                                "mean": float(v.mean()), "std": float(v.std()), "median": float(v.median())})
    if "material" in d.columns and d["material"].notna().any():
        v_all = d[target].astype(float)
        edges = np.histogram_bin_edges(v_all, bins=bins)
        for g, sub in d.groupby("material", dropna=True):
            counts, _ = np.histogram(sub[target].astype(float), bins=edges)
            out["by_material"].append({"material": str(g), "bins": [{"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])} for i in range(len(counts))]})
    return out


def trends(df: pd.DataFrame, target: str = "cof") -> dict[str, Any]:
    """Potential trends and interactions, stated with the appropriate caveats."""
    d = _usable(df, target)
    items = []
    for f in ("normal_load_n", "sliding_speed_m_s", "temperature_c", "time_s", "sliding_distance_m"):
        spec = STANDARD_FEATURES[f]
        if f not in d.columns or d[f].isna().all():
            items.append({"feature": f, "label": spec.label, "status": "not_recorded", "statement": f"{spec.label} was not recorded; no trend can be assessed."})
            continue
        s = d[f].astype(float)
        if not is_varying(s):
            items.append({"feature": f, "label": spec.label, "status": "constant", "value": float(s.mean()), "unit": spec.unit,
                          "statement": f"{spec.label} was fixed at {s.mean():.4g} {spec.unit}; its effect on {TARGET_INFO[target]['short']} cannot be learned from this dataset."})
            continue
        rho, p = stats.spearmanr(s, d[target])
        per_run = []
        if "test_run_id" in d.columns:
            for run, sub in d.groupby("test_run_id"):
                if sub[f].nunique() > 2:
                    r_, p_ = stats.spearmanr(sub[f], sub[target])
                    per_run.append({"run": str(run), "rho": to_jsonable(r_), "p_value": to_jsonable(p_)})
        items.append({"feature": f, "label": spec.label, "status": "varied", "spearman_rho": to_jsonable(rho), "p_value": to_jsonable(p), "n": int(len(d)),
                      "per_run": per_run, "statement": (f"{TARGET_INFO[target]['short']} shows a {'positive' if rho > 0 else 'negative'} monotonic association with {spec.label.lower()} "
                                                       f"(Spearman ρ = {rho:.2f}). Within-run autocorrelation makes the pooled p-value optimistic; "
                                                       f"{sum(1 for r in per_run if r['rho'] is not None and np.sign(r['rho']) == np.sign(rho))} of {len(per_run)} runs show the same sign.")})
    cat_items = []
    for f in ("material", "composition", "pore_size_class", "lubricant", "material_pair"):
        spec = STANDARD_FEATURES[f]
        if f not in d.columns or d[f].isna().all():
            cat_items.append({"feature": f, "label": spec.label, "status": "not_recorded", "statement": f"{spec.label} was not recorded."})
            continue
        levels = d[f].dropna().unique()
        if len(levels) < 2:
            cat_items.append({"feature": f, "label": spec.label, "status": "constant", "value": str(levels[0]),
                              "statement": f"All experiments used {spec.label.lower()} = '{levels[0]}'; no comparison is possible."})
            continue
        comp = material_comparison(d, target, f)
        cat_items.append({"feature": f, "label": spec.label, "status": "varied", "n_levels": int(len(levels)), "test": comp.get("test"),
                          "groups": [{"group": g["group"], "steady_mean": g["steady_mean"], "n_runs": g["n_runs"]} for g in comp["groups"]],
                          "statement": (comp["test"]["interpretation"] if comp.get("test") else "Not enough replicate runs for a between-group test.")})
    return {"target": target, "numeric": items, "categorical": cat_items,
            "disclaimer": "These are descriptive associations in the measured data. They describe this dataset only and do not establish causation."}


def overview(df: pd.DataFrame) -> dict[str, Any]:
    d = df[df["is_usable"]] if "is_usable" in df.columns else df
    ranges = {}
    for f in ("normal_load_n", "sliding_speed_m_s", "temperature_c", "time_s", "sliding_distance_m"):
        if f in d.columns and d[f].notna().any():
            s = d[f].astype(float)
            ranges[f] = {"min": float(s.min()), "max": float(s.max()), "unit": STANDARD_FEATURES[f].unit, "label": STANDARD_FEATURES[f].label, "varies": is_varying(s)}
        else:
            ranges[f] = {"min": None, "max": None, "unit": STANDARD_FEATURES[f].unit, "label": STANDARD_FEATURES[f].label, "varies": False, "not_recorded": True}
    def levels(col: str) -> list[str]:
        return sorted(map(str, d[col].dropna().unique())) if col in d.columns else []
    return {"n_rows": int(len(df)), "n_usable": int(len(d)), "n_runs": int(d["test_run_id"].nunique()) if "test_run_id" in d.columns else 0,
            "materials": levels("material"), "lubricants": levels("lubricant"), "material_pairs": levels("material_pair"), "ranges": ranges,
            "targets": {t: {"available": bool(t in d.columns and d[t].notna().any()), "n": int(d[t].notna().sum()) if t in d.columns else 0,
                            "mean": to_jsonable(d[t].mean()) if t in d.columns and d[t].notna().any() else None} for t in ("cof", "wear_rate")}}
