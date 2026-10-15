"""Candidate models, preprocessing pipeline, split strategies and evaluation metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from app.core.features import STANDARD_FEATURES
from app.services.preprocessing import is_varying


@dataclass
class Candidate:
    key: str
    label: str
    family: str
    estimator: Any
    grid: dict[str, list]
    quick_grid: dict[str, list]
    supports_tree_shap: bool = False
    notes: str = ""


def candidate_models(seed: int, n_jobs: int = 1) -> dict[str, Candidate]:
    return {
        "linear": Candidate(
            "linear", "Linear Regression", "linear", LinearRegression(), {}, {},
            notes="Ordinary least squares on scaled numeric and one-hot categorical features.",
        ),
        "polynomial": Candidate(
            "polynomial", "Polynomial Regression", "linear",
            Pipeline([("poly", PolynomialFeatures(include_bias=False)), ("ridge", Ridge(random_state=seed))]),
            {"model__poly__degree": [2, 3], "model__ridge__alpha": [0.01, 0.1, 1.0]},
            {"model__poly__degree": [2], "model__ridge__alpha": [0.1]},
            notes="Polynomial expansion (incl. material × time interactions) with ridge regularisation.",
        ),
        "random_forest": Candidate(
            "random_forest", "Random Forest", "tree",
            RandomForestRegressor(random_state=seed, n_jobs=n_jobs),
            {"model__n_estimators": [200], "model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 3, 5]},
            {"model__n_estimators": [100], "model__max_depth": [None], "model__min_samples_leaf": [3]},
            supports_tree_shap=True,
        ),
        "svr": Candidate(
            "svr", "Support Vector Regression", "kernel", SVR(kernel="rbf"),
            {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.005, 0.01, 0.02], "model__gamma": ["scale"]},
            {"model__C": [1.0], "model__epsilon": [0.01], "model__gamma": ["scale"]},
        ),
        "xgboost": Candidate(
            "xgboost", "XGBoost", "tree",
            XGBRegressor(random_state=seed, n_jobs=n_jobs, objective="reg:squarederror", verbosity=0),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1, 0.3], "model__max_depth": [3, 5]},
            {"model__n_estimators": [100], "model__learning_rate": [0.1], "model__max_depth": [3]},
            supports_tree_shap=True,
        ),
        "mlp": Candidate(
            "mlp", "Neural Network (MLP)", "neural",
            MLPRegressor(random_state=seed, max_iter=600, early_stopping=True, n_iter_no_change=20),
            {"model__hidden_layer_sizes": [(32, 32), (64, 64)], "model__alpha": [1e-4, 1e-3]},
            {"model__hidden_layer_sizes": [(32, 32)], "model__alpha": [1e-3]},
        ),
    }


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def build_pipeline(candidate: Candidate, numeric: list[str], categorical: list[str]) -> Pipeline:
    from sklearn.base import clone

    return Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", clone(candidate.estimator))])


# ----------------------------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------------------------

@dataclass
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    cv: Any
    groups_train: np.ndarray | None
    description: str
    warnings: list[str]
    test_groups: list[str]


def make_split(
    df: pd.DataFrame,
    strategy: str,
    seed: int,
    test_fraction: float = 0.2,
    group_col: str = "test_run_id",
    strata_col: str = "material",
    n_folds: int = 5,
) -> SplitResult:
    n = len(df)
    warnings: list[str] = []
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    has_groups = group_col in df.columns and df[group_col].notna().all() and df[group_col].nunique() >= 2

    if strategy == "auto":
        strategy = "grouped" if has_groups else "random"
        if not has_groups:
            warnings.append("No test-run identifiers with at least two runs; falling back to a row-wise random split.")

    if strategy == "grouped":
        if not has_groups:
            raise ValueError("Grouped splitting requires a test_run_id column with at least two distinct runs.")
        groups = df[group_col].astype(str).to_numpy()
        strata = df[strata_col].astype(str).to_numpy() if strata_col in df.columns and df[strata_col].notna().all() else np.array(["all"] * n)
        test_groups: list[str] = []
        for stratum in sorted(set(strata)):
            g_in = sorted(set(groups[strata == stratum]))
            if len(g_in) < 2:
                warnings.append(f"'{stratum}' has a single test run; it cannot be evaluated on an unseen run and stays in training.")
                continue
            k = max(1, int(round(len(g_in) * test_fraction)))
            k = min(k, len(g_in) - 1)
            chosen = rng.choice(g_in, size=k, replace=False).tolist()
            test_groups.extend(chosen)
        test_mask = np.isin(groups, test_groups)
        train_idx, test_idx = idx[~test_mask], idx[test_mask]
        groups_train = groups[train_idx]
        n_train_groups = len(set(groups_train))
        cv = GroupKFold(n_splits=max(2, min(n_folds, n_train_groups)))
        desc = (f"Whole test runs held out ({len(test_groups)} of {len(set(groups))} runs, one or more per {strata_col}); "
                f"{cv.get_n_splits()}-fold grouped cross-validation on the remaining runs for tuning and validation.")
        return SplitResult(train_idx, test_idx, "grouped", cv, groups_train, desc, warnings, test_groups)

    if strategy == "chronological":
        if "time_s" not in df.columns or df["time_s"].isna().all():
            raise ValueError("Chronological splitting requires a time_s column.")
        key = df[group_col].astype(str) if has_groups else pd.Series(["all"] * n, index=df.index)
        test_mask = np.zeros(n, dtype=bool)
        for _, sub in df.groupby(key):
            thr = sub["time_s"].quantile(1 - test_fraction)
            test_mask[sub.index[sub["time_s"] > thr]] = True
        train_idx, test_idx = idx[~test_mask], idx[test_mask]
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        desc = (f"Last {test_fraction:.0%} of sliding time of every run held out as test set; "
                f"{n_folds}-fold shuffled cross-validation on the earlier samples.")
        return SplitResult(train_idx, test_idx, "chronological", cv, None, desc, warnings, [])

    if strategy == "random":
        strat = df[strata_col].astype(str) if strata_col in df.columns and df[strata_col].notna().all() and df[strata_col].value_counts().min() >= 2 else None
        train_idx, test_idx = train_test_split(idx, test_size=test_fraction, random_state=seed, stratify=strat)
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        desc = (f"Row-wise random split ({test_fraction:.0%} test, stratified by {strata_col}); "
                f"{n_folds}-fold shuffled cross-validation on the training rows. Measures interpolation within known runs.")
        return SplitResult(np.sort(train_idx), np.sort(test_idx), "random", cv, None, desc, warnings, [])

    raise ValueError(f"Unknown split strategy '{strategy}'")


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, mape_eps: float | None = None) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    if n == 0:
        return {"r2": None, "mae": None, "rmse": None, "mape": None, "n": 0, "mape_coverage": 0.0}
    resid = y_true - y_pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(math.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else None
    eps = mape_eps if mape_eps is not None else max(1e-9, 0.01 * float(np.mean(np.abs(y_true))))
    mask = np.abs(y_true) > eps
    mape = float(100.0 * np.mean(np.abs(resid[mask] / y_true[mask]))) if mask.any() else None
    return {
        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape, "n": n,
        "mape_coverage": float(mask.mean()), "mape_eps": float(eps),
        "max_abs_error": float(np.max(np.abs(resid))), "bias": float(np.mean(resid)),
    }


def numeric_and_categorical(features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [f for f in features if STANDARD_FEATURES[f].dtype == "number"]
    categorical = [f for f in features if STANDARD_FEATURES[f].dtype == "category"]
    return numeric, categorical


def auto_select_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Pick informative inputs: present, varying, not the target or derived from it, not collinear duplicates."""
    notes: list[str] = []
    chosen: list[str] = []
    candidates_num = ["normal_load_n", "sliding_speed_m_s", "temperature_c", "time_s", "sliding_distance_m"]
    candidates_cat = ["material", "lubricant", "material_pair"]
    for f in candidates_cat:
        if f in df.columns and df[f].notna().any():
            if df[f].nunique() > 1:
                chosen.append(f)
            else:
                notes.append(f"{f} is constant ('{df[f].dropna().iloc[0]}') in this dataset and cannot be learned.")
    if "material" not in chosen:
        for f in ("composition", "pore_size_class"):
            if f in df.columns and df[f].notna().any() and df[f].nunique() > 1:
                chosen.append(f)
    for f in candidates_num:
        if f not in df.columns or df[f].isna().all():
            continue
        s = df[f].astype(float)
        if not is_varying(s):
            notes.append(f"{f} is fixed at about {s.dropna().mean():.4g} {STANDARD_FEATURES[f].unit} in this dataset; it is a fixed condition, not a learnable input.")
            continue
        collinear = False
        for g in [c for c in chosen if STANDARD_FEATURES[c].dtype == "number"]:
            c = np.corrcoef(s.fillna(s.median()), df[g].astype(float).fillna(df[g].median()))[0, 1]
            if abs(c) > 0.999:
                notes.append(f"{f} is perfectly collinear with {g} (r = {c:.4f}); only {g} is used.")
                collinear = True
                break
        if not collinear:
            chosen.append(f)
    if target in chosen:
        chosen.remove(target)
    return chosen, notes
