"""SHAP-based explanations aggregated back to the original (pre-encoding) features."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import shap

from app.core.features import STANDARD_FEATURES


def _transformed_feature_names(pipeline) -> list[str]:
    prep = pipeline.named_steps["prep"]
    return [str(n) for n in prep.get_feature_names_out()]


def _original_feature_of(transformed: str, features: list[str]) -> str:
    # names look like 'num__time_s' or 'cat__material_AC'
    body = transformed.split("__", 1)[1] if "__" in transformed else transformed
    for f in sorted(features, key=len, reverse=True):
        if body == f or body.startswith(f + "_"):
            return f
    return body


class ModelExplainer:
    """Wraps a fitted sklearn Pipeline([('prep', ...), ('model', ...)]) with a suitable SHAP explainer."""

    def __init__(self, pipeline, features: list[str], background: pd.DataFrame, family: str, max_background: int = 60):
        self.pipeline = pipeline
        self.features = features
        self.family = family
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["model"]
        self.t_names = _transformed_feature_names(pipeline)
        bg = background[features].sample(n=min(max_background, len(background)), random_state=0) if len(background) > max_background else background[features]
        self.X_bg = np.asarray(self.prep.transform(bg), dtype=float)
        self.method: str
        if family == "tree":
            self.explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")
            self.method = "TreeExplainer (exact Shapley values for tree ensembles)"
        elif family == "linear" and hasattr(self.model, "coef_"):
            self.explainer = shap.LinearExplainer(self.model, self.X_bg)
            self.method = "LinearExplainer (coefficient × centred feature value)"
        else:
            self.explainer = shap.PermutationExplainer(self.model.predict, self.X_bg)
            self.method = "PermutationExplainer (model-agnostic, background of %d training rows)" % len(self.X_bg)

    def _shap_matrix(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        Xt = np.asarray(self.prep.transform(X[self.features]), dtype=float)
        if isinstance(self.explainer, shap.PermutationExplainer):
            res = self.explainer(Xt, max_evals=max(2 * Xt.shape[1] + 1, 4 * Xt.shape[1] + 1), silent=True)
            values = np.asarray(res.values)
            base = float(np.mean(np.atleast_1d(res.base_values)))
        else:
            values = np.asarray(self.explainer.shap_values(Xt))
            ev = getattr(self.explainer, "expected_value", 0.0)
            base = float(np.mean(np.atleast_1d(ev)))
        if values.ndim == 3:
            values = values[:, :, 0]
        return values, base

    def aggregate(self, values: np.ndarray) -> pd.DataFrame:
        """Sum SHAP values of one-hot columns back to their original feature."""
        agg = pd.DataFrame(0.0, index=range(values.shape[0]), columns=self.features)
        for j, name in enumerate(self.t_names):
            orig = _original_feature_of(name, self.features)
            if orig in agg.columns:
                agg[orig] += values[:, j]
        return agg

    def global_importance(self, X: pd.DataFrame, max_rows: int = 300) -> dict[str, Any]:
        sample = X[self.features].sample(n=min(max_rows, len(X)), random_state=0) if len(X) > max_rows else X[self.features]
        values, base = self._shap_matrix(sample)
        agg = self.aggregate(values)
        mean_abs = agg.abs().mean().sort_values(ascending=False)
        importance = [
            {"feature": f, "label": STANDARD_FEATURES[f].label if f in STANDARD_FEATURES else f,
             "unit": STANDARD_FEATURES[f].unit if f in STANDARD_FEATURES else "",
             "mean_abs_shap": float(v), "mean_shap": float(agg[f].mean()),
             "share": float(v / mean_abs.sum()) if mean_abs.sum() > 0 else 0.0}
            for f, v in mean_abs.items()
        ]
        summary_sample: dict[str, list] = {}
        for f in self.features:
            vals = sample[f].to_numpy()
            pts = []
            for v, s in zip(vals, agg[f].to_numpy()):
                pts.append([None if (isinstance(v, float) and np.isnan(v)) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)), float(s)])
            summary_sample[f] = pts[:200]
        # per-level effect for categorical features (mean SHAP contribution of each level)
        level_effects: dict[str, list] = {}
        for f in self.features:
            if STANDARD_FEATURES.get(f) and STANDARD_FEATURES[f].dtype == "category":
                grp = pd.DataFrame({"level": sample[f].astype(str).to_numpy(), "shap": agg[f].to_numpy()}).groupby("level")["shap"]
                level_effects[f] = [{"level": k, "mean_shap": float(v), "n": int(c)} for (k, v), c in zip(grp.mean().items(), grp.count())]
        return {"method": self.method, "base_value": base, "n_samples": int(len(sample)), "importance": importance,
                "summary_sample": summary_sample, "level_effects": level_effects}

    def local(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        values, base = self._shap_matrix(X)
        agg = self.aggregate(values)
        preds = self.pipeline.predict(X[self.features])
        out = []
        for i in range(len(X)):
            contribs = [
                {"feature": f, "label": STANDARD_FEATURES[f].label if f in STANDARD_FEATURES else f,
                 "value": (None if pd.isna(X.iloc[i][f]) else (float(X.iloc[i][f]) if isinstance(X.iloc[i][f], (int, float, np.floating, np.integer)) else str(X.iloc[i][f]))),
                 "unit": STANDARD_FEATURES[f].unit if f in STANDARD_FEATURES else "",
                 "contribution": float(agg.iloc[i][f])}
                for f in self.features
            ]
            contribs.sort(key=lambda c: -abs(c["contribution"]))
            out.append({"base_value": base, "prediction": float(preds[i]), "contributions": contribs, "method": self.method})
        return out
