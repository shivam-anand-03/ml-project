# Machine-Learning Methodology

Implementation: `backend/app/ml/pipeline.py`, `backend/app/ml/explain.py`, `backend/app/services/training.py`, `backend/app/services/prediction.py`.

## Targets

Separate pipelines are trained per target: `cof` (coefficient of friction) and `wear_rate`. A target can only be trained when the dataset version contains measurements of it (`feature_summary.targets.<target>.available`). The bundled Kolev 2023 dataset supports `cof` only; wear-rate predictions are reported as *unavailable* rather than estimated from unrelated data.

## Feature selection

Automatic selection (default) keeps inputs that are present and **vary** in the data (a numeric input whose range is < 1 % of its magnitude is treated as a fixed condition, e.g. 0.4995–0.5027 m/s), drops features perfectly collinear with an already-selected one (|r| > 0.999, e.g. sliding distance vs time at constant speed), and excludes categorical attributes that are deterministic functions of `material` (composition, pore size class) when `material` itself is used. The user can override the feature list. For the bundled data the automatic choice is `material` + `time_s`; load, speed and lubricant become **fixed conditions** stored in the training domain.

## Preprocessing inside the pipeline (no leakage)

`ColumnTransformer` fitted **inside each cross-validation fold and on the training split only**:
- numeric: median imputation → standard scaling
- categorical: most-frequent imputation → one-hot encoding (unknown levels are rejected before they reach the pipeline)

## Candidate models and search grids

| Key | Model | Grid (standard) | Quick |
|---|---|---|---|
| `linear` | Linear Regression | – | – |
| `polynomial` | PolynomialFeatures + Ridge | degree ∈ {2, 3}, alpha ∈ {0.01, 0.1, 1} | degree 2, alpha 0.1 |
| `random_forest` | RandomForestRegressor | n_estimators 200, max_depth ∈ {None, 8, 16}, min_samples_leaf ∈ {1, 3, 5} | 100 trees, leaf 3 |
| `svr` | SVR (RBF) | C ∈ {0.1, 1, 10}, epsilon ∈ {0.005, 0.01, 0.02} | C 1, eps 0.01 |
| `xgboost` | XGBRegressor | n_estimators ∈ {100, 300}, learning_rate ∈ {0.05, 0.1, 0.3}, max_depth ∈ {3, 5} | 100, 0.1, 3 |
| `mlp` | MLPRegressor (early stopping) | hidden ∈ {(32,32), (64,64)}, alpha ∈ {1e-4, 1e-3} | (32,32), 1e-3 |

Tuning: `GridSearchCV`, scoring = negative RMSE, same CV splitter and same seed for every model. Polynomial degree 2 includes material × time interaction terms through the one-hot columns.

## Split strategies

| Strategy | Test set | Cross-validation | When |
|---|---|---|---|
| `grouped` (default when ≥ 2 runs exist) | whole test runs held out — one or more runs per material (≈ test fraction of the runs) | `GroupKFold` on the remaining runs (≤ 5 folds) | replicate / time-series runs: measures generalisation to an **unseen specimen** |
| `random` | row-wise stratified by material | `KFold(5, shuffle)` | reproduces the original authors' protocol; measures interpolation **within known runs** |
| `chronological` | last 20 % of each run's time | `KFold(5, shuffle)` on earlier samples | forecasting later behaviour |

A material with a single run cannot be evaluated on an unseen run; it stays in training and the limitation is logged. If the usable data has fewer than 20 rows, training is refused; with fewer than 200 training rows the held-out estimate is flagged unreliable and CV metrics are the primary evidence.

## Metrics

Computed by `regression_metrics` for **train**, **validation** (out-of-fold predictions on the training runs via `cross_val_predict` with the same grouped splitter) and **test** (untouched held-out set, evaluated once after selection):
- R², MAE, RMSE, MAPE. MAPE excludes rows whose measured value is below 1 % of the mean absolute target (e.g. run-in COF ≈ 0); the reported `mape_coverage` is the share of rows used.
- Per-fold RMSE of the selected hyper-parameters (`fold_rmse`), training duration.

**Model selection** uses validation (CV) RMSE only. The test set never influences selection.

## Prediction intervals

Split-conformal: the half-width is the `interval_level` (default 90 %) quantile of |out-of-fold residual| on the training runs. The empirical coverage on the held-out test set is stored and shown next to the interval so the user can judge calibration (`interval.test_coverage`). No other uncertainty claim is made.

## Domain validation at prediction time

- Categorical inputs must be a level seen in training (case-insensitive) — otherwise HTTP 422.
- Numeric inputs inside [min, max] → in domain; within `extrapolation_margin` × range outside → prediction with a *mild extrapolation* warning; further → **refused** (HTTP 422) unless `allow_far_extrapolation` is enabled in Settings, in which case a strong warning is attached.
- Fixed conditions (e.g. load = 50 N) supplied with a different value (beyond `fixed_condition_tolerance`) are treated as far extrapolation: the model has no evidence for them.
- Inputs that were not recorded in the data (e.g. temperature) cannot be used.

## Explainability

`ModelExplainer` wraps the fitted pipeline: `TreeExplainer` for Random Forest / XGBoost, `LinearExplainer` for linear regression, `PermutationExplainer` (model-agnostic, 60-row background) otherwise. SHAP values of one-hot columns are summed back to the original feature. Stored per model: mean |SHAP| importance, per-level mean effects of categorical features, a 200-row summary sample; local explanations are computed on demand. All explanations are labelled as descriptions of model behaviour, not physical causation.

## Persistence

`storage/models/model_<id>.joblib` holds: pipeline, feature schema, numeric/categorical lists, training config, dataset version id, training domain (ranges, levels, fixed conditions, unrecorded inputs), interval parameters, metrics, a 200-row background sample for SHAP, library versions. The database stores the same metadata plus per-split metrics, CV results, test/OOF predictions (down-sampled for plots) and global importance.

## Results on the bundled dataset (grouped split, quick search, seed 42)

From the test run executed during development (`tests/conftest.py` reproduces it): held-out-run R² for COF was about 0.45 (linear) to 0.71 (MLP/XGBoost/RF), validation (out-of-fold) R² about 0.31–0.43, RMSE ≈ 0.07. These are far below the ≈ 0.99 R² obtained with a row-wise random split (the original authors' protocol) because predicting an unseen replicate run is a genuinely harder task: run-to-run scatter of the steady-state COF is of the same order as the material differences. The application always shows the actual numbers of the current run; the figures above are not hard-coded anywhere.

## Known limitations (bundled dataset)

1. Only COF can be modelled; wear rate is unavailable.
2. Load, speed, lubricant and temperature were constant or unrecorded; their effects cannot be learned or claimed.
3. Three replicate runs per material → wide uncertainty on held-out-run metrics and low power for between-material tests.
4. Rows within a run are autocorrelated; pooled correlation p-values are optimistic (stated in the UI).
