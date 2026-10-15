# AI TriboPredict

**Data-Driven Prediction of Lubricant Friction and Wear Using Machine Learning — An Experimental and Machine Learning Approach to Data-Enabled Tribology.**

A full-stack, data-driven application that ingests pin-on-disk experimental data, validates and preprocesses it, trains and compares machine-learning models, and serves predictions, what-if simulations, SHAP explanations, an experimental validation loop and PDF/Excel reports. Every number displayed is computed from the stored experimental data; predictions are always labelled as model outputs, never as measurements.

| Layer | Stack |
|---|---|
| Frontend | Next.js 15 (App Router), TypeScript, Tailwind CSS 4, Recharts, TanStack Query |
| Backend | FastAPI, Pydantic 2, SQLAlchemy 2, Alembic, pandas, NumPy, scikit-learn, XGBoost, SHAP, joblib, ReportLab |
| Database | PostgreSQL 16 (Docker Compose) — SQLite for local development and tests |
| Deployment | Docker, Docker Compose, environment variables |

## Repository layout

```
ml-project/
├── backend/                 FastAPI service, ML pipeline, migrations, tests
│   ├── app/api/             REST endpoints (datasets, analysis, models, predict, validation, reports, dashboard, settings)
│   ├── app/services/        ingestion, mapping, preprocessing, quality, datasets, training, prediction, analysis, reports, jobs
│   ├── app/ml/              candidate models, splits, metrics, SHAP explainers
│   ├── app/core/features.py standard feature registry (data dictionary) and material catalogue
│   ├── alembic/             database migrations
│   └── tests/               pytest suite (runs on SQLite against the real bundled dataset)
├── frontend/                Next.js dashboard (11 pages)
├── data/source/             byte-for-byte copy of the Kolev 2023 Mendeley dataset (never modified)
├── docs/                    DATASET_AUDIT, DATA_DICTIONARY, ML_METHODOLOGY, API, DEPLOYMENT
├── docker-compose.yml       db + backend + frontend
├── run_dev.sh               one-command local run (SQLite, no Docker)
└── .env.example             environment variables for Docker Compose
```

## Quick start (local, no Docker)

Requirements: Python 3.12 (or 3.11), Node 20+, npm.

```bash
./run_dev.sh            # backend http://localhost:8000 (Swagger at /docs) + frontend http://localhost:3000
./run_dev.sh backend    # backend only
./run_dev.sh frontend   # frontend only
```

The script creates `backend/.venv`, installs pinned dependencies, uses SQLite (`backend/storage/tribopredict.db`) and starts both servers. Stop with Ctrl-C.

## Quick start (Docker Compose, PostgreSQL)

```bash
cp .env.example .env         # set POSTGRES_PASSWORD (and optionally API_KEY)
docker compose up --build    # db :5432, backend :8000, frontend :3000
```

The backend entrypoint waits for PostgreSQL, applies Alembic migrations and starts Uvicorn. See `docs/DEPLOYMENT.md`.

## Using the application with your Excel files

1. **Datasets → Upload experimental data.** Select one or more `.xlsx` / `.xls` / `.csv` / tribometer `.dwf` files (for example one workbook per test run). Every worksheet is inspected: dimensions, column types, missing values, duplicates, numeric ranges, a 100-row preview and run metadata parsed from worksheet names such as `AC_3_2_50N_80rpm_120mm`.
2. **Column mapping.** Suggested mappings from source columns to standardized features (time, friction force, load, speed, temperature, lubricant, material, sliding distance, COF, wear volume/mass/rate …) are shown; adjust them, add constants (e.g. normal load = 50 N, lubricant = dry) and declared units. A target must be available (COF, or friction force + load, or wear rate) — nothing is invented. Click **Create raw dataset version**.
3. **Versions & preprocessing.** Create a processed version: duplicates, non-finite values, negative friction readings, derived COF = FF / load, sliding distance = v · t, sliding speed from rpm and track diameter, run-in and outlier flags — every step is logged. The raw version is never changed. Export any version as CSV/XLSX and download the data-quality report.
4. **Train Models.** Choose target, candidate models, features (automatic by default), split strategy (grouped by test run recommended for replicate time series), search mode and folds. Training runs in the background.
5. **Model Evaluation.** Leaderboard with R², MAE, RMSE, MAPE for validation (out-of-fold CV) and the untouched test set, per-fold RMSE, actual-vs-predicted and residual plots. Promote a model to production explicitly.
6. **AI Predictions / What-If Simulator / Explainability.** Enter conditions (dropdowns and ranges come from the training data), get predictions with 90 % split-conformal intervals, domain warnings and SHAP contributions; sweep inputs, compare materials and scenarios.
7. **Experimental Validation.** Record measured results against predictions, approve them, add them to a new dataset version, retrain and compare before promoting.
8. **Reports.** Generate PDF or Excel reports from real stored results.

**One click to load the real data:** Datasets → *Import bundled Kolev 2023 dataset* imports the 12 processed workbooks from `data/source/` with run metadata, creates the raw and processed versions, and you can train immediately.

## The bundled dataset in one paragraph

Kolev (2023), *Friction coefficient data of open-cell AlSi10Mg and AlSi10Mg-Al2O3 materials with different pore sizes by pin-on-disk test and machine learning prediction*, Mendeley Data, DOI 10.17632/2356m76ktj.1 (CC BY 4.0). Dry sliding, 50 N, 0.5 m/s, 420 s; 4 materials (AC, AE = AlSi10Mg–Al2O3 composites with 800–1000 / 1000–1200 µm pores; C, E = unreinforced AlSi10Mg with the same pore classes) × 3 replicate runs × 462 samples of friction force vs time. **Only COF can be learned from it: there are no wear measurements, and load, speed, lubricant and temperature were not varied.** The application reports wear-rate predictions as *unavailable* for this dataset and restricts predictions to the tested conditions. Full audit: `docs/DATASET_AUDIT.md`.

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q
```

42 tests cover ingestion (xlsx/csv/dwf, empty files, unsupported types), column mapping and unit conversion, preprocessing edge cases (zero load, zero distance, duplicates, negative friction, missing targets, wear never derived from mass), dataset API, analysis endpoints, training on the real data, metrics safeguards, prediction validation (unknown categories, out-of-range and far-out-of-domain inputs, missing models), batch predictions, simulation, model promotion/archiving, the validation loop with dataset augmentation and retraining, PDF/Excel reports, dashboard and settings.

## Retraining and model versioning

- Every training run stores one **model version per candidate algorithm** with its pipeline (`storage/models/model_<id>.joblib`), feature schema, hyper-parameters, training domain, metrics, CV results and SHAP importance.
- The best model of a run (lowest cross-validation RMSE) is flagged `is_best_of_run`; predictions use the **production** model if one exists, otherwise the newest best candidate.
- Promotion is explicit (`POST /api/models/{id}/promote`); the previous production model is archived, never deleted or overwritten.
- To retrain after new experiments: approve validation records → create an augmented dataset version → train on it → compare on *Model Evaluation* → promote.

## Documentation

- `docs/DATASET_AUDIT.md` — Phase-1 audit of the supplied workbooks
- `docs/DATA_DICTIONARY.md` — standard features, units, validity ranges, equations
- `docs/ML_METHODOLOGY.md` — splits, tuning, metrics, intervals, explainability, limitations
- `docs/API.md` — endpoint reference (Swagger UI at `/docs`, OpenAPI at `/openapi.json`)
- `docs/DEPLOYMENT.md` — Docker Compose, environment variables, migrations, backups
