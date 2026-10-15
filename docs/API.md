# API Reference

Base URL `http://localhost:8000`. Interactive documentation: `/docs` (Swagger UI), `/redoc`, schema `/openapi.json`. When the backend is started with `API_KEY` set, every `/api/*` request must send `X-API-Key: <key>` (401 otherwise).

Error format: `{"detail": "<message>"}` with 400 (bad input / ingestion / training config), 404 (not found), 409 (conflict: no trained model, missing artifact, record already used), 422 (prediction refused or request validation), 500.

## Health & settings
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + database check |
| GET | `/api/settings` | current settings, defaults, environment |
| PUT | `/api/settings` | patch settings (extrapolation policy, interval level, training defaults, project info) |
| GET | `/api/settings/data-dictionary` | feature registry, material catalogue, equations |

## Datasets
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/datasets/upload` (multipart: `files[]`, `name`, `description`) | create dataset, store and inspect files |
| POST | `/api/datasets/{id}/files` | add files to a dataset |
| POST | `/api/datasets/import-bundled` | import the Kolev 2023 workbooks (raw + processed versions) |
| GET | `/api/datasets` | list datasets |
| GET | `/api/datasets/{id}` | dataset with files (inspection without preview) and versions |
| GET | `/api/datasets/{id}/files/{file_id}/inspection` | full inspection incl. 100-row preview, suggested mapping, parsed run metadata |
| POST | `/api/datasets/{id}/ingest` | body `{sheets:[{file_id, sheet_name, mapping, constants, units}], notes}` → raw version |
| POST | `/api/datasets/{id}/preprocess` | body `{version_id?, options, notes}` → processed version (options: `GET /api/datasets/preprocess-options`) |
| GET | `/api/datasets/{id}/versions`, `/api/datasets/versions/{version_id}` | versions, cleaning log, quality report, models trained on it |
| GET | `/api/datasets/{id}/preview?version_id&limit&offset&material&test_run_id&usable` | paged records |
| GET | `/api/datasets/{id}/quality?version_id` · `/quality/report.md` | quality report JSON / downloadable markdown |
| GET | `/api/datasets/{id}/export?version_id&format=csv|xlsx` | export a version |
| DELETE | `/api/datasets/{id}` | delete (refused while models reference it) |
| GET | `/api/datasets/feature-dictionary` | standard features |

## Experimental analysis (measured data)
`GET /api/analysis/{version_id}/overview | target-vs-time?target&group_by&bins | target-vs-feature?feature&target&group_by | material-comparison?target&group_by | correlation?target | distributions?target | trends?target`

Constant or unrecorded features return `varied: false` with an explanatory `reason` instead of a chart.

## Models & training
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/models/candidates` · `/api/models/metric-explanations` | algorithms + grids; metric definitions |
| POST | `/api/models/train` | body `{dataset_version_id, target, models[], features?, split_strategy, test_fraction, n_folds, search_mode, seed?, interval_level, notes?}` → 202 with run; executes in a background worker |
| GET | `/api/models/runs` · `/api/models/train/{run_id}` | runs, status, log, leaderboard, model summaries |
| GET | `/api/models?target&status&dataset_version_id` | model registry |
| GET | `/api/models/production` | production / latest-best / active model per target |
| GET | `/api/models/{id}` · `/api/models/{id}/metrics` | details, training domain; metrics with test/OOF predictions and fold RMSE |
| POST | `/api/models/{id}/promote` · `/archive` | explicit promotion (previous production archived) / archiving |
| GET | `/api/models/{id}/explain/global` | SHAP importance, level effects, summary sample |
| POST | `/api/models/{id}/explain/local` | body `{inputs}` → contributions for one prediction |
| GET | `/api/models/compare/explainability` | importance of the active COF and wear-rate models side by side |

## Predictions & simulation
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/predict/schema` | inputs (ranges, levels), fixed conditions, unrecorded inputs, active models |
| POST | `/api/predict` | body `{inputs, cof_model_id?, wear_model_id?, explain, save}` → COF and wear-rate results (each `available` or with `reason`), interval, warnings, explanation |
| POST | `/api/predict/batch` | body `{rows[], save}`; per-row `ok` / `error` |
| POST | `/api/predict/batch/upload` | CSV/XLSX file with standard feature columns |
| GET | `/api/predictions/history?limit&offset&source` · `/api/predictions/{id}` · `/api/predictions/{id}/export?format=json|csv` | history and export |
| POST | `/api/simulate` | body `{target, base_inputs, sweep:{feature,min,max,steps,by}?, compare?, model_id?}` → baseline, sensitivity, level comparisons, sweep curves |
| POST | `/api/simulate/compare` | body `{target, scenarios[]}` → side-by-side scenario table |

## Experimental validation
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/validation` | `{experiment_id, inputs, measured_cof?, measured_wear_rate?, prediction_id?, notes?}` → record with errors |
| GET | `/api/validation?status` · `/api/validation/summary` | records; error statistics (overall and per model) |
| PATCH | `/api/validation/{id}` | `{status: approved|rejected|pending, notes}` |
| DELETE | `/api/validation/{id}` | delete (unless already added to a version) |
| POST | `/api/validation/augment` | `{dataset_version_id, validation_ids[]}` → new augmented version |

## Reports & dashboard
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/reports/generate` | `{format: pdf|xlsx, title?, dataset_version_id?, model_version_ids?, prediction_ids?, include_predictions, include_validation, include_explainability}` |
| GET | `/api/reports` · `/api/reports/{id}/download` · DELETE `/api/reports/{id}` | history, download, delete |
| GET | `/api/dashboard/overview?dataset_version_id&material&lubricant&load_min&load_max&speed_min&speed_max&temp_min&temp_max` | overview statistics |

## Example

```bash
curl -s -X POST localhost:8000/api/datasets/import-bundled | jq .processed_version.id      # → 2
curl -s -X POST localhost:8000/api/models/train -H 'Content-Type: application/json' \
  -d '{"dataset_version_id":2,"target":"cof","models":["xgboost","random_forest"],"search_mode":"quick"}'
curl -s -X POST localhost:8000/api/predict -H 'Content-Type: application/json' \
  -d '{"inputs":{"material":"AC","time_s":200,"normal_load_n":50,"sliding_speed_m_s":0.5}}'
```
