# Deployment

## Docker Compose (recommended)

```bash
cp .env.example .env            # edit POSTGRES_PASSWORD, optionally API_KEY
docker compose up --build -d
docker compose logs -f backend  # wait for "Application startup complete"
```

Services:

| Service | Image | Port | Notes |
|---|---|---|---|
| `db` | postgres:16-alpine | 5432 | volume `pgdata`; health-checked |
| `backend` | built from `backend/Dockerfile` | 8000 | entrypoint waits for PostgreSQL, runs `alembic upgrade head`, starts Uvicorn; volume `storage` for uploads, models, reports, exports |
| `frontend` | built from `frontend/Dockerfile` (standalone Next.js) | 3000 | `NEXT_PUBLIC_API_URL` is baked in at build time |

Open http://localhost:3000 → Datasets → *Import bundled Kolev 2023 dataset* → Train Models.

### Environment variables

| Variable | Default | Used by |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / `POSTGRES_PORT` | tribo / change_me / tribopredict / 5432 | db, backend |
| `DATABASE_URL` | derived in compose: `postgresql+psycopg://…@db:5432/…` | backend |
| `STORAGE_DIR` | `/app/storage` | backend |
| `BUNDLED_DATASET_DIR` | `/data/source/kolev_2023_pin_on_disk` | backend |
| `API_KEY` | empty (no auth) | backend requires `X-API-Key`; frontend sends it (`NEXT_PUBLIC_API_KEY`) |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | backend |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | frontend build |
| `RANDOM_SEED` | 42 | backend |
| `TRAINING_WORKERS` | 1 | background training threads |
| `N_JOBS` | 2 | scikit-learn / XGBoost parallelism |
| `MAX_UPLOAD_MB` | 50 | upload limit |

Secrets are read from the environment only; nothing is stored in source code.

### Migrations

Schema changes are managed with Alembic (`backend/alembic/`). The initial migration creates all 12 tables with indexes. To add a migration after changing `app/models.py`:

```bash
cd backend && DATABASE_URL=postgresql+psycopg://tribo:***@localhost:5432/tribopredict .venv/bin/alembic revision --autogenerate -m "describe change"
.venv/bin/alembic upgrade head
```

The application also calls `Base.metadata.create_all` at startup for SQLite development databases; this is a no-op when migrations have already been applied.

### Backups

- Database: `docker compose exec db pg_dump -U tribo tribopredict > backup.sql`
- Files (uploads, model artifacts, reports): back up the `storage` volume (`docker run --rm -v ml-project_storage:/s -v $PWD:/b alpine tar czf /b/storage.tgz /s`).

## Local development without Docker

`./run_dev.sh` — backend with SQLite on :8000 (auto-reload) and frontend on :3000. Environment for the backend is read from `backend/.env` (created from `backend/.env.example`).

To use a local PostgreSQL instead, set in `backend/.env`:

```
DATABASE_URL=postgresql+psycopg://tribo:tribo_secret@localhost:5432/tribopredict
```

and run `cd backend && .venv/bin/alembic upgrade head` once.

## Running the tests

```bash
cd backend && .venv/bin/python -m pytest -q      # SQLite in a temp dir; trains on the real bundled data (~15 s)
cd frontend && npx tsc --noEmit && npm run build  # type-check and production build
```

## Production notes

- Set `API_KEY` and restrict `CORS_ORIGINS` to the frontend origin.
- Put the services behind a reverse proxy with TLS (e.g. Caddy / nginx) and forward `/api` to the backend.
- Training runs on a thread pool inside the backend process; for heavy workloads increase `N_JOBS` or run a dedicated backend replica for training.
- Model artifacts are version-stamped with library versions; retrain after upgrading scikit-learn / XGBoost majors.
