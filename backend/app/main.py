"""AI TriboPredict — FastAPI application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import analysis, dashboard, datasets, models, predict, reports, settings as settings_api, validation
from app.config import get_settings
from app.db import init_db
from app.services.ingestion import IngestionError
from app.services.jobs import get_runner
from app.services.mapping import MappingError
from app.services.prediction import ModelUnavailable, PredictionError
from app.services.training import TrainingError

logger = logging.getLogger("tribopredict")

DESCRIPTION = """
**AI TriboPredict** — data-driven prediction of friction (and wear, when measured) from pin-on-disk experiments.

Workflow: upload / import a dataset → map columns → preprocess → train & compare models → predict, simulate, explain →
record experimental validation → augment dataset → retrain → promote. Every number shown is computed from the stored data;
predictions are always labelled as model outputs, never as measurements.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    init_db()  # idempotent; Alembic migrations are used for schema evolution on PostgreSQL
    logger.info("AI TriboPredict started (db=%s)", settings.database_url.split("://", 1)[0])
    yield
    get_runner().shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, description=DESCRIPTION, lifespan=lifespan,
                  docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(PredictionError)
    async def _pred_err(_: Request, exc: PredictionError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ModelUnavailable)
    async def _model_err(_: Request, exc: ModelUnavailable):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(IngestionError)
    async def _ingest_err(_: Request, exc: IngestionError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(MappingError)
    async def _map_err(_: Request, exc: MappingError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(TrainingError)
    async def _train_err(_: Request, exc: TrainingError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _validation_err(_: Request, exc: RequestValidationError):
        errors = [{"loc": [str(x) for x in e.get("loc", [])], "msg": str(e.get("msg", "")), "type": str(e.get("type", ""))} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "Request validation failed: " + "; ".join(f"{'.'.join(e['loc'][1:]) or 'body'}: {e['msg']}" for e in errors), "errors": errors})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"})

    app.include_router(settings_api.health_router)
    app.include_router(datasets.router)
    app.include_router(analysis.router)
    app.include_router(models.router)
    app.include_router(predict.router)
    app.include_router(validation.router)
    app.include_router(reports.router)
    app.include_router(dashboard.router)
    app.include_router(settings_api.router)
    return app


app = create_app()
