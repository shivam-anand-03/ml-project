"""Test fixtures: isolated SQLite database + storage directory, synchronous training, real bundled dataset."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="tribopredict_tests_"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["STORAGE_DIR"] = str(_TMP / "storage")
os.environ["TRAINING_SYNC"] = "true"
os.environ["N_JOBS"] = "1"
os.environ["API_KEY"] = ""
os.environ["BUNDLED_DATASET_DIR"] = str(Path(__file__).resolve().parents[2] / "data" / "source" / "kolev_2023_pin_on_disk")

from fastapi.testclient import TestClient  # noqa: E402

from app.db import init_db, reset_engine  # noqa: E402
from app.main import app  # noqa: E402

DATA_DIR = Path(os.environ["BUNDLED_DATASET_DIR"])


@pytest.fixture(scope="session", autouse=True)
def _db():
    reset_engine()
    init_db()
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="session")
def client(_db):
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def bundled(client):
    """Import the real Kolev 2023 dataset once per session."""
    r = client.post("/api/datasets/import-bundled")
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture(scope="session")
def trained(client, bundled):
    """Train a quick model comparison on the real processed data once per session."""
    body = {"dataset_version_id": bundled["processed_version"]["id"], "target": "cof",
            "models": ["linear", "random_forest", "xgboost"], "split_strategy": "grouped", "search_mode": "quick"}
    r = client.post("/api/models/train", json=body)
    assert r.status_code == 202, r.text
    run = client.get(f"/api/models/train/{r.json()['id']}").json()
    assert run["status"] == "completed", run
    return run
