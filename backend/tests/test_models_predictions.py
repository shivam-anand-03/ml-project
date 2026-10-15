"""Training, evaluation metrics, model versioning, prediction API, simulation, validation loop, reports."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.pipeline import regression_metrics


def test_regression_metrics_safeguards():
    m = regression_metrics(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    assert m["r2"] == 1.0 and m["rmse"] == 0.0 and m["mape"] == 0.0
    assert m["mape_coverage"] == pytest.approx(2 / 3)  # the zero row is excluded from MAPE
    assert regression_metrics(np.array([]), np.array([]))["n"] == 0
    assert regression_metrics(np.array([1.0, 1.0]), np.array([1.0, 2.0]))["r2"] is None  # constant truth


def test_training_requires_available_target(client, bundled):
    r = client.post("/api/models/train", json={"dataset_version_id": bundled["processed_version"]["id"], "target": "wear_rate", "models": ["linear"]})
    assert r.status_code == 400 and "wear_rate" in r.json()["detail"]


def test_training_rejects_unknown_model(client, bundled):
    r = client.post("/api/models/train", json={"dataset_version_id": bundled["processed_version"]["id"], "target": "cof", "models": ["bogus"]})
    assert r.status_code == 400


def test_training_results_are_real_and_comparable(client, trained):
    assert trained["status"] == "completed"
    lb = trained["summary"]["leaderboard"]
    assert {e["algorithm"] for e in lb} == {"linear", "random_forest", "xgboost"}
    assert all(e["error"] is None for e in lb)
    assert trained["summary"]["features"] == ["material", "time_s"]
    assert trained["summary"]["split"]["strategy"] == "grouped"
    assert len(trained["summary"]["split"]["test_groups"]) == 4
    best = trained["summary"]["best_algorithm"]
    assert min(lb, key=lambda e: e["validation_rmse"])["algorithm"] == best
    for m in trained["models"]:
        for split in ("train", "validation", "test"):
            assert m["metrics"][split]["rmse"] > 0 and m["metrics"][split]["mae"] > 0
            assert -1 <= m["metrics"][split]["r2"] <= 1
        assert m["training_duration_s"] > 0
        assert m["n_train"] == 3696 and m["n_test"] == 1848
        assert m["interval"]["half_width"] > 0


def test_model_endpoints(client, trained):
    mid = trained["summary"]["best_model_version_id"]
    r = client.get(f"/api/models/{mid}")
    assert r.status_code == 200 and r.json()["artifact_available"] is True
    dom = r.json()["training_domain"]
    assert dom["features"]["material"]["levels"].keys() == {"AC", "AE", "C", "E"}
    assert dom["fixed_conditions"]["normal_load_n"]["value"] == 50.0
    assert "temperature_c" in dom["not_recorded"]
    r = client.get(f"/api/models/{mid}/metrics")
    assert r.status_code == 200 and r.json()["test_predictions"] and r.json()["fold_rmse"]
    r = client.get(f"/api/models/{mid}/explain/global")
    assert r.status_code == 200
    imp = {i["feature"]: i["mean_abs_shap"] for i in r.json()["importance"]}
    assert set(imp) == {"material", "time_s"} and all(v >= 0 for v in imp.values())
    r = client.post(f"/api/models/{mid}/explain/local", json={"inputs": {"material": "C", "time_s": 300}})
    assert r.status_code == 200 and len(r.json()["contributions"]) == 2
    assert client.get("/api/models/candidates").status_code == 200
    assert client.get("/api/models/metric-explanations").json()["r2"]
    assert client.get("/api/models", params={"target": "cof"}).json()
    assert client.get("/api/models/99999").status_code == 404


def test_predict_and_history(client, trained):
    r = client.post("/api/predict", json={"inputs": {"material": "AC", "time_s": 200, "normal_load_n": 50, "sliding_speed_m_s": 0.5}})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["cof"]["available"] is True and 0 < p["cof"]["value"] < 1
    assert p["cof"]["interval"]["low"] < p["cof"]["value"] < p["cof"]["interval"]["high"]
    assert p["wear_rate"]["available"] is False and "No trained" in p["wear_rate"]["reason"]
    assert p["in_domain"] is True and p["warnings"] == []
    assert p["cof"]["explanation"]["contributions"]
    assert "not an experimentally measured" in p["note"]
    pid = p["prediction_id"]
    r = client.get("/api/predictions/history")
    assert r.status_code == 200 and r.json()["items"][0]["id"] == pid
    assert client.get(f"/api/predictions/{pid}").status_code == 200
    assert client.get(f"/api/predictions/{pid}/export", params={"format": "csv"}).status_code == 200
    assert client.get("/api/predictions/999999").status_code == 404
    assert client.get("/api/predict/schema").json()["targets"]["cof"]["available"] is True


def test_predict_validation_errors(client, trained):
    assert client.post("/api/predict", json={"inputs": {"material": "ZZ", "time_s": 10}}).status_code == 422  # unknown category
    assert client.post("/api/predict", json={"inputs": {"material": "AC"}}).status_code == 422  # missing input
    assert client.post("/api/predict", json={"inputs": {"material": "AC", "time_s": "abc"}}).status_code == 422  # invalid number
    assert client.post("/api/predict", json={"inputs": {"material": "AC", "time_s": -5}}).status_code == 422  # negative time
    assert client.post("/api/predict", json={"inputs": {"material": "AC", "time_s": 5000}}).status_code == 422  # far out of range
    assert client.post("/api/predict", json={"inputs": {"material": "AC", "time_s": 100, "normal_load_n": 0}}).status_code == 422  # zero load
    assert client.post("/api/predict", json={"inputs": {"material": "AC", "time_s": 100, "normal_load_n": 120}}).status_code == 422  # fixed condition violated
    r = client.post("/api/predict", json={"inputs": {"material": "AC", "time_s": 430}, "save": False})
    assert r.status_code == 200 and r.json()["cof"]["extrapolation"] == "mild" and r.json()["warnings"]
    assert client.post("/api/predict", json={"inputs": {}}).status_code == 422


def test_batch_predictions(client, trained):
    rows = [{"material": m, "time_s": t} for m in ("AC", "AE", "C", "E") for t in (50, 400)] + [{"material": "Q", "time_s": 10}]
    r = client.post("/api/predict/batch", json={"rows": rows})
    assert r.status_code == 200 and r.json()["n"] == 9 and r.json()["n_ok"] == 8
    assert r.json()["results"][-1]["ok"] is False
    df = pd.DataFrame(rows[:3])
    r = client.post("/api/predict/batch/upload", files={"file": ("batch.csv", df.to_csv(index=False).encode(), "text/csv")})
    assert r.status_code == 200 and r.json()["n_ok"] == 3


def test_simulation(client, trained):
    r = client.post("/api/simulate", json={"base_inputs": {"material": "E", "time_s": 100}, "sweep": {"feature": "time_s", "min": 0, "max": 420, "steps": 12, "by": "material"}})
    assert r.status_code == 200, r.text
    s = r.json()
    assert len(s["comparisons"]["material"]) == 4 and len(s["sweep"]["series"]) == 4 and len(s["sweep"]["series"][0]["points"]) == 12
    assert s["sensitivity"][0]["feature"] == "time_s"
    assert "not measurements" in s["note"]
    r = client.post("/api/simulate/compare", json={"scenarios": [{"material": "AC", "time_s": 100}, {"material": "C", "time_s": 100}, {"material": "AC", "time_s": 9999}]})
    assert r.status_code == 200 and r.json()["scenarios"][2]["ok"] is False and r.json()["scenarios"][0]["delta_vs_first"] == 0
    assert client.post("/api/simulate", json={"base_inputs": {"material": "E", "time_s": 100}, "sweep": {"feature": "material"}}).status_code == 422


def test_promote_archive_and_resolution(client, trained):
    ids = trained["summary"]["model_version_ids"]
    best = trained["summary"]["best_model_version_id"]
    other = next(i for i in ids if i != best)
    r = client.post(f"/api/models/{other}/promote")
    assert r.status_code == 200 and r.json()["status"] == "production"
    assert client.get("/api/models/production").json()["cof"]["active"]["id"] == other
    r = client.post("/api/predict", json={"inputs": {"material": "AC", "time_s": 100}, "save": False})
    assert r.json()["cof"]["model"]["id"] == other
    r = client.post(f"/api/models/{best}/promote")
    assert r.status_code == 200
    assert client.get(f"/api/models/{other}").json()["status"] == "archived"  # previous production archived, not overwritten
    assert client.get("/api/models/production").json()["cof"]["active"]["id"] == best
    r = client.post(f"/api/models/{best}/archive")
    assert r.status_code == 200
    assert client.get("/api/models/production").json()["cof"]["active"] is None or client.get("/api/models/production").json()["cof"]["active"]["id"] != best
    client.post(f"/api/models/{best}/promote")  # restore for later tests


def test_validation_loop_and_augmentation(client, trained, bundled):
    r = client.post("/api/validation", json={"experiment_id": "EXP-001", "inputs": {"material": "AC", "time_s": 300}, "measured_cof": 0.5})
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["predicted_cof"] is not None and v["cof_abs_error"] == pytest.approx(abs(0.5 - v["predicted_cof"]))
    assert v["status"] == "pending"
    assert client.post("/api/validation", json={"experiment_id": "E2", "inputs": {"material": "AC", "time_s": 300}}).status_code == 400
    assert client.post("/api/validation", json={"experiment_id": "E3", "inputs": {"material": "AC", "time_s": 300}, "measured_cof": 5}).status_code == 422
    r = client.get("/api/validation/summary")
    assert r.status_code == 200 and r.json()["cof"]["n"] >= 1
    r = client.post("/api/validation/augment", json={"dataset_version_id": bundled["processed_version"]["id"], "validation_ids": [v["id"]]})
    assert r.status_code == 409  # not approved yet
    r = client.patch(f"/api/validation/{v['id']}", json={"status": "approved", "notes": "ok"})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    r = client.post("/api/validation/augment", json={"dataset_version_id": bundled["processed_version"]["id"], "validation_ids": [v["id"]]})
    assert r.status_code == 201, r.text
    aug = r.json()
    assert aug["kind"] == "augmented" and aug["record_count"] == 5545
    assert client.get(f"/api/validation/{v['id']}").status_code in (404, 405)
    assert client.patch(f"/api/validation/{v['id']}", json={"status": "rejected"}).status_code == 409
    r = client.post("/api/models/train", json={"dataset_version_id": aug["id"], "target": "cof", "models": ["linear"], "search_mode": "quick"})
    assert r.status_code == 202
    run = client.get(f"/api/models/train/{r.json()['id']}").json()
    assert run["status"] == "completed"


def test_reports_pdf_and_xlsx(client, trained):
    for fmt in ("pdf", "xlsx"):
        r = client.post("/api/reports/generate", json={"format": fmt, "title": f"test {fmt}"})
        assert r.status_code == 201, r.text
        rid = r.json()["id"]
        d = client.get(f"/api/reports/{rid}/download")
        assert d.status_code == 200 and len(d.content) > 1000
        if fmt == "pdf":
            assert d.content[:4] == b"%PDF"
    assert client.get("/api/reports").status_code == 200 and len(client.get("/api/reports").json()) >= 2
    assert client.get("/api/reports/99999/download").status_code == 404


def test_dashboard_and_settings(client, trained, bundled):
    vid = bundled["processed_version"]["id"]
    r = client.get("/api/dashboard/overview", params={"dataset_version_id": vid})
    assert r.status_code == 200
    d = r.json()
    assert d["data"]["n_runs"] == 12 and d["models"]["cof"]["active"] is not None
    assert d["models"]["wear_rate"]["active"] is None and "wear" in d["models"]["wear_rate"]["reason"].lower()
    assert client.get("/api/dashboard/overview").status_code == 200  # default (latest) version also works
    r = client.get("/api/dashboard/overview", params={"dataset_version_id": vid, "material": "AC"})
    assert r.json()["filtered"]["materials"] == ["AC"]
    r = client.get("/api/settings")
    assert r.status_code == 200 and r.json()["settings"]["allow_far_extrapolation"] is False
    r = client.put("/api/settings", json={"allow_far_extrapolation": True})
    assert r.status_code == 200 and r.json()["settings"]["allow_far_extrapolation"] is True
    r = client.post("/api/predict", json={"inputs": {"material": "AC", "time_s": 5000}, "save": False})
    assert r.status_code == 200 and r.json()["cof"]["extrapolation"] == "far"
    client.put("/api/settings", json={"allow_far_extrapolation": False})
    assert client.put("/api/settings", json={"default_models": ["nope"]}).status_code == 400
    assert client.get("/api/settings/data-dictionary").status_code == 200


def test_small_dataset_limits(client):
    df = pd.DataFrame({"load": [10, 20, 30, 40, 50], "cof": [0.1, 0.2, 0.3, 0.4, 0.5]})
    r = client.post("/api/datasets/upload", data={"name": "tiny"}, files={"files": ("tiny.csv", df.to_csv(index=False).encode(), "text/csv")})
    ds_id, f = r.json()["dataset"]["id"], r.json()["files"][0]
    v = client.post(f"/api/datasets/{ds_id}/ingest", json={"sheets": [{"file_id": f["id"], "sheet_name": "data", "mapping": {"load": "normal_load_n", "cof": "cof"}}]}).json()
    r = client.post("/api/models/train", json={"dataset_version_id": v["id"], "target": "cof", "models": ["linear"], "search_mode": "quick"})
    assert r.status_code == 202
    run = client.get(f"/api/models/train/{r.json()['id']}").json()
    assert run["status"] == "failed" and "at least 20" in run["error"]


def test_missing_model_for_target_is_explicit(client, bundled):
    r = client.post("/api/simulate", json={"target": "wear_rate", "base_inputs": {"material": "AC", "time_s": 1}})
    assert r.status_code == 409
