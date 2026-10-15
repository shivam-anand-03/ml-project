"""Dataset API: upload, mapping, validation, preprocessing, versions, export."""
from __future__ import annotations

import io

import pandas as pd


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["database"] is True


def test_upload_inspect_ingest_preprocess_export(client):
    df = pd.DataFrame({"Load (N)": [20, 20, 50, 50, 80, 80], "Speed (m/s)": [0.5, 1.0, 0.5, 1.0, 0.5, 1.0], "Lubricant": ["A", "A", "B", "B", "A", "B"],
                       "COF": [0.08, 0.07, 0.09, 0.085, 0.1, 0.095], "Run": ["r1", "r2", "r3", "r4", "r5", "r6"]})
    r = client.post("/api/datasets/upload", data={"name": "csv test", "description": "d"}, files={"files": ("exp.csv", _csv_bytes(df), "text/csv")})
    assert r.status_code == 201, r.text
    payload = r.json()
    ds_id = payload["dataset"]["id"]
    f = payload["files"][0]
    sheet = f["inspection"]["sheets"][0]
    assert sheet["suggested_mapping"]["COF"]["feature"] == "cof"
    assert sheet["suggested_mapping"]["Run"]["feature"] == "test_run_id"
    r = client.get(f"/api/datasets/{ds_id}/files/{f['id']}/inspection")
    assert r.status_code == 200 and len(r.json()["inspection"]["sheets"][0]["preview"]) == 6

    mapping = {c: v["feature"] for c, v in sheet["suggested_mapping"].items()}
    r = client.post(f"/api/datasets/{ds_id}/ingest", json={"sheets": [{"file_id": f["id"], "sheet_name": "data", "mapping": mapping, "constants": {"material_pair": "Steel-Al"}}]})
    assert r.status_code == 201, r.text
    raw = r.json()
    assert raw["kind"] == "raw" and raw["record_count"] == 6
    r = client.get(f"/api/datasets/{ds_id}/preview", params={"version_id": raw["id"], "limit": 3})
    assert r.status_code == 200 and len(r.json()["rows"]) == 3 and r.json()["total"] == 6
    assert r.json()["rows"][0]["material_pair"] == "Steel-Al"
    r = client.post(f"/api/datasets/{ds_id}/preprocess", json={"version_id": raw["id"], "options": {"outlier_zscore": 0}})
    assert r.status_code == 201, r.text
    proc = r.json()
    assert proc["kind"] == "processed" and proc["parent_version_id"] == raw["id"]
    r = client.get(f"/api/datasets/{ds_id}/quality", params={"version_id": proc["id"]})
    assert r.status_code == 200 and r.json()["quality_report"]["n_rows"] == 6
    r = client.get(f"/api/datasets/{ds_id}/quality/report.md")
    assert r.status_code == 200 and "Data-quality report" in r.text
    for fmt in ("csv", "xlsx"):
        r = client.get(f"/api/datasets/{ds_id}/export", params={"version_id": proc["id"], "format": fmt})
        assert r.status_code == 200 and len(r.content) > 50
    r = client.get(f"/api/datasets/{ds_id}")
    assert r.status_code == 200 and len(r.json()["versions"]) == 2
    r = client.get(f"/api/datasets/versions/{proc['id']}")
    assert r.status_code == 200 and r.json()["cleaning_log"]
    r = client.delete(f"/api/datasets/{ds_id}")
    assert r.status_code == 204
    assert client.get(f"/api/datasets/{ds_id}").status_code == 404


def test_ingest_without_any_target_is_rejected(client):
    df = pd.DataFrame({"Load (N)": [10, 20], "Speed": [1, 2]})
    r = client.post("/api/datasets/upload", data={"name": "no target"}, files={"files": ("x.csv", _csv_bytes(df), "text/csv")})
    assert r.status_code == 201
    ds_id, f = r.json()["dataset"]["id"], r.json()["files"][0]
    r = client.post(f"/api/datasets/{ds_id}/ingest", json={"sheets": [{"file_id": f["id"], "sheet_name": "data", "mapping": {"Load (N)": "normal_load_n", "Speed": "sliding_speed_m_s"}}]})
    assert r.status_code == 400
    assert "No target can be formed" in r.json()["detail"]


def test_empty_file_upload_rejected(client):
    r = client.post("/api/datasets/upload", data={"name": "empty"}, files={"files": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_unsupported_file_type_rejected(client):
    r = client.post("/api/datasets/upload", data={"name": "pdf"}, files={"files": ("doc.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 400


def test_bundled_import_matches_audit(client, bundled):
    raw, proc = bundled["raw_version"], bundled["processed_version"]
    assert raw["record_count"] == 5544 and proc["usable_record_count"] == 5544
    fs = proc["feature_summary"]
    assert fs["n_runs"] == 12 and fs["n_materials"] == 4
    assert fs["targets"]["cof"]["available"] is True
    assert fs["targets"]["wear_rate"]["available"] is False
    assert fs["inputs"]["normal_load_n"]["varies"] is False
    assert fs["inputs"]["time_s"]["varies"] is True
    r = client.get(f"/api/datasets/{bundled['dataset']['id']}/preview", params={"version_id": proc["id"], "material": "AC", "limit": 5})
    assert r.status_code == 200 and all(row["material"] == "AC" for row in r.json()["rows"])
    assert r.json()["rows"][0]["cof"] is not None


def test_feature_dictionary_and_options(client):
    r = client.get("/api/datasets/feature-dictionary")
    assert r.status_code == 200 and any(f["name"] == "cof" for f in r.json())
    r = client.get("/api/datasets/preprocess-options")
    assert r.status_code == 200 and "negative_friction" in r.json()


def test_analysis_endpoints_on_real_data(client, bundled):
    vid = bundled["processed_version"]["id"]
    r = client.get(f"/api/analysis/{vid}/target-vs-time")
    assert r.status_code == 200 and len(r.json()["groups"]) == 4 and len(r.json()["runs"]) == 12
    r = client.get(f"/api/analysis/{vid}/target-vs-feature", params={"feature": "normal_load_n"})
    assert r.status_code == 200 and r.json()["varied"] is False and r.json()["fixed_value"] == 50.0
    r = client.get(f"/api/analysis/{vid}/target-vs-feature", params={"feature": "temperature_c"})
    assert r.status_code == 200 and r.json()["available"] is False
    r = client.get(f"/api/analysis/{vid}/target-vs-feature", params={"feature": "time_s"})
    assert r.status_code == 200 and r.json()["varied"] is True and r.json()["points"]
    r = client.get(f"/api/analysis/{vid}/material-comparison")
    assert r.status_code == 200 and r.json()["test"]["name"].startswith("Kruskal")
    assert {g["group"] for g in r.json()["groups"]} == {"AC", "AE", "C", "E"}
    r = client.get(f"/api/analysis/{vid}/correlation")
    assert r.status_code == 200 and r.json()["available"] is True
    r = client.get(f"/api/analysis/{vid}/distributions")
    assert r.status_code == 200 and r.json()["by_material"]
    r = client.get(f"/api/analysis/{vid}/trends")
    assert r.status_code == 200
    statuses = {i["feature"]: i["status"] for i in r.json()["numeric"]}
    assert statuses["normal_load_n"] == "constant" and statuses["temperature_c"] == "not_recorded" and statuses["time_s"] == "varied"
    r = client.get(f"/api/analysis/{vid}/overview")
    assert r.status_code == 200 and r.json()["materials"] == ["AC", "AE", "C", "E"]
    assert client.get(f"/api/analysis/{vid}/target-vs-time", params={"target": "wear_rate"}).status_code == 404
