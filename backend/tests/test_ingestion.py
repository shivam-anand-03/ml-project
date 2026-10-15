"""Excel / CSV / DWF ingestion, inspection and column mapping."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.services.ingestion import IngestionError, inspect_file, read_table
from app.services.mapping import MappingError, apply_mapping, parse_run_name, suggest_feature, suggest_mapping
from tests.conftest import DATA_DIR

PROCESSED = DATA_DIR / "Pin-on-disk_data" / "Processed_files"
RAW = DATA_DIR / "Pin-on-disk_data" / "Raw_files"


def test_read_processed_workbook_has_time_and_friction():
    df = read_table(PROCESSED / "AC_3_2.xlsx")
    assert list(df.columns) == ["TIME", "FF"]
    assert len(df) == 462
    assert df["TIME"].is_monotonic_increasing


def test_inspect_workbook_reports_sheet_columns_and_run_metadata():
    insp = inspect_file(PROCESSED / "AC_3_2.xlsx")
    sheet = insp["sheets"][0]
    assert sheet["sheet_name"] == "AC_3_2_50N_80rpm_120mm"
    assert sheet["n_rows"] == 462 and sheet["n_cols"] == 2
    assert sheet["parsed_run"]["material"] == "AC"
    assert sheet["parsed_run"]["normal_load_n"] == 50.0
    assert sheet["parsed_run"]["rpm"] == 80.0
    assert sheet["parsed_run"]["track_diameter_mm"] == 120.0
    assert abs(sheet["parsed_run"]["sliding_speed_m_s"] - 0.5027) < 1e-3
    assert sheet["suggested_mapping"]["TIME"]["feature"] == "time_s"
    assert sheet["suggested_mapping"]["FF"]["feature"] == "friction_force_n"
    assert len(sheet["preview"]) == 100


def test_read_raw_dwf_matches_processed_except_sign():
    raw = read_table(RAW / "C_5_1.dwf")
    proc = read_table(PROCESSED / "C_5_1.xlsx")
    assert len(raw) == len(proc) == 462
    assert (raw["FF"] < 0).sum() == 10
    assert (raw["FF"].abs() - proc["FF"]).abs().max() < 1e-9


def test_csv_with_semicolon_and_decimal_comma(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("Load (N);Speed (m/s);COF\n50;0,5;0,42\n60;0,5;0,45\n")
    df = read_table(p)
    assert df.shape == (2, 3)
    assert df["COF"].tolist() == pytest.approx([0.42, 0.45])
    m = suggest_mapping(list(df.columns))
    assert m["Load (N)"]["feature"] == "normal_load_n"
    assert m["Speed (m/s)"]["feature"] == "sliding_speed_m_s"
    assert m["COF"]["feature"] == "cof"


def test_empty_excel_file_is_reported_not_crashed(tmp_path: Path):
    p = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(p, index=False)
    insp = inspect_file(p)
    assert insp["sheets"][0]["is_empty"] is True
    assert insp["sheets"][0]["n_rows"] == 0


def test_unsupported_extension_rejected(tmp_path: Path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    with pytest.raises(IngestionError):
        inspect_file(p)


def test_synonym_mapping_variants():
    assert suggest_feature("Average Coefficient Of Friction AC")[0] == "cof"
    assert suggest_feature("Average Time E")[0] == "time_s"
    assert suggest_feature("Normal_Load")[0] == "normal_load_n"
    assert suggest_feature("Temperature (°C)")[0] == "temperature_c"
    assert suggest_feature("Specific wear rate")[0] == "wear_rate"
    assert suggest_feature("Lubricant type")[0] == "lubricant"
    assert suggest_feature("totally unknown column")[0] is None


def test_parse_run_name_variants():
    assert parse_run_name("AE_6_6_50N_0.502_80rpm_120m")["sliding_speed_m_s"] == 0.502
    assert parse_run_name("E_6")["test_run_id"] == "E_6"
    assert parse_run_name("Sheet1") == {}
    assert parse_run_name("") == {}
    assert parse_run_name("C_7_1_50N_137rpm_70mm")["composition"] == "AlSi10Mg"


def test_apply_mapping_with_constants_and_units():
    df = pd.DataFrame({"t": [0, 1, 2], "F": [1.0, 2.0, 3.0], "note": ["a", "b", "c"]})
    out = apply_mapping(df, {"t": "time_s", "F": "friction_force_n", "note": None}, {"normal_load_n": 5, "material": "X"}, {"normal_load_n": "kgf"})
    assert out["normal_load_n"].iloc[0] == pytest.approx(5 * 9.80665)
    assert out["material"].tolist() == ["X"] * 3
    assert out["extra"].iloc[1] == {"note": "b"}
    assert out["cof"].isna().all()


def test_apply_mapping_rejects_duplicate_and_unknown_features():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(MappingError):
        apply_mapping(df, {"a": "cof", "b": "cof"})
    with pytest.raises(MappingError):
        apply_mapping(df, {"a": "not_a_feature"})
    with pytest.raises(MappingError):
        apply_mapping(df, {"zzz": "cof"})
