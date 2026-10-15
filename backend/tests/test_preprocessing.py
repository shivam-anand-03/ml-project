"""Data validation and documented preprocessing operations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.features import RECORD_COLUMNS
from app.services.preprocessing import PreprocessOptions, feature_summary, is_varying, preprocess
from app.services.quality import quality_report


def frame(**cols) -> pd.DataFrame:
    n = max(len(v) for v in cols.values())
    df = pd.DataFrame({c: [np.nan] * n for c in RECORD_COLUMNS})
    for c, v in cols.items():
        df[c] = v
    return df


def _log(log, step):
    return next(e for e in log if e["step"] == step)


def test_cof_derived_from_friction_and_load_and_zero_load_flagged():
    df = frame(test_run_id=["r1"] * 4, material=["AC"] * 4, time_s=[1, 2, 3, 4], friction_force_n=[10, 11, 12, 13], normal_load_n=[50, 50, 0, 50])
    out, log = preprocess(df)
    assert out["cof"].iloc[0] == pytest.approx(0.2)
    assert np.isnan(out["cof"].iloc[2])
    assert not out["is_usable"].iloc[2]
    assert "zero_normal_load" in out["flags"].iloc[2]
    assert _log(log, "zero_load")["rows_affected"] == 1


def test_zero_sliding_distance_and_speed_derivation():
    df = frame(test_run_id=["r1"] * 3, time_s=[0, 10, 20], rpm=[100, 100, 100], track_diameter_mm=[100, 100, 100], cof=[0.1, 0.2, 0.3])
    out, _ = preprocess(df)
    assert out["sliding_speed_m_s"].iloc[0] == pytest.approx(np.pi * 0.1 * 100 / 60)
    assert out["sliding_distance_m"].iloc[0] == 0.0
    assert out["sliding_distance_m"].iloc[2] == pytest.approx(out["sliding_speed_m_s"].iloc[2] * 20)


def test_duplicates_removed_and_logged():
    df = frame(test_run_id=["r1"] * 3, time_s=[1, 1, 2], cof=[0.3, 0.3, 0.31])
    out, log = preprocess(df)
    assert len(out) == 2
    assert _log(log, "drop_duplicates")["rows_affected"] == 1


def test_negative_friction_strategies():
    df = frame(test_run_id=["r1"] * 3, time_s=[1, 2, 3], friction_force_n=[-0.05, 20, 21], normal_load_n=[50] * 3)
    out_abs, _ = preprocess(df, PreprocessOptions(negative_friction="abs"))
    assert out_abs["friction_force_n"].iloc[0] == pytest.approx(0.05)
    out_drop, _ = preprocess(df, PreprocessOptions(negative_friction="drop"))
    assert not out_drop["is_usable"].iloc[0]
    out_keep, _ = preprocess(df, PreprocessOptions(negative_friction="keep"))
    assert out_keep["friction_force_n"].iloc[0] == pytest.approx(-0.05)
    with pytest.raises(ValueError):
        PreprocessOptions.from_dict({"negative_friction": "nonsense"})


def test_missing_target_rows_unusable():
    df = frame(test_run_id=["r1"] * 2, time_s=[1, 2], cof=[0.3, np.nan])
    out, _ = preprocess(df)
    assert out["is_usable"].tolist() == [True, False]


def test_wear_rate_never_derived_from_mass():
    df = frame(test_run_id=["r1"] * 2, time_s=[1, 2], cof=[0.3, 0.3], wear_mass_mg=[1.0, 2.0], normal_load_n=[50, 50], sliding_distance_m=[10, 20])
    out, log = preprocess(df)
    assert out["wear_rate"].isna().all()
    assert any(e["step"] == "wear_rate_not_derived" for e in log)


def test_material_attributes_and_run_in_flags():
    df = frame(test_run_id=["AC_1"] * 6, material=["AC"] * 6, time_s=[1, 2, 3, 4, 5, 6], friction_force_n=[0.01, 0.02, 20, 21, 22, 23], normal_load_n=[50] * 6)
    out, _ = preprocess(df)
    assert out["composition"].iloc[0] == "AlSi10Mg-Al2O3"
    assert out["pore_size_class"].iloc[0] == "800-1000 um"
    assert "run_in" in out["flags"].iloc[0] and "run_in" not in out["flags"].iloc[3]
    out2, _ = preprocess(df, PreprocessOptions(exclude_run_in=True))
    assert out2["is_usable"].tolist() == [False, False, True, True, True, True]


def test_is_varying_relative_tolerance():
    assert not is_varying(pd.Series([0.4995, 0.5027, 0.5011]))
    assert is_varying(pd.Series([0.4, 0.6]))
    assert not is_varying(pd.Series([50.0, 50.0]))


def test_feature_summary_and_quality_report():
    df = frame(test_run_id=["r1"] * 3 + ["r2"] * 3, material=["A"] * 3 + ["B"] * 3, time_s=[1, 2, 3] * 2, friction_force_n=[10, 11, 12] * 2, normal_load_n=[50] * 6)
    out, _ = preprocess(df)
    fs = feature_summary(out)
    assert fs["targets"]["cof"]["available"] and not fs["targets"]["wear_rate"]["available"]
    assert fs["inputs"]["material"]["levels"] == ["A", "B"]
    assert fs["inputs"]["normal_load_n"]["varies"] is False
    q = quality_report(out)
    assert q["n_rows"] == 6 and q["usable_rows"] == 6
    assert q["summary"]["n_runs"] == 2
    assert 0 <= q["score"] <= 100
    assert quality_report(out.iloc[0:0])["n_rows"] == 0
