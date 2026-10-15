"""Column-name normalisation, mapping suggestions and run-name parsing."""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from app.core.features import MATERIAL_CATALOGUE, RECORD_COLUMNS, STANDARD_FEATURES

UNIT_CONVERSIONS: dict[str, dict[str, float]] = {
    # standard feature -> {declared source unit: multiplier to the standard unit}
    "normal_load_n": {"N": 1.0, "kN": 1000.0, "kgf": 9.80665, "gf": 0.00980665, "lbf": 4.448222},
    "sliding_speed_m_s": {"m/s": 1.0, "mm/s": 0.001, "cm/s": 0.01, "m/min": 1 / 60.0, "mm/min": 1 / 60000.0},
    "temperature_c": {"°C": 1.0, "C": 1.0},
    "time_s": {"s": 1.0, "ms": 0.001, "min": 60.0, "h": 3600.0},
    "sliding_distance_m": {"m": 1.0, "mm": 0.001, "km": 1000.0, "cm": 0.01},
    "track_diameter_mm": {"mm": 1.0, "m": 1000.0, "cm": 10.0},
    "friction_force_n": {"N": 1.0, "kN": 1000.0, "kgf": 9.80665},
    "wear_volume_mm3": {"mm3": 1.0, "mm³": 1.0, "cm3": 1000.0, "m3": 1e9},
    "wear_mass_mg": {"mg": 1.0, "g": 1000.0, "kg": 1e6},
}


def normalise(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)  # drop bracketed units
    s = s.replace("µ", "u").replace("°", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_SYNONYM_INDEX: list[tuple[str, str]] = sorted(
    ((normalise(syn), f.name) for f in STANDARD_FEATURES.values() for syn in f.synonyms),
    key=lambda x: -len(x[0]),
)


def suggest_feature(column: str) -> tuple[str | None, float]:
    """Return (standard feature, confidence) for one source column name."""
    n = normalise(column)
    if not n:
        return None, 0.0
    if n in STANDARD_FEATURES:
        return n, 1.0
    for syn, feat in _SYNONYM_INDEX:
        if n == syn:
            return feat, 0.95
    # 'average coefficient of friction ac' -> cof ; 'average time e' -> time_s
    for syn, feat in _SYNONYM_INDEX:
        if len(syn) >= 3 and (n.startswith(syn + " ") or n.endswith(" " + syn)):
            return feat, 0.8
    for syn, feat in _SYNONYM_INDEX:
        if len(syn) >= 4 and syn in n:
            return feat, 0.6
    return None, 0.0


def suggest_mapping(columns: list[str]) -> dict[str, dict[str, Any]]:
    used: set[str] = set()
    out: dict[str, dict[str, Any]] = {}
    for col in columns:
        feat, conf = suggest_feature(col)
        if feat in used:
            feat, conf = None, 0.0
        if feat:
            used.add(feat)
        out[str(col)] = {"feature": feat, "confidence": conf}
    return out


_RUN_TOKEN_LOAD = re.compile(r"^(\d+(?:\.\d+)?)\s*n$", re.I)
_RUN_TOKEN_RPM = re.compile(r"^(\d+(?:\.\d+)?)\s*rpm$", re.I)
_RUN_TOKEN_DIAM = re.compile(r"^(\d+(?:\.\d+)?)\s*mm?$", re.I)
_RUN_TOKEN_SPEED = re.compile(r"^(\d+\.\d+)(?:\s*m\s*/?\s*s)?$", re.I)
_RUN_TOKEN_INT = re.compile(r"^\d+$")
_RUN_TOKEN_MAT = re.compile(r"^[A-Za-z][A-Za-z0-9\-]*$")


def parse_run_name(name: str | None) -> dict[str, Any]:
    """Parse names such as 'AC_3_2_50N_80rpm_120mm' or 'AE_6_6_50N_0.502_80rpm_120m'.

    Returns the metadata that can be read from the name; nothing is guessed.
    """
    if not name:
        return {}
    tokens = [t for t in re.split(r"[_\s]+", str(name).strip()) if t]
    # a run name needs a short alphabetic material code followed by at least one numeric/parameter token
    if len(tokens) < 2 or not _RUN_TOKEN_MAT.match(tokens[0]) or len(tokens[0]) > 6 or re.search(r"\d", tokens[0]):
        return {}
    if not any(re.match(r"^\d", t) for t in tokens[1:]):
        return {}
    meta: dict[str, Any] = {"material": tokens[0].upper()}
    specimen: list[str] = []
    for tok in tokens[1:]:
        if m := _RUN_TOKEN_LOAD.match(tok):
            meta["normal_load_n"] = float(m.group(1))
        elif m := _RUN_TOKEN_RPM.match(tok):
            meta["rpm"] = float(m.group(1))
        elif m := _RUN_TOKEN_DIAM.match(tok):
            meta["track_diameter_mm"] = float(m.group(1))
        elif m := _RUN_TOKEN_SPEED.match(tok):
            meta["sliding_speed_m_s"] = float(m.group(1))
        elif _RUN_TOKEN_INT.match(tok):
            specimen.append(tok)
        else:
            break
    meta["test_run_id"] = "_".join([meta["material"], *specimen]) if specimen else meta["material"]
    if "rpm" in meta and "track_diameter_mm" in meta and "sliding_speed_m_s" not in meta:
        meta["sliding_speed_m_s"] = round(math.pi * meta["track_diameter_mm"] / 1000.0 * meta["rpm"] / 60.0, 4)
        meta["sliding_speed_derived"] = True
    cat = MATERIAL_CATALOGUE.get(meta["material"])
    if cat:
        meta["composition"] = cat["composition"]
        meta["pore_size_class"] = cat["pore_size_class"]
    return meta


class MappingError(ValueError):
    pass


def apply_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    constants: dict[str, Any] | None = None,
    units: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a standardized record frame from a source frame.

    mapping:   {source column -> standard feature}
    constants: {standard feature -> constant value} applied to every row of this sheet (e.g. load = 50 N)
    units:     {standard feature -> declared source unit} converted to the standard unit
    """
    constants = constants or {}
    units = units or {}
    bad = [f for f in list(mapping.values()) + list(constants.keys()) if f and f not in STANDARD_FEATURES]
    if bad:
        raise MappingError(f"Unknown standard feature(s): {sorted(set(bad))}")
    missing_cols = [c for c in mapping if c not in df.columns]
    if missing_cols:
        raise MappingError(f"Mapped source column(s) not found in sheet: {missing_cols}")
    targets_seen = [f for f in mapping.values() if f]
    dupes = {f for f in targets_seen if targets_seen.count(f) > 1}
    if dupes:
        raise MappingError(f"Several source columns map to the same feature: {sorted(dupes)}")

    out = pd.DataFrame(index=df.index)
    extra_cols = [c for c in df.columns if c not in mapping or not mapping[c]]
    for src, feat in mapping.items():
        if not feat:
            continue
        spec = STANDARD_FEATURES[feat]
        col = df[src]
        if spec.dtype == "number":
            if col.dtype == object:
                col = col.astype(str).str.replace(",", ".", regex=False)
            col = pd.to_numeric(col, errors="coerce")
            factor = UNIT_CONVERSIONS.get(feat, {}).get(units.get(feat, ""), None)
            if units.get(feat) and factor is None:
                raise MappingError(f"Unknown unit '{units[feat]}' for {feat}")
            if factor is not None:
                col = col * factor
        else:
            col = col.astype("string").str.strip()
            col = col.where(col.notna() & (col != ""), None)
        out[feat] = col
    for feat, value in constants.items():
        if value is None or value == "":
            continue
        spec = STANDARD_FEATURES[feat]
        if spec.dtype == "number":
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise MappingError(f"Constant for {feat} must be numeric, got {value!r}") from exc
            factor = UNIT_CONVERSIONS.get(feat, {}).get(units.get(feat, ""), None)
            if factor is not None:
                value = value * factor
        else:
            value = str(value).strip()
        out[feat] = value
    for feat in RECORD_COLUMNS:
        if feat not in out.columns:
            out[feat] = np.nan if STANDARD_FEATURES[feat].dtype == "number" else None
    out["extra"] = [
        {str(c): (None if (isinstance(v, float) and math.isnan(v)) else v) for c, v in zip(extra_cols, row)}
        for row in df[extra_cols].itertuples(index=False, name=None)
    ] if extra_cols else [{} for _ in range(len(df))]
    out["row_index"] = np.arange(len(df))
    return out[RECORD_COLUMNS + ["extra", "row_index"]]
