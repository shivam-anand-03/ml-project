# Data Dictionary

All uploaded columns are mapped onto the standard features below (`backend/app/core/features.py`). The same registry drives ingestion suggestions, validation ranges, training-domain checks and the data dictionary shown on the Settings page (`GET /api/settings/data-dictionary`).

| Feature | Label | Unit | Type | Role | Validity range | Definition / source |
|---|---|---|---|---|---|---|
| `test_run_id` | Test run ID | – | category | identifier | – | One tribometer run / specimen. Used for grouped train–test splitting. Parsed from worksheet names (e.g. `AC_3_2`). |
| `material` | Material | – | category | input | – | Material code of the specimen (AC, AE, C, E in the bundled dataset). |
| `composition` | Composition | – | category | input | – | Derived from the material catalogue (AlSi10Mg vs AlSi10Mg-Al2O3). Empty for unknown codes. |
| `pore_size_class` | Pore size class | µm | category | input | – | Derived from the material catalogue (800-1000 um, 1000-1200 um). |
| `material_pair` | Material pair | – | category | input | – | Pin / disc pair, when recorded. |
| `lubricant` | Lubricant | – | category | input | – | Lubricant type; `dry` when none. |
| `normal_load_n` | Normal load | N | number | input | 0 – 5000 | Applied normal load. Accepted source units: N, kN, kgf, gf, lbf. |
| `sliding_speed_m_s` | Sliding speed | m/s | number | input | 0 – 100 | Linear sliding speed. Derived as π·D·n/60 from rpm and track diameter when not given. Units: m/s, mm/s, cm/s, m/min, mm/min. |
| `temperature_c` | Temperature | °C | number | input | −100 – 1500 | Test temperature. |
| `time_s` | Sliding time | s | number | input | 0 – 1e7 | Elapsed time in the run. Units: s, ms, min, h. |
| `sliding_distance_m` | Sliding distance | m | number | input | 0 – 1e7 | Derived as speed × time when not measured. Units: m, mm, cm, km. |
| `rpm` | Rotational speed | rpm | number | metadata | 0 – 100000 | Disc rotation speed (from worksheet name). |
| `track_diameter_mm` | Wear track diameter | mm | number | metadata | 0 – 10000 | Wear track diameter (from worksheet name). |
| `friction_force_n` | Friction force | N | number | measurement | −1e4 – 1e4 | Measured tangential force (`FF` in the bundled files). |
| `cof` | Coefficient of friction | – | number | **target** | 0 – 2 | COF = friction force / normal load. |
| `wear_volume_mm3` | Wear volume | mm³ | number | measurement | ≥ 0 | Volume loss. |
| `wear_mass_mg` | Wear mass | mg | number | measurement | ≥ 0 | Mass loss. Never converted to volume without a documented density. |
| `wear_rate` | Wear rate | declared by dataset | number | **target** | ≥ 0 | Specific wear rate k = V / (F·s) or the definition declared with the dataset. Never derived automatically. |

Additional per-record fields: `extra` (unmapped source columns as JSON, including `_source_file` and `_source_sheet`), `flags` (list, see below), `is_usable`, `source` (`measured` | `validation`), `row_index`.

## Flags written by preprocessing

| Flag | Meaning |
|---|---|
| `derived:cof` | COF computed from friction force and load |
| `derived:sliding_distance` | distance computed from speed × time |
| `derived:sliding_speed` | speed computed from rpm and track diameter |
| `negative_friction_abs` | negative friction reading replaced by its absolute value |
| `negative_friction` | negative friction reading kept or marked unusable (option) |
| `zero_normal_load` | load ≤ 0 → COF cannot be formed → unusable |
| `out_of_valid_range:<feature>` | value outside the physical validity range → unusable |
| `non_finite:<feature>` | inf/NaN replaced by missing |
| `missing_target` | neither COF nor wear rate → unusable |
| `run_in` | friction force < 5 % of the run's steady-state median (kept by default) |
| `outlier:<target>` | robust z-score > 4 within the run (kept by default) |
| `validation` | record added from an approved experimental validation |

## Equations

| Quantity | Formula | Units | Note |
|---|---|---|---|
| Coefficient of friction | COF = F_friction / F_normal | – | per sample |
| Specific wear rate | k = V / (F × s) | mm³/(N·m) with V in mm³, F in N, s in m; in SI m³/(N·m) = m²/N | requires wear volume, load and distance; the definition and unit must be confirmed against the laboratory method before use |
| Sliding distance | s = v × t | m | |
| Sliding speed from rotation | v = π × D × n / 60 | m/s | D in m, n in rpm |

## Material catalogue (bundled dataset)

| Code | Composition | Pore size class | Source |
|---|---|---|---|
| AC | AlSi10Mg-Al2O3 | 800-1000 um | Mendeley dataset description |
| AE | AlSi10Mg-Al2O3 | 1000-1200 um | Mendeley dataset description |
| C | AlSi10Mg | 800-1000 um | Mendeley dataset description |
| E | AlSi10Mg | 1000-1200 um | Mendeley dataset description |

Codes not in the catalogue keep empty composition / pore size; nothing is guessed.

## Worksheet-name parsing

`<MATERIAL>_<specimen ints>_<load>N_<speed>_<rpm>rpm_<diameter>mm` (tokens optional, any order after the material code), e.g. `AC_3_2_50N_80rpm_120mm` → material AC, run AC_3_2, load 50 N, 80 rpm, track diameter 120 mm, speed 0.5027 m/s (derived). `AE_6_6_50N_0.502_80rpm_120m` → speed 0.502 m/s read directly. Names without a numeric token (e.g. `Sheet1`) yield no metadata.
