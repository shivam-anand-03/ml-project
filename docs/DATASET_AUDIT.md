# Dataset Audit — Phase 1

**Dataset inspected:** `Friction coefficient data of open-cell AlSi10Mg and AlSi10Mg-Al2O3 materials with different pore sizes by pin-on-disk test and machine learning prediction`
**Source:** Mendeley Data, Mihail Kolev, Institute of Metal Science, Equipment and Technologies "Acad. A. Balevski", Bulgarian Academy of Sciences. DOI 10.17632/2356m76ktj.1, published 30 May 2023, licence CC BY 4.0.
**Related article:** Data in Brief 50 (2023) 109489, DOI 10.1016/j.dib.2023.109489.
**Copy used by the application:** `data/source/kolev_2023_pin_on_disk/` (byte-for-byte copy; the original download is never modified).
**Audit date:** 2026-10-15

## 1. Inventory

| Folder | Files | Content |
|---|---|---|
| `Pin-on-disk_data/Raw_files` | 12 × `.dwf` | Raw tribometer export. Plain text, `;`-separated, decimal comma, CR or CRLF line endings, each line quoted. Columns `TIME`, `FF`. 462 data rows each. |
| `Pin-on-disk_data/Processed_files` | 12 × `.xlsx` | One worksheet per file, 462 rows × 2 columns (`TIME`, `FF`). Worksheet name encodes test parameters, e.g. `AC_3_2_50N_80rpm_120mm`. |
| `Pin-on-disk_data/COF_calculation` | 8 × `.xlsx`, 1 × `.py`, 2 × `.png` | Per-material average COF (mean of the 3 replicate runs at each sample index) and average time. 462 rows × 1 column each. |
| `Prediction/Input_files` | 4 × `.xlsx` | `time`, `COF` — identical to the averaged COF files (max abs difference 0.0). Columns C–H are formatted but empty. `pred_COF_AC.xlsx` has one extra empty row. |
| `Prediction/Output_data` | 4 × `.xlsx`, 4 × `.txt`, 4 × `.png` | Original authors' XGBoost actual/predicted values for test and validation sets (93 rows each) and metrics. |
| `Prediction/Python_COF_prediction` | 1 × `.py` | Original authors' XGBoost script (row-wise random 60/20/20 split, GridSearchCV over `eta` and `n_estimators`). |

## 2. Experimental design (from the dataset description and worksheet names)

| Parameter | Value | How determined |
|---|---|---|
| Test type | Pin-on-disk, **dry** sliding, room temperature | Mendeley description |
| Normal load | **50 N (constant in all runs)** | Description; worksheet names (`50N`); COF script divides FF by 50 |
| Sliding speed | **0.5 m/s (constant)** | Description; verified: π × track diameter × rpm / 60 = 0.500–0.503 m/s for every run |
| Test duration | 420 s | Description; last timestamp 419.9–420.2 s |
| Sliding distance | ≈ 210 m per run (0.5 m/s × 420 s) | Derived |
| Sampling | ≈ 0.909 s interval, 462 samples per run | Measured from `TIME` |
| Track diameter / rotation speed | 60–120 mm / 80–160 rpm, varies per run | Worksheet names |
| Lubricant | **None (dry)** — no lubricant variable exists | Description |
| Temperature | Room temperature, **not recorded numerically** | Description |
| Counterbody / pin material | **Not stated in the dataset** | — |
| Wear volume / wear mass / wear rate | **Not present in any file** | Inventory |

### Materials (categorical factor, 4 levels)

| Code | Material | Pore size | Runs (worksheet suffix) |
|---|---|---|---|
| `AC` | AlSi10Mg–Al2O3 composite | 800–1000 µm | AC_3_2 (80 rpm, 120 mm), AC_3_3 (87 rpm, 110 mm), AC_3_4 (96 rpm, 100 mm) |
| `C` | AlSi10Mg (unreinforced) | 800–1000 µm | C_5_1 (120 rpm, 80 mm), C_5_2 (160 rpm, 60 mm), C_7_1 (137 rpm, 70 mm) |
| `AE` | AlSi10Mg–Al2O3 composite | 1000–1200 µm | AE_3_2 (87 rpm, 110 mm), AE_4_1 (96 rpm, 100 mm), AE_6_6 (80 rpm, 120 mm) |
| `E` | AlSi10Mg (unreinforced) | 1000–1200 µm | E_3_1 (106 rpm, 90 mm), E_6 (87 rpm, 110 mm), E_6_3 (106 rpm, 90 mm) |

The material code therefore decomposes into two derived factors: **composition** (`AlSi10Mg` vs `AlSi10Mg-Al2O3`) and **pore size class** (`800-1000 um` vs `1000-1200 um`).

## 3. Columns, units and standard-feature mapping

| Source column | Unit | Standard feature | Role |
|---|---|---|---|
| `TIME` | s | `time_s` | input (continuous) |
| `FF` | N | `friction_force_n` | measured output |
| derived `FF / 50` | – | `cof` | **target** (COF = friction force / normal load) |
| worksheet name | – | `test_run_id`, `material`, `normal_load_n`, `rpm`, `track_diameter_mm` | metadata / grouping |
| derived | m/s | `sliding_speed_m_s` = π·D·n/60 | input (constant ≈ 0.5) |
| derived | m | `sliding_distance_m` = speed × time | input (continuous) |
| derived | – | `composition`, `pore_size_class` | input (categorical) |

## 4. Data quality

| Check | Result |
|---|---|
| Rows (12 runs × 462) | 5 544 measured rows |
| Missing values | 0 in every file |
| Duplicate rows | 0 within any run (the 26–63 "duplicates" in the averaged COF files are repeated values at different times, not duplicate records) |
| Negative friction force | 17 values in the raw `.dwf` files (C_5_1: 10, E_3_1: 5, E_6_3: 2), all between −0.01 N and −0.18 N during the first seconds. In the processed `.xlsx` files they appear as the **absolute value**. The application keeps the processed values and flags these rows. |
| Zero friction force | 3 values (sensor at rest before pin engagement) |
| Run-in transient | Every run starts with FF ≈ 0–1 N for the first 1–8 s before the pin engages, then rises to a steady state of 20–30 N. These are genuine measurements, not errors; they are kept and flagged as `run_in` so that steady-state statistics can be computed separately. |
| COF range | 0.000–0.617 (max in AE_3_2 at a spike); steady-state means (t > 210 s): AC 0.516, AE 0.484, C 0.543, E 0.499 |
| Timestamp alignment | Replicates are sampled at the same indices (max timestamp difference < 0.3 s), which is why the original authors averaged by index. |
| Outliers | No physically implausible values. Spikes up to COF 0.6 are retained. |

## 5. Usable targets and features for machine learning

**Available target:** `cof` (coefficient of friction), continuous, 5 544 observations.

**Unavailable targets:** wear rate / specific wear rate / wear volume / wear mass. No file contains any wear quantity, so wear-rate models **cannot be trained** from this dataset and the application reports wear-rate predictions as *unavailable* for it. The pipeline supports a wear-rate target as soon as a dataset that contains it is uploaded.

**Informative inputs:** `material` (and its derived `composition`, `pore_size_class`), `time_s` / `sliding_distance_m`. Track diameter and rpm vary only between replicate runs of the same material and are confounded with the replicate, so they are excluded from the default feature set.

**Constant inputs:** `normal_load_n` = 50 N, `sliding_speed_m_s` ≈ 0.5 m/s. A model cannot learn their effect; the prediction interface restricts them to the experimental domain and refuses extrapolation.

**Not recorded:** lubricant (dry test), temperature (room temperature, no value), humidity, counterbody.

## 6. Structure relevant to validation

The data are **repeated time-series measurements**: 12 runs, 3 replicate runs per material. Rows within one run are strongly autocorrelated. A row-wise random split (as in the original authors' script) measures interpolation within a known run and gives R² ≈ 0.99. The application therefore uses **grouped splitting by `test_run_id`** as the default strategy (whole runs are held out; one run per material forms the test set; grouped K-fold on the remaining runs for tuning), and offers the row-wise random split as a documented alternative for comparison with the original publication.

## 7. Original authors' reported XGBoost metrics (row-wise split, per material)

| Material | Test R² | Test RMSE | Test MAE | Val R² |
|---|---|---|---|---|
| AC | 0.9971 | 0.0048 | 0.0037 | 0.9934 |
| AE | 0.9936 | 0.0063 | 0.0040 | 0.9965 |
| C | 0.9944 | 0.0078 | 0.0052 | 0.9875 |
| E | 0.9929 | 0.0086 | 0.0048 | 0.9906 |

These values are quoted from the dataset's `*_performance_metrics.txt` files for reference only; the application never displays them as its own results.

## 8. Limitations to communicate in the application

1. Only COF can be predicted. Wear rate is unavailable.
2. Load, speed, lubricant and temperature were not varied, so no load/speed/temperature trend can be learned or claimed.
3. Predictions are valid only for the four tested materials, 50 N, 0.5 m/s, dry sliding, 0–420 s.
4. With only three replicate runs per material, held-out-run performance is an estimate with wide uncertainty.
