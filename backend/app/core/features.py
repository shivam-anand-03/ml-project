"""Standard feature registry (data dictionary) and material catalogue.

Every uploaded column is mapped onto one of these standardized features. The registry is the single
source of truth for units, validity ranges and roles used by ingestion, training and prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    unit: str
    dtype: str  # number | category
    role: str  # input | target | measurement | identifier | metadata
    description: str
    valid_min: float | None = None
    valid_max: float | None = None
    synonyms: tuple[str, ...] = field(default_factory=tuple)


_FEATURES = [
    Feature("test_run_id", "Test run ID", "", "category", "identifier",
            "Identifier of one tribometer run / specimen. Used for grouped train–test splitting.",
            synonyms=("run", "run id", "test id", "test run", "experiment", "experiment id", "specimen id",
                      "sample id", "trial")),
    Feature("material", "Material", "", "category", "input",
            "Material code of the tested specimen (e.g. AC, AE, C, E in the Kolev 2023 dataset).",
            synonyms=("material", "specimen", "sample", "alloy", "material code", "disc material")),
    Feature("composition", "Composition", "", "category", "input",
            "Material composition derived from the material code (e.g. AlSi10Mg vs AlSi10Mg-Al2O3).",
            synonyms=("composition", "reinforcement")),
    Feature("pore_size_class", "Pore size class", "µm", "category", "input",
            "Pore size class of the open-cell material derived from the material code.",
            synonyms=("pore size", "pore size class", "porosity class", "pore")),
    Feature("material_pair", "Material pair", "", "category", "input",
            "Pin / disc material pair (e.g. Steel-Al).",
            synonyms=("material pair", "pair", "tribo pair", "pin disc", "counterbody", "pin material")),
    Feature("lubricant", "Lubricant", "", "category", "input",
            "Lubricant type. 'dry' when no lubricant was used.",
            synonyms=("lubricant", "lube", "oil", "lubrication", "lubricant type", "grease")),
    Feature("normal_load_n", "Normal load", "N", "number", "input",
            "Applied normal load.", valid_min=0.0, valid_max=5000.0,
            synonyms=("load", "normal load", "fn", "f n", "applied load", "normal force", "load n", "force")),
    Feature("sliding_speed_m_s", "Sliding speed", "m/s", "number", "input",
            "Linear sliding speed at the wear track.", valid_min=0.0, valid_max=100.0,
            synonyms=("speed", "sliding speed", "velocity", "linear velocity", "sliding velocity", "v", "speed m s")),
    Feature("temperature_c", "Temperature", "°C", "number", "input",
            "Test temperature.", valid_min=-100.0, valid_max=1500.0,
            synonyms=("temperature", "temp", "t c", "temperature c")),
    Feature("time_s", "Sliding time", "s", "number", "input",
            "Elapsed sliding time since the start of the run.", valid_min=0.0, valid_max=1e7,
            synonyms=("time", "t", "sliding time", "duration", "elapsed time", "time s", "average time")),
    Feature("sliding_distance_m", "Sliding distance", "m", "number", "input",
            "Accumulated sliding distance (speed × time when not measured directly).", valid_min=0.0, valid_max=1e7,
            synonyms=("distance", "sliding distance", "s m", "distance m")),
    Feature("rpm", "Rotational speed", "rpm", "number", "metadata",
            "Disc rotational speed.", valid_min=0.0, valid_max=100000.0,
            synonyms=("rpm", "rotational speed", "rotation speed", "n rpm")),
    Feature("track_diameter_mm", "Wear track diameter", "mm", "number", "metadata",
            "Diameter of the wear track on the disc.", valid_min=0.0, valid_max=10000.0,
            synonyms=("track diameter", "diameter", "wear track diameter", "track", "radius")),
    Feature("friction_force_n", "Friction force", "N", "number", "measurement",
            "Measured tangential friction force.", valid_min=-1e4, valid_max=1e4,
            synonyms=("ff", "friction force", "friction", "f f", "tangential force", "friction n", "fr")),
    Feature("cof", "Coefficient of friction", "–", "number", "target",
            "Coefficient of friction = friction force / normal load.", valid_min=0.0, valid_max=2.0,
            synonyms=("cof", "coefficient of friction", "friction coefficient", "mu", "µ",
                      "average coefficient of friction")),
    Feature("wear_volume_mm3", "Wear volume", "mm³", "number", "measurement",
            "Volume loss of the specimen.", valid_min=0.0, valid_max=1e9,
            synonyms=("wear volume", "volume loss", "wear volume mm3", "volume")),
    Feature("wear_mass_mg", "Wear mass", "mg", "number", "measurement",
            "Mass loss of the specimen.", valid_min=0.0, valid_max=1e9,
            synonyms=("wear mass", "mass loss", "weight loss", "mass wear", "wear mg")),
    Feature("wear_rate", "Wear rate", "declared by dataset", "number", "target",
            "Specific wear rate k = V / (F · s) or another wear-rate definition declared with the dataset.",
            valid_min=0.0, valid_max=1e9,
            synonyms=("wear rate", "specific wear rate", "k", "ws", "wear rate k", "wear")),
]

STANDARD_FEATURES: dict[str, Feature] = {f.name: f for f in _FEATURES}

NUMERIC_FEATURES = [f.name for f in STANDARD_FEATURES.values() if f.dtype == "number"]
CATEGORICAL_FEATURES = [f.name for f in STANDARD_FEATURES.values() if f.dtype == "category"]
INPUT_FEATURES = [f.name for f in STANDARD_FEATURES.values() if f.role == "input"]
TARGETS = ["cof", "wear_rate"]
RECORD_COLUMNS = list(STANDARD_FEATURES.keys())

TARGET_INFO = {
    "cof": {"label": "Coefficient of friction", "unit": "–", "short": "COF"},
    "wear_rate": {"label": "Wear rate", "unit": "declared by dataset", "short": "Wear rate"},
}

# Material catalogue for the bundled Kolev 2023 dataset. Codes not listed here keep composition / pore
# size empty — nothing is invented for unknown materials.
MATERIAL_CATALOGUE: dict[str, dict[str, str]] = {
    "AC": {"composition": "AlSi10Mg-Al2O3", "pore_size_class": "800-1000 um",
           "description": "Open-cell AlSi10Mg-Al2O3 composite, pore size 800-1000 µm"},
    "AE": {"composition": "AlSi10Mg-Al2O3", "pore_size_class": "1000-1200 um",
           "description": "Open-cell AlSi10Mg-Al2O3 composite, pore size 1000-1200 µm"},
    "C": {"composition": "AlSi10Mg", "pore_size_class": "800-1000 um",
          "description": "Open-cell AlSi10Mg, pore size 800-1000 µm"},
    "E": {"composition": "AlSi10Mg", "pore_size_class": "1000-1200 um",
          "description": "Open-cell AlSi10Mg, pore size 1000-1200 µm"},
}


def feature_dictionary() -> list[dict]:
    return [
        {
            "name": f.name,
            "label": f.label,
            "unit": f.unit,
            "dtype": f.dtype,
            "role": f.role,
            "description": f.description,
            "valid_min": f.valid_min,
            "valid_max": f.valid_max,
            "synonyms": list(f.synonyms),
        }
        for f in STANDARD_FEATURES.values()
    ]
