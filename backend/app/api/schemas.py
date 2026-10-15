"""Pydantic request / response models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SheetSpecIn(BaseModel):
    file_id: int
    sheet_name: str | None = None
    mapping: dict[str, str | None] = Field(default_factory=dict, description="source column -> standard feature (or null)")
    constants: dict[str, Any] = Field(default_factory=dict, description="standard feature -> constant value for every row")
    units: dict[str, str] = Field(default_factory=dict, description="standard feature -> declared source unit")


class IngestRequest(BaseModel):
    sheets: list[SheetSpecIn] = Field(min_length=1)
    notes: str | None = None


class PreprocessRequest(BaseModel):
    version_id: int | None = Field(default=None, description="Parent version; defaults to the dataset's latest version")
    options: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class TrainRequest(BaseModel):
    dataset_version_id: int
    target: Literal["cof", "wear_rate"] = "cof"
    models: list[str] = Field(default_factory=lambda: ["linear", "polynomial", "random_forest", "svr", "xgboost", "mlp"])
    features: list[str] | None = None
    split_strategy: Literal["auto", "grouped", "random", "chronological"] = "auto"
    test_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    n_folds: int = Field(default=5, ge=2, le=10)
    search_mode: Literal["standard", "quick"] = "standard"
    seed: int | None = None
    interval_level: float = Field(default=0.9, gt=0.5, lt=1.0)
    notes: str | None = None


class PredictRequest(BaseModel):
    inputs: dict[str, Any]
    cof_model_id: int | None = None
    wear_model_id: int | None = None
    explain: bool = True
    save: bool = True

    @field_validator("inputs")
    @classmethod
    def _non_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("inputs must not be empty")
        return v


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=5000)
    cof_model_id: int | None = None
    wear_model_id: int | None = None
    save: bool = False


class SweepSpec(BaseModel):
    feature: str
    min: float | None = None
    max: float | None = None
    steps: int = Field(default=50, ge=2, le=500)
    by: str | None = None


class SimulateRequest(BaseModel):
    target: Literal["cof", "wear_rate"] = "cof"
    base_inputs: dict[str, Any]
    sweep: SweepSpec | None = None
    compare: dict[str, list[str]] | None = None
    model_id: int | None = None


class CompareScenariosRequest(BaseModel):
    target: Literal["cof", "wear_rate"] = "cof"
    scenarios: list[dict[str, Any]] = Field(min_length=1, max_length=20)
    model_id: int | None = None


class LocalExplainRequest(BaseModel):
    inputs: dict[str, Any]


class ValidationCreate(BaseModel):
    experiment_id: str = Field(min_length=1, max_length=120)
    inputs: dict[str, Any]
    measured_cof: float | None = None
    measured_wear_rate: float | None = None
    prediction_id: int | None = None
    cof_model_id: int | None = None
    wear_model_id: int | None = None
    notes: str | None = None

    @field_validator("measured_cof")
    @classmethod
    def _cof_range(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 2):
            raise ValueError("measured_cof must be between 0 and 2")
        return v

    @field_validator("measured_wear_rate")
    @classmethod
    def _wear_range(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("measured_wear_rate must be ≥ 0")
        return v


class ValidationUpdate(BaseModel):
    status: Literal["pending", "approved", "rejected"] | None = None
    notes: str | None = None


class AugmentRequest(BaseModel):
    dataset_version_id: int
    validation_ids: list[int] = Field(min_length=1)
    notes: str | None = None


class ReportRequest(BaseModel):
    format: Literal["pdf", "xlsx"] = "pdf"
    title: str | None = None
    dataset_version_id: int | None = None
    model_version_ids: list[int] | None = None
    prediction_ids: list[int] | None = None
    include_predictions: int = Field(default=20, ge=0, le=500)
    include_validation: bool = True
    include_explainability: bool = True


class SettingsPatch(BaseModel):
    extrapolation_margin: float | None = Field(default=None, ge=0, le=1)
    allow_far_extrapolation: bool | None = None
    fixed_condition_tolerance: float | None = Field(default=None, ge=0, le=1)
    interval_level: float | None = Field(default=None, gt=0.5, lt=1)
    default_split_strategy: Literal["auto", "grouped", "random", "chronological"] | None = None
    default_search_mode: Literal["standard", "quick"] | None = None
    default_models: list[str] | None = None
    default_test_fraction: float | None = Field(default=None, ge=0.05, le=0.5)
    default_n_folds: int | None = Field(default=None, ge=2, le=10)
    random_seed: int | None = None
    project: dict[str, str] | None = None
