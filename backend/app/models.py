"""Relational schema for AI TriboPredict (SQLAlchemy 2.0 declarative)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(40), default="upload")  # upload | bundled | validation
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    files: Mapped[list[DatasetFile]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    versions: Mapped[list[DatasetVersion]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", order_by="DatasetVersion.version_number"
    )


class DatasetFile(Base):
    __tablename__ = "dataset_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(300))
    stored_path: Mapped[str] = mapped_column(String(600))
    file_type: Mapped[str] = mapped_column(String(20))  # xlsx | xls | csv | dwf
    size_bytes: Mapped[int] = mapped_column(Integer)
    inspection: Mapped[dict] = mapped_column(JSON, default=dict)  # sheets, columns, previews, suggestions
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    dataset: Mapped[Dataset] = relationship(back_populates="files")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    version_number: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20))  # raw | processed | augmented
    parent_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    usable_record_count: Mapped[int] = mapped_column(Integer, default=0)
    column_mapping: Mapped[dict] = mapped_column(JSON, default=dict)  # per file/sheet mapping + constants
    cleaning_log: Mapped[list] = mapped_column(JSON, default=list)
    quality_report: Mapped[dict] = mapped_column(JSON, default=dict)
    feature_summary: Mapped[dict] = mapped_column(JSON, default=dict)  # available targets, varying features, ranges
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    dataset: Mapped[Dataset] = relationship(back_populates="versions")
    records: Mapped[list[ExperimentalRecord]] = relationship(
        back_populates="version", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (Index("ix_dataset_versions_dataset_version", "dataset_id", "version_number", unique=True),)


class ExperimentalRecord(Base):
    """One measured observation, stored once per dataset version with standardized columns."""

    __tablename__ = "experimental_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), index=True
    )
    test_run_id: Mapped[str | None] = mapped_column(String(120), index=True)
    material: Mapped[str | None] = mapped_column(String(120), index=True)
    composition: Mapped[str | None] = mapped_column(String(120))
    pore_size_class: Mapped[str | None] = mapped_column(String(60))
    material_pair: Mapped[str | None] = mapped_column(String(120))
    lubricant: Mapped[str | None] = mapped_column(String(120))
    normal_load_n: Mapped[float | None] = mapped_column(Float)
    sliding_speed_m_s: Mapped[float | None] = mapped_column(Float)
    temperature_c: Mapped[float | None] = mapped_column(Float)
    time_s: Mapped[float | None] = mapped_column(Float)
    sliding_distance_m: Mapped[float | None] = mapped_column(Float)
    rpm: Mapped[float | None] = mapped_column(Float)
    track_diameter_mm: Mapped[float | None] = mapped_column(Float)
    friction_force_n: Mapped[float | None] = mapped_column(Float)
    cof: Mapped[float | None] = mapped_column(Float)
    wear_volume_mm3: Mapped[float | None] = mapped_column(Float)
    wear_mass_mg: Mapped[float | None] = mapped_column(Float)
    wear_rate: Mapped[float | None] = mapped_column(Float)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    is_usable: Mapped[bool] = mapped_column(Boolean, default=True)
    source: Mapped[str] = mapped_column(String(20), default="measured")  # measured | validation
    row_index: Mapped[int | None] = mapped_column(Integer)

    version: Mapped[DatasetVersion] = relationship(back_populates="records")

    __table_args__ = (
        Index("ix_records_version_material", "dataset_version_id", "material"),
        Index("ix_records_version_run", "dataset_version_id", "test_run_id"),
    )


class FeatureDefinition(Base):
    """Data dictionary: definition, unit, source and validity range of each standard feature."""

    __tablename__ = "feature_definitions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)
    label: Mapped[str] = mapped_column(String(120))
    unit: Mapped[str] = mapped_column(String(40), default="")
    dtype: Mapped[str] = mapped_column(String(20))  # number | category
    role: Mapped[str] = mapped_column(String(20))  # input | target | measurement | identifier | metadata
    description: Mapped[str] = mapped_column(Text, default="")
    valid_min: Mapped[float | None] = mapped_column(Float)
    valid_max: Mapped[float | None] = mapped_column(Float)
    source_columns: Mapped[list] = mapped_column(JSON, default=list)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    target: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued | running | completed | failed
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    log: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    models: Mapped[list[ModelVersion]] = relationship(back_populates="training_run")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"), index=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    target: Mapped[str] = mapped_column(String(40), index=True)
    algorithm: Mapped[str] = mapped_column(String(40))
    version_label: Mapped[str] = mapped_column(String(80))
    artifact_path: Mapped[str] = mapped_column(String(600))
    feature_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    training_domain: Mapped[dict] = mapped_column(JSON, default=dict)
    cv_results: Mapped[dict] = mapped_column(JSON, default=dict)
    global_importance: Mapped[dict] = mapped_column(JSON, default=dict)
    interval: Mapped[dict] = mapped_column(JSON, default=dict)  # conformal half-width and method
    status: Mapped[str] = mapped_column(String(20), default="candidate")  # candidate | production | archived
    is_best_of_run: Mapped[bool] = mapped_column(Boolean, default=False)
    training_duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    n_train: Mapped[int] = mapped_column(Integer, default=0)
    n_test: Mapped[int] = mapped_column(Integer, default=0)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    training_run: Mapped[TrainingRun] = relationship(back_populates="models")
    metrics: Mapped[list[ModelMetric]] = relationship(back_populates="model", cascade="all, delete-orphan")


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    split: Mapped[str] = mapped_column(String(20))  # train | validation | test
    metric: Mapped[str] = mapped_column(String(20))  # r2 | mae | rmse | mape
    value: Mapped[float | None] = mapped_column(Float)
    n_samples: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)

    model: Mapped[ModelVersion] = relationship(back_populates="metrics")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    cof_model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), index=True)
    wear_model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    dataset_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    inputs: Mapped[dict] = mapped_column(JSON, default=dict)
    predicted_cof: Mapped[float | None] = mapped_column(Float)
    cof_interval: Mapped[dict | None] = mapped_column(JSON)
    predicted_wear_rate: Mapped[float | None] = mapped_column(Float)
    wear_rate_unit: Mapped[str | None] = mapped_column(String(40))
    in_domain: Mapped[bool] = mapped_column(Boolean, default=True)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    explanation: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(20), default="single")  # single | batch | simulation
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ValidationRecord(Base):
    __tablename__ = "validation_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int | None] = mapped_column(ForeignKey("predictions.id"))
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    experiment_id: Mapped[str] = mapped_column(String(120), index=True)
    inputs: Mapped[dict] = mapped_column(JSON, default=dict)
    measured_cof: Mapped[float | None] = mapped_column(Float)
    measured_wear_rate: Mapped[float | None] = mapped_column(Float)
    predicted_cof: Mapped[float | None] = mapped_column(Float)
    predicted_wear_rate: Mapped[float | None] = mapped_column(Float)
    cof_abs_error: Mapped[float | None] = mapped_column(Float)
    cof_pct_error: Mapped[float | None] = mapped_column(Float)
    wear_abs_error: Mapped[float | None] = mapped_column(Float)
    wear_pct_error: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | approved | rejected
    notes: Mapped[str | None] = mapped_column(Text)
    added_to_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(10))  # pdf | xlsx
    title: Mapped[str] = mapped_column(String(200))
    file_path: Mapped[str] = mapped_column(String(600))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    dataset_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
