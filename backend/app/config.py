"""Application configuration loaded from environment variables (never from source code)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI TriboPredict"
    app_version: str = "1.0.0"
    environment: str = "development"

    # Persistence
    database_url: str = "sqlite:///./storage/tribopredict.db"
    storage_dir: Path = Path("./storage")

    # Bundled experimental dataset (Kolev 2023, copied into the repository)
    bundled_dataset_dir: Path = Path("../data/source/kolev_2023_pin_on_disk")

    # Security
    api_key: str | None = None  # when set, every /api request must send X-API-Key
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Reproducibility and compute
    random_seed: int = 42
    training_workers: int = 1
    training_sync: bool = False  # run training inline (tests / debugging)
    max_upload_mb: int = 50
    n_jobs: int = 2

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def models_dir(self) -> Path:
        return self.storage_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.storage_dir / "reports"

    @property
    def exports_dir(self) -> Path:
        return self.storage_dir / "exports"

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.uploads_dir, self.models_dir, self.reports_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
