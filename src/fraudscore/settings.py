"""Single source of configuration. Every variable is listed in `.env.example`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    data_path: Path = ROOT / "data" / "raw" / "creditcard.parquet"
    sample_path: Path = ROOT / "data" / "sample" / "creditcard_sample.parquet"
    models_dir: Path = ROOT / "models"
    reports_dir: Path = ROOT / "reports"
    monitoring_config: Path = ROOT / "monitoring.yaml"

    # Business costs, in currency units per decision. Used to pick the alert threshold.
    cost_fp: float = Field(default=5.0, ge=0, description="Cost of reviewing a legitimate alert.")
    cost_fn: float = Field(default=150.0, ge=0, description="Average loss on a missed fraud.")

    # Modelling
    seed: int = 42
    shap_background_rows: int = 2000
    bootstrap_resamples: int = 1000

    # Serving
    max_batch: int = Field(default=1000, ge=1)
    rate_limit_per_minute: int = Field(default=600, ge=1)
    rate_limit_burst: int = Field(default=100, ge=1)
    p95_latency_budget_ms: float = 50.0

    def resolved_data_path(self) -> Path:
        """Full dataset when downloaded, otherwise the committed sample."""
        return self.data_path if self.data_path.exists() else self.sample_path


@lru_cache
def get_settings() -> Settings:
    return Settings()
