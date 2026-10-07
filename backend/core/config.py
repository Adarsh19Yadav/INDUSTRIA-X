"""
backend/core/config.py
======================
Centralised application settings loaded from environment variables.

All secrets are read from the environment — never hard-coded.
Uses pydantic-settings for type-safe, validated configuration.
"""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.

    Values are read from environment variables (case-insensitive).
    A .env file is loaded automatically if present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_SECRET_KEY: str = "dev-secret-key-change-in-production"
    API_KEY: str = "dev-api-key-change-in-production"
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://industria:industria_pass@localhost:5432/industria_x"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── LLM Provider ─────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["watsonx", "openai"] = "watsonx"
    WATSONX_API_KEY: str = ""
    WATSONX_PROJECT_ID: str = ""
    WATSONX_URL: str = "https://us-south.ml.cloud.ibm.com"
    WATSONX_MODEL_ID: str = "ibm/granite-13b-instruct-v2"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./backend/chroma_db"

    # ── Sensor Simulation ─────────────────────────────────────────────────────
    SENSOR_STREAM_INTERVAL_SECONDS: float = 2.0
    SENSOR_DATASET_PATH: str = "./data/processed/sensor_stream.parquet"

    # ── ML ────────────────────────────────────────────────────────────────────
    ML_ARTIFACTS_DIR: str = "./ml/artifacts"
    ACTIVE_FAILURE_MODEL: Literal["logistic_regression", "random_forest", "xgboost"] = (
        "random_forest"
    )
    ANOMALY_THRESHOLD: float = 0.5
    FAILURE_PROBABILITY_ALERT_THRESHOLD: float = 0.6
    RISK_SCORE_ALERT_THRESHOLD: float = 70.0

    # ── Rate Limiting (Phase 8) ───────────────────────────────────────────────
    # Maximum requests per client per window for the agent and assess endpoints.
    # If Redis is unavailable the limits are silently bypassed.
    RATE_LIMIT_AGENT_PER_MINUTE: int = 30
    RATE_LIMIT_ASSESS_PER_MINUTE: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── Derived properties ───────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @field_validator("APP_SECRET_KEY")
    @classmethod
    def warn_default_secret(cls, v: str) -> str:
        if v == "dev-secret-key-change-in-production":
            import warnings

            warnings.warn(
                "APP_SECRET_KEY is using the default development value. "
                "Set a strong secret in production.",
                stacklevel=2,
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Return cached Settings instance.

    Use this as a FastAPI dependency: Depends(get_settings).
    The lru_cache ensures settings are loaded once per process.
    """
    return Settings()


# Module-level convenience instance used by non-FastAPI code (e.g. ML scripts)
settings = get_settings()
