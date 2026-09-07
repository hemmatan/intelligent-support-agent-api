"""Application configuration loaded from namespaced environment variables."""

import json
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy import URL

SQLITE_DEFAULT_NAME = "db.sqlite3"
DEVELOPMENT_SECRET_KEY = "development-only-change-me-32-bytes"


class Environment(StrEnum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Validated application settings."""

    PROJECT_NAME: str = "DornaShop Support API"
    PROJECT_DESCRIPTION: str = "Customer support API for the DornaShop storefront"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False

    # JWT
    SECRET_KEY: SecretStr = SecretStr(DEVELOPMENT_SECRET_KEY)
    # Pinned: the minimum secret length below is chosen for HS256.
    ALGORITHM: Literal["HS256"] = "HS256"
    JWT_ISSUER: str = "dornashop-api"
    JWT_AUDIENCE: str = "dornashop-users"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, gt=0)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, gt=0)

    # Retrieval. Absent, searching stays lexical: an optional second ranker
    # being unconfigured is not a reason to refuse to start.
    HUGGINGFACE_API_TOKEN: SecretStr | None = None
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0)

    # How long a reading of the shop's records stays worth sending, and how
    # long past that it stays worth showing a colleague. Neither is tuned:
    # nothing caches yet, so every reading is taken now and lands on the
    # first rung regardless. They are settings rather than constants because
    # the day something does cache, this is the dial.
    COMMERCE_FRESHNESS_TTL_SECONDS: float = Field(default=900.0, gt=0)
    COMMERCE_READABLE_FOR_SECONDS: float = Field(default=21600.0, gt=0)

    # CORS. NoDecode suppresses the JSON pre-parse that pydantic-settings
    # applies to list fields, so the validator below sees the raw string.
    CORS_ORIGINS: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    # Database
    DB_ENGINE: Literal["sqlite", "postgresql"] = "sqlite"
    DB_USER: str = ""
    DB_PASSWORD: SecretStr = SecretStr("")
    DB_HOST: str = ""
    DB_PORT: int | None = Field(default=None, ge=1, le=65535)
    DB_NAME: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DORNASHOP_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, value: str | list[str]) -> list[str]:
        """Accept JSON arrays or comma-separated origins from the environment."""
        if isinstance(value, list):
            return value
        value = value.strip()
        if not value.startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("CORS_ORIGINS contains invalid JSON") from exc
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed
        raise ValueError("CORS_ORIGINS must contain only strings")

    @field_validator("API_V1_PREFIX")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """Reject prefixes that FastAPI's router would refuse at startup."""
        if not value.startswith("/"):
            raise ValueError("API_V1_PREFIX must start with '/'")
        if value.endswith("/"):
            raise ValueError("API_V1_PREFIX must not end with '/'")
        return value

    @model_validator(mode="after")
    def validate_postgresql_settings(self) -> Self:
        """Require full credentials when PostgreSQL is selected."""
        if self.DB_ENGINE != "postgresql":
            return self
        missing = [
            name
            for name, value in (
                ("DB_USER", self.DB_USER),
                ("DB_PASSWORD", self.DB_PASSWORD.get_secret_value()),
                ("DB_HOST", self.DB_HOST),
                ("DB_NAME", self.DB_NAME),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "PostgreSQL requires " + ", ".join(f"DORNASHOP_{n}" for n in missing)
            )
        return self

    @model_validator(mode="after")
    def validate_production_security(self) -> Self:
        """Fail fast when production uses unsafe development defaults."""
        if self.ENVIRONMENT is not Environment.PRODUCTION:
            return self
        secret = self.SECRET_KEY.get_secret_value()
        if len(secret) < 32 or secret == DEVELOPMENT_SECRET_KEY:
            raise ValueError(
                "Production SECRET_KEY must contain at least 32 characters"
            )
        if "*" in self.CORS_ORIGINS:
            raise ValueError("Wildcard CORS origins are forbidden in production")
        if self.DEBUG:
            raise ValueError("DEBUG must be disabled in production")
        return self

    def _url(self, database: str) -> URL:
        """Build a driver URL, letting SQLAlchemy escape the credentials."""
        if self.DB_ENGINE == "sqlite":
            return URL.create("sqlite+aiosqlite", database=database)
        return URL.create(
            "postgresql+asyncpg",
            username=self.DB_USER,
            password=self.DB_PASSWORD.get_secret_value(),
            host=self.DB_HOST,
            port=5432 if self.DB_PORT is None else self.DB_PORT,
            database=database,
        )

    @property
    def DATABASE_URL(self) -> URL:
        """Async SQLAlchemy URL. Rendering it redacts the password by default."""
        return self._url(self.DB_NAME or SQLITE_DEFAULT_NAME)


settings = Settings()
