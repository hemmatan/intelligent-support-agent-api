"""Configuration is loaded and validated through the real environment path."""

import pytest
from sqlalchemy import make_url

from app.core.config import Environment, Settings

PRODUCTION_SECRET = "x" * 32


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient DORNASHOP_* variables so results cannot drift."""
    import os

    for name in [key for key in os.environ if key.startswith("DORNASHOP_")]:
        monkeypatch.delenv(name, raising=False)


def build(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    """Construct Settings the way the application does: from the environment."""
    for name, value in env.items():
        monkeypatch.setenv(f"DORNASHOP_{name}", value)
    # _env_file is a runtime option of BaseSettings that its signature does
    # not declare, so the ignore is required rather than masking a defect.
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_unprefixed_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("SECRET_KEY", "someone-elses-key")
    settings = build(monkeypatch)
    assert settings.DEBUG is False
    assert settings.SECRET_KEY.get_secret_value() != "someone-elses-key"


@pytest.mark.parametrize(
    "raw",
    [
        "https://a.example,https://b.example",
        '["https://a.example","https://b.example"]',
        " https://a.example , https://b.example ",
    ],
)
def test_cors_origins_accept_both_documented_formats(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    settings = build(monkeypatch, CORS_ORIGINS=raw)
    assert settings.CORS_ORIGINS == ["https://a.example", "https://b.example"]


def test_cors_origins_reject_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        build(monkeypatch, CORS_ORIGINS='["unterminated"')


def test_postgresql_url_escapes_reserved_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = build(
        monkeypatch,
        DB_ENGINE="postgresql",
        DB_USER="app",
        DB_PASSWORD="p@ss/word:100%",
        DB_HOST="db",
        DB_NAME="shop",
    )
    url = settings.DATABASE_URL
    assert url.password == "p@ss/word:100%"
    assert "p@ss" not in str(url)
    assert make_url(url.render_as_string(hide_password=False)).host == "db"


def test_postgresql_requires_complete_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="PostgreSQL requires"):
        build(monkeypatch, DB_ENGINE="postgresql", DB_USER="app")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ACCESS_TOKEN_EXPIRE_MINUTES", "0"),
        ("ACCESS_TOKEN_EXPIRE_MINUTES", "-5"),
        ("REFRESH_TOKEN_EXPIRE_DAYS", "0"),
    ],
)
def test_token_lifetimes_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    with pytest.raises(ValueError):
        build(monkeypatch, **{name: value})


@pytest.mark.parametrize("prefix", ["api/v1", "/api/v1/", "/"])
def test_api_prefix_must_be_router_compatible(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    with pytest.raises(ValueError, match="API_V1_PREFIX"):
        build(monkeypatch, API_V1_PREFIX=prefix)


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "SECRET_KEY"),
        ({"SECRET_KEY": "tooshort"}, "SECRET_KEY"),
        ({"SECRET_KEY": PRODUCTION_SECRET, "CORS_ORIGINS": "*"}, "Wildcard CORS"),
        ({"SECRET_KEY": PRODUCTION_SECRET, "DEBUG": "true"}, "DEBUG"),
    ],
)
def test_production_rejects_unsafe_configuration(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build(monkeypatch, ENVIRONMENT="production", **env)


def test_production_accepts_explicit_secure_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = build(
        monkeypatch,
        ENVIRONMENT="production",
        SECRET_KEY=PRODUCTION_SECRET,
        CORS_ORIGINS="https://shop.example.com",
    )
    assert settings.ENVIRONMENT is Environment.PRODUCTION
    assert settings.DEBUG is False
