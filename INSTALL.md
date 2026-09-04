# Install UV

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Install the project

`uv sync` creates `.venv` and installs exactly what `uv.lock` pins. Use
`--frozen` so the environment matches the one the tests run against.

```sh
uv sync --extra dev --frozen
```

Omit `--extra dev` for runtime dependencies only. Drop `--frozen` only when you
intend to re-resolve, which updates `uv.lock`.

# Run the app

```sh
uv run python main.py
```

# Adding Packages to pyproject.toml

```sh
uv add <package-name>
```

# Detect schema/ Generate migrations changes

```sh
alembic revision --autogenerate
```

# run migrations

```sh
alembic upgrade head
```

# Regenerate requirements.txt from the lock file

`uv pip compile` re-resolves and drifts from `uv.lock`. Export instead, so the
pinned versions match exactly:

```sh
uv export --frozen --format requirements-txt --extra dev \
  --no-hashes --no-emit-project --output-file requirements.txt
```

# run all tests

```sh
pytest
```
