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

For the full local and Docker walkthrough, see
[Installation in the README](README.md#installation).

# Configure the app

Copy the documented development settings before running any command that loads
the application:

```sh
cp .env.example .env
```

# Initialize the database

Apply the existing migrations before starting the application or seeding an
account:

```sh
uv run alembic upgrade head
```

# Seed the commerce demonstration

This is optional for policy questions and required to reach the invented
commerce records:

```sh
uv run python -m app.seed
```

# Run the app

```sh
uv run python main.py
```

# Common development commands

Add a package to `pyproject.toml` and update the lock file:

```sh
uv add <package-name>
```

Generate a migration after changing the database models:

```sh
uv run alembic revision --autogenerate -m "describe the schema change"
```

Run all tests:

```sh
uv run pytest
```
