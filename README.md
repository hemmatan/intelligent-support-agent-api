<p align="center">
  <img src="assets/dorna-shop-logo.png" alt="DornaShop" width="320">
</p>

<h1 align="center">DornaShop Support Agent</h1>

<p align="center">
  An evidence-grounded support API: every answer cites the approved policy it
  came from, and anything it cannot ground is routed to a person instead.
</p>


A FastAPI service for DornaShop's intelligent customer-support agent. The
current foundation provides authentication, authorization, asynchronous database
access, migrations, and Docker-based development workflows.

The agent's design and the reasoning behind it are in
[docs/architecture.md](docs/architecture.md).

## Features

- **Modern Python**: Type hints, async/await syntax, and the latest FastAPI features
- **JWT Authentication**: Short-lived access tokens and rotating, revocable refresh tokens
- **SQLAlchemy with Async**: Fully async database operations using SQLAlchemy 2.0+
- **Alembic Migrations**: Database schema migrations with Alembic
- **Role Model and Authorization Foundation**: Database-backed `customer`, `support_agent` and `admin` roles, with staff enforcement available but not yet required by any endpoint
- **Versioned API**: Public application routes are grouped under `/api/v1`
- **Docker Development Workflow**: Containerized local setup; see the Docker section for current limitations
- **Developer-friendly**: Auto-reload, debugging, and development tools
- **Validated Configuration**: Namespaced settings with production secret and CORS safeguards
- **Grounded Answers**: Every customer-facing sentence is approved, versioned and content-hashed; figures come from structured claims, never from prose
- **Four Honest Outcomes**: Answer, clarify, escalate or hold for review, each with a reason code, a rendered message and a recorded case

## Project Structure

```
.
├── alembic/                 # Database migrations
├── app/                     # Main application package
│   ├── api/                 # API endpoints
│   ├── core/                # Core functionality (config, security)
│   ├── db/                  # Database session and base
│   ├── models/              # SQLAlchemy models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic
│   └── utils/               # Utility functions
├── docker-compose.yml       # Baseline local/demo Compose configuration
├── docker-compose.dev.yml   # Local development configuration with reload
├── Dockerfile               # Development-oriented application image
├── alembic.ini              # Alembic configuration
├── .env.example             # Documented configuration template
├── docs/architecture.md     # Support-agent design decisions
├── main.py                  # Application entry point
├── pyproject.toml           # Project dependencies and metadata
├── start.sh                 # Baseline container startup script
└── start-dev.sh             # Development startup script
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for the recommended setup; see [INSTALL.md](INSTALL.md)
- Docker (optional)

## Installation

### Using Docker for local development

> **Development only:** The current image and Compose configurations are for
> local development and demonstrations. They are not production-ready yet.

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repository-directory>
   ```

2. Start the application with Docker Compose:
   ```bash
   # Development with auto-reload
   docker-compose -f docker-compose.dev.yml up --build

   # Baseline local run without auto-reload
   docker-compose up --build
   ```

3. The API will be available at http://localhost:8000

### Local Development

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repository-directory>
   ```

2. Install the locked dependencies:
   ```bash
   uv sync --extra dev --frozen
   ```
   `--frozen` installs exactly what `uv.lock` pins, so the environment matches
   the one the tests and CI run against. Without `uv`, a virtual environment
   plus `pip install -e ".[dev]"` works, but re-resolves and may drift from the
   lock file.

3. Set up environment variables (copy `.env.example` to `.env` and edit):
   ```
   DORNASHOP_ENVIRONMENT=development
   DORNASHOP_DEBUG=true
   DORNASHOP_SECRET_KEY=replace-with-at-least-32-random-characters
   DORNASHOP_DB_ENGINE=sqlite  # or postgresql
   # For PostgreSQL, add these:
   # DORNASHOP_DB_USER=postgres
   # DORNASHOP_DB_PASSWORD=password
   # DORNASHOP_DB_HOST=localhost
   # DORNASHOP_DB_PORT=5432
   # DORNASHOP_DB_NAME=app
   ```

4. Run migrations:
   ```bash
   alembic upgrade head
   ```

5. Start the application:
   ```bash
   uvicorn main:app --reload
   ```

6. The API will be available at http://localhost:8000

## API Documentation

Once the application is running, you can access:

- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc
- OpenAPI schema: http://localhost:8000/api/v1/openapi.json

## API Endpoints

### Authentication

- `POST /api/v1/auth/signup` - Register a customer
- `POST /api/v1/auth/login` - Authenticate and obtain access and refresh tokens
- `POST /api/v1/auth/token/refresh` - Rotate a refresh token and obtain a new token pair
- `POST /api/v1/auth/logout` - Revoke a refresh token
- `GET /api/v1/auth/me` - Get the current user using an access token
- `POST /api/v1/auth/api-tokens` - Create an API token that is stored only as a hash
- `GET /api/v1/auth/api-me` - Get the current user using an API token

### Support

- `POST /api/v1/support/messages` - Ask the agent a question

Send a question as a signed-in customer. The language of the reply comes from
the account, not the request.

```jsonc
POST /api/v1/support/messages
Authorization: Bearer <access token>

{
  "message": "How long do I have to return a jacket?",
  "order_id": "ORD-4471",          // optional
  "product_reference": "SKU-9"     // optional
}
```

Four things can come back, and **all of them are `200`**. Being asked a
question, being passed to a person and being held for checking are decisions
about a request that was understood; the `route` says which. Status codes are
kept for a malformed body (`422`) and an unknown caller (`401`).

Every reply carries a `case`, which is the record the decision was written
into before the reply was sent.

**`direct_response`** — answered from approved wording, with the evidence it
rests on:

```json
{
  "route": "direct_response",
  "case": "9c8e2f1a-...",
  "intent": "return_policy",
  "reply": "Returns are accepted within 30 days of delivery.",
  "citations": [
    {
      "source": "knowledge_base",
      "reference": "kb:returns.standard.en.v1",
      "content_hash": "sha256:..."
    }
  ],
  "wording": ["say:return_window.en.v1@sha256:..."],
  "reliability": {
    "level": "acceptable",
    "ordinal": 2,
    "scale": 3,
    "factors": {"authority": "ready", "coverage": "ready", "relevance": "acceptable"}
  }
}
```

**`clarification`** — something is missing that the customer can supply:

```json
{
  "route": "clarification",
  "case": "9c8e2f1a-...",
  "reason": "missing_order_id",
  "message": "Please send us your order number and we will look it up.",
  "wording": "say:ask_for_order_number.en.v1@sha256:..."
}
```

**`human_escalation`** — a person takes it, and a case is waiting for them:

```json
{
  "route": "human_escalation",
  "case": "9c8e2f1a-...",
  "reasons": ["payment_dispute"],
  "message": "We have passed this to a member of our team to handle personally.",
  "wording": "say:handed_to_a_specialist.en.v1@sha256:...",
  "reliability": null
}
```

**`internal_review`** — nothing is wrong with the request; something is wrong
with us, and somebody here finishes it. Same shape as an escalation, with
`route: "internal_review"`.

`reason` and `message` are not alternatives. The code is stable and
machine-readable — branch on it, count it, assert against it — while the
message is written for a person and translated. Reading the message to work
out what happened means reading the wrong field.

### Staff

Requires the `support_agent` or `admin` role. This is the other half of
escalating: the queue exists so that telling a customer somebody is dealing
with their message is true.

- `GET /api/v1/support/cases` - Requests still waiting for a person, oldest first
- `POST /api/v1/support/cases/{reference}/claim` - Put your name against one
- `POST /api/v1/support/cases/{reference}/resolve` - Close it, recording what was done

A case carries what the decision rested on — the message, the order number and
product reference the customer supplied, the sources the request was permitted
to read, the route and reasons, the words they received, the evidence cited and
the rating each dimension earned — so nobody has to write back for something
already given.

Claiming and resolving are conditional writes, so two people cannot both be
told a case is theirs. Who took it on and who finished it are recorded
separately: covering a colleague's shift should credit the person who did the
work. What happens after that — drafting, editing, replying to the customer —
is outside this API. Answered requests do
not appear: they are records, not work. Resolving twice is refused, because the
second note would replace the account of whoever did it.

### System

- `GET /health` - Health check endpoint

## Configuration

The application is configured through `DORNASHOP_`-prefixed environment
variables, which can be set in a `.env` file. Unprefixed variables such as
`DEBUG` and `SECRET_KEY` are intentionally ignored.

| Variable | Description | Default |
|----------|-------------|---------|
| `DORNASHOP_ENVIRONMENT` | Runtime environment: `development` or `production` | `development` |
| `DORNASHOP_DEBUG` | Application debug configuration flag | `false` |
| `DORNASHOP_SECRET_KEY` | JWT signing secret; production requires at least 32 characters and rejects the development default | Development-only value |
| `DORNASHOP_ALGORITHM` | JWT signing algorithm; pinned, as the minimum secret length is chosen for it | `HS256` |
| `DORNASHOP_JWT_ISSUER` | Expected JWT issuer | `dornashop-api` |
| `DORNASHOP_JWT_AUDIENCE` | Expected JWT audience | `dornashop-users` |
| `DORNASHOP_ACCESS_TOKEN_EXPIRE_MINUTES` | Access-token lifetime in minutes; must be positive | `15` |
| `DORNASHOP_REFRESH_TOKEN_EXPIRE_DAYS` | Refresh-token lifetime in days; must be positive | `7` |
| `DORNASHOP_CORS_ORIGINS` | JSON array or comma-separated allowed origins | Local ports `3000` and `8000` |
| `DORNASHOP_DB_ENGINE` | Database engine: `sqlite` or `postgresql` | `sqlite` |
| `DORNASHOP_DB_USER` | PostgreSQL user | `""` |
| `DORNASHOP_DB_PASSWORD` | PostgreSQL password | `""` |
| `DORNASHOP_DB_HOST` | PostgreSQL host | `""` |
| `DORNASHOP_DB_PORT` | PostgreSQL port, `1`-`65535` | `5432` when omitted |
| `DORNASHOP_DB_NAME` | Database name, or SQLite file path. Required for PostgreSQL | `db.sqlite3` for SQLite |

In production, set `DORNASHOP_ENVIRONMENT=production`, provide a unique secret,
and list explicit CORS origins. The application refuses to start rather than
serve traffic with an unsafe configuration. It rejects, at startup:

- the development secret, or any secret under 32 characters, in production;
- wildcard CORS origins in production;
- debug mode in production;
- PostgreSQL selected without a complete set of credentials;
- non-positive token lifetimes, out-of-range ports, and API prefixes the
  router would refuse.

## Roles

- `customer`: default role for customer-facing access.
- `support_agent`: staff role for support operations.
- `admin`: staff role with administrative authority.

Account activation is stored separately from role membership. Authorization
reloads the user from the database so that role and activation changes take
effect even while an older access token still exists.

## Development

### Running Tests

```bash
pytest
```

### Continuous Integration

Every push to `main` and every pull request runs the same gate as
`pre-commit`, plus two checks a local run cannot cover: the declared
dependency floors are installed and imported (`--resolution lowest-direct`),
and the migrations are applied and reversed. See
[.github/workflows/ci.yml](.github/workflows/ci.yml).

### Code Quality Tools

The project uses several tools to ensure code quality:

- **Ruff**: Linting, import sorting and formatting
- **mypy**: Static type checking
- **pre-commit**: Runs both on every commit

```bash
ruff check .          # lint
ruff format .         # format
mypy .                # type check
pre-commit install    # run all of the above on each commit
```

## Database

The application supports SQLite for development and PostgreSQL for production. The default is SQLite.

### Migrations

To create a new migration after changing models:

```bash
alembic revision --autogenerate -m "Description of changes"
```

To apply migrations:

```bash
alembic upgrade head
```

## Docker

The current Docker setup is intentionally development-oriented:

- `docker-compose.yml`: Baseline local/demo setup without auto-reload.
- `docker-compose.dev.yml`: Local development setup with source mounting and hot-reload.

The build context is filtered by `.dockerignore`: local environment files,
virtual environments, repository history, databases and generated caches are
never sent to the builder. Runtime assets remain available to the image.

`requirements.txt` is the locked production dependency set used by the image.
`requirements-dev.txt` adds the test, lint and type-check tools used during
development; those tools are not installed in the runtime image.

The image uses a multi-stage `python:3.11-slim` build. Dependencies are
installed outside the runtime stage, which receives only the virtual
environment, application code, migrations, startup script and served assets.

The runtime process uses the dedicated `dornashop` account with UID/GID 10001.
It owns the working directory so the local SQLite default can create its
database; dependencies and application files remain root-owned and read-only
to the process.

Docker probes `/health` from inside the container every 30 seconds, after a
10-second startup grace period. The probe uses Python's standard library, so
the image does not carry a separate HTTP client solely for health checks.

After migrations succeed, `start.sh` replaces its shell process with Uvicorn.
Uvicorn therefore runs as PID 1 and receives container termination signals
directly during a graceful stop.

The setup is not suitable for production as it stands: `docker-compose.yml`
bind-mounts the source tree. Known work before a production deployment:

- PostgreSQL with persistent storage and migrations run as a one-shot service.
- A production Compose file without source-code bind mounts.

## Known limitations

Deliberate, and recorded rather than hidden:

- **Staff authorization is defined but unused.** The `support_agent` and `admin`
  roles and the staff dependency exist; no endpoint requires them yet. They are
  in place for the support-agent work that follows.
- **Refresh tokens are never pruned.** Revoked and expired rows accumulate. A
  periodic cleanup is needed before this runs for any length of time.
- **API-token authentication writes on every request.** Each call updates
  `last_used_at`, so a read costs a write.
- **Logout requires a live access token.** A client whose access token has
  expired cannot revoke its still-valid refresh token without refreshing first.
- The Docker limitations listed above.

## Acknowledgements

Built on a FastAPI starter template by Clément Malige, provided as the
starting point for this project. The initial commit is that template
unmodified; everything after it is this project's work. The template
declared the MIT licence without including licence text, and that
declaration is left exactly as provided.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b ft/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin ft/my-feature`
