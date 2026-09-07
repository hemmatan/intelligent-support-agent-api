FROM ghcr.io/astral-sh/uv:0.11.15 AS uv

FROM python:3.11-slim AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DORNASHOP_DB_NAME=/var/lib/dornashop/db.sqlite3 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN groupadd --gid 10001 dornashop \
    && useradd --uid 10001 --gid 10001 --no-create-home \
        --home-dir /app --shell /usr/sbin/nologin --no-log-init dornashop \
    && install --directory --owner=dornashop --group=dornashop \
        /var/lib/dornashop

COPY --from=builder /opt/venv /opt/venv
COPY alembic ./alembic
COPY app ./app
COPY assets ./assets
COPY alembic.ini main.py start.sh ./

RUN chmod 0555 start.sh

EXPOSE 8000

USER dornashop

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).close()"]

CMD ["./start.sh"]
