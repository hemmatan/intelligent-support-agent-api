FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv /opt/venv
COPY requirements.txt /tmp/requirements.txt
RUN pip install --requirement /tmp/requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY alembic ./alembic
COPY app ./app
COPY assets ./assets
COPY alembic.ini main.py start.sh ./

RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]
