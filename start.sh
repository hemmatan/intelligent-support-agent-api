#!/bin/sh

# Run DB migrations
alembic upgrade head

# Replace the shell so Uvicorn receives container stop signals directly.
exec uvicorn main:app --host 0.0.0.0 --port 8000
