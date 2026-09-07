#!/bin/sh

set -e

# This convenience path is for the single-replica development stack only.
alembic upgrade head

# Replace the shell so the reload process receives container stop signals.
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
