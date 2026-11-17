FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend:/app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /tmp/backend-requirements.txt
COPY ml/requirements.txt /tmp/ml-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/backend-requirements.txt -r /tmp/ml-requirements.txt
COPY backend ./backend
COPY ml ./ml
COPY alembic.ini ./alembic.ini
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/models && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "alembic -c alembic.ini upgrade head && uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000"]
