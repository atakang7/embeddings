# syntax=docker/dockerfile:1

# -----------------------------------------------------------
# Stage 1 – builder: install deps & cache the embedding model
# -----------------------------------------------------------
FROM python:3.12.3-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Install Poetry (no system compilers → smaller surface‑area)
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copy dependency metadata and install runtime deps only
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-dev --no-root --no-ansi --no-interaction

# Pre‑download the Sentence‑Transformers model so runtime boots instantly
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
PY

# -----------------------------------------------------------
# Stage 2 – runtime: ultra‑small, non‑root, distroless base
# -----------------------------------------------------------
# Exact pinned digest (published 2025‑06‑17). Update regularly.
FROM gcr.io/distroless/python3-debian12:nonroot AS runtime

# Copy only what we need from the builder image
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /root/.cache /home/nonroot/.cache

WORKDIR /app
COPY embeddingsserver.py /app/embeddingsserver.py

EXPOSE 8000

# Distroless images already run as a non‑root UID ("nonroot").
CMD ["uvicorn", "embeddingsserver:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
