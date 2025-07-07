#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run.sh – Convenience launcher for the FastAPI embeddings service
# -----------------------------------------------------------------------------
# * Installs dependencies via Poetry if a venv isn’t present.
# * Pre‑downloads the embedding model (first run only) so the API starts fast.
# * Exposes a couple of tunables via env vars (PORT, WORKERS, MODEL).
#
# Usage:
#   chmod +x run.sh
#   ./run.sh            # starts on http://localhost:8000
#   PORT=9000 WORKERS=4 ./run.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# -------- Configurable env vars --------
: "${PORT:=8000}"        # Listener port
: "${WORKERS:=2}"       # Uvicorn workers
: "${MODEL:=all-MiniLM-L6-v2}"  # Embedding model ID

# -------- Ensure Poetry exists --------
if ! command -v poetry &>/dev/null; then
  echo "[run.sh] Poetry not found – installing..." >&2
  pip install --no-cache-dir "poetry==1.8.2"
fi

# -------- Install project deps (if first run) --------
if ! poetry env info --path &>/dev/null; then
  echo "[run.sh] Installing Python dependencies via Poetry..." >&2
  poetry install --no-root --no-interaction --no-ansi
fi

# -------- Pre‑download the embedding model --------
python - <<PY
from sentence_transformers import SentenceTransformer
SentenceTransformer('${MODEL}')
PY

# -------- Launch Uvicorn (production‑ish defaults) --------
exec poetry run uvicorn embeddingsserver:app \
  --host 0.0.0.0 --port "${PORT}" --workers "${WORKERS}"
