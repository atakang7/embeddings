"""embeddingsserver.py
A minimal FastAPI service that exposes a single /embed endpoint to generate text
embeddings. The Sentence‑Transformers model is loaded once at startup and kept
in memory for ultra‑fast requests.

Usage:
    # Install dependencies
    pip install fastapi uvicorn sentence-transformers torch

    # Run the server (default port 8000)
    python embeddingsserver.py

    # Or change the port / model with environment variables
    PORT=9000 EMBEDDING_MODEL=thenlper/gte-large python embeddingsserver.py

Request example (curl):
    curl -X POST http://localhost:8000/embed \
         -H "Content-Type: application/json" \
         -d '{"texts": ["Hello world", "Fast embeddings"]}'
"""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import uvicorn

# ---------------------------------------------------------------------------
# Model loading — happens **once** at process start.
# ---------------------------------------------------------------------------
MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
except Exception as ex:  # pragma: no cover
    # Fail fast if model cannot be loaded.
    raise RuntimeError(f"Failed to load embedding model '{MODEL_NAME}': {ex}") from ex

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Embeddings Server", version="1.0.0")

class TextsRequest(BaseModel):
    """Input schema expecting a non‑empty list of texts."""

    texts: List[str]

class EmbeddingsResponse(BaseModel):
    """Response schema returning a list of embeddings in row‑major order."""

    embeddings: List[List[float]]

@app.post("/embed", response_model=EmbeddingsResponse)
async def embed(req: TextsRequest) -> EmbeddingsResponse:  # noqa: D401
    """Return embeddings for the provided *texts* as 32‑bit floats."""

    if not req.texts:  # Guard against empty payloads.
        raise HTTPException(status_code=400, detail="'texts' list must not be empty")

    # Batch‑encode (no gradients, GPU if available).
    with torch.inference_mode():
        vectors = model.encode(
            req.texts,
            batch_size=min(32, len(req.texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

    return EmbeddingsResponse(embeddings=vectors)

# ---------------------------------------------------------------------------
# Entry‑point helper so the file can be executed directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port: int = int(os.getenv("PORT", "8000"))
    # Using a single worker avoids multiple model loads. Scale horizontally with
    # container replicas if you need more throughput.
    uvicorn.run(
        "embeddingsserver:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
    )
