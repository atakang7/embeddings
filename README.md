# Embeddings Server

A minimal, high-throughput FastAPI service exposing a single `/embed` endpoint that returns normalized 32-bit float embeddings using a Sentence-Transformers model.

Optimized for cold-start time and low-latency inference. Suitable for production use behind a load balancer or internal service mesh.

## Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- pip dependencies:
  - `fastapi`
  - `uvicorn`
  - `sentence-transformers`
  - `torch`

Install:

```bash
pip install fastapi uvicorn sentence-transformers torch
```

## Usage

Start the server (defaults to port 8000, model: `all-mpnet-base-v2`):

```bash
python embeddingsserver.py
```

Customize via environment variables:

```bash
PORT=9000 EMBEDDING_MODEL=thenlper/gte-large python embeddingsserver.py
```

## API

### POST /embed

**Request:**

```json
{
  "texts": ["Hello world", "Another sentence"]
}
```

**Response:**

```json
{
  "embeddings": [[...], [...]]
}
```

Each entry is a 32-bit float vector (length depends on the model, e.g., 768).

## Design Notes

- Model is instantiated once at startup and remains in memory.
- Inference is batched and runs on GPU if available (`torch.cuda.is_available()`).
- Single-worker mode by default — scale horizontally with replicas if needed.
- Embeddings are normalized (L2).

## Example curl

```bash
curl -X POST http://localhost:8000/embed \
     -H "Content-Type: application/json" \
     -d '{"texts": ["FastAPI is fast", "Embeddings FTW"]}'
```

## Production Considerations

- Run behind a reverse proxy (e.g., Envoy or NGINX).
- Consider containerizing for orchestration (Docker/K8s).
- Use health checks to monitor model load success.

