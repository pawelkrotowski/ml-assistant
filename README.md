# prodtramtsty — RAG Assistant (FastAPI · Postgres/pgvector · Ollama/Azure)

**prodtramtsty** is a minimalist, production‑ready Retrieval‑Augmented Generation (RAG) service.
It ingests your documents, builds vector embeddings, retrieves the most relevant chunks, and answers questions with an LLM (Ollama by default; Azure OpenAI optional).

---

## ✨ Features

- **End‑to‑end RAG**: ingest → chunk → embed → vector search → generate
- **Database**: PostgreSQL 16 + **pgvector** (ANN via IVFFLAT)
- **Embeddings (local)**: `BAAI/bge-small-en-v1.5` (384‑d) via `sentence-transformers`
- **LLM providers**:
  - **Ollama** (default): streaming responses, retry logic
  - **Azure OpenAI** (Chat Completions)
- **FastAPI** endpoints: `/ingest`, `/chat`, `/stats`, `/health`
- **Citations toggle**: enable/disable inline `[1] [2] …` references
- **MMR re‑ranking** (optional) to diversify retrieved context
- **Docker Compose** stack with healthchecks and hot‑reload for API

---

## 🧱 Architecture

```
         ┌─────────────┐    upload            ┌─────────────┐
User/UI ─▶   FastAPI   ├──────────────────────▶   /ingest    │
         │   (API)     │                      └─────┬───────┘
         └────┬────────┘                            │
              │  embed_texts()                      │ chunks + embeddings
              │                                      ▼
              │                               ┌─────────────┐
              │        top‑k cosine           │ Postgres +  │
  /chat ──────┼──────────────────────────────▶│  pgvector   │
              │                               └─────────────┘
              │ build prompt (contexts)               │
              ▼                                       
        LLM provider (Ollama/Azure)  ◀───────────────┘
              │  streaming output
              ▼
           Answer JSON
```

---

## 📁 Repository Layout

```
ml-assistant/
  api/
    main.py          # FastAPI app (ingest, chat, stats, health)
    rag.py           # prompt builder + LLM clients (Ollama/Azure)
    embed.py         # local embeddings (bge-small)
    chunking.py      # PDF/TXT/MD -> chunks
    db.py            # SQLAlchemy models & pgvector indexes
  db-init/
    001-extensions.sql   # CREATE EXTENSION vector
  docker-compose.yml
  requirements.txt
  .env.example
  README.md
```

---

## 🚀 Quick Start (Docker Compose)

1) **Configure environment**
```bash
cp .env.example .env
# minimally set:
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://ollama:11434
# OLLAMA_MODEL=phi3:mini            # good default (CPU); use llama3.1:8b with GPU
# DATABASE_URL=postgresql+psycopg://postgres:postgres@db:5432/ml_assistant
```

2) **Start services**
```bash
docker compose up -d
```

3) **Pull a model inside the Ollama container**
```bash
docker exec -it ml_ollama ollama pull phi3:mini
# or (once GPU is enabled): ollama pull llama3.1:8b
```

4) **Health check**
```bash
curl http://localhost:8000/health
```

5) **Ingest a document**
```bash
curl -F "file=@./your.pdf" http://localhost:8000/ingest
# -> {"ok": true, "file": "your.pdf", "chunks": N}
```

6) **Ask a question**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize key points.","k":4,"language":"en"}'
```

---

## 💻 Local Development (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run Postgres with pgvector yourself (or via Docker)
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ml_assistant"

# Use local Ollama daemon on host
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434

uvicorn api.main:app --reload
```

---

## 🔌 API Reference

### `POST /ingest`
Uploads one file (PDF/TXT/MD). The service chunks, embeds, and stores vectors.

- **Form-data**: `file` (required)
- **Response**
  ```json
  { "ok": true, "file": "your.pdf", "chunks": 42 }
  ```

### `POST /chat`
Ask a question; the server retrieves vector top‑k chunks, builds a RAG prompt, and calls the LLM.

- **Request**
  ```json
  {
    "message": "List 3 bird species and one fact each.",
    "k": 6,
    "language": "en",
    "citations": false   // optional; overrides global setting
  }
  ```
- **Response**
  ```json
  {
    "answer": "...",
    "contexts": [
      { "source": "your.pdf", "index": 12, "content": "..." },
      { "source": "your.pdf", "index": 7,  "content": "..." }
    ]
  }
  ```

### `GET /stats`
Returns counts of documents and chunks.
```json
{ "documents": 3, "chunks": 128 }
```

### `GET /health`
Liveness/readiness probe.
```json
{ "status": "ok" }
```

---

## ⚙️ Configuration (Environment Variables)

**Core**
- `DATABASE_URL` — SQLAlchemy URL to Postgres (with pgvector)
- `LLM_PROVIDER` — `ollama` | `azure`
- `MAX_CONTEXT_CHARS` — clamp each chunk length in prompt (default `1600`)
- `RAG_CITATIONS` — `1` (default) to enable `[n]` markers, `0` to disable

**Ollama**
- `OLLAMA_BASE_URL` — `http://ollama:11434` (in Compose) or `http://host.docker.internal:11434`
- `OLLAMA_MODEL` — e.g., `phi3:mini`, `qwen2.5:3b-instruct`, `llama3.1:8b`
- `OLLAMA_NUM_PREDICT` — max generated tokens (default `256`)
- `LLM_TIMEOUT` — read timeout in seconds (default `300`)

**Azure OpenAI**
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT` (e.g., `gpt-4o-mini`)
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (default `2024-02-15-preview`)

**Embeddings**
- Local model: `BAAI/bge-small-en-v1.5` (384‑d).  
  If you change the embedding model, also update `EMBED_DIM` in `api/db.py`.

---

## 🟩 GPU with Ollama (Docker)

1. On host, verify drivers:
   ```bash
   nvidia-smi
   ```
2. Install **NVIDIA Container Toolkit** and verify Docker GPU:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
3. In `docker-compose.yml` under `ollama`:
   ```yaml
   gpus: all
   environment:
     NVIDIA_VISIBLE_DEVICES: "all"
     NVIDIA_DRIVER_CAPABILITIES: "compute,utility"
     OLLAMA_KEEP_ALIVE: "4h"
   ```
4. Recreate service:
   ```bash
   docker compose up -d --force-recreate --no-deps ollama
   ```
> The `ollama/ollama` image doesn’t include `nvidia-smi`. Use the CUDA test image above to validate GPU access.

**Alternative**: run Ollama on the host (GPU) and set:
```
OLLAMA_BASE_URL=http://host.docker.internal:11434
```
plus:
```yaml
api:
  extra_hosts: ["host.docker.internal:host-gateway"]
```

---

## ⚡ Performance Tips

- CPU: use a **small model** (`phi3:mini`, `qwen2.5:3b-instruct`).
- GPU: step up to `llama3.1:8b`.
- Keep `k` small (3–6) and clamp chunks with `MAX_CONTEXT_CHARS` (e.g., `1200`).
- Limit generation: `OLLAMA_NUM_PREDICT=256`.
- After bulk ingest:
  ```sql
  VACUUM ANALYZE chunks;
  -- For large collections, recreate IVFFLAT with higher `lists` (~sqrt(N))
  ```
- Enable MMR re‑ranking to reduce redundancy in retrieved chunks.

---

## 🛡️ Security & Privacy

- Add an API key header (e.g., `X-API-Key`) and validate it in `main.py`.
- Restrict uploads (size/type) and deduplicate by file hash.
- Log minimal PII, consider encryption at rest for sensitive data.
- For multi‑tenant scenarios, add `tenant_id` to tables and filter in queries.

---

## 🗺️ Roadmap

- Toggle for Azure/OpenAI embeddings
- Multi-tenant namespaces
- Better PDF parsing (tables/images)
- Minimal web UI (Next.js) for upload/chat/context
- Eval suite & monitoring dashboards
- CI/CD, container hardening

---

## 🤝 Contributing

Issues and PRs are welcome. Please include:
- Repro steps
- Environment details (OS, Docker, GPU, `.env` minus secrets)
- Logs and stack traces

---

## 📄 License

Choose a license (e.g., MIT) before publishing publicly.

---

## ☕ Quick Commands

```bash
# Health
curl http://localhost:8000/health

# Ingest
curl -F "file=@./your.pdf" http://localhost:8000/ingest

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize the document.","k":4,"language":"en"}'

# Stats
curl http://localhost:8000/stats
```
