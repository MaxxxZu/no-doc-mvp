# no-doc-mvp
Norwegian document MVP (FastAPI + Hugging Face, uv-native)
FastAPI service for zero-shot classification and NER on Norwegian documents.
Ready for local run and deployment (Docker). Swagger UI available at /docs.

## Features

MVP service for **Norwegian** document processing:
- **/classify** — zero‑shot categorization via `NbAiLab/nb-bert-base-mnli`
- **/ner** — named-entity extraction (ORG/LOC/PER, etc.) via `Kushtrim/norbert3-large-ner`
- **/healthz** — status and active models
- Pydantic Settings (SettingsConfigDict), reads from .env/ENV (NDM_ prefix)
- Examples in Swagger schemas, CORS enabled

No GPU needed. CPU-only.

## Stack
- FastAPI, Uvicorn
- Hugging Face Transformers
- `uv` (dependency manager) with `pyproject.toml` + `uv.lock`
- Docker + docker-compose

## Quickstart (Docker Compose)

```bash
cp .env.example .env
uv lock            # generates uv.lock (optional but recommended)
docker compose build --no-cache
docker compose up
```

## Test calls

### health
```bash
curl http://localhost:8080/healthz
```

## classify
```bash
curl -s http://localhost:8080/classify -H "Content-Type: application/json" \
  -d '{"text":"Søknad om igangsettingstillatelse for nybygg ...","labels":["rammetillatelse","igangsettingstillatelse","annet"],"multi_label":false}'
```

### ner
```bash
curl -s http://localhost:8080/ner -H "Content-Type: application/json" \
  -d '{"text":"AF Gruppen er hovedentreprenør. Arkitekt er Snøhetta AS i Oslo. BRA 120 m², tomteareal 900 m²."}'
```

## Local run (without Docker)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
```

## Configuration

All settings are read from .env / environment (see config.py):
- NDM_ZSC_MODEL — zero‑shot model (default: NbAiLab/nb-bert-base-mnli)
- NDM_NER_MODEL — NER model (default: Kushtrim/norbert3-large-ner)
- NDM_ZSC_MULTI_LABEL — allow multiple labels (default: true)
- NDM_ZSC_MAX_LEN — max input chars (default: 2048)
- NDM_HF_HOME — HF cache (improves cold start)
- NDM_PORT — server port (default: 8080)
- NDM_CORS_ORIGINS=http://localhost:3000,https://my-ror-app.com

## Notes

- Models are cached in ./.hf (mounted to /app/.hf in container). You can delete it to free space; models will re-download on next start.
- If you need strictly CPU PyTorch wheels, adjust the Dockerfile as commented.


## Open:
- [Swagger](http://localhost:8080/docs)
- [ReDoc](http://localhost:8080/redoc)
- [Health](http://localhost:8080/healthz)


# 🚀 Deployment Guide ( Render, fastest & free)

## 1. Prepare the repo

- Push the `no-doc-mvp/` folder to a **public GitHub repository**.
- Make sure you have:
  - `Dockerfile`
  - `.env.example`

---

## 2. Create the service on Render

1. Log in to [render.com](https://render.com).
2. Go to **New → Web Service → Build & deploy from a Git repo**.
3. Select your repository and the desired branch.
4. Set the following options:
   - **Environment:** `Docker`
   - **Instance:** `Free (CPU)`
   - **Port:** Render will automatically set `$PORT`

---

## 3. Environment Variables

Add the following in **Environment Variables**:

```env
NDM_ZSC_MODEL=NbAiLab/nb-bert-base-mnli
NDM_NER_MODEL=Kushtrim/norbert3-large-ner
NDM_ZSC_MULTI_LABEL=false
# optional
NDM_HF_HOME=/app/.hf
NDM_CORS_ORIGINS=http://localhost:3000,https://my-ror-app.com
```

## 4. Health Check & Verification

- **Health Check Path:** `/healthz`
- After deployment, open `https://<your-app>.onrender.com/docs` in your browser — you can test the API directly there.

