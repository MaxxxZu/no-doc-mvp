# no-doc-mvp
Norwegian document MVP (FastAPI + Hugging Face, uv-native)

MVP service for **Norwegian** document processing:
- **/classify** — zero‑shot categorization via `NbAiLab/nb-bert-base-mnli`
- **/ner** — named-entity extraction (ORG/LOC/PER, etc.) via `Kushtrim/norbert3-large-ner`

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

```bash
curl http://localhost:8080/healthz

curl -s http://localhost:8080/classify -H "Content-Type: application/json" \
  -d '{"text":"Søknad om igangsettingstillatelse for nybygg ...","labels":["rammetillatelse","igangsettingstillatelse","annet"],"multi_label":false}'

curl -s http://localhost:8080/ner -H "Content-Type: application/json" \
  -d '{"text":"AF Gruppen er hovedentreprenør. Arkitekt er Snøhetta AS i Oslo."}'
```

## Local run (without Docker)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
```

## Configuration

All settings are read from .env / environment (see config.py):
- ZSC_MODEL — zero‑shot model (default: NbAiLab/nb-bert-base-mnli)
- NER_MODEL — NER model (default: Kushtrim/norbert3-large-ner)
- ZSC_MULTI_LABEL — allow multiple labels (default: true)
- ZSC_MAX_LEN — max input chars (default: 2048)
- HF_HOME, TRANSFORMERS_CACHE — HF caches (improves cold start)
- PORT — server port (default: 8080)

## Notes

- Models are cached in ./.hf (mounted to /app/.hf in container). You can delete it to free space; models will re-download on next start.
- If you need strictly CPU PyTorch wheels, adjust the Dockerfile as commented.
