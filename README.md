# no-doc-mvp
Norwegian document MVP (FastAPI + Hugging Face, uv-native)
FastAPI service for zero-shot classification and NER on Norwegian documents.
Local CPU inference verified on macOS arm64; Linux container verification is pending.
Swagger UI available at /docs.

## Features

MVP service for **Norwegian** document processing:
- **/classify** — zero‑shot categorization via `NbAiLab/nb-bert-base-mnli`
- **/ner** — named-entity extraction (ORG/LOC/PER, etc.) via `NbAiLab/nb-bert-base-ner`
- **/healthz** — status and active models
- Pydantic Settings (SettingsConfigDict), reads from .env/ENV (NDM_ prefix)
- Examples in Swagger schemas, CORS enabled

No GPU needed. CPU-only.

## Stack
- FastAPI, Uvicorn
- Python 3.11, PyTorch 2.10.0, NumPy 2.3.2, Transformers 5.10.1
- `uv==0.13.6` (dependency manager) with `pyproject.toml` + `uv.lock`
- Docker + docker-compose

## Quickstart (Docker Compose)

```bash
cp .env.example .env
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
uv sync --frozen
uv run --no-sync python app.py
```

## Real-model verification

Install `uv==0.13.6` in your tooling environment, then use `uv sync --frozen --extra dev`.
Fast tests (`uv run --no-sync pytest`) use fake models and never establish NLP accuracy.
The real smoke is a separate, explicit operation that downloads the two pinned models:

```bash
mkdir -p /tmp/no-doc-smoke-cache
uv run --no-sync python scripts/real_model_smoke.py \
  --cache-dir /tmp/no-doc-smoke-cache --report /tmp/no-doc-smoke.json
```

Use a dedicated cache directory. The script checks CPU loading, both zero-shot modes,
MNLI mapping, NER offsets, API-schema JSON serialization and tensor-to-NumPy, then
repeats loading and inference in a fresh offline process with socket access blocked.
It does not read `.env` or run through pytest fixtures. Model/tokenizer revisions are
shared with the service defaults. Both model cards declare CC BY 4.0; the
classification repository also contains an Apache 2.0 `LICENSE` at the pinned
revision. This licensing inconsistency remains unresolved; see the evidence report.
See [stage 2 evidence](docs/STAGE2_VERIFICATION.md) for exact versions, revisions,
source links, commands, warnings and platform limitations.

## Configuration

Settings use case-insensitive names in both process environment and `.env`.
Priority: constructor arguments > process environment > `.env` > defaults.
Within each source, `PORT` overrides `NDM_PORT`; otherwise `NDM_*` overrides
legacy names. For example, process `NDM_PORT` beats dotenv `PORT`.

Supported legacy aliases: `ZSC_MODEL`, `NER_MODEL`, `ZSC_MULTI_LABEL`,
`ZSC_MAX_LEN`, `HF_HOME`, and `PORT`. Unprefixed `HOST` and `CORS_ORIGINS`
are not supported. Unknown dotenv keys are ignored.

Configuration (see `config.py` and `.env.example`):

- NDM_ZSC_MODEL — zero‑shot model (default: NbAiLab/nb-bert-base-mnli)
- NDM_NER_MODEL — NER model (default: NbAiLab/nb-bert-base-ner)
- NDM_ZSC_REVISION / NDM_NER_REVISION — full 40-character commit SHAs (pinned defaults
  in `model_specs.py`; update the corresponding revision when overriding a model ID)
- NDM_ZSC_MULTI_LABEL — allow multiple labels (default: true)
- NDM_ZSC_MAX_LEN — max input chars (default: 2048)
- NDM_HF_HOME — HF cache (improves cold start)
- NDM_HOST — local server host (default: 0.0.0.0)
- NDM_PORT — server port, 1–65535 (default: 8080; PORT takes priority)
- NDM_CORS_ORIGINS — JSON array, e.g. `NDM_CORS_ORIGINS=["http://localhost:3000"]`

Compatibility changes: CORS now defaults to `[]` (no origins allowed), and an
explicit `[]` stays empty. CSV origins are rejected; use a JSON array. Wildcard
access requires explicitly setting `["*"]`. Ports outside 1–65535, nonpositive
`NDM_ZSC_MAX_LEN`, and blank model IDs fail configuration loading before startup.
Previously inconsistent ENV/dotenv alias and case handling is now uniform.

`make run` and `python app.py` use the configured host/port without environment
synchronization. Direct `uvicorn app:app` uses Uvicorn's own binding options;
Docker/Compose launch configuration is unchanged in this stage.

## Notes

- Models are cached in ./.hf (mounted to /app/.hf in container). You can delete it to free space; models will re-download on next start.
- The lock selects PyTorch CPU wheels for Linux x86_64 and PyPI arm64 wheels for macOS.
  Other platforms are outside the locked target set. Linux container execution is pending.


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
NDM_NER_MODEL=NbAiLab/nb-bert-base-ner
NDM_ZSC_MULTI_LABEL=false
# optional
NDM_HF_HOME=/app/.hf
NDM_CORS_ORIGINS=["http://localhost:3000","https://my-ror-app.com"]
```

## 4. Health Check & Verification

- **Health Check Path:** `/healthz`
- After deployment, open `https://<your-app>.onrender.com/docs` in your browser — you can test the API directly there.

