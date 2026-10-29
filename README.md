# no-doc-mvp

A FastAPI service for zero-shot classification and named entity recognition on
Norwegian text. It applies pretrained Hugging Face models on CPU, with an
application-owned model lifecycle and bounded inference concurrency.

The default classification labels cover Norwegian planning and construction
documents. Clients can also provide their own candidate labels.

## Architecture

- **API:** FastAPI routes and Pydantic request/response schemas.
- **Services:** classification and entity-extraction logic.
- **Runtime:** sequential startup loading, readiness, a shared inference slot and
  graceful shutdown.
- **Models:** pinned BERT/MNLI and NER artifacts loaded with fast tokenizers,
  safetensors and remote code disabled.
- **Configuration:** Pydantic Settings, environment variables and optional `.env`.

Python 3.11, PyTorch 2.10.0, NumPy 2.3.2 and Transformers 5.10.1 are locked with
`uv==0.13.6`. The lock targets macOS arm64 and Linux x86_64; Linux selects PyTorch
CPU wheels. Other platforms are outside the current locked target set.

## Local quickstart

Prerequisites: Python 3.11, `uv==0.13.6`, network access for the first model download,
and disk/RAM capacity for both models. Run these commands from the repository root:

```bash
uv sync --frozen
NDM_HF_HOME=.hf uv run --no-sync python app.py
```

The service loads both pinned models before accepting requests. Later starts reuse
cached artifacts. The local launcher uses the configured host and port, defaulting
to `0.0.0.0:8080`, with one Uvicorn worker.

Configuration can also be placed in `.env`; see [.env.example](.env.example).
For local use, choose a writable cache path such as `.hf` rather than the template's
container path `/app/.hf`.

- [Swagger UI](http://localhost:8080/docs)
- [ReDoc](http://localhost:8080/redoc)
- [Liveness](http://localhost:8080/healthz)
- [Readiness](http://localhost:8080/readyz)

## API

Inputs are extracted text supplied as JSON. Examples below are illustrative inputs;
model quality has not been evaluated on a labeled benchmark.

| Endpoint | Purpose |
| --- | --- |
| `POST /classify` | Ranked candidate labels, scores and the top result |
| `POST /ner` | Entity groups, text, scores and character offsets |
| `GET /healthz` | Cheap liveness and configured model IDs |
| `GET /readyz` | 200 when both required models are ready; otherwise 503 |

```bash
curl http://localhost:8080/healthz

curl http://localhost:8080/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Oslo kommune behandler en søknad om byggetillatelse.","labels":["byggesøknad","sport"],"multi_label":false}'

curl http://localhost:8080/ner \
  -H 'Content-Type: application/json' \
  -d '{"text":"Ola Nordmann arbeider i Oslo kommune i Norge."}'
```

Both inference routes share one slot per process. Busy requests receive 503 with
`Retry-After: 1`. CPU work runs in a worker thread; cancelling an HTTP request does
not stop a running PyTorch call. Shutdown waits for active inference before clearing
runtime references. Immediate return of all RAM to the OS is not guaranteed.

## Configuration

Names are case-insensitive in process environment and `.env`.
Priority: constructor arguments > process environment > `.env` > defaults.
Within each source, `PORT` overrides `NDM_PORT`; other `NDM_*` names override legacy
aliases. Process `NDM_PORT`, for example, takes precedence over dotenv `PORT`.

| Variable | Default / meaning |
| --- | --- |
| `NDM_HOST` | `0.0.0.0`; local launcher binding |
| `NDM_PORT` / `PORT` | `8080`; range 1–65535 |
| `NDM_CORS_ORIGINS` | `[]`; JSON array of allowed origins |
| `NDM_HF_HOME` | Unset; artifact cache directory passed directly as `cache_dir` |
| `NDM_OFFLINE` | `false`; when true, incomplete cache fails startup |
| `NDM_CPU_THREADS` | `2`; PyTorch intra-op threads, range 1–256 |
| `NDM_ZSC_MODEL` | `NbAiLab/nb-bert-base-mnli` |
| `NDM_NER_MODEL` | `NbAiLab/nb-bert-base-ner` |
| `NDM_ZSC_REVISION` / `NDM_NER_REVISION` | Full commit SHAs pinned in [model_specs.py](model_specs.py) |
| `NDM_ZSC_MULTI_LABEL` | `true`; request-level override supported |
| `NDM_ZSC_MAX_LEN` | `2048`; maximum original classification text length in code points |
| `NDM_NER_MAX_LEN` | `2048`; maximum original NER text length in code points |
| `NDM_BODY_MAX_BYTES` | `65536`; maximum received HTTP body bytes |

Supported legacy aliases are `ZSC_MODEL`, `NER_MODEL`, `ZSC_MULTI_LABEL`,
`ZSC_MAX_LEN`, `HF_HOME` and `PORT`. Unprefixed `HOST` and `CORS_ORIGINS` are ignored.
Unknown dotenv keys are ignored. Invalid ports, nonpositive classification limits,
blank model IDs/cache paths and malformed revisions fail configuration loading.

CORS example: `NDM_CORS_ORIGINS=["http://localhost:3000"]`. An empty list permits no
cross-origin clients; wildcard access must be explicit. CSV and `null` are rejected.
CORS is a browser policy, not API authentication.

`NDM_HF_HOME` and its legacy alias select the artifact directory itself: no `hub/`
suffix is appended. To reuse an older cache at `<HF_HOME>/hub`, select that directory
directly. CPU thread policy is process-wide; apps in one process must agree.
Changing a model ID requires a matching reviewed revision and a fresh model check.

`make run` uses the same local launcher. Direct `uvicorn app:app` uses Uvicorn's
own host/port options.

## Tests and reproducible smoke checks

```bash
uv sync --frozen --extra dev
uv run --no-sync pytest
uv run --no-sync flake8 .
uv run --no-sync isort . --check-only --filter-files
```

Fast tests use fake models and cover configuration, CORS, lifecycle, concurrent
admission, cancellation, shutdown and API behavior. They do not download weights
or establish NLP accuracy.

The separate model smoke downloads the pinned models into a dedicated cache,
checks CPU inference, both zero-shot modes, MNLI mapping, NER offsets and JSON
serialization, then repeats loading/inference in a fresh offline process:

```bash
mkdir -p /tmp/no-doc-smoke-cache
uv run --no-sync python scripts/real_model_smoke.py \
  --cache-dir /tmp/no-doc-smoke-cache --report /tmp/no-doc-smoke.json
```

After this cache has been populated, check the application lifecycle and both API
endpoints offline without starting a listening server:

```bash
uv run --no-sync python scripts/runtime_offline_smoke.py \
  --cache-dir /tmp/no-doc-smoke-cache/artifacts --report /tmp/no-doc-runtime-smoke.json
```

Both commands write reports outside the repository. Local verification on macOS
arm64 with Python 3.11 covered real CPU inference, offline cache reuse, application
startup/readiness and graceful shutdown. Offline smoke checks block socket/DNS
access and require local model artifacts.

## Current limitations

- Text is passed unchanged to inference. NER offsets are Python Unicode code point
  indices into the original text, with inclusive `start` and exclusive `end`;
  `word` is the corresponding original substring.
- Both endpoints reject text above their character limits. Tokenization without
  truncation checks each classification document/hypothesis pair and each NER
  document, including special tokens, against the model and tokenizer capacity.
  Unsupported model capacity fails inference. Chunking is not implemented.
- Missing or null labels select defaults; otherwise provide 1–32 unique labels,
  each 1–128 characters after trimming outer whitespace. Labels retain case.
  The 2048-character limits bound preprocessing, while the 65536-byte body limit
  leaves room for JSON encoding and labels; token limits can reject shorter text.
- Errors use `{"error":{"code":"...","message":"..."}}`: invalid input and
  `text_too_long` use 422, `body_too_large` uses 413, `runtime_unavailable` and
  `inference_busy` use 503 with `Retry-After`, and `inference_failed` uses 500.
  Invalid JSON or schemas use `invalid_input` (422).
- Model accuracy, latency and memory requirements have not been benchmarked.
  Successful smoke checks establish operability, not quality metrics.
- Docker and Compose configurations are included, but container startup, cache
  permissions and Linux inference have not been verified. Use the local path above
  for the currently verified workflow; hosting tier suitability is unmeasured.

## License and model terms

The project source is licensed under [MIT](LICENSE). Model terms are separate;
model weights are downloaded at runtime and are not included in this repository.

- [NER model card](https://huggingface.co/NbAiLab/nb-bert-base-ner/blob/98083f61535300a7fb6d8b72c26873d47e81ac93/README.md): CC BY 4.0.
- The classifier [model card](https://huggingface.co/NbAiLab/nb-bert-base-mnli/blob/171cd472d61d91ad6609530997e63ea98515efcb/README.md)
  declares CC BY 4.0, while its [LICENSE](https://huggingface.co/NbAiLab/nb-bert-base-mnli/blob/171cd472d61d91ad6609530997e63ea98515efcb/LICENSE)
  contains Apache 2.0 at the same revision. This inconsistency remains unresolved;
  no established dual-license grant or fully cleared model-usage rights are claimed.
