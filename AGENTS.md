# Contributor instructions

## Project structure

This project provides zero-shot classification and named entity recognition for
Norwegian text using FastAPI and Hugging Face Transformers.

- `app.py`: application factory, routes and lifespan.
- `config.py`: validated application settings.
- `models/`: Pydantic HTTP request and response schemas.
- `services/`: inference services, artifact loading and model runtime.
- `model_specs.py`: default model IDs and pinned revisions.
- `tests/`: fast tests using fake model dependencies.
- `scripts/`: explicitly invoked real-model smoke checks.

## Engineering conventions

- Preserve separation between HTTP handling, inference logic and model lifecycle.
  Prefer small changes consistent with the existing architecture.
- Keep models owned by the application runtime; do not load them at import time
  or construct new pipelines per request.
- Preserve explicit revisions, CPU execution, safetensors loading and disabled
  remote code. Review model IDs and revisions together when changing artifacts.
- Keep the default fast suite independent of model downloads and external network.
  Separate real-model checks from mocked unit and integration tests.
- With development dependencies available, run the relevant tests, Flake8 and
  isort checks using the existing environment without implicit dependency sync.
- Keep code and public documentation in English. Document actual behavior and
  relevant limitations; report only verification that was actually performed.
- Preserve unrelated changes and keep documentation aligned with affected APIs,
  settings and launch commands.

## Repository hygiene

- Do not include secrets, private documents or personal machine paths in source,
  logs, examples or documentation. Use synthetic test data and portable commands.
- Keep environment files, caches, downloaded weights and generated verification
  reports out of Git and container build contexts. Preserve the safe `.env.example`.
- Public instructions must not depend on private files or link to ignored reports.
- Treat project-source and model licenses separately; retain model attribution
  and document unresolved model terms without assuming a license grant.
