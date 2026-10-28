# Stage 2: dependencies and early model verification

Status: implemented; awaiting independent review. Evidence date: 2026-10-28.
Baseline HEAD: `50558e2`. Actual branch: `chore--Verified-nlp-stack`, differing
from requested `chore/verified-nlp-stack`; no branch operations performed.

## Changes and version decisions

| Component | Before (project lock/tool declarations) | After | Reason |
| --- | --- | --- | --- |
| Python | 3.11 | 3.11 (executed: 3.11.11) | No compatibility obstacle |
| NumPy | 2.3.2 (transitive) | 2.3.2 (explicit) | Baseline tensor-to-NumPy and inference passed; no ABI failure observed on macOS |
| PyTorch | 2.3.1 | 2.10.0 macOS / 2.10.0+cpu Linux | Minimum patched release for upstream GHSA-63cw-57p8-fm3p; dedicated Linux CPU index |
| Transformers | 4.43.3 | 5.10.1 | Known loading/configuration vulnerabilities; first non-yanked patch of the 5.10 security floor reviewed below |
| tokenizers | 0.19.1 | 0.22.2 | Transformers constraint |
| huggingface-hub | 0.34.4 | 1.33.0 | Transformers constraint |
| safetensors | 0.6.2 | 0.6.2 | Existing lock version retained; loaders require safetensors |
| uv | Docker 0.4.20; CI unpinned; local 0.5.7 | 0.13.6 everywhere declared | Exact required-version, Docker tool tag and CI installer agree |

The initial environment installed the exact three baseline core pins separately,
not the entire old lock: torch 2.3.1, NumPy 2.3.2, Transformers 4.43.3. Its resolved
transitives (including Hub 0.36.2 and safetensors 0.8.0) are recorded in baseline
JSON reports. This is evidence for the core tuple on this platform, not proof of
an old frozen installation on every platform. Imports, a tiny random BERT forward,
tensor.numpy(), and both real selected model inferences passed. NumPy was not downgraded.
The final environment was installed from the actual final lock with `--frozen`.
No `--upgrade` or blanket refresh was requested; necessary transitive changes and
platform pruning account for the lock diff. Web/API dependencies retained their pins.

Security sources and scope:

- [PyTorch upstream advisory GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6): torch <=2.5.1 affected; 2.6.0 fixes an earlier weights-only RCE.
- [PyTorch upstream advisory GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p): <=2.9.1 affected; >=2.10.0 patched. Merely moving to 2.6 would leave this known issue.
- [Transformers upstream configuration fix](https://github.com/huggingface/transformers/commit/0720e206c6ba28887e4d60ef60a6a089f6c1cc76) addresses the shared config-file ReDoS path ([GHSA-q2wp-rjmx-x6x9](https://github.com/advisories/GHSA-q2wp-rjmx-x6x9), patched in 4.51.0).
- [Transformers upstream model-loading fix](https://github.com/huggingface/transformers/commit/a7f8e7ff37d87d1a1a0c8cf607971c607741452f) addresses malicious attention config ([GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72), patched in 5.3.0).
- [Transformers upstream tokenizer save fix](https://github.com/huggingface/transformers/commit/eaaaf8494dd5386634ae37d1d122212fdc315be5) addresses chat-template path traversal ([GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w), patched in 5.10.0). This path is not used by this BERT inference service; choosing the patched line avoids carrying the known issue into local tooling. Transformers 5.10.0 is [yanked on PyPI](https://pypi.org/pypi/transformers/5.10.0/json) for an erroneous old-main release; 5.10.1 is not yanked.

This targeted review does not establish that the complete dependency graph has no
vulnerabilities. HTTP-stack security review/upgrades are outside this stage.
Revisions, no remote code, and safetensors-only loading further restrict the
model-loading surface. Changing model IDs/revisions requires a new review/smoke.

## Model selection

| Task | ID | Full revision | Declared license |
| --- | --- | --- | --- |
| Zero-shot | NbAiLab/nb-bert-base-mnli | 171cd472d61d91ad6609530997e63ea98515efcb | Card: CC BY 4.0; LICENSE file: Apache 2.0 (unresolved conflict) |
| NER | NbAiLab/nb-bert-base-ner | 98083f61535300a7fb6d8b72c26873d47e81ac93 | CC BY 4.0 |

Primary sources:

- Classification [model card](https://huggingface.co/NbAiLab/nb-bert-base-mnli/blob/171cd472d61d91ad6609530997e63ea98515efcb/README.md), [config](https://huggingface.co/NbAiLab/nb-bert-base-mnli/blob/171cd472d61d91ad6609530997e63ea98515efcb/config.json), [tokenizer config](https://huggingface.co/NbAiLab/nb-bert-base-mnli/blob/171cd472d61d91ad6609530997e63ea98515efcb/tokenizer_config.json).
- NER [model card](https://huggingface.co/NbAiLab/nb-bert-base-ner/blob/98083f61535300a7fb6d8b72c26873d47e81ac93/README.md), [config](https://huggingface.co/NbAiLab/nb-bert-base-ner/blob/98083f61535300a7fb6d8b72c26873d47e81ac93/config.json), [tokenizer config](https://huggingface.co/NbAiLab/nb-bert-base-ner/blob/98083f61535300a7fb6d8b72c26873d47e81ac93/tokenizer_config.json).
- Both cards declare [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The classification repository separately contains an [Apache 2.0 LICENSE](https://huggingface.co/NbAiLab/nb-bert-base-mnli/blob/171cd472d61d91ad6609530997e63ea98515efcb/LICENSE) at the same revision. This is conflicting source evidence, not an established dual-license grant. The authoritative licensing scope needs clarification from the model publisher; no claim of fully resolved classification licensing is made. The NER repository inventory has no separate LICENSE file; its card declares CC BY 4.0. Both repositories were publicly downloadable without credentials; access does not resolve licensing. Retain attribution and license references when sharing under the declared terms. Do not infer suitability/accuracy from licensing.

Both selected repositories were anonymously downloadable at the full revisions.
They use built-in `BertForSequenceClassification` / `BertForTokenClassification`.
WordPiece vocab/tokenizer assets load as fast BERT tokenizers with the pinned
Transformers/tokenizers stack. No SentencePiece, protobuf, pandas or custom Python
model code is required. Both have safetensors weights. Position capacity is 512;
long-text behavior is not implemented or established by this short smoke.

MNLI mapping: contradiction=0, neutral=1, entailment=2. Harness checks config mapping
and pipeline entailment ID before testing single-label and multi-label outputs.
NER config has O plus B/I labels PER, ORG, GPE_LOC, PROD, LOC, GPE_ORG, DRV, EVT,
MISC. Fast tokenizer offset_mapping and aggregated entity offsets were verified.
The card suggests `first`; existing service aggregation `simple` is retained.

The original default [Kushtrim/norbert3-large-ner card](https://huggingface.co/Kushtrim/norbert3-large-ner/blob/6d6a6b3df1aa9ace8eba4c1a9876d24a8ff34e93/README.md)
(API SHA `6d6a6b3df1aa9ace8eba4c1a9876d24a8ff34e93`) declares CC BY 4.0 and documents
`trust_remote_code=True` for both tokenizer and model; repository inventory includes
`configuration_norbert.py` and `modeling_norbert.py`. Config retrieval returned HTTP
401 at both the pinned raw URL and main resolve URL, although the API and main
README were accessible. Architecture/config compatibility could therefore not be
fully verified. No credentials were sought and its weights were not downloaded.
The already-proposed example NER is an accessible built-in BERT alternative.
No assertion is made that NorBERT can never work or that the replacement is more accurate.

`model_specs.py` is the shared source of reviewed default artifacts. Settings add
`NDM_ZSC_REVISION` / `NDM_NER_REVISION`, requiring lowercase full commit SHAs.
Overriding a model ID also requires selecting its matching revision. Services pass
explicit model and tokenizer IDs plus `revision` into pipeline; the inspected
Transformers loader passes that revision to both model and tokenizer loaders.
Fast spy tests verify arguments. A separate real offline service check verified
both loaded config commit hashes and fast tokenizer construction.

## Commands and executed checks

All environments/caches below were new paths under `/private/tmp/no-doc-stage2`.
The existing `.venv`, `.env`, system installations and existing model caches were
not modified. Bootstrap `uv` was the existing 0.5.7; the final tool is isolated.
Commands below run from repository root unless stated otherwise.

```bash
# Bootstrap original-core and tooling environments (no project sync).
uv venv --python /Users/maksim/.local/share/uv/python/cpython-3.11.11-macos-aarch64-none/bin/python3.11 /private/tmp/no-doc-stage2/baseline
UV_CACHE_DIR=/private/tmp/no-doc-stage2/uv-cache uv pip install --python /private/tmp/no-doc-stage2/baseline/bin/python torch==2.3.1 numpy==2.3.2 transformers==4.43.3
UV_CACHE_DIR=/private/tmp/no-doc-stage2/uv-cache uv pip install --python /private/tmp/no-doc-stage2/baseline/bin/python pydantic==2.8.2
uv venv --python /Users/maksim/.local/share/uv/python/cpython-3.11.11-macos-aarch64-none/bin/python3.11 /private/tmp/no-doc-stage2/tooling
UV_CACHE_DIR=/private/tmp/no-doc-stage2/uv-cache uv pip install --python /private/tmp/no-doc-stage2/tooling/bin/python uv==0.13.6

mkdir -p /private/tmp/no-doc-stage2/model-cache
/private/tmp/no-doc-stage2/baseline/bin/python scripts/real_model_smoke.py --cache-dir /private/tmp/no-doc-stage2/model-cache --report /private/tmp/no-doc-stage2/baseline-smoke.json

UV_CACHE_DIR=/private/tmp/no-doc-stage2/uv-cache /private/tmp/no-doc-stage2/tooling/bin/uv lock
UV_CACHE_DIR=/private/tmp/no-doc-stage2/uv-cache UV_PROJECT_ENVIRONMENT=/private/tmp/no-doc-stage2/verified /private/tmp/no-doc-stage2/tooling/bin/uv sync --frozen --extra dev --python /Users/maksim/.local/share/uv/python/cpython-3.11.11-macos-aarch64-none/bin/python3.11
/private/tmp/no-doc-stage2/verified/bin/python scripts/real_model_smoke.py --cache-dir /private/tmp/no-doc-stage2/model-cache --report /private/tmp/no-doc-stage2/verified-smoke.json

UV_CACHE_DIR=/private/tmp/no-doc-stage2/uv-cache UV_PROJECT_ENVIRONMENT=/private/tmp/no-doc-stage2/linux-target /private/tmp/no-doc-stage2/tooling/bin/uv sync --frozen --no-dev --python /Users/maksim/.local/share/uv/python/cpython-3.11.11-macos-aarch64-none/bin/python3.11 --python-platform x86_64-manylinux_2_28 --dry-run
UV_CACHE_DIR=/private/tmp/no-doc-stage2/uv-cache /private/tmp/no-doc-stage2/tooling/bin/uv lock --check --offline
UV_CACHE_DIR=/private/tmp/no-doc-stage2/uv-cache UV_PROJECT_ENVIRONMENT=/private/tmp/no-doc-stage2/verified /private/tmp/no-doc-stage2/tooling/bin/uv sync --frozen --extra dev --offline

/private/tmp/no-doc-stage2/verified/bin/python -m pytest
env PORT=invalid NDM_PORT=invalid NDM_ZSC_MODEL= HF_HOME=/nonexistent/hostile-cache /private/tmp/no-doc-stage2/verified/bin/python -m pytest -ra -W default -o addopts=''
/private/tmp/no-doc-stage2/verified/bin/python -m flake8 .
/private/tmp/no-doc-stage2/verified/bin/python -m isort . --check-only --filter-files
git diff --check
```

Results:

- macOS 26.5 arm64, Python 3.11.11, CPU only, torch threads=2.
- Baseline and final: both model loads, single-label and multi-label inference,
  MNLI mapping, tensor-to-NumPy, NER structure/offsets, Pydantic JSON serialization
  passed. Reports in [evidence/stage2](evidence/stage2).
- Both runs automatically started fresh offline children: `local_files_only=True`,
  offline flags set before imports, socket connection/DNS methods blocked. Both
  models loaded and inferred again without network access.
- Final sample NER spans: PER [0,12), GPE_ORG [24,36), GPE_LOC [39,44).
  These are bounded offsets for the fixed smoke input, not accuracy evidence.
- The underlying NER pipeline returns NumPy scores; JSON compatibility is verified
  through the existing API Pydantic schema, not claimed for raw `json.dumps(pipe(...))`.
- Real production service loaders were additionally called from a temporary cwd
  with offline flags, HF_HUB_CACHE pointing at the dedicated artifacts, blocked
  sockets and `Settings(_env_file=None)`. Both inference calls passed, loaded config
  hashes matched the configured revisions, and both tokenizers were fast.
- Frozen final install: 56 packages installed. Offline frozen repeat checked 56
  packages with no resolution/download; offline lock check resolved 59 packages
  (universal graph includes platform/extra alternatives) without changing the lock.
- Fast suite: 80 passed, 9 warnings. Hostile ENV diagnostic run: 80 passed,
  17 warnings. flake8, isort and diff checks pass after formatting the new harness.

Warnings / intermediate failures:

- Baseline: deprecated TRANSFORMERS_CACHE and gradient_checkpointing config;
  no predefined tokenizer maximum length (short inputs only).
- Final Hub: anonymous-request rate-limit notice; execution succeeded without credentials.
- Fast tests: pre-existing Pydantic `Field(example=...)` deprecations and
  AnyIO/Starlette TestClient unclosed memory-stream ResourceWarnings in diagnostic
  mode; warning counts depend on garbage collection. Not fixed in this stage.
- Resolver normalized an invalid upstream version specifier. Transformers 5.10.0
  was initially tried, warned as yanked, and replaced before the final installation.
- First draft smoke failed after successful model inference due to importing an
  incorrectly named schema (`Entity` instead of `NerEntity`). Corrected; complete
  baseline and final online/offline runs passed afterward.
- Initial isort check rejected the new harness import order; formatted and rechecked.
- Direct sandbox DNS was unavailable; authorized network operations used normal
  sandbox escalation. One source-inspection helper failed on an unnecessary
  `packaging` import, then succeeded with stdlib only. No dependency was installed
  into the system interpreter to work around that helper error.

## Linux evidence and deferred work

[Linux dry-run output](evidence/stage2/linux-selection.txt) selects
`torch==2.10.0+cpu` among 46 production packages. Lock source is explicitly
`https://download.pytorch.org/whl/cpu`; the cp311 artifact is
`torch-2.10.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`, SHA256
`b7cb1ec66cefb90fd7b676eac72cfda3b8d4e4d0cacd7a531963bc2e0a9710ab`.
No nvidia-* or triton package remains in the universal lock. macOS arm64 uses the
PyPI wheel. The lock supports only these two platform families; Windows, Linux
arm64 and macOS Intel are not claimed supported.
See [uv's official PyTorch index guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

This is Linux dependency resolution/artifact-selection evidence, not a Linux install,
Linux inference or container-runtime test. Docker was not run. Docker changes in
this stage only align the uv version and copy the new shared model-spec module.
The existing Docker non-frozen fallback, launcher and cache permissions still need
stage 5 work/verification; CI execution is also unverified. Lifecycle, concurrency,
chunking, stripping/offset corrections for arbitrary API inputs, load benchmarks,
accuracy and broad evaluation remain outside this stage.

Stage 2 DoD: core tuple/platform and frozen macOS install verified; Linux CPU selection
verified; both real models and offline reload verified; revisions and conflicting classification-license sources documented. The DoD
requirement for an unambiguous classification license remains open pending publisher
clarification; implementation is complete, acceptance is not claimed.
Independent acceptance is pending. Linux container verification remains an explicit
stage 5 gate and is not marked complete by these results.


## Final handoff note

Touched files: pyproject.toml, uv.lock, config.py, model_specs.py,
services/classification_service.py, services/ner_service.py, .env.example,
tests/test_settings.py, tests/test_services_unit.py, scripts/real_model_smoke.py,
README.md, Dockerfile (uv tag and shared-module COPY only),
.github/workflows/ci.yml (uv pin only), this report and docs/evidence/stage2/*.
The local ignored docs/IMPLEMENTATION_ROADMAP.md was also updated.

A proposed additional smoke rerun after tightening draft span/serialization checks
was not executed: automatic approval review failed because the workspace was out
of credits. This was a review-service failure, not an unsafe-action finding.
The unverified draft changes were reverted to the previously successful harness;
retained reports correspond to that verified implementation. No approval bypass
or further network operation was attempted.
