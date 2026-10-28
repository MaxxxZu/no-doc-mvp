# Project instructions

## Project and communication

- This is a small Norwegian text classification and named entity recognition API
  built with FastAPI and Hugging Face Transformers.
- Communicate with the user in Russian. Keep code, identifiers, and public project
  documentation in English unless requested otherwise.
- Preserve the existing separation of API, services, Pydantic schemas, and settings.
  Prefer focused changes over a rewrite or unnecessary infrastructure.
- Describe only implemented capabilities. Do not call the project a RAG or
  generative LLM system unless the code actually implements those capabilities.

## Starting work

- Read applicable instructions, inspect the current branch and working tree, and
  read the relevant source files before editing. Preserve all user changes.
- Work only within the scope of the current user request. Opening the repository
  or reading these instructions does not authorize implementation work.
- The audit findings, implementation stages, acceptance criteria, and execution
  prompts belong in docs/IMPLEMENTATION_ROADMAP.md, not in this file. That document
  will be added separately. If present, read it before roadmap-related work;
  if absent, do not create it or reconstruct its contents without a user request.
- For requested implementation work, proceed incrementally and verify the result
  before moving to the next authorized scope. Do not expand the task implicitly.

## Authorization and safety

- Never commit, push, pull, merge, or deploy unless explicitly requested.
- Do not create or switch branches unless requested or already authorized.
- Running tests, installing/updating dependencies, downloading models, starting
  Docker, and making external network requests require explicit authorization
  covering those operations. A written plan is not authorization to execute it.
- Existing authorization in the current session persists; do not ask for the same
  permission repeatedly. Report checks that could not be performed.
- Tool commands may install dependencies implicitly. For authorized checks use an
  existing environment without automatic dependency sync unless installation is
  also authorized. Do not trigger model downloads merely by importing or testing.
- Do not overwrite the user's .env, expose secret values, log input documents,
  delete model caches, or change existing cache permissions without authorization.
- Use normal approval mechanisms for filesystem and sandbox restrictions. Do not
  work around them. Do not spawn subagents unless explicitly requested by the
  user or applicable instructions.

## Verification and handoff

- Distinguish static findings from behavior verified by execution. Do not claim
  tests passed, models loaded, accuracy was measured, or Docker works without
  having performed the corresponding authorized checks.
- Use fake model dependencies for fast tests; keep actual model inference and
  downloads in a separate explicitly invoked verification path.
- Never assess NLP accuracy from model names or mocked responses. Record the
  environment and relevant model/dependency versions for real-model evidence.
- Keep documentation consistent with actual behavior and verified limitations.
- At handoff summarize changed behavior, touched files, checks actually executed,
  their results, remaining limitations, and any unperformed required checks.
