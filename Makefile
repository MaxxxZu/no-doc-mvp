.PHONY: help lock rebuild up down logs sync sync-dev lint format test cov run

help:
	@echo "Targets:"
	@echo "  lock        - generate/refresh uv.lock"
	@echo "  sync        - install prod deps (uv)"
	@echo "  sync-dev    - install prod + dev deps (uv --extra dev)"
	@echo "  rebuild     - docker compose build --no-cache"
	@echo "  up          - docker compose up --build"
	@echo "  down        - docker compose down"
	@echo "  logs        - docker compose logs -f"
	@echo "  lint        - run flake8 and isort --check"
	@echo "  format      - run isort (apply) "
	@echo "  test        - run pytest"
	@echo "  cov         - run pytest with coverage"
	@echo "  run         - run API locally via uvicorn"

# -------- uv deps --------
lock:
	uv lock

sync:
	uv sync

sync-dev:
	uv sync --extra dev

# -------- Docker --------
up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

rebuild:
	docker compose build --no-cache


# -------- Quality --------
lint:
	uv run flake8 .
	uv run isort . --check-only --filter-files

format:
	uv run isort .

test:
	uv run pytest

cov:
	uv run pytest --cov=services --cov=app --cov-report=term-missing

# -------- Local run --------
run:
	uv run --no-sync python app.py
