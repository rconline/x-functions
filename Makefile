.DEFAULT_GOAL := help

PY ?= python3
PIP ?= $(PY) -m pip
PYTEST ?= $(PY) -m pytest

.PHONY: help install install-dev test test-governed lint fmt clean docker-up docker-down preflight

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime deps (editable)
	$(PIP) install -e .

install-dev: ## Install dev + governance extras
	$(PIP) install -e ".[dev,governance]"

test: ## Unit tests only (no Spark sql, no HTTP)
	$(PYTEST) tests/unit -v

test-governed: ## Integration tests against the docker stack
	SPARK_AI_GOVERNED_TESTS=1 $(PYTEST) tests/integration -v -m "governed"

test-all: ## Unit + integration
	$(PYTEST) tests -v

lint: ## Ruff lint
	$(PY) -m ruff check src tests

fmt: ## Ruff format
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

clean: ## Remove build artefacts and caches
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -exec rm -rf {} +

docker-up: ## Start the stripped gravitino-playground stack
	cd docker && docker compose up -d

docker-down: ## Stop the stack
	cd docker && docker compose down -v

preflight: ## Run §19.0 preflight checks
	$(PY) scripts/preflight.py
