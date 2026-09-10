.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install install-vllm install-hf data validate validate-full compat smoke test test-network lint fmt report clean clean-runs distclean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Typical first run:"
	@echo "  make install data smoke"

venv: ## Create the virtualenv
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

install: venv ## Install the package with dev extras (CPU only, no GPU deps)
	$(BIN)/pip install -e ".[dev]"
	@echo "Installed. Run 'make data' next."

install-vllm: install ## Add the in-process vLLM backend (needs a CUDA GPU)
	$(BIN)/pip install -e ".[vllm]"

install-hf: install ## Add the transformers backend incl. bitsandbytes INT4
	$(BIN)/pip install -e ".[hf]"

data: ## Download the 18 Indic splits and verify row counts
	$(BIN)/python scripts/download_data.py

validate: ## Preflight: prompts, data, row counts, model configs
	$(BIN)/cmb-indic validate --check-rows

compat: ## CPU-only check that model architectures + chat templates work
	$(BIN)/python scripts/check_compat.py --all

validate-full: ## Also probe the model server and the Hugging Face ids
	$(BIN)/cmb-indic validate --check-rows --check-server --check-hub

smoke: ## End-to-end pipeline check with the fake backend (no GPU, ~10s)
	$(BIN)/cmb-indic run --suite smoke --run-id smoke

test: ## Run the test suite (offline)
	$(BIN)/python -m pytest tests -q -m "not network"

test-network: ## Also verify the registry against the live Hugging Face dataset
	$(BIN)/python -m pytest tests -q -m network

lint: ## Lint and check formatting
	$(BIN)/ruff check src tests scripts
	$(BIN)/black --check src tests scripts

fmt: ## Auto-format and auto-fix
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/black src tests scripts

report: ## Rebuild results/ from the newest run
	@latest=$$(ls -1dt runs/*/ 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then echo "No runs found. Try 'make smoke'."; exit 1; fi; \
	echo "Reporting on $$latest"; \
	$(BIN)/cmb-indic report --run-dir "$$latest"

clean: ## Remove caches and build artefacts (keeps runs/ and data/)
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-runs: ## Delete all generations and scores -- destructive
	rm -rf runs results

distclean: clean ## Also remove the venv and downloaded data
	rm -rf $(VENV) data/raw data/cache
