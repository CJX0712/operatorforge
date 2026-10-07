# OperatorForge developer tasks.
# All commands run from the repository root.

PY := python
VENV := python -m venv .venv

.PHONY: help install install-dev test lint format check demo benchmark clean

help:
	@echo "Targets:"
	@echo "  install      Install runtime deps from requirements.lock.txt"
	@echo "  install-dev  Install runtime + dev (pytest, ruff) deps"
	@echo "  test         Run the full pytest suite"
	@echo "  lint         ruff check (source + delivery files)"
	@echo "  format       ruff format (source + delivery files)"
	@echo "  check        lint + format --check"
	@echo "  demo         Run the deterministic demo (writes benchmark.json)"
	@echo "  benchmark    Run the full CLI benchmark"
	@echo "  clean        Remove caches"

install:
	$(PY) -m pip install -r requirements.lock.txt

install-dev: install
	$(PY) -m pip install "pytest==9.1.1" "pytest-cov==7.1.0" "ruff==0.16.10"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check core data operators training eval pipeline operatorforge examples

format:
	$(PY) -m ruff format core data operators training eval pipeline operatorforge examples

check: lint
	$(PY) -m ruff format --check core data operators training eval pipeline operatorforge examples

demo:
	$(PY) examples/run_demo.py

benchmark:
	$(PY) cli.py benchmark --output benchmark.json

clean:
	$(PY) -m ruff clean || true
	rm -rf .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
