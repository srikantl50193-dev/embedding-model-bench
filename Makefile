.PHONY: setup lint format typecheck test check env data data-verify baseline benchmark

setup:
	bash scripts/setup.sh

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run ty check

test:
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test

env:
	uv run emb-bench env

data:
	uv run emb-bench data

data-verify:
	uv run emb-bench data-verify

baseline:
	uv run emb-bench baseline

benchmark:
	uv run emb-bench benchmark
