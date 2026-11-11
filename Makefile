.PHONY: setup lint format typecheck test check env data data-verify baseline benchmark prefixes matryoshka quantization generate generate-sample prepare-training finetune

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

prefixes:
	uv run emb-bench prefixes

matryoshka:
	uv run emb-bench matryoshka

quantization:
	uv run emb-bench quantization

# Time 20 chunks before committing to the full run.
generate-sample:
	uv run emb-bench generate --limit 20

# The full run, resumable. caffeinate keeps the Mac awake until it finishes.
generate:
	caffeinate -i uv run emb-bench generate

prepare-training:
	uv run emb-bench prepare-training

finetune:
	uv run emb-bench finetune
