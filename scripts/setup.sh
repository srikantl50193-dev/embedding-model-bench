#!/usr/bin/env bash
# Create or repair .venv from uv.lock. Run through `make setup`.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it with the official installer, then run this again:" >&2
    echo >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The chunker is imported from Fine Print, which uv.lock expects to find next to this repo.
if [ ! -f ../fine-print/pyproject.toml ]; then
    echo "fine-print was not found next to this repo. Clone it first:" >&2
    echo >&2
    echo "    git clone https://github.com/srikantl50193-dev/fine-print ../fine-print" >&2
    exit 1
fi

uv sync --locked
uv run pre-commit install
echo "Environment ready in .venv. Next: make data"
