"""Read and write the project's JSON and JSONL files through pydantic models."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    """Read one model per non-empty line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> None:
    """Write one model per line, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")


def read_json[M: BaseModel](path: Path, model: type[M]) -> M:
    """Read one model from a JSON file."""
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def write_json(path: Path, item: BaseModel) -> None:
    """Write one model as indented JSON, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(item.model_dump_json(indent=2) + "\n", encoding="utf-8")
