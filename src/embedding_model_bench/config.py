"""All settings in one place, overridable with EMB_BENCH_* environment variables or .env.

There is deliberately no cache directory setting: models come from the shared Hugging Face
cache and the Ollama store, so nothing is downloaded twice. Chunking settings are absent too,
because they are read from Fine Print (see `corpus.py`).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings. Defaults work when fine-print is a sibling of this repo."""

    model_config = SettingsConfigDict(env_prefix="EMB_BENCH_", env_file=".env", extra="ignore")

    # Paths
    fine_print_dir: Path = PROJECT_ROOT.parent / "fine-print"
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"

    # SciFact, pinned so that the benchmark can be reproduced exactly.
    scifact_corpus_repo: str = "BeIR/scifact"
    scifact_corpus_revision: str = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
    scifact_qrels_repo: str = "BeIR/scifact-qrels"
    scifact_qrels_revision: str = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"

    # Fine Print's services, which must be stopped before anything heavy runs.
    fine_print_api_port: int = Field(default=8000, gt=0)
    fine_print_ui_port: int = Field(default=8501, gt=0)

    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e4b"
    ollama_timeout: float = Field(default=3.0, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment only once."""
    return Settings()
