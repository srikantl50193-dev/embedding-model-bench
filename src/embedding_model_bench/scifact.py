"""BEIR SciFact, the out-of-domain benchmark, read from the shared Hugging Face cache."""

import csv
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq

from embedding_model_bench.config import Settings
from embedding_model_bench.models import SciFact

CORPUS_FILE = "corpus/corpus-00000-of-00001.parquet"
QUERIES_FILE = "queries/queries-00000-of-00001.parquet"
QRELS_FILE = "test.tsv"

type Fetch = Callable[[str, str, str], Path]


def fetch_from_hub(repo: str, filename: str, revision: str) -> Path:
    """Return a dataset file's path in the shared cache, downloading it only if it's absent."""
    # Imported here because the CLI shouldn't pay for it on commands that don't download.
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return Path(
        hf_hub_download(repo_id=repo, filename=filename, revision=revision, repo_type="dataset")
    )


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Read a BEIR relevance file, keeping only the judgments marked relevant."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = score
    return qrels


def load_scifact(settings: Settings, fetch: Fetch = fetch_from_hub) -> SciFact:
    """Load the test split: the whole corpus, and only the queries that have judgments.

    A document's text is its title and body on separate lines, which is how Fine Print
    joins a section title to its chunk.

    Args:
        settings: Where the dataset is, and which revision.
        fetch: Returns the local path of a file, given its repo, name and revision.

    Returns:
        The dataset.
    """
    corpus_repo, corpus_revision = settings.scifact_corpus_repo, settings.scifact_corpus_revision
    corpus_rows = pq.read_table(fetch(corpus_repo, CORPUS_FILE, corpus_revision)).to_pylist()
    query_rows = pq.read_table(fetch(corpus_repo, QUERIES_FILE, corpus_revision)).to_pylist()
    qrels = read_qrels(
        fetch(settings.scifact_qrels_repo, QRELS_FILE, settings.scifact_qrels_revision)
    )
    return SciFact(
        corpus={str(row["_id"]): f"{row['title']}\n{row['text']}".strip() for row in corpus_rows},
        queries={str(row["_id"]): row["text"] for row in query_rows if str(row["_id"]) in qrels},
        qrels=qrels,
    )
