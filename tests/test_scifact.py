from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from embedding_model_bench.config import Settings
from embedding_model_bench.scifact import (
    CORPUS_FILE,
    QRELS_FILE,
    QUERIES_FILE,
    load_scifact,
    read_qrels,
)


def test_read_qrels_drops_judgments_of_zero(tmp_path: Path) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text("query-id\tcorpus-id\tscore\n1\t10\t1\n1\t11\t0\n2\t12\t2\n", encoding="utf-8")
    assert read_qrels(path) == {"1": {"10": 1}, "2": {"12": 2}}


def test_load_keeps_the_whole_corpus_but_only_judged_queries(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.parquet"
    queries = tmp_path / "queries.parquet"
    qrels = tmp_path / "test.tsv"
    pq.write_table(
        pa.table({"_id": ["10", "11"], "title": ["T", ""], "text": ["Body", "Only body"]}), corpus
    )
    pq.write_table(
        pa.table({"_id": ["1", "2"], "title": ["", ""], "text": ["Judged", "Train only"]}), queries
    )
    qrels.write_text("query-id\tcorpus-id\tscore\n1\t10\t1\n", encoding="utf-8")
    files = {CORPUS_FILE: corpus, QUERIES_FILE: queries, QRELS_FILE: qrels}
    seen_revisions = []

    def fetch(_repo: str, filename: str, revision: str) -> Path:
        seen_revisions.append(revision)
        return files[filename]

    settings = Settings()
    scifact = load_scifact(settings, fetch)

    assert scifact.corpus == {"10": "T\nBody", "11": "Only body"}
    assert scifact.queries == {"1": "Judged"}
    assert scifact.qrels == {"1": {"10": 1}}
    assert set(seen_revisions) == {
        settings.scifact_corpus_revision,
        settings.scifact_qrels_revision,
    }
