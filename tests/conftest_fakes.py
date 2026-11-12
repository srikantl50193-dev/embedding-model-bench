import hashlib
import re
from collections.abc import Sequence

import numpy as np
from fine_print.models import Chunk

from embedding_model_bench.benchmark import BenchmarkInputs
from embedding_model_bench.models import EvalQuestion, ModelSpec, SciFact

DIM = 64


class FakeEncoder:
    """A bag-of-words hash: texts that share words get similar vectors, with no model involved."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def num_parameters(self) -> int:
        return 1_000_000

    def weights_bytes(self) -> int:
        return 4 * 1024**2


SPEC = ModelSpec(key="fake", repo="org/fake", revision="abc", dim=64, query_prefix="q: ")


def make_chunk(chunk_id: str, text: str, doc_type: str = "terms", section: str = "Misc") -> Chunk:
    return Chunk(
        text=text,
        company="acme",
        doc_type=doc_type,
        section=section,
        chunk_id=chunk_id,
        strategy="section",
        start_char=0,
        end_char=len(text),
    )


CHUNKS = [
    make_chunk("a0", "we do not provide refunds", section="Refunds"),
    make_chunk("a1", "we keep data as long as necessary", "privacy", "Retention"),
    *[make_chunk(f"f{i}", f"unrelated filler {i}") for i in range(12)],
]
QUESTIONS = [
    EvalQuestion(
        question="provide refunds", company="acme", doc_type="terms", keywords=["refunds"]
    ),
    EvalQuestion(question="keep data", company="acme", doc_type="privacy", keywords=["as long as"]),
]
SCIFACT = SciFact(
    corpus={str(i): f"science document {i} topic{i}" for i in range(12)},
    queries={"q1": "topic3"},
    qrels={"q1": {"3": 1}},
)


INPUTS = BenchmarkInputs(CHUNKS, QUESTIONS, SCIFACT, "commit", "fingerprint")
