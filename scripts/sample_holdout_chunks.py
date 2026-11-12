"""Draw the chunks the fresh held-out questions are written from, at random.

Picking passages by hand would favour clear, easy ones. The draw is seeded so that it can be
repeated, takes two chunks per document, and skips short chunks and any chunk that already
answers a tuning question, so that the fresh set asks about different facts.
"""

import json
import random
from pathlib import Path

from embedding_model_bench.config import get_settings
from embedding_model_bench.corpus import load_chunks

SEED = 20261104
PER_DOCUMENT = 2
MIN_CHARS = 400


def main() -> None:
    link_dir = get_settings().data_dir / "fine_print"
    _, chunks = load_chunks(link_dir / "raw")
    lines = Path(link_dir / "tuning_questions.jsonl").read_text(encoding="utf-8").splitlines()
    tuning = [json.loads(line) for line in lines if line.strip()]

    def answers_tuning(chunk) -> bool:
        text = " ".join(chunk.text.lower().split())
        return any(
            q["company"] == chunk.company
            and q["doc_type"] == chunk.doc_type
            and all(k.lower() in text for k in q["keywords"])
            for q in tuning
        )

    rng = random.Random(SEED)
    documents = sorted({(c.company, c.doc_type) for c in chunks})
    for company, doc_type in documents:
        pool = [
            c
            for c in chunks
            if (c.company, c.doc_type) == (company, doc_type)
            and len(c.text) >= MIN_CHARS
            and not answers_tuning(c)
        ]
        for chunk in rng.sample(pool, PER_DOCUMENT):
            print(f"\n### {chunk.chunk_id} | {chunk.section}\n{chunk.text}")


if __name__ == "__main__":
    main()
