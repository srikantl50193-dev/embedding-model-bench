"""Print what a model retrieved for the tuning questions it missed, for reading by hand.

Tuning questions only. Fine Print's keyword test can fail a chunk that does answer the question
in other words, so this shows what the test can't: whether the retrieved text answers it.

Usage: uv run python scripts/inspect_misses.py granite-small-en-r2 [--chars 700]
"""

from typing import Annotated

import numpy as np
import typer

from embedding_model_bench.benchmark import load_tuning_questions
from embedding_model_bench.config import get_settings
from embedding_model_bench.corpus import load_chunks
from embedding_model_bench.data import LINK_FOLDER
from embedding_model_bench.embedding import Encoder, embed_cached
from embedding_model_bench.metrics import is_relevant, top_k
from embedding_model_bench.models import ModelSpec
from embedding_model_bench.registry import MODELS

K = 5


def _not_cached() -> Encoder:
    raise SystemExit("Vectors aren't cached. Run `make benchmark` first.")


def main(key: str, chars: Annotated[int, typer.Option(help="Text shown per chunk.")] = 700) -> None:
    """Show the top 5 for every question this model missed at hit@5."""
    settings = get_settings()
    spec: ModelSpec = MODELS[key]
    link_dir = settings.data_dir / LINK_FOLDER
    _, chunks = load_chunks(link_dir / "raw")
    questions = load_tuning_questions(link_dir / "tuning_questions.jsonl")
    cache_dir = settings.data_dir / "embeddings"
    chunk_vectors = embed_cached(
        spec, [c.embedding_text() for c in chunks], spec.passage_prefix, cache_dir, _not_cached
    )[0]
    query_vectors = embed_cached(
        spec, [q.question for q in questions], spec.query_prefix, cache_dir, _not_cached
    )[0]
    ranked = np.asarray(top_k(query_vectors, chunk_vectors, K))

    misses = 0
    for question, row in zip(questions, ranked.tolist(), strict=True):
        top = [chunks[i] for i in row]
        if any(is_relevant(c.text, c.company, c.doc_type, question) for c in top):
            continue
        misses += 1
        print(f"\n=== Q: {question.question}")
        print(f"    want: {question.company}/{question.doc_type}, keywords {question.keywords}")
        for rank, chunk in enumerate(top, start=1):
            right_doc = chunk.company == question.company and chunk.doc_type == question.doc_type
            if not right_doc:
                print(f"  {rank}. [other doc] {chunk.company}/{chunk.doc_type} | {chunk.section}")
                continue
            text = " ".join(chunk.text.split())
            found = [k for k in question.keywords if k.lower() in text.lower()]
            print(f"  {rank}. [right doc] {chunk.section} | keywords found: {found}")
            print(f"     {text[:chars]}")
    print(f"\n{key}: {misses} of {len(questions)} tuning questions missed at hit@{K}.")


if __name__ == "__main__":
    typer.run(main)
