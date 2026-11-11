from collections.abc import Sequence

import numpy as np
import pytest

from conftest_fakes import CHUNKS, FakeEncoder, make_chunk
from embedding_model_bench import training_data
from embedding_model_bench.synthetic import SyntheticRecord, text_sha256
from embedding_model_bench.training_data import (
    format_problem,
    leakage,
    overlaps,
    prepare_training_data,
)

ENCODER = FakeEncoder()
CHUNK_VECTORS = ENCODER.encode([c.embedding_text() for c in CHUNKS])


def embed(texts: Sequence[str]) -> np.ndarray:
    return ENCODER.encode(list(texts))


def record(chunk_index: int, *questions: str) -> SyntheticRecord:
    chunk = CHUNKS[chunk_index]
    return SyntheticRecord(
        chunk_id=chunk.chunk_id,
        company=chunk.company,
        doc_type=chunk.doc_type,
        text_sha256=text_sha256(chunk.text),
        questions=list(questions),
        model="fake",
        model_digest="d",
        prompt_version=1,
        seconds=1.0,
        output_tokens=1,
    )


def test_format_checks() -> None:
    assert format_problem("Does Acme provide refunds?", "acme") is None
    assert format_problem("Does Acme provide refunds", "acme") == "no question mark"
    assert format_problem("Acme refunds?", "acme") == "too short or long"
    assert format_problem("Does the company provide refunds?", "acme") == (
        "doesn't name the company"
    )
    assert format_problem("What does this passage say about Acme?", "acme") == (
        "refers to the passage"
    )
    assert format_problem("Does PayPal charge fees here?", "paypal") is None


def test_leakage_catches_exact_and_reworded_copies_and_counts_per_set() -> None:
    questions = ["Does Acme give refunds?", "How long does Acme keep data?", "Is Acme fun?"]
    eval_sets = {
        "tuning": ["does acme give refunds"],
        "holdout": ["How long will Acme keep my data around?"],
    }
    vectors = {name: embed(qs) for name, qs in eval_sets.items()}
    leaks, counts = leakage(questions, embed(questions), eval_sets, vectors)
    assert leaks == [True, True, False]
    tuning, holdout = counts
    assert (tuning.exact, tuning.dropped) == (1, 1)
    assert holdout.dropped == 1
    assert holdout.exact == 0


def test_overlap_needs_the_same_document_and_shared_characters() -> None:
    a = make_chunk("a", "x" * 10).model_copy(update={"start_char": 0, "end_char": 100})
    b = a.model_copy(update={"chunk_id": "b", "start_char": 90, "end_char": 200})
    c = a.model_copy(update={"chunk_id": "c", "start_char": 100, "end_char": 200})
    other_doc = b.model_copy(update={"doc_type": "privacy"})
    assert overlaps(a, b)
    assert not overlaps(a, c)
    assert not overlaps(a, other_doc)


def test_leaked_questions_never_reach_training_and_their_text_isnt_reported() -> None:
    secret = "Does Acme provide refunds for anything?"
    records = [record(0, secret, "Can Acme refunds be provided?")]
    examples, report = prepare_training_data(
        records, CHUNKS, CHUNK_VECTORS, embed, {"fresh holdout": [secret]}
    )
    assert secret not in {e.query for e in examples}
    assert report.dropped_leakage >= 1
    assert secret not in report.model_dump_json()


def test_a_question_the_baseline_cant_trace_back_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fake corpus is smaller than the real cut-off, so every chunk would be in range.
    monkeypatch.setattr(training_data, "ROUNDTRIP_K", 1)
    records = [record(0, "Does Acme provide refunds?", "Is Acme unrelated filler 3?")]
    examples, report = prepare_training_data(records, CHUNKS, CHUNK_VECTORS, embed, {})
    assert [e.query for e in examples] == ["Does Acme provide refunds?"]
    assert report.dropped_roundtrip == 1


def test_negatives_are_never_the_source_or_a_copy_of_it() -> None:
    copy = make_chunk("a0-copy", CHUNKS[0].text, doc_type="privacy", section="Refunds")
    chunks = [*CHUNKS, copy]
    vectors = ENCODER.encode([c.embedding_text() for c in chunks])
    examples, _ = prepare_training_data(
        [record(0, "Does Acme provide refunds?")], chunks, vectors, embed, {}
    )
    (example,) = examples
    assert example.negative_chunk_id not in {"a0", "a0-copy"}
    assert example.positive == CHUNKS[0].embedding_text()


def test_each_source_chunk_is_in_exactly_one_split() -> None:
    records = [record(i, f"What does Acme say about filler {i}?") for i in range(2, 14)]
    examples, report = prepare_training_data(records, CHUNKS, CHUNK_VECTORS, embed, {})
    splits: dict[str, set[str]] = {}
    for example in examples:
        splits.setdefault(example.chunk_id, set()).add(example.split)
    assert all(len(s) == 1 for s in splits.values())
    assert report.num_train + report.num_validation == len(examples)
    assert report.num_validation >= 1
