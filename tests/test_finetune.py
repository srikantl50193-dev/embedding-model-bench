from pathlib import Path

from conftest_fakes import CHUNKS, make_chunk
from embedding_model_bench.config import PROJECT_ROOT
from embedding_model_bench.embedding import resolve_repo
from embedding_model_bench.finetune import (
    choose_unseen_companies,
    data_sha256,
    project_relative,
    select_examples,
    subset_comparison,
    training_columns,
    validation_task,
)
from embedding_model_bench.models import EvalQuestion, RetrievalScores
from embedding_model_bench.training_data import TrainingExample


def question(company: str) -> EvalQuestion:
    return EvalQuestion(question="q?", company=company, doc_type="terms", keywords=["x"])


def example(company: str = "acme", negative_id: str | None = "f1", **update) -> TrainingExample:
    fields = {
        "query": "Does Acme refund?",
        "positive": "Refunds\nwe do not provide refunds",
        "negative": None if negative_id is None else "Misc\nunrelated filler 1",
        "chunk_id": "a0",
        "negative_chunk_id": negative_id,
        "company": company,
        "doc_type": "terms",
        "split": "train",
    }
    return TrainingExample.model_validate(fields | update)


def scores(ranks: list[int | None]) -> RetrievalScores:
    return RetrievalScores(
        num_queries=len(ranks),
        num_without_relevant=0,
        hit_at_k={},
        recall_at_k={},
        mrr=0.0,
        ndcg_at_10=0.0,
        first_relevant_ranks=ranks,
    )


def test_unseen_companies_are_three_with_the_target_number_of_questions() -> None:
    counts = {"a": 4, "b": 4, "c": 3, "d": 3, "e": 3, "f": 2, "g": 1}
    questions = [question(c) for c, n in counts.items() for _ in range(n)]
    chosen = choose_unseen_companies(questions)
    assert len(chosen) == 3
    assert sum(counts[c] for c in chosen) == 9
    assert chosen == choose_unseen_companies(list(reversed(questions)))


def test_examples_touching_an_excluded_company_are_dropped() -> None:
    other = make_chunk("o1", "other text").model_copy(update={"company": "other"})
    chunks = [*CHUNKS, other]
    kept = select_examples(
        [
            example(),
            example(company="other", chunk_id="o1"),
            example(negative_id="o1"),
            example(negative_id=None),
        ],
        chunks,
        ["other"],
    )
    assert len(kept) == 2
    assert all(e.company == "acme" and e.negative_chunk_id != "o1" for e in kept)


def test_examples_with_and_without_negatives_go_to_separate_columns_with_the_prefix() -> None:
    triplets, pairs = training_columns([example(), example(negative_id=None)], "query: ")
    assert triplets["anchor"] == ["query: Does Acme refund?"]
    assert len(triplets["negative"]) == 1
    assert pairs == {"anchor": ["query: Does Acme refund?"], "positive": [example().positive]}


def test_validation_counts_identical_copies_as_relevant_and_searches_everything() -> None:
    copy = make_chunk("a0-copy", "we do not provide\nrefunds", doc_type="privacy")
    chunks = [*CHUNKS, copy]
    queries, corpus, relevant = validation_task([example(split="validation")], chunks, "q: ")
    assert queries == {"q0": "q: Does Acme refund?"}
    assert relevant == {"q0": {"a0", "a0-copy"}}
    assert len(corpus) == len(chunks)


def test_the_data_hash_changes_with_the_data() -> None:
    assert data_sha256([example()]) != data_sha256([example(query="Other?")])


def test_a_subset_counts_only_its_companies() -> None:
    questions = [question("a"), question("a"), question("b")]
    result = subset_comparison("x", ["a"], questions, scores([1, None, 1]), scores([9, 2, None]))
    assert (result.num_queries, result.baseline_hits_at_5, result.model_hits_at_5) == (2, 1, 1)
    assert (result.wins, result.losses) == (1, 1)


def test_saved_models_are_recorded_relative_to_the_project(tmp_path: Path) -> None:
    assert project_relative(PROJECT_ROOT / "models" / "all") == "models/all"
    assert project_relative(tmp_path / "elsewhere") == str(tmp_path / "elsewhere")


def test_a_relative_model_resolves_to_its_folder_and_a_hub_id_is_unchanged() -> None:
    assert resolve_repo("BAAI/bge-small-en-v1.5") == "BAAI/bge-small-en-v1.5"
    assert resolve_repo("tests") == str(PROJECT_ROOT / "tests")
