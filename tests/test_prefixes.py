from pathlib import Path

import pytest

from conftest_fakes import INPUTS, SPEC, FakeEncoder
from embedding_model_bench.models import ModelSpec
from embedding_model_bench.prefixes import (
    CARD,
    NONE,
    PASSAGE_ONLY,
    QUERY_ONLY,
    prefix_table,
    run_prefix_ablation,
    variants_for,
)

BOTH_SIDES = SPEC.model_copy(update={"query_prefix": "q: ", "passage_prefix": "p: "})
NO_PREFIX = SPEC.model_copy(update={"query_prefix": "", "passage_prefix": ""})


def factory(_spec: ModelSpec, _device: str) -> FakeEncoder:
    return FakeEncoder()


def test_a_query_only_model_gets_two_variants() -> None:
    assert [name for name, _, _ in variants_for(SPEC)] == [CARD, NONE]


def test_a_two_sided_model_also_gets_the_one_sided_variants() -> None:
    variants = variants_for(BOTH_SIDES)
    assert variants == [
        (CARD, "q: ", "p: "),
        (NONE, "", ""),
        (QUERY_ONLY, "q: ", ""),
        (PASSAGE_ONLY, "", "p: "),
    ]


def test_a_model_without_prefixes_has_nothing_to_test(tmp_path: Path) -> None:
    assert variants_for(NO_PREFIX) == []
    with pytest.raises(ValueError, match="no prefixes"):
        run_prefix_ablation(NO_PREFIX, INPUTS, tmp_path, factory, "cpu")


def test_each_variant_is_compared_with_the_card(tmp_path: Path) -> None:
    result = run_prefix_ablation(BOTH_SIDES, INPUTS, tmp_path, factory, "cpu")
    assert [v.name for v in result.variants] == [CARD, NONE, QUERY_ONLY, PASSAGE_ONLY]
    card = result.variants[0]
    assert (card.tuning_vs_card.wins, card.tuning_vs_card.losses) == (0, 0)
    assert card.tuning.num_queries == len(INPUTS.questions)


def test_shared_vectors_are_encoded_once(tmp_path: Path) -> None:
    encoder = FakeEncoder()
    run_prefix_ablation(BOTH_SIDES, INPUTS, tmp_path, lambda _s, _d: encoder, "cpu")
    # Four text collections, each with and without a prefix.
    assert encoder.calls == 8


def test_a_rerun_loads_no_model(tmp_path: Path) -> None:
    run_prefix_ablation(BOTH_SIDES, INPUTS, tmp_path, factory, "cpu")

    def must_not_load(_spec: ModelSpec, _device: str) -> FakeEncoder:
        raise AssertionError("everything should be cached")

    run_prefix_ablation(BOTH_SIDES, INPUTS, tmp_path, must_not_load, "cpu")


def test_table_names_each_variant_and_the_question_count(tmp_path: Path) -> None:
    table = prefix_table([run_prefix_ablation(SPEC, INPUTS, tmp_path, factory, "cpu")])
    assert CARD in table
    assert NONE in table
    assert f"n={len(INPUTS.questions)}" in table
