"""Measure what each model's query and passage prefixes are actually worth."""

from collections.abc import Callable, Sequence
from pathlib import Path

from embedding_model_bench.benchmark import BenchmarkInputs, encode_inputs, score_encoded
from embedding_model_bench.compare import (
    SELECTION_K,
    PairedComparison,
    noise_note,
    paired_comparison,
)
from embedding_model_bench.embedding import EncoderFactory, cached_embedder
from embedding_model_bench.machine import peak_rss_mb
from embedding_model_bench.models import FrozenModel, ModelSpec, RetrievalScores
from embedding_model_bench.report import markdown_table

CARD = "as on the card"
NONE = "no prefixes"
QUERY_ONLY = "query prefix only"
PASSAGE_ONLY = "passage prefix only"


class PrefixVariant(FrozenModel):
    """One combination of prefixes, compared with using them as the model card says."""

    name: str
    query_prefix: str
    passage_prefix: str
    tuning: RetrievalScores
    scifact: RetrievalScores
    tuning_vs_card: PairedComparison
    scifact_vs_card: PairedComparison


class PrefixResult(FrozenModel):
    """Every prefix combination tested for one model."""

    spec: ModelSpec
    fine_print_commit: str
    corpus_fingerprint: str
    variants: list[PrefixVariant]
    peak_rss_mb: float


def variants_for(spec: ModelSpec) -> list[tuple[str, str, str]]:
    """List the combinations to test as (name, query prefix, passage prefix).

    A model with no prefixes has nothing to test. The one-sided combinations are only distinct
    for models that prefix both sides.
    """
    query, passage = spec.query_prefix, spec.passage_prefix
    if not query and not passage:
        return []
    variants = [(CARD, query, passage), (NONE, "", "")]
    if query and passage:
        variants += [(QUERY_ONLY, query, ""), (PASSAGE_ONLY, "", passage)]
    return variants


def run_prefix_ablation(
    spec: ModelSpec,
    inputs: BenchmarkInputs,
    cache_dir: Path,
    encoder_factory: EncoderFactory,
    device: str,
    log: Callable[[str], None] = lambda _message: None,
) -> PrefixResult:
    """Score every prefix combination for one model, on both corpora.

    Args:
        spec: The model.
        inputs: The corpora and questions.
        cache_dir: Where vectors are cached. Only combinations not already there are encoded.
        encoder_factory: Loads the model, which only happens if something has to be encoded.
        device: Where to encode.
        log: Receives progress messages.

    Returns:
        The scores of each combination, each compared with the first.

    Raises:
        ValueError: If the model has no prefixes to test.
    """
    combinations = variants_for(spec)
    if not combinations:
        raise ValueError(f"{spec.key} has no prefixes to test")
    embed = cached_embedder(spec, cache_dir, lambda: encoder_factory(spec, device))
    scored = []
    for name, query_prefix, passage_prefix in combinations:
        log(f"{name}: query {query_prefix!r}, passage {passage_prefix!r}")
        encoded = encode_inputs(inputs, embed, query_prefix, passage_prefix)
        scored.append((name, query_prefix, passage_prefix, *score_encoded(inputs, encoded)))

    card_tuning, card_scifact = scored[0][3], scored[0][4]
    return PrefixResult(
        spec=spec,
        fine_print_commit=inputs.fine_print_commit,
        corpus_fingerprint=inputs.corpus_fingerprint,
        variants=[
            PrefixVariant(
                name=name,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
                tuning=tuning,
                scifact=scifact,
                tuning_vs_card=paired_comparison(card_tuning, tuning),
                scifact_vs_card=paired_comparison(card_scifact, scifact),
            )
            for name, query_prefix, passage_prefix, tuning, scifact in scored
        ],
        peak_rss_mb=round(peak_rss_mb(), 1),
    )


def prefix_table(results: Sequence[PrefixResult]) -> str:
    """Render the combinations. Wins and losses are against using the prefixes as documented."""
    headers = [
        "Model",
        "Prefixes",
        f"Tuning hit@{SELECTION_K}",
        "Wins / losses",
        "p",
        "MRR",
        "SciFact nDCG@10",
        "SciFact wins / losses",
        "p",
    ]
    rows = [
        [
            result.spec.key,
            v.name,
            f"{v.tuning.hit_at_k[SELECTION_K]:.3f}",
            f"{v.tuning_vs_card.wins} / {v.tuning_vs_card.losses}",
            f"{v.tuning_vs_card.sign_test_p:.3f}",
            f"{v.tuning.mrr:.3f}",
            f"{v.scifact.ndcg_at_10:.3f}",
            f"{v.scifact_vs_card.wins} / {v.scifact_vs_card.losses}",
            f"{v.scifact_vs_card.sign_test_p:.3f}",
        ]
        for result in results
        for v in result.variants
    ]
    footer = [noise_note(results[0].variants[0].tuning.num_queries)] if results else []
    return markdown_table(headers, rows, footer)
