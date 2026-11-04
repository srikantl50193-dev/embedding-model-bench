"""Measure what each model's query and passage prefixes are actually worth."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from embedding_model_bench.benchmark import BenchmarkInputs, score_scifact, score_tuning
from embedding_model_bench.compare import PairedComparison, noise_note, paired_comparison
from embedding_model_bench.embedding import Encoder, EncoderFactory, embed_cached
from embedding_model_bench.machine import peak_rss_mb
from embedding_model_bench.models import FrozenModel, ModelSpec, RetrievalScores

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
    loaded: list[Encoder] = []

    def encoder() -> Encoder:
        if not loaded:
            loaded.append(encoder_factory(spec, device))
        return loaded[0]

    def embed(texts: Sequence[str], prefix: str) -> np.ndarray:
        return embed_cached(spec, texts, prefix, cache_dir, encoder)[0]

    chunk_texts = [chunk.embedding_text() for chunk in inputs.chunks]
    query_texts = [question.question for question in inputs.questions]
    document_ids, query_ids = list(inputs.scifact.corpus), list(inputs.scifact.queries)
    document_texts = list(inputs.scifact.corpus.values())
    scifact_query_texts = list(inputs.scifact.queries.values())

    scored = []
    for name, query_prefix, passage_prefix in combinations:
        log(f"{name}: query {query_prefix!r}, passage {passage_prefix!r}")
        tuning = score_tuning(
            embed(query_texts, query_prefix),
            embed(chunk_texts, passage_prefix),
            inputs.chunks,
            inputs.questions,
        )
        scifact = score_scifact(
            embed(scifact_query_texts, query_prefix),
            embed(document_texts, passage_prefix),
            query_ids,
            document_ids,
            inputs.scifact,
        )
        scored.append((name, query_prefix, passage_prefix, tuning, scifact))

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
        "Tuning hit@5",
        "Wins / losses",
        "p",
        "MRR",
        "SciFact nDCG@10",
        "SciFact wins / losses",
        "p",
    ]
    rows = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for result in results:
        for v in result.variants:
            tuning, scifact = v.tuning_vs_card, v.scifact_vs_card
            cells = [
                result.spec.key,
                v.name,
                f"{v.tuning.hit_at_k[5]:.3f}",
                f"{tuning.wins} / {tuning.losses}",
                f"{tuning.sign_test_p:.3f}",
                f"{v.tuning.mrr:.3f}",
                f"{v.scifact.ndcg_at_10:.3f}",
                f"{scifact.wins} / {scifact.losses}",
                f"{scifact.sign_test_p:.3f}",
            ]
            rows.append("| " + " | ".join(cells) + " |")
    if results:
        rows += ["", noise_note(results[0].variants[0].tuning.num_queries)]
    return "\n".join(rows) + "\n"
