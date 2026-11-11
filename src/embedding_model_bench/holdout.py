"""Score the final configurations on held-out questions. Run once, and report as it comes.

The same code scores any question set, so it is first checked on the tuning set, where every
configuration's result is already known. The real run then differs only in the questions.

Embedding happens in one process and scoring in another: a model may have to be loaded to embed
questions no model has seen, and FAISS, which binary search uses, can't share a process with
torch on macOS.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

from fine_print.models import Chunk

from embedding_model_bench.benchmark import score_vectors, tuning_relevant
from embedding_model_bench.compare import (
    SELECTION_K,
    PairedComparison,
    noise_note,
    paired_comparison,
)
from embedding_model_bench.embedding import Encoder, cached_embedder
from embedding_model_bench.files import read_json
from embedding_model_bench.matryoshka import truncate
from embedding_model_bench.metrics import DEEPEST, score
from embedding_model_bench.models import EvalQuestion, FrozenModel, ModelSpec, RetrievalScores
from embedding_model_bench.quantization import Scheme
from embedding_model_bench.registry import BGE_BASE, BGE_SMALL, MODELS
from embedding_model_bench.report import markdown_table

FINE_TUNED_RUN = "all"


class HoldoutConfig(FrozenModel):
    """One configuration under test."""

    name: str
    key: str
    dim: int
    scheme: Scheme
    fine_tuned: bool = False


BASELINE = HoldoutConfig(
    name="bge-small 384d float32 (baseline)", key=BGE_SMALL.key, dim=384, scheme=Scheme.FLOAT32
)
# The order and choice were fixed in results/recommendation_frozen.md before any held-out run.
CONFIGS = [
    HoldoutConfig(name="bge-base 768d float32", key=BGE_BASE.key, dim=768, scheme=Scheme.FLOAT32),
    HoldoutConfig(
        name="bge-base 768d binary + float32 rescoring (recommended)",
        key=BGE_BASE.key,
        dim=768,
        scheme=Scheme.BINARY_RESCORE_FLOAT,
    ),
    HoldoutConfig(name="bge-base 256d float32", key=BGE_BASE.key, dim=256, scheme=Scheme.FLOAT32),
    HoldoutConfig(
        name="bge-small 384d float32, fine-tuned",
        key=BGE_SMALL.key,
        dim=384,
        scheme=Scheme.FLOAT32,
        fine_tuned=True,
    ),
    BASELINE,
]


class HoldoutRow(FrozenModel):
    """One configuration's scores on one question set."""

    config: HoldoutConfig
    scores: RetrievalScores
    vs_baseline: PairedComparison


class HoldoutSet(FrozenModel):
    """Every configuration on one question set."""

    name: str
    note: str
    num_questions: int
    sha256: str
    rows: list[HoldoutRow]


class HoldoutResult(FrozenModel):
    """The held-out run, with what it was run on."""

    fine_print_commit: str
    corpus_fingerprint: str
    sets: list[HoldoutSet]


def config_spec(config: HoldoutConfig, finetune_dir: Path) -> ModelSpec:
    """Return the model to embed with. A fine-tuned model is pinned to its weights' hash."""
    if not config.fine_tuned:
        return MODELS[config.key]
    from embedding_model_bench.finetune import FineTuneResult  # noqa: PLC0415

    return read_json(finetune_dir / f"{FINE_TUNED_RUN}.json", FineTuneResult).spec


def embed_all(
    specs: Sequence[ModelSpec],
    question_texts: Sequence[str],
    chunks: Sequence[Chunk],
    cache_dir: Path,
    load: Callable[[ModelSpec], Encoder],
) -> None:
    """Put every vector the scoring step needs into the cache, loading models only on a miss."""
    chunk_texts = [chunk.embedding_text() for chunk in chunks]
    for spec in specs:
        embed = cached_embedder(spec, cache_dir, lambda spec=spec: load(spec))
        embed(question_texts, spec.query_prefix)
        embed(chunk_texts, spec.passage_prefix)


def score_config(
    config: HoldoutConfig,
    spec: ModelSpec,
    questions: Sequence[EvalQuestion],
    chunks: Sequence[Chunk],
    relevant: Sequence[set[int]],
    cache_dir: Path,
    load: Callable[[], Encoder],
) -> RetrievalScores:
    """Score one configuration exactly as the experiment that measured it on tuning did.

    Float32 uses the benchmark's exact numpy search; binary uses the quantization step's FAISS
    index. Vectors shorter than the model's are truncated as in the Matryoshka step.
    """
    embed = cached_embedder(spec, cache_dir, load)
    query_vectors = embed([q.question for q in questions], spec.query_prefix)
    chunk_vectors = embed([c.embedding_text() for c in chunks], spec.passage_prefix)
    if config.dim < spec.dim:
        query_vectors = truncate(query_vectors, config.dim)
        chunk_vectors = truncate(chunk_vectors, config.dim)
    if config.scheme == Scheme.FLOAT32:
        return score_vectors(query_vectors, chunk_vectors, relevant)
    from embedding_model_bench.quantization import SearchIndex  # noqa: PLC0415 - loads FAISS

    ranked = SearchIndex(config.scheme, chunk_vectors).search(query_vectors, DEEPEST)
    return score(ranked.tolist(), relevant)


def score_set(  # noqa: PLR0913
    name: str,
    note: str,
    sha256: str,
    questions: Sequence[EvalQuestion],
    chunks: Sequence[Chunk],
    specs: dict[str, ModelSpec],
    *,
    cache_dir: Path,
    load: Callable[[], Encoder],
) -> HoldoutSet:
    """Score every configuration on one question set, each compared with the baseline."""
    relevant = tuning_relevant(chunks, questions)
    scores = {
        c.name: score_config(c, specs[c.name], questions, chunks, relevant, cache_dir, load)
        for c in CONFIGS
    }
    baseline = scores[BASELINE.name]
    return HoldoutSet(
        name=name,
        note=note,
        num_questions=len(questions),
        sha256=sha256,
        rows=[
            HoldoutRow(
                config=c,
                scores=scores[c.name],
                vs_baseline=paired_comparison(baseline, scores[c.name]),
            )
            for c in CONFIGS
        ],
    )


def holdout_table(result: HoldoutResult) -> str:
    """Render each question set, with wins and losses against the baseline."""
    parts = []
    for held in result.sets:
        rows = [
            [
                row.config.name,
                f"{row.scores.hits(1)}/{held.num_questions}",
                f"{row.scores.hits(SELECTION_K)}/{held.num_questions}",
                f"{row.scores.hits(10)}/{held.num_questions}",
                f"{row.scores.mrr:.3f}",
                f"{row.vs_baseline.wins} / {row.vs_baseline.losses}",
                f"{row.vs_baseline.sign_test_p:.3f}",
            ]
            for row in held.rows
        ]
        without = held.rows[0].scores.num_without_relevant
        footer = [noise_note(held.num_questions)]
        if without:
            footer.append(f"{without} questions have no chunk that passes the keyword test.")
        parts.append(
            f"## {held.name}\n\n{held.note}\n\n"
            + markdown_table(
                [
                    "Configuration",
                    "hit@1",
                    f"hit@{SELECTION_K}",
                    "hit@10",
                    "MRR",
                    f"Wins / losses vs baseline (hit@{SELECTION_K})",
                    "Sign test p",
                ],
                rows,
                footer,
            )
        )
    return "\n".join(parts)
