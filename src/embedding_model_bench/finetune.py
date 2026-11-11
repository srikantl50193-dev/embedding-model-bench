"""Fine-tune bge-small on the synthetic questions, and compare it with the baseline.

Training uses MultipleNegativesRankingLoss, in its cached form (GradCache), which gives the same
loss over the whole batch but runs the backward pass in mini-batches: without it, a batch of 32
triplets needed more memory than this 16 GB machine has. Each question must score its own chunk
above its hard negative and above every other chunk in the batch. Examples with and without a
hard negative are two datasets sampled in proportion, so no example needs a made-up negative. No
batch holds the same text twice, because each chunk has two questions and a repeated positive
would be counted as a wrong answer. The validation questions are searched against the whole
corpus twice an epoch, and training stops when nDCG@10 stops improving.

Two runs:

- `all`: trained on every company.
- `unseen`: trained without three companies, chosen before any results, to see whether gains
  carry over to documents the model never trained on. About 9 tuning questions, so a rough
  signal only.
"""

import hashlib
import itertools
import random
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from fine_print.models import Chunk

from embedding_model_bench.benchmark import BenchmarkInputs, encode_inputs, score_encoded
from embedding_model_bench.compare import (
    SELECTION_K,
    PairedComparison,
    noise_note,
    paired_comparison,
)
from embedding_model_bench.config import PROJECT_ROOT
from embedding_model_bench.embedding import Encoder, cached_embedder
from embedding_model_bench.files import read_jsonl
from embedding_model_bench.models import (
    BenchmarkResult,
    EvalQuestion,
    FrozenModel,
    ModelSpec,
    RetrievalScores,
)
from embedding_model_bench.registry import BGE_SMALL
from embedding_model_bench.report import markdown_table
from embedding_model_bench.synthetic import text_sha256
from embedding_model_bench.training_data import TrainingExample

UNSEEN_COUNT = 3
UNSEEN_TARGET_QUESTIONS = 9
UNSEEN_SEED = 20261111
TRAIN_SEED = 42
BEST_METRIC = "eval_validation_cosine_ndcg@10"


class TrainConfig(FrozenModel):
    """One fine-tuning run. Hyperparameters are common defaults, not tuned."""

    name: str
    base: ModelSpec = BGE_SMALL
    excluded_companies: tuple[str, ...] = ()
    max_epochs: int = 4
    batch_size: int = 32
    mini_batch_size: int = 8  # Only limits memory; the loss is the same as without caching.
    empty_cache_steps: int = 10  # Return PyTorch's cached GPU memory every this many steps.
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1  # Passed as transformers 5's warmup_steps, where below 1 is a ratio.
    evals_per_epoch: int = 2
    patience: int = 2
    max_steps: int = -1  # A positive value cuts training short, for a smoke test.
    seed: int = TRAIN_SEED


class EvalPoint(FrozenModel):
    """Validation quality at one point in training."""

    step: int
    epoch: float
    ndcg_at_10: float


class TrainingRun(FrozenModel):
    """What happened in training, and where the model is."""

    config: TrainConfig
    model_dir: str  # Relative to the project root when saved inside it, as it normally is.
    weights_sha256: str
    num_triplets: int
    num_pairs: int
    num_validation: int
    data_sha256: str
    history: list[EvalPoint]
    best_step: int
    stopped_at_step: int
    seconds: float
    peak_rss_mb: float


def choose_unseen_companies(questions: Sequence[EvalQuestion]) -> list[str]:
    """Pick the companies to hold out, with as close to the target number of questions as possible.

    Ties are broken at random with a fixed seed, so the choice doesn't depend on any result.
    """
    counts = Counter(question.company for question in questions)
    combos = list(itertools.combinations(sorted(counts), UNSEEN_COUNT))
    gap = {c: abs(sum(counts[x] for x in c) - UNSEEN_TARGET_QUESTIONS) for c in combos}
    smallest = min(gap.values())
    best = [c for c in combos if gap[c] == smallest]
    return list(random.Random(UNSEEN_SEED).choice(best))


def select_examples(
    examples: Sequence[TrainingExample], chunks: Sequence[Chunk], excluded: Sequence[str]
) -> list[TrainingExample]:
    """Drop examples about excluded companies, including ones whose negative comes from them."""
    company = {chunk.chunk_id: chunk.company for chunk in chunks}
    return [
        e
        for e in examples
        if e.company not in excluded
        and (e.negative_chunk_id is None or company[e.negative_chunk_id] not in excluded)
    ]


def training_columns(
    examples: Sequence[TrainingExample], query_prefix: str
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return the triplet and pair columns, with the query prefix the model expects."""
    triplets: dict[str, list[str]] = {"anchor": [], "positive": [], "negative": []}
    pairs: dict[str, list[str]] = {"anchor": [], "positive": []}
    for example in examples:
        target = triplets if example.negative else pairs
        target["anchor"].append(query_prefix + example.query)
        target["positive"].append(example.positive)
        if example.negative:
            target["negative"].append(example.negative)
    return triplets, pairs


def validation_task(
    examples: Sequence[TrainingExample], chunks: Sequence[Chunk], query_prefix: str
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Return the queries, the whole corpus and the relevant chunks, for the evaluator.

    A chunk's identical copies elsewhere in the corpus count as relevant too.
    """
    copies: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        copies[text_sha256(chunk.text)].add(chunk.chunk_id)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    queries, relevant = {}, {}
    for number, example in enumerate(examples):
        key = f"q{number}"
        queries[key] = query_prefix + example.query
        relevant[key] = copies[text_sha256(by_id[example.chunk_id].text)]
    corpus = {chunk.chunk_id: chunk.embedding_text() for chunk in chunks}
    return queries, corpus, relevant


def data_sha256(examples: Sequence[TrainingExample]) -> str:
    """Hash the training data, so a result can be tied to exactly what it was trained on."""
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.model_dump_json().encode())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without reading it into memory at once."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def project_relative(path: Path) -> str:
    """Return a path relative to the project root if it is inside it, so results stay portable."""
    resolved = path.resolve()
    return (
        str(resolved.relative_to(PROJECT_ROOT))
        if resolved.is_relative_to(PROJECT_ROOT)
        else str(path)
    )


def fine_tuned_spec(run: TrainingRun) -> ModelSpec:
    """Describe a fine-tuned model so the embedding cache keys on its exact weights."""
    base = run.config.base
    return base.model_copy(
        update={
            "key": f"{base.key}-ft-{run.config.name}",
            "repo": run.model_dir,
            "revision": run.weights_sha256,
        }
    )


def train(
    config: TrainConfig,
    examples: Sequence[TrainingExample],
    chunks: Sequence[Chunk],
    models_dir: Path,
    device: str,
    log: Callable[[str], None] = lambda _message: None,
) -> TrainingRun:
    """Fine-tune the base model and save the best checkpoint.

    Args:
        config: The run.
        examples: Every training and validation example. Excluded companies are removed here.
        chunks: The whole corpus, which validation searches.
        models_dir: Where to save. Each run gets its own folder, replaced if it exists.
        device: Where to train.
        log: Receives progress messages.

    Returns:
        A record of the run.
    """
    # Imported here: they load torch and are slow, and only this step needs them.
    import torch  # noqa: PLC0415
    from datasets import Dataset, DatasetDict  # noqa: PLC0415
    from sentence_transformers import (  # noqa: PLC0415
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator  # noqa: PLC0415
    from sentence_transformers.sentence_transformer.losses import (  # noqa: PLC0415
        CachedMultipleNegativesRankingLoss,
    )
    from sentence_transformers.sentence_transformer.training_args import (  # noqa: PLC0415
        BatchSamplers,
        MultiDatasetBatchSamplers,
    )
    from transformers import EarlyStoppingCallback  # noqa: PLC0415

    from embedding_model_bench.machine import peak_rss_mb  # noqa: PLC0415

    kept = select_examples(examples, chunks, config.excluded_companies)
    train_examples = [e for e in kept if e.split == "train"]
    validation = [e for e in kept if e.split == "validation"]
    prefix = config.base.query_prefix
    triplets, pairs = training_columns(train_examples, prefix)
    log(f"{len(triplets['anchor'])} triplets, {len(pairs['anchor'])} pairs, {len(validation)} val")

    model = SentenceTransformer(
        config.base.repo,
        revision=config.base.revision,
        device=device,
        local_files_only=True,  # The benchmark already put it in the shared cache.
        # PyTorch's fused attention on MPS can't apply dropout, which training uses.
        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
    )
    queries, corpus, relevant = validation_task(validation, chunks, prefix)
    evaluator = InformationRetrievalEvaluator(
        # Its hints accept many input types, and dict is invariant, so plain str needs a cast.
        queries=cast("dict[str, Any]", queries),
        corpus=cast("dict[str, Any]", corpus),
        relevant_docs=relevant,
        name="validation",
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[5, 10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
        show_progress_bar=False,
        batch_size=config.batch_size,
    )
    steps_per_epoch = max(1, (len(train_examples)) // config.batch_size)
    eval_steps = max(1, steps_per_epoch // config.evals_per_epoch)
    if config.max_steps > 0:
        eval_steps = max(1, min(eval_steps, config.max_steps // 2))

    out_dir = models_dir / config.name
    checkpoints = models_dir / f"{config.name}-checkpoints"
    for folder in (out_dir, checkpoints):
        shutil.rmtree(folder, ignore_errors=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=config.max_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_ratio,
        seed=config.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        logging_steps=eval_steps,
        report_to="none",
        torch_empty_cache_steps=config.empty_cache_steps,
    )
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=config.mini_batch_size)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=DatasetDict(
            {"triplets": Dataset.from_dict(triplets), "pairs": Dataset.from_dict(pairs)}
        ),
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )
    before = evaluator(model)
    history = [
        EvalPoint(step=0, epoch=0.0, ndcg_at_10=round(before[BEST_METRIC.removeprefix("eval_")], 4))
    ]
    log(f"step 0: validation nDCG@10 {history[0].ndcg_at_10}")
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    for entry in trainer.state.log_history:
        if BEST_METRIC in entry:
            history.append(
                EvalPoint(
                    step=int(entry["step"]),
                    epoch=round(float(entry["epoch"]), 2),
                    ndcg_at_10=round(float(entry[BEST_METRIC]), 4),
                )
            )
            log(f"step {entry['step']}: validation nDCG@10 {entry[BEST_METRIC]:.4f}")
    best = max(history[1:], key=lambda point: point.ndcg_at_10, default=history[0])

    model.save(str(out_dir))
    shutil.rmtree(checkpoints, ignore_errors=True)
    run = TrainingRun(
        config=config,
        model_dir=project_relative(out_dir),
        weights_sha256=file_sha256(out_dir / "model.safetensors"),
        num_triplets=len(triplets["anchor"]),
        num_pairs=len(pairs["anchor"]),
        num_validation=len(validation),
        data_sha256=data_sha256(kept),
        history=history,
        best_step=best.step,
        stopped_at_step=int(trainer.state.global_step),
        seconds=round(seconds, 1),
        peak_rss_mb=round(peak_rss_mb(), 1),
    )
    (out_dir / "training_run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run


def load_examples(path: Path) -> list[TrainingExample]:
    """Read one split's examples."""
    return read_jsonl(path, TrainingExample)


class SubsetComparison(FrozenModel):
    """Baseline against a fine-tuned model on some of the tuning questions."""

    name: str
    companies: list[str]
    num_queries: int
    baseline_hits_at_5: int
    model_hits_at_5: int
    wins: int
    losses: int
    sign_test_p: float


class FineTuneResult(FrozenModel):
    """A fine-tuned model's scores, compared with the baseline on the same questions."""

    run: TrainingRun
    spec: ModelSpec
    unseen_companies: list[str]
    tuning: RetrievalScores
    scifact: RetrievalScores
    tuning_vs_baseline: PairedComparison
    scifact_vs_baseline: PairedComparison
    subsets: list[SubsetComparison]


def subset_comparison(
    name: str,
    companies: Sequence[str],
    questions: Sequence[EvalQuestion],
    baseline: RetrievalScores,
    model: RetrievalScores,
    k: int = SELECTION_K,
) -> SubsetComparison:
    """Compare two models on the tuning questions about some companies."""
    rows = [i for i, q in enumerate(questions) if q.company in companies]
    paired = paired_comparison(baseline, model, k, rows)
    baseline_hits, model_hits = baseline.hit_flags(k), model.hit_flags(k)
    return SubsetComparison(
        name=name,
        companies=sorted(companies),
        num_queries=paired.num_queries,
        baseline_hits_at_5=sum(baseline_hits[i] for i in rows),
        model_hits_at_5=sum(model_hits[i] for i in rows),
        wins=paired.wins,
        losses=paired.losses,
        sign_test_p=paired.sign_test_p,
    )


def evaluate_fine_tuned(
    run: TrainingRun,
    inputs: BenchmarkInputs,
    baseline: BenchmarkResult,
    unseen: Sequence[str],
    cache_dir: Path,
    load_encoder: Callable[[ModelSpec], Encoder],
) -> FineTuneResult:
    """Score a fine-tuned model exactly as the benchmark scores every model."""
    spec = fine_tuned_spec(run)
    embed = cached_embedder(spec, cache_dir, lambda: load_encoder(spec))
    encoded = encode_inputs(inputs, embed, spec.query_prefix, spec.passage_prefix)
    tuning, scifact = score_encoded(inputs, encoded)
    seen = sorted({q.company for q in inputs.questions} - set(unseen))
    return FineTuneResult(
        run=run,
        spec=spec,
        unseen_companies=sorted(unseen),
        tuning=tuning,
        scifact=scifact,
        tuning_vs_baseline=paired_comparison(baseline.tuning, tuning),
        scifact_vs_baseline=paired_comparison(baseline.scifact, scifact),
        subsets=[
            subset_comparison(
                "unseen companies", unseen, inputs.questions, baseline.tuning, tuning
            ),
            subset_comparison("other companies", seen, inputs.questions, baseline.tuning, tuning),
        ],
    )


def finetune_table(results: Sequence[FineTuneResult], baseline: BenchmarkResult) -> str:
    """Render each fine-tuned model beside the baseline, overall and by company group."""
    n = baseline.tuning.num_queries
    overall = [
        [
            f"{baseline.spec.key} (baseline)",
            "-",
            f"{baseline.tuning.hits(SELECTION_K)}/{n}",
            "-",
            "-",
            f"{baseline.tuning.mrr:.3f}",
            f"{baseline.scifact.ndcg_at_10:.3f}",
            "-",
            "-",
        ]
    ]
    for r in results:
        excluded = r.run.config.excluded_companies
        trained = "all companies" if not excluded else f"all but {', '.join(sorted(excluded))}"
        t, s = r.tuning_vs_baseline, r.scifact_vs_baseline
        overall.append(
            [
                r.spec.key,
                trained,
                f"{r.tuning.hits(SELECTION_K)}/{n}",
                f"{t.wins} / {t.losses}",
                f"{t.sign_test_p:.3f}",
                f"{r.tuning.mrr:.3f}",
                f"{r.scifact.ndcg_at_10:.3f}",
                f"{s.wins} / {s.losses}",
                f"{s.sign_test_p:.3f}",
            ]
        )
    groups = [
        [
            r.spec.key,
            g.name,
            ", ".join(g.companies),
            g.num_queries,
            g.baseline_hits_at_5,
            g.model_hits_at_5,
            f"{g.wins} / {g.losses}",
            f"{g.sign_test_p:.3f}",
        ]
        for r in results
        for g in r.subsets
    ]
    first = markdown_table(
        [
            "Model",
            "Trained on",
            f"Tuning hit@{SELECTION_K}",
            "Wins / losses",
            "p",
            "MRR",
            "SciFact nDCG@10",
            "SciFact wins / losses",
            "p",
        ],
        overall,
    )
    second = markdown_table(
        ["Model", "Group", "Companies", "n", "Baseline", "Fine-tuned", "Wins / losses", "p"],
        groups,
        [noise_note(n)],
    )
    note = "By company group (tuning hit@5; wins and losses against the baseline):"
    return f"{first}\n{note}\n\n{second}"
