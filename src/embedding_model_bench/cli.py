"""The command line interface. Heavy libraries are imported inside commands, to start quickly."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import psutil
import typer

from embedding_model_bench.config import Settings, get_settings
from embedding_model_bench.preflight import HeavyStep, run_preflight

if TYPE_CHECKING:
    from fine_print.models import Chunk

    from embedding_model_bench.benchmark import BenchmarkInputs
    from embedding_model_bench.embedding import Encoder
    from embedding_model_bench.holdout import HoldoutSet
    from embedding_model_bench.models import (
        BenchmarkResult,
        DataManifest,
        EvalQuestion,
        MachineState,
    )

app = typer.Typer(no_args_is_help=True, add_completion=False)

GIB = 1024**3
# Fine Print's README, section retrieval on the tuning set.
FINE_PRINT_PUBLISHED = {"hit@1": 0.58, "hit@3": 0.65, "hit@5": 0.65, "hit@10": 0.68, "MRR": 0.618}
FINETUNE_RUNS = ("all", "unseen")
SMOKE_STEPS = 4
MPS_HIGH_WATERMARK = "0.5"
MPS_LOW_WATERMARK = "0.4"


# Helpers shared by the commands.


def _lock_path(settings: Settings) -> Path:
    return settings.results_dir / ".heavy.lock"


def _require_preflight(step: HeavyStep, settings: Settings) -> None:
    """Print the checks and stop if any fails."""
    checks = run_preflight(step, settings)
    if not all(check.ok for check in checks):
        for check in checks:
            typer.echo(f"[{'ok' if check.ok else 'STOP'}] {check.name}: {check.detail}", err=True)
        raise typer.Exit(1)


def _verified_data(settings: Settings) -> tuple[DataManifest, list[Chunk]]:
    """Check Fine Print is unchanged, and return the manifest and the chunks made to check it."""
    from embedding_model_bench.data import FinePrintDataError, verify_manifest  # noqa: PLC0415

    try:
        return verify_manifest(settings.fine_print_dir, settings.data_dir)
    except FinePrintDataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


def _run_each(command: str, items: Iterable[str], *extra: str) -> list[str]:
    """Run a hidden command once per item, each in its own process, and return the failures.

    A process per model keeps each one's peak memory its own, and frees it afterwards.
    """
    base = [sys.executable, "-m", "embedding_model_bench.cli", command]
    return [item for item in items if subprocess.run([*base, item, *extra], check=False).returncode]


def _exit_if_failed(failed: list[str]) -> None:
    if failed:
        typer.echo(f"Failed: {', '.join(failed)}", err=True)
        raise typer.Exit(1)


def _report_provisional(key: str, machine: MachineState) -> None:
    """Say so if competing apps or low memory mean the timings can't be final."""
    if machine.timings_provisional:
        typer.echo(
            f"[{key}] Timings are PROVISIONAL: competing apps: "
            f"{', '.join(machine.competing_apps) or 'none'}, {machine.available_gb} GB "
            f"available, {machine.swap_used_gb} GB swap in use."
        )


def _logger(prefix: str) -> Callable[[str], None]:
    """Return a function that prints progress to stderr, labelled with the model or run."""

    def log(message: str) -> None:
        typer.echo(f"[{prefix}] {message}", err=True)

    return log


def _baseline(settings: Settings) -> BenchmarkResult:
    """Load bge-small's benchmark result, which fine-tuning is compared with."""
    from embedding_model_bench.files import read_json  # noqa: PLC0415
    from embedding_model_bench.models import BenchmarkResult  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL  # noqa: PLC0415

    return read_json(settings.results_dir / "models" / f"{BGE_SMALL.key}.json", BenchmarkResult)


def _devices() -> list[str]:
    """Return the devices to time: MPS first where it exists, then always CPU."""
    import torch  # noqa: PLC0415

    return ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]


def _prepared_inputs() -> BenchmarkInputs:
    """Run the preflight checks, then load the corpora and tuning questions.

    The preflight also refuses while Ollama holds a model in memory.
    """
    from embedding_model_bench.benchmark import (  # noqa: PLC0415
        BenchmarkInputs,
        load_tuning_questions,
    )
    from embedding_model_bench.data import LINK_FOLDER  # noqa: PLC0415
    from embedding_model_bench.scifact import load_scifact  # noqa: PLC0415

    settings = get_settings()
    _require_preflight("benchmark", settings)
    manifest, chunks = _verified_data(settings)
    return BenchmarkInputs(
        chunks=chunks,
        questions=load_tuning_questions(settings.data_dir / LINK_FOLDER / "tuning_questions.jsonl"),
        scifact=load_scifact(settings),
        fine_print_commit=manifest.fine_print_commit,
        corpus_fingerprint=manifest.corpus_fingerprint,
    )


# Setup and data.


@app.command()
def env() -> None:
    """Report the hardware, accelerator, Ollama and which models are already cached."""
    import torch  # noqa: PLC0415
    from huggingface_hub import scan_cache_dir  # noqa: PLC0415

    typer.echo(f"Platform: {platform.platform()} ({platform.machine()})")
    typer.echo(f"RAM: {psutil.virtual_memory().total / GIB:.0f} GB")
    typer.echo(f"Python: {sys.version.split()[0]}")
    typer.echo(f"torch: {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")
    for check in run_preflight("generation", get_settings()):
        typer.echo(f"{check.name}: {check.detail}")
    for repo in sorted(scan_cache_dir().repos, key=lambda r: r.repo_id):
        typer.echo(f"Cached {repo.repo_type}: {repo.repo_id} ({repo.size_on_disk_str})")


@app.command()
def data() -> None:
    """Symlink Fine Print's data, record a manifest, and fetch SciFact into the shared cache."""
    from embedding_model_bench.data import (  # noqa: PLC0415
        FinePrintDataError,
        build_manifest,
        link_data,
        save_manifest,
    )
    from embedding_model_bench.scifact import load_scifact  # noqa: PLC0415

    settings = get_settings()
    try:
        link_dir = link_data(settings.fine_print_dir, settings.data_dir)
    except FinePrintDataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    manifest = build_manifest(settings.fine_print_dir, link_dir)
    save_manifest(manifest, settings.data_dir)
    typer.echo(f"Linked {link_dir} -> {settings.fine_print_dir}")
    typer.echo(f"Fine Print commit: {manifest.fine_print_commit[:10]}")
    typer.echo(f"Documents: {manifest.num_documents}, section chunks: {manifest.num_chunks}")
    typer.echo(f"Corpus fingerprint: {manifest.corpus_fingerprint[:16]}")
    typer.echo(
        f"Questions: {manifest.num_tuning_questions} tuning, "
        f"{manifest.num_holdout_questions} held-out (counted, not read)"
    )
    scifact = load_scifact(settings)
    typer.echo(
        f"SciFact test split: {len(scifact.corpus)} documents, {len(scifact.queries)} queries"
    )


@app.command(name="data-verify")
def data_verify() -> None:
    """Check that Fine Print hasn't changed since its data was linked."""
    manifest, _ = _verified_data(get_settings())
    typer.echo(f"Fine Print is unchanged ({manifest.num_chunks} chunks).")


@app.command()
def preflight(
    step: Annotated[HeavyStep, typer.Argument(help="The kind of step about to run.")],
) -> None:
    """Check that a heavy step can run, and exit with an error if it shouldn't yet."""
    checks = run_preflight(step, get_settings())
    for check in checks:
        typer.echo(f"[{'ok' if check.ok else 'STOP'}] {check.name}: {check.detail}")
    if not all(check.ok for check in checks):
        raise typer.Exit(1)


# Benchmarks.


def _measure(spec_key: str) -> BenchmarkResult:
    """Measure one model, in this process."""
    from embedding_model_bench.benchmark import run_benchmark  # noqa: PLC0415
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    with heavy_step(_lock_path(settings)):
        return run_benchmark(
            MODELS[spec_key],
            inputs,
            settings.data_dir / "embeddings",
            SentenceTransformerEncoder,
            _devices(),
            log=_logger(spec_key),
        )


@app.command()
def baseline() -> None:
    """Measure bge-small and check that it reproduces Fine Print's published tuning scores."""
    from embedding_model_bench.benchmark import summary_table  # noqa: PLC0415
    from embedding_model_bench.files import write_json  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL  # noqa: PLC0415

    result = _measure(BGE_SMALL.key)
    out_dir = get_settings().results_dir / "baseline"
    write_json(out_dir / f"{BGE_SMALL.key}.json", result)
    table = summary_table([result])
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)

    tuning = result.tuning
    measured = {f"hit@{k}": tuning.hit_at_k[k] for k in (1, 3, 5, 10)} | {"MRR": tuning.mrr}
    for name, published in FINE_PRINT_PUBLISHED.items():
        digits = 3 if name == "MRR" else 2
        status = "ok" if round(measured[name], digits) == published else "DIFFERENT"
        typer.echo(f"[{status}] {name}: measured {measured[name]:.3f}, Fine Print {published}")
    _report_provisional(result.spec.key, result.machine)


@app.command(name="benchmark-one", hidden=True)
def benchmark_one(key: str) -> None:
    """Measure one model. `benchmark` runs this once per process, to isolate peak memory."""
    from embedding_model_bench.files import write_json  # noqa: PLC0415

    result = _measure(key)
    write_json(get_settings().results_dir / "models" / f"{key}.json", result)
    _report_provisional(key, result.machine)


@app.command()
def benchmark(
    model: Annotated[
        list[str] | None, typer.Option(help="Only these models. Defaults to all of them.")
    ] = None,
) -> None:
    """Measure the models, one process each. Rerunning is quick, and re-measures the timings."""
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    keys = model or list(MODELS)
    unknown = [key for key in keys if key not in MODELS]
    if unknown:
        typer.echo(f"Unknown models: {', '.join(unknown)}. Known: {', '.join(MODELS)}", err=True)
        raise typer.Exit(1)
    failed = _run_each("benchmark-one", keys)
    write_model_summary()
    _exit_if_failed(failed)


def write_model_summary() -> None:
    """Rebuild the tables and report the current leader from every saved result."""
    from embedding_model_bench.benchmark import summary_table  # noqa: PLC0415
    from embedding_model_bench.compare import (  # noqa: PLC0415
        apply_selection_rule,
        comparison_table,
    )
    from embedding_model_bench.files import read_json  # noqa: PLC0415
    from embedding_model_bench.models import BenchmarkResult  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL, MODELS  # noqa: PLC0415

    out_dir = get_settings().results_dir / "models"
    paths = [out_dir / f"{key}.json" for key in MODELS if (out_dir / f"{key}.json").exists()]
    results = [read_json(path, BenchmarkResult) for path in paths]
    if not results:
        return
    report = summary_table(results)
    if any(r.spec.key == BGE_SMALL.key for r in results):
        report += "\n" + comparison_table(results, BGE_SMALL.key)
    leader = apply_selection_rule(results)
    report += f"\nSelection rule, on these models only: {leader.spec.key}.\n"
    if any(r.machine.timings_provisional for r in results):
        report += "\nTimings are provisional: competing apps were open or memory was low.\n"
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    typer.echo(report)


@app.command(name="prefixes-one", hidden=True)
def prefixes_one(key: str) -> None:
    """Test one model's prefixes. `prefixes` runs this once per process."""
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.files import write_json  # noqa: PLC0415
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415
    from embedding_model_bench.prefixes import run_prefix_ablation  # noqa: PLC0415
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    with heavy_step(_lock_path(settings)):
        result = run_prefix_ablation(
            MODELS[key],
            inputs,
            settings.data_dir / "embeddings",
            SentenceTransformerEncoder,
            _devices()[0],
            log=_logger(key),
        )
    write_json(settings.results_dir / "prefixes" / f"{key}.json", result)


@app.command()
def prefixes() -> None:
    """Measure each model with and without its prefixes. This measures quality, not speed."""
    from embedding_model_bench.files import read_json  # noqa: PLC0415
    from embedding_model_bench.prefixes import (  # noqa: PLC0415
        PrefixResult,
        prefix_table,
        variants_for,
    )
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    keys = [key for key, spec in MODELS.items() if variants_for(spec)]
    failed = _run_each("prefixes-one", keys)
    out_dir = get_settings().results_dir / "prefixes"
    paths = [out_dir / f"{key}.json" for key in keys if (out_dir / f"{key}.json").exists()]
    results = [read_json(path, PrefixResult) for path in paths]
    table = prefix_table(results)
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)
    for result in results:
        typer.echo(f"Peak RSS, {result.spec.key}: {result.peak_rss_mb:.0f} MB")
    _exit_if_failed(failed)


@app.command()
def matryoshka() -> None:
    """Measure recall against storage at shorter vector lengths. This doesn't measure speed."""
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.files import write_json  # noqa: PLC0415
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415
    from embedding_model_bench.matryoshka import (  # noqa: PLC0415
        TRUNCATIONS,
        matryoshka_table,
        run_matryoshka,
    )
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    out_dir = settings.results_dir / "matryoshka"
    results = []
    with heavy_step(_lock_path(settings)):
        for key, truncation in TRUNCATIONS.items():
            spec = MODELS[key]
            result = run_matryoshka(
                spec,
                truncation,
                inputs,
                settings.data_dir / "embeddings",
                lambda spec=spec: SentenceTransformerEncoder(spec, _devices()[0]),
                log=_logger(key),
            )
            write_json(out_dir / f"{key}.json", result)
            results.append(result)
    table = matryoshka_table(results)
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)
    typer.echo(f"Peak RSS: {max(r.peak_rss_mb for r in results):.0f} MB")


@app.command()
def quantization() -> None:
    """Measure float32, int8 and binary vectors: quality, memory and search time."""
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.files import write_json  # noqa: PLC0415
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415
    from embedding_model_bench.quantization import (  # noqa: PLC0415
        CONFIGS,
        quantization_table,
        run_quantization,
    )
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    out_dir = settings.results_dir / "quantization"
    results = []
    with heavy_step(_lock_path(settings)):
        for config in CONFIGS:
            spec = MODELS[config.key]
            name = f"{config.key}-{config.dim}"
            result = run_quantization(
                spec,
                config.dim,
                inputs,
                settings.data_dir / "embeddings",
                lambda spec=spec: SentenceTransformerEncoder(spec, _devices()[0]),
                log=_logger(name),
            )
            write_json(out_dir / f"{name}.json", result)
            results.append(result)
    table = quantization_table(results)
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)
    typer.echo(f"Peak RSS: {max(r.peak_rss_mb for r in results):.0f} MB")
    for result in results:
        _report_provisional(f"{result.spec.key}-{result.dim}", result.machine)


@app.command()
def cost() -> None:
    """Project storage, query latency and embedding time to 100k and 1M chunks. Timing-sensitive."""
    from embedding_model_bench.cost import CONFIGS, cost_table, run_cost  # noqa: PLC0415
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.files import read_json, write_json  # noqa: PLC0415
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415
    from embedding_model_bench.models import BenchmarkResult  # noqa: PLC0415
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    out_dir = settings.results_dir / "cost"
    results = []
    with heavy_step(_lock_path(settings)):
        for config in CONFIGS:
            spec = MODELS[config.key]
            benchmark = read_json(
                settings.results_dir / "models" / f"{spec.key}.json", BenchmarkResult
            )
            result = run_cost(
                config,
                spec,
                benchmark,
                inputs,
                settings.data_dir / "embeddings",
                settings.data_dir / "cost_scratch",
                lambda spec=spec: SentenceTransformerEncoder(spec, _devices()[0]),
                log=_logger("cost"),
            )
            slug = f"{config.key}-{config.dim}-{config.scheme.name.lower()}"
            write_json(out_dir / f"{slug}.json", result)
            results.append(result)
    table = cost_table(results)
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)
    typer.echo(f"Peak RSS: {max(r.peak_rss_mb for r in results):.0f} MB")
    for result in results:
        _report_provisional(result.config.name, result.machine)


# The held-out run.

HOLDOUT_RESULT = "result.json"
FRESH_NOTE = (
    "Headline. 30 questions written by Claude from randomly sampled chunks, frozen (hash-checked) "
    "before any benchmark ran, and never scored before this run."
)
SPENT_NOTE = (
    "Previously used: Fine Print scored these 20 questions for its v2 and v3, so this is not a "
    "clean estimate. Reported separately, as agreed."
)


def _holdout_sets(
    settings: Settings, inputs: BenchmarkInputs, tuning_check: bool
) -> list[tuple[str, str, str, list[EvalQuestion]]]:
    """Return each question set to score, as (name, note, sha256, questions)."""
    from embedding_model_bench.data import LINK_FOLDER  # noqa: PLC0415
    from embedding_model_bench.files import read_jsonl  # noqa: PLC0415
    from embedding_model_bench.models import EvalQuestion  # noqa: PLC0415

    link_dir = settings.data_dir / LINK_FOLDER

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if tuning_check:
        path = link_dir / "tuning_questions.jsonl"
        return [
            (
                "Tuning set (pipeline check)",
                "Must match the stored results.",
                sha(path),
                list(inputs.questions),
            )
        ]
    fresh = settings.fresh_holdout_path
    if sha(fresh) != settings.fresh_holdout_sha256:
        typer.echo("The fresh held-out file doesn't match its frozen hash.", err=True)
        raise typer.Exit(1)
    spent = link_dir / "holdout" / "questions.jsonl"  # Its hash was checked with the manifest.
    return [
        ("Fresh held-out set", FRESH_NOTE, sha(fresh), read_jsonl(fresh, EvalQuestion)),
        (
            "Fine Print's held-out set (previously used)",
            SPENT_NOTE,
            sha(spent),
            read_jsonl(spent, EvalQuestion),
        ),
    ]


@app.command(name="holdout-embed", hidden=True)
def holdout_embed(
    tuning_check: Annotated[bool, typer.Option(help="Use the tuning set.")] = False,
) -> None:
    """Embed the held-out questions with every configuration's model. Loads models on a miss."""
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.holdout import CONFIGS, config_spec, embed_all  # noqa: PLC0415
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    specs = list(
        {c.name: config_spec(c, settings.results_dir / "finetune") for c in CONFIGS}.values()
    )
    texts = [
        q.question
        for _, _, _, questions in _holdout_sets(settings, inputs, tuning_check)
        for q in questions
    ]
    device = _devices()[0]
    with heavy_step(_lock_path(settings)):
        embed_all(
            specs,
            texts,
            inputs.chunks,
            settings.data_dir / "embeddings",
            lambda spec: SentenceTransformerEncoder(spec, device),
        )


def _never_load() -> Encoder:
    raise RuntimeError("A vector wasn't cached: run the embedding step first.")


@app.command(name="holdout-score", hidden=True)
def holdout_score(
    tuning_check: Annotated[bool, typer.Option(help="Use the tuning set.")] = False,
) -> None:
    """Score every configuration from cached vectors only. Never loads a model."""
    from embedding_model_bench.files import write_json  # noqa: PLC0415
    from embedding_model_bench.holdout import (  # noqa: PLC0415
        CONFIGS,
        HoldoutResult,
        config_spec,
        holdout_table,
        score_set,
    )

    settings = get_settings()
    inputs = _prepared_inputs()
    specs = {c.name: config_spec(c, settings.results_dir / "finetune") for c in CONFIGS}
    sets = [
        score_set(
            name,
            note,
            sha,
            questions,
            inputs.chunks,
            specs,
            cache_dir=settings.data_dir / "embeddings",
            load=_never_load,
        )
        for name, note, sha, questions in _holdout_sets(settings, inputs, tuning_check)
    ]
    result = HoldoutResult(
        fine_print_commit=inputs.fine_print_commit,
        corpus_fingerprint=inputs.corpus_fingerprint,
        sets=sets,
    )
    table = holdout_table(result)
    typer.echo(table)
    if tuning_check:
        _check_against_tuning(result.sets[0], settings)
        return
    out_dir = settings.results_dir / "holdout"
    write_json(out_dir / HOLDOUT_RESULT, result)
    (out_dir / "summary.md").write_text(table, encoding="utf-8")


def _check_against_tuning(held: HoldoutSet, settings: Settings) -> None:
    """Compare each configuration's tuning scores with the experiment that first measured them."""
    from embedding_model_bench.files import read_json  # noqa: PLC0415
    from embedding_model_bench.finetune import FineTuneResult  # noqa: PLC0415
    from embedding_model_bench.matryoshka import MatryoshkaResult  # noqa: PLC0415
    from embedding_model_bench.models import BenchmarkResult  # noqa: PLC0415
    from embedding_model_bench.quantization import QuantizationResult, Scheme  # noqa: PLC0415
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    results = settings.results_dir
    failures = []
    for row in held.rows:
        config = row.config
        if config.fine_tuned:
            expected = read_json(results / "finetune" / "all.json", FineTuneResult).tuning
        elif config.scheme != Scheme.FLOAT32:
            quant = read_json(
                results / "quantization" / f"{config.key}-{config.dim}.json", QuantizationResult
            )
            expected = next(s.tuning for s in quant.schemes if s.scheme == config.scheme)
        elif config.dim < MODELS[config.key].dim:
            mat = read_json(results / "matryoshka" / f"{config.key}.json", MatryoshkaResult)
            expected = next(v.tuning for v in mat.variants if v.dim == config.dim)
        else:
            expected = read_json(results / "models" / f"{config.key}.json", BenchmarkResult).tuning
        status = "ok" if row.scores == expected else "DIFFERENT"
        if status != "ok":
            failures.append(config.name)
        typer.echo(f"[{status}] {config.name}")
    if failures:
        raise typer.Exit(1)


@app.command()
def holdout(
    tuning_check: Annotated[
        bool, typer.Option(help="Run the same steps on the tuning set, to check the pipeline.")
    ] = False,
) -> None:
    """Score the final configurations on the held-out sets. Runs once; refuses a second time."""
    settings = get_settings()
    done = settings.results_dir / "holdout" / HOLDOUT_RESULT
    if not tuning_check and done.exists():
        typer.echo(
            f"The held-out run has already happened ({done}). It is run once only.", err=True
        )
        raise typer.Exit(1)
    extra = ["--tuning-check"] if tuning_check else []
    base = [sys.executable, "-m", "embedding_model_bench.cli"]
    for step in ("holdout-embed", "holdout-score"):
        if subprocess.run([*base, step, *extra], check=False).returncode:
            typer.echo(f"{step} failed.", err=True)
            raise typer.Exit(1)


# Synthetic data and fine-tuning.


@app.command()
def generate(
    limit: Annotated[
        int | None, typer.Option(help="Stop after this many new chunks. Omit to finish the run.")
    ] = None,
) -> None:
    """Generate synthetic questions with Ollama. Safe to stop and rerun: it resumes."""
    import httpx  # noqa: PLC0415

    from embedding_model_bench.machine import heavy_step, peak_rss_mb  # noqa: PLC0415
    from embedding_model_bench.synthetic import (  # noqa: PLC0415
        OllamaGenerator,
        output_path,
        run_generation,
    )

    settings = get_settings()
    _require_preflight("generation", settings)
    _, chunks = _verified_data(settings)
    with heavy_step(_lock_path(settings)), httpx.Client() as client:
        generator = OllamaGenerator(
            settings.ollama_url, settings.ollama_model, settings.ollama_generate_timeout, client
        )
        path = output_path(settings.data_dir, generator.digest)
        typer.echo(f"Writing to {path}")
        try:
            summary = run_generation(
                chunks,
                path,
                generator,
                settings.ollama_model,
                generator.digest,
                limit=limit,
                log=lambda message: typer.echo(message, err=True),
            )
        except KeyboardInterrupt:
            typer.echo("Stopped. Everything finished so far is saved; rerun to continue.")
            raise typer.Exit(130) from None
        finally:
            generator.unload()
    typer.echo(summary.model_dump_json(indent=2))
    typer.echo(f"Peak RSS of this process: {peak_rss_mb():.0f} MB (Ollama's memory is separate)")


def _questions(path: Path) -> list[str]:
    """Read just the question text from a question file, of any of the eval formats."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["question"] for line in lines if line.strip()]


@app.command(name="prepare-training")
def prepare_training() -> None:
    """Filter the synthetic questions, check them for leakage and mine hard negatives."""
    from embedding_model_bench.data import LINK_FOLDER  # noqa: PLC0415
    from embedding_model_bench.embedding import (  # noqa: PLC0415
        SentenceTransformerEncoder,
        cached_embedder,
    )
    from embedding_model_bench.files import write_json, write_jsonl  # noqa: PLC0415
    from embedding_model_bench.machine import heavy_step, peak_rss_mb  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL  # noqa: PLC0415
    from embedding_model_bench.synthetic import (  # noqa: PLC0415
        PROMPT_VERSION,
        load_records,
        unique_chunks,
    )
    from embedding_model_bench.training_data import prepare_training_data  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    files = sorted((settings.data_dir / "synthetic").glob(f"questions-v{PROMPT_VERSION}-*.jsonl"))
    if len(files) != 1:
        typer.echo(f"Expected one synthetic question file, found {len(files)}", err=True)
        raise typer.Exit(1)
    records = load_records(files[0])
    expected = len(unique_chunks(inputs.chunks))
    if len(records) < expected:
        typer.echo(f"Only {len(records)} of {expected} chunks have questions yet.", err=True)
        typer.echo("Finish `make generate` first.", err=True)
        raise typer.Exit(1)

    fresh = settings.fresh_holdout_path.read_bytes()
    if hashlib.sha256(fresh).hexdigest() != settings.fresh_holdout_sha256:
        typer.echo("The fresh held-out file doesn't match its frozen hash.", err=True)
        raise typer.Exit(1)

    link_dir = settings.data_dir / LINK_FOLDER
    # Used only to keep these questions out of training data. Never printed or saved.
    eval_sets = {
        "tuning": inputs.question_texts,
        "tuning out of scope": _questions(link_dir / "tuning_out_of_scope.jsonl"),
        "Fine Print held-out (spent)": _questions(link_dir / "holdout" / "questions.jsonl"),
        "Fine Print held-out out of scope": _questions(link_dir / "holdout" / "out_of_scope.jsonl"),
        "fresh held-out": _questions(settings.fresh_holdout_path),
    }

    embed = cached_embedder(
        BGE_SMALL,
        settings.data_dir / "embeddings",
        lambda: SentenceTransformerEncoder(BGE_SMALL, _devices()[0]),
    )
    with heavy_step(_lock_path(settings)):
        examples, report = prepare_training_data(
            records,
            inputs.chunks,
            embed(inputs.chunk_texts, BGE_SMALL.passage_prefix),
            lambda texts: embed(texts, BGE_SMALL.query_prefix),
            eval_sets,
            log=lambda message: typer.echo(message, err=True),
        )

    out_dir = settings.data_dir / "training"
    for split in ("train", "validation"):
        write_jsonl(out_dir / f"{split}.jsonl", (e for e in examples if e.split == split))
    write_json(settings.results_dir / "synthetic" / "report.json", report)
    typer.echo(report.model_dump_json(indent=2))
    typer.echo(f"Peak RSS: {peak_rss_mb():.0f} MB")


@app.command(name="finetune-one", hidden=True)
def finetune_one(
    name: str,
    smoke: Annotated[bool, typer.Option(help="Train 4 steps only, to test the pipeline.")] = False,
) -> None:
    """Fine-tune and score one run. `finetune` runs this once per process."""
    # Cap the GPU memory pool before torch starts, so that running short raises an error instead
    # of pushing everything else into swap. Ratios are of macOS's recommended GPU working set.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", MPS_HIGH_WATERMARK)
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", MPS_LOW_WATERMARK)
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.files import write_json  # noqa: PLC0415
    from embedding_model_bench.finetune import (  # noqa: PLC0415
        TrainConfig,
        choose_unseen_companies,
        evaluate_fine_tuned,
        load_examples,
        train,
    )
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    data = settings.data_dir / "training"
    examples = load_examples(data / "train.jsonl") + load_examples(data / "validation.jsonl")
    unseen = choose_unseen_companies(inputs.questions)
    config = TrainConfig(
        name=f"smoke-{name}" if smoke else name,
        excluded_companies=tuple(unseen) if name == "unseen" else (),
        max_steps=SMOKE_STEPS if smoke else -1,
    )
    baseline = _baseline(settings)
    device = _devices()[0]
    log = _logger(config.name)
    with heavy_step(_lock_path(settings)):
        run = train(config, examples, inputs.chunks, settings.models_dir, device, log)
        log(f"trained in {run.seconds:.0f} s; scoring")
        result = evaluate_fine_tuned(
            run,
            inputs,
            baseline,
            unseen,
            settings.data_dir / "embeddings",
            lambda spec: SentenceTransformerEncoder(spec, device),
        )
    out_dir = settings.models_dir if smoke else settings.results_dir / "finetune"
    write_json(out_dir / f"{config.name}.json", result)
    log(f"peak RSS {run.peak_rss_mb:.0f} MB")


@app.command()
def finetune(
    smoke: Annotated[bool, typer.Option(help="Train 4 steps per run, as a test.")] = False,
) -> None:
    """Fine-tune bge-small on all companies, then without three, and compare with the baseline."""
    from embedding_model_bench.files import read_json  # noqa: PLC0415
    from embedding_model_bench.finetune import FineTuneResult, finetune_table  # noqa: PLC0415

    settings = get_settings()
    failed = _run_each("finetune-one", FINETUNE_RUNS, *(["--smoke"] if smoke else []))
    if smoke:
        typer.echo("Smoke test done." if not failed else f"Smoke test failed: {failed}")
        raise typer.Exit(1 if failed else 0)
    out_dir = settings.results_dir / "finetune"
    paths = [out_dir / f"{n}.json" for n in FINETUNE_RUNS if (out_dir / f"{n}.json").exists()]
    results = [read_json(path, FineTuneResult) for path in paths]
    if results:
        table = finetune_table(results, _baseline(settings))
        (out_dir / "summary.md").write_text(table, encoding="utf-8")
        typer.echo(table)
    _exit_if_failed(failed)


if __name__ == "__main__":
    app()
