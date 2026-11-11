"""The command line interface. Heavy libraries are imported inside commands, to start quickly."""

import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import psutil
import typer

from embedding_model_bench.config import get_settings
from embedding_model_bench.preflight import HeavyStep, run_preflight

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from embedding_model_bench.benchmark import BenchmarkInputs
    from embedding_model_bench.models import BenchmarkResult, MachineState

app = typer.Typer(no_args_is_help=True, add_completion=False)

GIB = 1024**3
# Fine Print's README, section retrieval on the tuning set.
FINE_PRINT_PUBLISHED = {"hit@1": 0.58, "hit@3": 0.65, "hit@5": 0.65, "hit@10": 0.68, "MRR": 0.618}


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
    from embedding_model_bench.data import FinePrintDataError, verify_manifest  # noqa: PLC0415

    settings = get_settings()
    try:
        manifest = verify_manifest(settings.fine_print_dir, settings.data_dir)
    except FinePrintDataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Fine Print is unchanged ({manifest.num_chunks} chunks).")


def _prepared_inputs() -> BenchmarkInputs:
    """Run the preflight checks, then load the corpora and tuning questions."""
    from embedding_model_bench.benchmark import (  # noqa: PLC0415
        BenchmarkInputs,
        load_tuning_questions,
    )
    from embedding_model_bench.corpus import load_chunks  # noqa: PLC0415
    from embedding_model_bench.data import (  # noqa: PLC0415
        LINK_FOLDER,
        FinePrintDataError,
        verify_manifest,
    )
    from embedding_model_bench.scifact import load_scifact  # noqa: PLC0415

    settings = get_settings()
    checks = run_preflight("benchmark", settings)
    if not all(check.ok for check in checks):
        for check in checks:
            typer.echo(f"[{'ok' if check.ok else 'STOP'}] {check.name}: {check.detail}", err=True)
        raise typer.Exit(1)
    try:
        manifest = verify_manifest(settings.fine_print_dir, settings.data_dir)
    except FinePrintDataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error

    link_dir = settings.data_dir / LINK_FOLDER
    _, chunks = load_chunks(link_dir / "raw")
    return BenchmarkInputs(
        chunks=chunks,
        questions=load_tuning_questions(link_dir / "tuning_questions.jsonl"),
        scifact=load_scifact(settings),
        fine_print_commit=manifest.fine_print_commit,
        corpus_fingerprint=manifest.corpus_fingerprint,
    )


def _measure(spec_key: str) -> BenchmarkResult:
    """Measure one model, in this process."""
    from embedding_model_bench.benchmark import run_benchmark  # noqa: PLC0415
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    with heavy_step(settings.results_dir / ".heavy.lock"):
        return run_benchmark(
            MODELS[spec_key],
            inputs,
            settings.data_dir / "embeddings",
            SentenceTransformerEncoder,
            _devices(),
            log=lambda message: typer.echo(f"[{spec_key}] {message}", err=True),
        )


def _report_provisional(result: BenchmarkResult) -> None:
    """Say so if competing apps or low memory mean the timings can't be final."""
    if result.machine.timings_provisional:
        apps = ", ".join(result.machine.competing_apps) or "none"
        typer.echo(
            f"[{result.spec.key}] Timings are PROVISIONAL: competing apps: {apps}, "
            f"{result.machine.available_gb} GB available, "
            f"{result.machine.swap_used_gb} GB swap in use."
        )


@app.command()
def baseline() -> None:
    """Measure bge-small and check that it reproduces Fine Print's published tuning scores."""
    from embedding_model_bench.benchmark import summary_table  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL  # noqa: PLC0415

    result = _measure(BGE_SMALL.key)
    out_dir = get_settings().results_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{BGE_SMALL.key}.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    table = summary_table([result])
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)

    tuning = result.tuning
    measured = {
        "hit@1": tuning.hit_at_k[1],
        "hit@3": tuning.hit_at_k[3],
        "hit@5": tuning.hit_at_k[5],
        "hit@10": tuning.hit_at_k[10],
        "MRR": tuning.mrr,
    }
    for name, published in FINE_PRINT_PUBLISHED.items():
        digits = 3 if name == "MRR" else 2
        status = "ok" if round(measured[name], digits) == published else "DIFFERENT"
        typer.echo(f"[{status}] {name}: measured {measured[name]:.3f}, Fine Print {published}")
    _report_provisional(result)


@app.command(name="benchmark-one", hidden=True)
def benchmark_one(key: str) -> None:
    """Measure one model. `benchmark` runs this once per process, to isolate peak memory."""
    result = _measure(key)
    out_dir = get_settings().results_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{key}.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _report_provisional(result)


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
    failed = []
    for key in keys:
        command = [sys.executable, "-m", "embedding_model_bench.cli", "benchmark-one", key]
        if subprocess.run(command, check=False).returncode != 0:
            failed.append(key)
    write_model_summary()
    if failed:
        typer.echo(f"Failed: {', '.join(failed)}", err=True)
        raise typer.Exit(1)


def write_model_summary() -> None:
    """Rebuild the tables and report the current leader from every saved result."""
    from embedding_model_bench.benchmark import summary_table  # noqa: PLC0415
    from embedding_model_bench.compare import (  # noqa: PLC0415
        apply_selection_rule,
        comparison_table,
    )
    from embedding_model_bench.models import BenchmarkResult  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL, MODELS  # noqa: PLC0415

    out_dir = get_settings().results_dir / "models"
    paths = [out_dir / f"{key}.json" for key in MODELS if (out_dir / f"{key}.json").exists()]
    results = [
        BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8")) for path in paths
    ]
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
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415
    from embedding_model_bench.prefixes import run_prefix_ablation  # noqa: PLC0415
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()
    with heavy_step(settings.results_dir / ".heavy.lock"):
        result = run_prefix_ablation(
            MODELS[key],
            inputs,
            settings.data_dir / "embeddings",
            SentenceTransformerEncoder,
            _devices()[0],
            log=lambda message: typer.echo(f"[{key}] {message}", err=True),
        )
    out_dir = settings.results_dir / "prefixes"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{key}.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


@app.command()
def prefixes() -> None:
    """Measure each model with and without its prefixes. This measures quality, not speed."""
    from embedding_model_bench.prefixes import (  # noqa: PLC0415
        PrefixResult,
        prefix_table,
        variants_for,
    )
    from embedding_model_bench.registry import MODELS  # noqa: PLC0415

    keys = [key for key, spec in MODELS.items() if variants_for(spec)]
    failed = []
    for key in keys:
        command = [sys.executable, "-m", "embedding_model_bench.cli", "prefixes-one", key]
        if subprocess.run(command, check=False).returncode != 0:
            failed.append(key)
    out_dir = get_settings().results_dir / "prefixes"
    paths = [out_dir / f"{key}.json" for key in keys if (out_dir / f"{key}.json").exists()]
    results = [PrefixResult.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]
    table = prefix_table(results)
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)
    for result in results:
        typer.echo(f"Peak RSS, {result.spec.key}: {result.peak_rss_mb:.0f} MB")
    if failed:
        typer.echo(f"Failed: {', '.join(failed)}", err=True)
        raise typer.Exit(1)


@app.command()
def matryoshka() -> None:
    """Measure recall against storage at shorter vector lengths. This doesn't measure speed."""
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
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
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with heavy_step(settings.results_dir / ".heavy.lock"):
        for key, truncation in TRUNCATIONS.items():
            spec = MODELS[key]
            result = run_matryoshka(
                spec,
                truncation,
                inputs,
                settings.data_dir / "embeddings",
                lambda spec=spec: SentenceTransformerEncoder(spec, _devices()[0]),
                log=lambda message, key=key: typer.echo(f"[{key}] {message}", err=True),
            )
            (out_dir / f"{key}.json").write_text(
                result.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            results.append(result)
    table = matryoshka_table(results)
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)
    typer.echo(f"Peak RSS: {max(r.peak_rss_mb for r in results):.0f} MB")


@app.command()
def quantization() -> None:
    """Measure float32, int8 and binary vectors: quality, memory and search time."""
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
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
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with heavy_step(settings.results_dir / ".heavy.lock"):
        for config in CONFIGS:
            spec = MODELS[config.key]
            name = f"{config.key}-{config.dim}"
            result = run_quantization(
                spec,
                config.dim,
                inputs,
                settings.data_dir / "embeddings",
                lambda spec=spec: SentenceTransformerEncoder(spec, _devices()[0]),
                log=lambda message, name=name: typer.echo(f"[{name}] {message}", err=True),
            )
            (out_dir / f"{name}.json").write_text(
                result.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            results.append(result)
    table = quantization_table(results)
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    typer.echo(table)
    typer.echo(f"Peak RSS: {max(r.peak_rss_mb for r in results):.0f} MB")
    for result in results:
        _report_provisional_machine(result.spec.key, result.machine)


@app.command()
def generate(
    limit: Annotated[
        int | None, typer.Option(help="Stop after this many new chunks. Omit to finish the run.")
    ] = None,
) -> None:
    """Generate synthetic questions with Ollama. Safe to stop and rerun: it resumes."""
    import httpx  # noqa: PLC0415

    from embedding_model_bench.corpus import load_chunks  # noqa: PLC0415
    from embedding_model_bench.data import (  # noqa: PLC0415
        LINK_FOLDER,
        FinePrintDataError,
        verify_manifest,
    )
    from embedding_model_bench.machine import heavy_step, peak_rss_mb  # noqa: PLC0415
    from embedding_model_bench.synthetic import (  # noqa: PLC0415
        OllamaGenerator,
        output_path,
        run_generation,
    )

    settings = get_settings()
    checks = run_preflight("generation", settings)
    if not all(check.ok for check in checks):
        for check in checks:
            typer.echo(f"[{'ok' if check.ok else 'STOP'}] {check.name}: {check.detail}", err=True)
        raise typer.Exit(1)
    try:
        verify_manifest(settings.fine_print_dir, settings.data_dir)
    except FinePrintDataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    _, chunks = load_chunks(settings.data_dir / LINK_FOLDER / "raw")

    with (
        heavy_step(settings.results_dir / ".heavy.lock"),
        httpx.Client() as client,
    ):
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


@app.command(name="prepare-training")
def prepare_training() -> None:
    """Filter the synthetic questions, check them for leakage and mine hard negatives."""
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    from embedding_model_bench.data import LINK_FOLDER  # noqa: PLC0415
    from embedding_model_bench.embedding import (  # noqa: PLC0415
        SentenceTransformerEncoder,
        embed_cached,
    )
    from embedding_model_bench.machine import heavy_step, peak_rss_mb  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL  # noqa: PLC0415
    from embedding_model_bench.synthetic import (  # noqa: PLC0415
        PROMPT_VERSION,
        load_records,
        unique_chunks,
    )
    from embedding_model_bench.training_data import prepare_training_data  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()  # Also refuses while Ollama holds a model in memory.
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

    def questions(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["question"] for line in lines if line.strip()]

    link_dir = settings.data_dir / LINK_FOLDER
    # Used only to keep these questions out of training data. Never printed or saved.
    eval_sets = {
        "tuning": [q.question for q in inputs.questions],
        "tuning out of scope": questions(link_dir / "tuning_out_of_scope.jsonl"),
        "Fine Print held-out (spent)": questions(link_dir / "holdout" / "questions.jsonl"),
        "Fine Print held-out out of scope": questions(link_dir / "holdout" / "out_of_scope.jsonl"),
        "fresh held-out": questions(settings.fresh_holdout_path),
    }

    cache_dir = settings.data_dir / "embeddings"
    loaded: list[SentenceTransformerEncoder] = []

    def encoder() -> SentenceTransformerEncoder:
        if not loaded:
            loaded.append(SentenceTransformerEncoder(BGE_SMALL, _devices()[0]))
        return loaded[0]

    def embed_queries(texts: Sequence[str]) -> np.ndarray:
        return embed_cached(BGE_SMALL, texts, BGE_SMALL.query_prefix, cache_dir, encoder)[0]

    with heavy_step(settings.results_dir / ".heavy.lock"):
        chunk_texts = [chunk.embedding_text() for chunk in inputs.chunks]
        chunk_vectors = embed_cached(
            BGE_SMALL, chunk_texts, BGE_SMALL.passage_prefix, cache_dir, encoder
        )[0]
        examples, report = prepare_training_data(
            records,
            inputs.chunks,
            chunk_vectors,
            embed_queries,
            eval_sets,
            log=lambda message: typer.echo(message, err=True),
        )

    out_dir = settings.data_dir / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        lines = [e.model_dump_json() for e in examples if e.split == split]
        (out_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    results = settings.results_dir / "synthetic"
    results.mkdir(parents=True, exist_ok=True)
    (results / "report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(report.model_dump_json(indent=2))
    typer.echo(f"Peak RSS: {peak_rss_mb():.0f} MB")


FINETUNE_RUNS = ("all", "unseen")
SMOKE_STEPS = 4


@app.command(name="finetune-one", hidden=True)
def finetune_one(
    name: str,
    smoke: Annotated[bool, typer.Option(help="Train 4 steps only, to test the pipeline.")] = False,
) -> None:
    """Fine-tune and score one run. `finetune` runs this once per process."""
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.finetune import (  # noqa: PLC0415
        TrainConfig,
        choose_unseen_companies,
        evaluate_fine_tuned,
        load_examples,
        train,
    )
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415
    from embedding_model_bench.models import BenchmarkResult  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL  # noqa: PLC0415

    settings = get_settings()
    inputs = _prepared_inputs()  # Also refuses while Ollama holds a model in memory.
    data = settings.data_dir / "training"
    examples = load_examples(data / "train.jsonl") + load_examples(data / "validation.jsonl")
    unseen = choose_unseen_companies(inputs.questions)
    config = TrainConfig(
        name=f"smoke-{name}" if smoke else name,
        excluded_companies=tuple(unseen) if name == "unseen" else (),
        max_steps=SMOKE_STEPS if smoke else -1,
    )
    baseline = BenchmarkResult.model_validate_json(
        (settings.results_dir / "models" / f"{BGE_SMALL.key}.json").read_text(encoding="utf-8")
    )
    device = _devices()[0]
    log = lambda message: typer.echo(f"[{config.name}] {message}", err=True)  # noqa: E731
    with heavy_step(settings.results_dir / ".heavy.lock"):
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{config.name}.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    log(f"peak RSS {run.peak_rss_mb:.0f} MB")


@app.command()
def finetune(
    smoke: Annotated[
        bool, typer.Option(help="Train 4 steps per run, to test the pipeline.")
    ] = False,
) -> None:
    """Fine-tune bge-small on all companies, then without three, and compare with the baseline."""
    from embedding_model_bench.finetune import FineTuneResult, finetune_table  # noqa: PLC0415
    from embedding_model_bench.models import BenchmarkResult  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL  # noqa: PLC0415

    settings = get_settings()
    failed = []
    for name in FINETUNE_RUNS:
        command = [sys.executable, "-m", "embedding_model_bench.cli", "finetune-one", name]
        if subprocess.run([*command, *(["--smoke"] if smoke else [])], check=False).returncode:
            failed.append(name)
    if smoke:
        typer.echo("Smoke test done." if not failed else f"Smoke test failed: {failed}")
        raise typer.Exit(1 if failed else 0)
    out_dir = settings.results_dir / "finetune"
    results = [
        FineTuneResult.model_validate_json((out_dir / f"{n}.json").read_text(encoding="utf-8"))
        for n in FINETUNE_RUNS
        if (out_dir / f"{n}.json").exists()
    ]
    baseline = BenchmarkResult.model_validate_json(
        (settings.results_dir / "models" / f"{BGE_SMALL.key}.json").read_text(encoding="utf-8")
    )
    if results:
        table = finetune_table(results, baseline)
        (out_dir / "summary.md").write_text(table, encoding="utf-8")
        typer.echo(table)
    if failed:
        typer.echo(f"Failed: {', '.join(failed)}", err=True)
        raise typer.Exit(1)


def _report_provisional_machine(key: str, machine: MachineState) -> None:
    """Say so if competing apps or low memory mean timings can't be final."""
    if machine.timings_provisional:
        typer.echo(
            f"[{key}] Timings are PROVISIONAL: competing apps: "
            f"{', '.join(machine.competing_apps) or 'none'}, {machine.available_gb} GB available."
        )


def _devices() -> list[str]:
    """Return the devices to time: MPS first where it exists, then always CPU."""
    import torch  # noqa: PLC0415

    return ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"]


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


if __name__ == "__main__":
    app()
