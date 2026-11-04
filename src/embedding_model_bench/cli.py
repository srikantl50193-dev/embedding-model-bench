"""The command line interface. Heavy libraries are imported inside commands, to start quickly."""

import platform
import sys
from typing import Annotated

import psutil
import typer

from embedding_model_bench.config import get_settings
from embedding_model_bench.preflight import HeavyStep, run_preflight

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


@app.command()
def baseline() -> None:
    """Measure bge-small and check that it reproduces Fine Print's published tuning scores."""
    from embedding_model_bench.benchmark import (  # noqa: PLC0415
        BenchmarkInputs,
        load_tuning_questions,
        run_benchmark,
        summary_table,
    )
    from embedding_model_bench.corpus import load_chunks  # noqa: PLC0415
    from embedding_model_bench.data import (  # noqa: PLC0415
        LINK_FOLDER,
        FinePrintDataError,
        verify_manifest,
    )
    from embedding_model_bench.embedding import SentenceTransformerEncoder  # noqa: PLC0415
    from embedding_model_bench.machine import heavy_step  # noqa: PLC0415
    from embedding_model_bench.registry import BGE_SMALL  # noqa: PLC0415
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
    questions = load_tuning_questions(link_dir / "tuning_questions.jsonl")
    with heavy_step(settings.results_dir / ".heavy.lock"):
        result = run_benchmark(
            BGE_SMALL,
            BenchmarkInputs(
                chunks=chunks,
                questions=questions,
                scifact=load_scifact(settings),
                fine_print_commit=manifest.fine_print_commit,
                corpus_fingerprint=manifest.corpus_fingerprint,
            ),
            settings.data_dir / "embeddings",
            SentenceTransformerEncoder,
            _devices(),
            log=lambda message: typer.echo(message, err=True),
        )

    out_dir = settings.results_dir / "baseline"
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
    if result.machine.timings_provisional:
        apps = ", ".join(result.machine.competing_apps) or "none"
        typer.echo(
            f"Timings are PROVISIONAL: competing apps: {apps}, "
            f"{result.machine.available_gb} GB available, "
            f"{result.machine.swap_used_gb} GB swap in use."
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
