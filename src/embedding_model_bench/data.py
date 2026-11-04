"""Link Fine Print's data in place and check that it is never changed.

Everything is symlinked rather than copied, which saves disk but also means a careless write
under `data/fine_print/` would land in ../fine-print. Nothing here opens those files for
writing, and `verify_manifest` detects a change to the Fine Print tree from any cause.
"""

import hashlib
import os
import subprocess
from pathlib import Path

from embedding_model_bench.corpus import corpus_fingerprint, load_chunks
from embedding_model_bench.models import DataManifest

SOURCES = {
    "raw": Path("data/raw"),
    "tuning_questions.jsonl": Path("eval/questions.jsonl"),
    "tuning_out_of_scope.jsonl": Path("eval/out_of_scope.jsonl"),
    "holdout": Path("eval/holdout"),
}
FILES_TO_HASH = {
    "tuning_questions.jsonl": Path("eval/questions.jsonl"),
    "tuning_out_of_scope.jsonl": Path("eval/out_of_scope.jsonl"),
    "holdout/questions.jsonl": Path("eval/holdout/questions.jsonl"),
    "holdout/out_of_scope.jsonl": Path("eval/holdout/out_of_scope.jsonl"),
}
# .git changes on any git command there, and .venv isn't part of the project's content.
UNWATCHED_DIRS = {".git", ".venv"}
LINK_FOLDER = "fine_print"
MANIFEST_NAME = "manifest.json"


class FinePrintDataError(RuntimeError):
    """Fine Print's data is missing, or differs from when it was linked."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_lines(path: Path) -> int:
    """Count non-empty lines, which reveals nothing about what the questions say."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def check_sources(fine_print_dir: Path) -> None:
    """Check that everything this project reads from Fine Print exists.

    Raises:
        FinePrintDataError: If a path is missing, naming each one and how to fix it.
    """
    required = [*SOURCES.values(), *FILES_TO_HASH.values()]
    missing = [
        str(fine_print_dir / path) for path in required if not (fine_print_dir / path).exists()
    ]
    if missing:
        listed = "\n  ".join(dict.fromkeys(missing))
        raise FinePrintDataError(
            f"Fine Print's data was not found:\n  {listed}\n"
            "Clone fine-print next to this repo and run `make data` there, or set "
            "EMB_BENCH_FINE_PRINT_DIR to where it is."
        )
    if not any((fine_print_dir / SOURCES["raw"]).glob("*.md")):
        raise FinePrintDataError(
            f"{fine_print_dir / SOURCES['raw']} has no documents. Run `make data` in fine-print."
        )


def link_data(fine_print_dir: Path, data_dir: Path) -> Path:
    """Create the symlinks to Fine Print's data, replacing older symlinks only.

    Args:
        fine_print_dir: The Fine Print repo.
        data_dir: This project's data folder.

    Returns:
        The folder holding the symlinks.

    Raises:
        FinePrintDataError: If a source is missing, or a real file is in a symlink's place.
    """
    check_sources(fine_print_dir)
    link_dir = data_dir / LINK_FOLDER
    link_dir.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        link = link_dir / name
        if link.exists() and not link.is_symlink():
            raise FinePrintDataError(f"{link} is not a symlink, so it won't be replaced.")
        link.unlink(missing_ok=True)
        target = os.path.relpath((fine_print_dir / source).resolve(), link_dir.resolve())
        link.symlink_to(target)
    return link_dir


def tree_digest(fine_print_dir: Path) -> str:
    """Hash the path, size and modified time of every file, which detects any write."""
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(fine_print_dir):
        dirs[:] = sorted(d for d in dirs if d not in UNWATCHED_DIRS)
        for name in sorted(files):
            path = Path(root) / name
            stat = path.lstat()
            line = f"{path.relative_to(fine_print_dir)}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
            digest.update(line.encode())
    return digest.hexdigest()


def _git(fine_print_dir: Path, *args: str) -> str:
    """Run a git command there, without the index refresh that would write to its .git."""
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(fine_print_dir), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def build_manifest(fine_print_dir: Path, link_dir: Path) -> DataManifest:
    """Record what the linked data is, including a fingerprint of the chunks.

    The held-out questions are hashed and counted but not parsed.
    """
    num_documents, chunks = load_chunks(link_dir / "raw")
    return DataManifest(
        fine_print_commit=_git(fine_print_dir, "rev-parse", "HEAD") or "unknown",
        fine_print_dirty=bool(_git(fine_print_dir, "status", "--porcelain")),
        file_sha256={name: _sha256(fine_print_dir / p) for name, p in FILES_TO_HASH.items()},
        tree_digest=tree_digest(fine_print_dir),
        num_documents=num_documents,
        num_chunks=len(chunks),
        corpus_fingerprint=corpus_fingerprint(chunks),
        num_tuning_questions=_count_lines(link_dir / "tuning_questions.jsonl"),
        num_holdout_questions=_count_lines(link_dir / "holdout" / "questions.jsonl"),
    )


def save_manifest(manifest: DataManifest, data_dir: Path) -> Path:
    """Write the manifest to this project's data folder, and return its path."""
    path = data_dir / MANIFEST_NAME
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def verify_manifest(fine_print_dir: Path, data_dir: Path) -> DataManifest:
    """Check that Fine Print hasn't changed since its data was linked.

    Returns:
        The saved manifest.

    Raises:
        FinePrintDataError: If there is no manifest, or the corpus, questions or tree changed.
    """
    path = data_dir / MANIFEST_NAME
    if not path.exists():
        raise FinePrintDataError(f"{path} doesn't exist. Run `emb-bench data` first.")
    saved = DataManifest.model_validate_json(path.read_text(encoding="utf-8"))
    current = build_manifest(fine_print_dir, data_dir / LINK_FOLDER)
    changed = [
        field
        for field in ("corpus_fingerprint", "file_sha256", "num_chunks", "tree_digest")
        if getattr(saved, field) != getattr(current, field)
    ]
    if changed:
        raise FinePrintDataError(
            f"Fine Print changed since it was linked ({', '.join(changed)}). If only "
            "tree_digest differs and you ran Fine Print yourself, run `emb-bench data` again. "
            "Otherwise this project wrote to it, which is a bug."
        )
    return saved
