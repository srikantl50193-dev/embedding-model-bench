import sys
from pathlib import Path

import pytest

from embedding_model_bench.corpus import STRATEGY, chunk_settings, corpus_fingerprint, load_chunks
from embedding_model_bench.data import (
    FinePrintDataError,
    build_manifest,
    link_data,
    save_manifest,
    tree_digest,
    verify_manifest,
)


def test_importing_the_package_disables_bytecode_writes() -> None:
    assert sys.dont_write_bytecode is True


def test_missing_fine_print_names_the_path_and_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FinePrintDataError, match=r"EMB_BENCH_FINE_PRINT_DIR") as error:
        link_data(tmp_path / "nowhere", tmp_path / "data")
    assert str(tmp_path / "nowhere" / "data" / "raw") in str(error.value)


def test_empty_raw_folder_is_an_error(fine_print_tree: Path, tmp_path: Path) -> None:
    (fine_print_tree / "data" / "raw" / "acme_terms.md").unlink()
    with pytest.raises(FinePrintDataError, match="no documents"):
        link_data(fine_print_tree, tmp_path / "data")


def test_links_are_relative_symlinks_to_the_sources(fine_print_tree: Path, tmp_path: Path) -> None:
    link_dir = link_data(fine_print_tree, tmp_path / "data")
    raw = link_dir / "raw"
    assert raw.is_symlink()
    assert not raw.readlink().is_absolute()
    assert raw.resolve() == (fine_print_tree / "data" / "raw").resolve()


def test_linking_twice_works_but_never_replaces_a_real_file(
    fine_print_tree: Path, tmp_path: Path
) -> None:
    link_dir = link_data(fine_print_tree, tmp_path / "data")
    link_data(fine_print_tree, tmp_path / "data")
    (link_dir / "raw").unlink()
    (link_dir / "raw").mkdir()
    with pytest.raises(FinePrintDataError, match="not a symlink"):
        link_data(fine_print_tree, tmp_path / "data")


def test_linking_and_building_a_manifest_leave_fine_print_untouched(
    fine_print_tree: Path, tmp_path: Path
) -> None:
    before = tree_digest(fine_print_tree)
    link_dir = link_data(fine_print_tree, tmp_path / "data")
    build_manifest(fine_print_tree, link_dir)
    assert tree_digest(fine_print_tree) == before


def test_manifest_records_counts(fine_print_tree: Path, tmp_path: Path) -> None:
    manifest = build_manifest(fine_print_tree, link_data(fine_print_tree, tmp_path / "data"))
    assert manifest.num_documents == 1
    assert manifest.num_chunks == 2
    assert manifest.num_tuning_questions == 2
    assert manifest.num_holdout_questions == 1
    assert set(manifest.file_sha256) >= {"holdout/questions.jsonl", "tuning_questions.jsonl"}


def test_verify_passes_when_unchanged_and_fails_after_a_write(
    fine_print_tree: Path, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    save_manifest(build_manifest(fine_print_tree, link_data(fine_print_tree, data_dir)), data_dir)
    manifest, chunks = verify_manifest(fine_print_tree, data_dir)
    assert manifest.num_chunks == len(chunks) == 2

    (fine_print_tree / "eval" / "questions.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FinePrintDataError, match="file_sha256"):
        verify_manifest(fine_print_tree, data_dir)


def test_verify_without_a_manifest_says_what_to_run(fine_print_tree: Path, tmp_path: Path) -> None:
    with pytest.raises(FinePrintDataError, match="emb-bench data"):
        verify_manifest(fine_print_tree, tmp_path / "data")


def test_chunking_uses_fine_print_defaults_and_ignores_its_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINE_PRINT_CHUNK_SIZE", "50")
    assert chunk_settings() == (1000, 150)
    assert STRATEGY == "section"


def test_fingerprint_changes_with_the_embedded_text(fine_print_tree: Path) -> None:
    raw = fine_print_tree / "data" / "raw"
    _, chunks = load_chunks(raw)
    assert corpus_fingerprint(chunks) == corpus_fingerprint(load_chunks(raw)[1])
    changed = chunks[0].model_copy(update={"section": "Something else"})
    assert corpus_fingerprint([changed, *chunks[1:]]) != corpus_fingerprint(chunks)
