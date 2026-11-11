# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- `.venv` is managed by uv. Run things with `uv run ...` or through the Makefile.
- Add dependencies with `uv add <package>` or `uv add --dev <package>`, which also update `uv.lock`.
- Never use `pip` or `pip3`: on this machine `pip3` resolves to the global Python. Never install globally.
- Python is pinned to 3.14.
- This repo's git email is the GitHub noreply address, set in `.git/config`. The global setting is a personal address, so check `git config user.email` in any new clone.
- Commit messages follow Conventional Commits.

## Project

Answers a client question: which embedding model should we use for our documents, and is fine-tuning worth it? Everything runs locally and free. The corpus is Fine Print's Terms of Service and privacy policies, with BEIR SciFact as an out-of-domain check.

## Fine Print is read-only

`../fine-print` is an editable path dependency and the source of the data. **Never write into it.**

- `data/fine_print/` holds symlinks into it, so a write there lands in Fine Print. Only ever read from that folder.
- Importing the package sets `sys.dont_write_bytecode`, because importing `fine_print` would otherwise write `.pyc` files there. Keep that first in `__init__.py`.
- Import only `fine_print.chunking`, `.ingest`, `.models` and `.config`. `.evaluate`, `.store` and `.embed` load chromadb or torch.
- Chunk settings are read from Fine Print's field defaults (`corpus.chunk_settings`), not repeated here.
- The chunker is editable, so a change in Fine Print changes the chunks. The chunk fingerprint and Fine Print's commit are recorded in `data/manifest.json`. `make data-verify` fails if either the chunks or any file in the tree changed, and must pass after every step.
- Run git there with `--no-optional-locks`, which stops status commands writing to its index.
- The one exception is the final integration experiment, which uses a new branch in a separate worktree and is never merged.

## Sharing downloads

- Models and SciFact come from the shared Hugging Face cache, and Ollama models from its own store. Never set a custom cache directory.
- Report what is cached and the extra disk needed before downloading models.
- Reuse Fine Print's bge-small vectors only if they are provably identical: same model revision, settings and chunk text. Its Chroma collections don't record a model or revision, so today they fail that test and are recomputed.

## Memory

This machine has 16 GB of RAM, and `gemma4:e4b` needs 6-7 GB.

- Run `uv run emb-bench preflight benchmark` or `preflight generation` before benchmarks, training or synthetic generation. If Fine Print's API or UI is running, ask the user to stop it. Never stop it yourself.
- Never run Ollama generation at the same time as a benchmark or training.
- Report peak RAM for each step.

## Evaluation rules

- **Choose models and settings on the tuning set only** (`data/fine_print/tuning_questions.jsonl`).
- **Run the held-out set once, at the very end,** for the final 2-3 configurations, and report it as-is. Change nothing because of what it shows.
- **Don't read held-out questions except for the leakage check.** Counting and hashing them is fine.
- **The headline held-out result comes from `eval/fresh_holdout/`,** 30 questions frozen before any benchmark ran. They were written by Claude, so never describe them as human or independent. Never edit them: a test checks the file's hash.
- **`eval/holdout/` in Fine Print is already spent:** it was scored for its v2 and v3, and those scores are in its README. Report it separately, labelled as previously used, and never as a clean estimate.
- **Timings count only when no competing app is running.** A result with `timings_provisional: true` has valid quality scores, but its speeds can't be reported as final. Remind the user to close Chrome and Teams before each timing-sensitive step, and re-check memory and swap.
- **No eval question, tuning or held-out, may appear in training data.** Drop synthetic queries that are near-duplicates of eval questions, and report how many were dropped.
- **Report the number of questions with every result,** and say when a gap is within one or two questions, which is likely noise. One question is 3.2 points on the tuning set (31) and 5 points on the Fine Print held-out set (20).
- **Selection rule, fixed before any results:** the highest tuning recall@5 wins. Between configurations within two questions of each other, the smaller and faster one wins.
- Fine Print's "recall@k" means at least one chunk in the top k from the right document and containing every keyword. Label it that way. SciFact uses standard BEIR recall.

## Commands

```bash
make setup        # create or repair .venv from uv.lock, and install the git hooks
make check        # lint (ruff), typecheck (ty) and test (pytest, coverage at least 70%), as CI does
make format       # ruff fix and format
make env          # hardware, MPS, Ollama, and what's in the shared cache
make data         # symlink Fine Print's data, write the manifest, fetch SciFact
make data-verify  # check Fine Print hasn't changed
make baseline     # measure bge-small and compare it with Fine Print's published scores
make benchmark    # measure every model, one process each; a rerun re-measures timings only
make prefixes     # score each model with and without its prefixes
make matryoshka   # score shorter vectors against storage, from cached vectors
uv run pytest tests/test_x.py::test_y   # a single test
```

Tests use no network, models, Ollama or real data, only tiny fakes. They do need `../fine-print`, because the chunker is imported from it. CI clones Fine Print at a pinned commit and installs CPU-only torch on Linux. Fakes such as `FakeEncoder` and `make_chunk` are in `tests/conftest_fakes.py`, and `tests/conftest.py` builds a tiny fake Fine Print tree.

## Architecture

- **`cli.py`** is the entry point (`emb-bench`, built with Typer). It imports torch and sentence-transformers inside each command, so `--help` and `preflight` start fast. Keep heavy imports out of module scope there.
- **One process per model or run.** `benchmark`, `prefixes` and `finetune` run hidden `*-one` commands as subprocesses (`cli._run_each`), so each model's peak RAM is measured alone and its memory is freed afterwards. Each writes `results/<step>/<model>.json`, then the parent writes `summary.md`.
- **Heavy-step lock.** Model work runs inside `machine.heavy_step(results/.heavy.lock)`, an OS `flock` that refuses while another process holds it. The kernel releases it when the holder dies, so a leftover file never blocks. Synthetic generation and training use the same lock.
- **Vector cache** (`embedding.embed_cached`, in `data/embeddings/`). The key hashes the repo, revision, prefix and exact text. That is why reruns only re-measure timings, and why `matryoshka` needs no model. Prefixes are added in `embed_cached`, not through sentence-transformers' `prompts`, so the key covers them. Files are written to a temporary name and then renamed.
- **Models** are pinned to exact revisions in `registry.py` (`MODELS`). Add a model there, not in the CLI.
- **Scoring:** `metrics.py` does exact search, Fine Print's keyword recall@k and BEIR metrics. `compare.py` does paired per-question comparisons and `noise_note`, the "within N questions" wording. Use these for any new experiment, not ad hoc comparisons.
- **Data:** `corpus.py` chunks with Fine Print's own code and fingerprints the result. `data.py` creates the symlinks and writes and checks `data/manifest.json`. `scifact.py` reads pinned SciFact revisions from the HF cache.
- **Settings** are in `config.py` and can be overridden with `EMB_BENCH_*` or `.env`. The fresh held-out file's SHA-256 is pinned there.
- **Experiment modules** take an `EncoderFactory` or `Callable[[], Encoder]` and load the model only on a cache miss, so tests can pass a fake encoder. Build on the shared pieces rather than repeating them:
  - `embedding.cached_embedder` embeds through the cache and loads the model at most once.
  - `benchmark.BenchmarkInputs` computes texts and relevance sets once per process; `encode_inputs` and `score_encoded` encode and score both corpora, with an optional transform such as truncation.
  - `models.RetrievalScores.hits(k)` / `hit_flags(k)` count hits; `compare.paired_comparison(..., rows=)` compares on a subset.
  - `report.markdown_table` renders every summary; `files.read_json`, `read_jsonl`, `write_json` and `write_jsonl` read and write pydantic models.
- **Regression check after refactors:** recompute results from the cache with a factory that raises on a cache miss, and compare with the committed JSON. Quality fields must match exactly; only timings, peak RSS and machine state may differ. FAISS code must run in a separate process from anything that loads torch.

## Status

- **Done:** setup, data, fresh held-out set frozen, baseline, model benchmark, prefix ablation. bge-small reproduces Fine Print's published tuning scores, and SciFact nDCG@10 is within 0.011 of MTEB's for all six models. Results are in `results/models/`. Quality scores are final, but timings are provisional because Chrome was open.
- **Decisions:** EmbeddingGemma was replaced by `intfloat/e5-base-v2`, because there is no Hugging Face login on this machine. Nomic runs without `trust_remote_code`, which current transformers doesn't need; the README must say so. The unseen-companies fine-tuning test is in, labelled as a rough signal of about 9 questions.
- **Granite finding:** a hand read of the misses (`results/misses/hand_read.md`, not blind) shows the keyword test undercounts every model by 3 to 5 questions, and doesn't explain Granite's gap. Credited hit@5: bge-base 27/31, bge-small 25/31, Granite 16/31. Reported scores still use the keyword test.
- **Prefix finding:** keep prefixes as on each card. Only Arctic depends on its prefix, and no combination beat the card by more than noise. Results are in `results/prefixes/`.
- **Matryoshka finding:** results are in `results/matryoshka/`. bge-base, not trained for truncation, keeps 23/31 down to 384 dimensions and 22/31 at 256, but SciFact falls steadily (.740 to .695 at 256, 3 wins / 23 losses). Applied to lengths, the selection rule picks bge-base at 256; keep full-length bge-base and bge-small as comparisons in the final configurations.
- **Timings are final:** `make benchmark` was rerun with no competing apps (Arctic and Nomic twice, after Chrome relaunched itself). Results are in `results/models/`.
- **Quantization finding:** results are in `results/quantization/` (FAISS, exact indexes, 1 thread, 100k vectors). int8 is free in quality but slower than float32 in FAISS flat search. Binary with float32 rescoring keeps tuning hit@5 for every configuration at 1/32 of the memory, but costs SciFact quality. Pure binary loses clearly on SciFact.
- **Duplicate chunks:** 43 Lyft chunks are identical in its terms and privacy files (`results/data/duplicate_chunks.md`). Exact ties go to the lower index everywhere; flipping that changes no hit@5 but moves MRR by about 0.03 for every model.
- **torch and FAISS can't share a process on macOS** (two OpenMP runtimes abort it). `quantization.load_faiss` refuses if torch is loaded. Keep torch out of tests that use FAISS.
- **Synthetic data done:** 1,520 unique chunks, 3,036 questions from gemma4:e4b in 67 minutes (`make generate`, resumable; data in `data/synthetic/`). `make prepare-training` writes `data/training/{train,validation}.jsonl` and `results/synthetic/report.json`: 2,124 train and 249 validation examples, split by chunk. Leakage dropped 427 questions (226 near tuning, 71 near the spent held-out set, 174 near the fresh held-out set; sets overlap). The leakage cosine was tightened from 0.85 to 0.80 and the round-trip cut-off widened from 20 to 100 after reading samples, before any training; reasons are in `training_data.REVISIONS` and the report. 724 examples have no hard negative and rely on in-batch negatives.
- **Fine-tuning finding:** results are in `results/finetune/`. Validation nDCG@10 on synthetic questions rose from 0.535 to 0.661, but on the 31 human tuning questions hit@5 went 20 -> 22 (3 wins / 1 loss, p 0.63: noise), MRR fell 0.618 -> 0.603, and SciFact nDCG@10 fell 0.713 -> 0.685 (5 / 16, p 0.027). Unseen companies (Discord, Spotify, Zoom; n=9): 5 -> 5 for both models. Untuned bge-base (23/31) still leads. Training needed the cached loss, mini-batch 8, eager attention, MPS cache emptying and an MPS memory cap; plain settings thrashed swap.
- **Quantization was rerun after the refactor** on a clean machine: quality identical, search p95 within about 3%, peak RSS 1.1-1.3 GB.
- **Cost finding:** results are in `results/cost/` (exact FAISS, 1 thread, repeated real vectors; clean machine). At 1M chunks, bge-base float32 needs 3.1 GB of RAM and 62 ms search p95; binary rescored with float32 vectors from disk needs 96 MB of RAM and 3.7 ms, with tuning hit@5 unchanged (23/31) and SciFact nDCG@10 0.732 against 0.740. int8 is the slowest in FAISS flat search (129 ms at 1M). Embedding 1M chunks is projected at 5.7 h for bge-base and 2.2 h for bge-small on this M1 Pro's GPU. The cost run's peak RSS (7.5 GB) counts memory-mapped corpus pages it read.
- **Next:** the held-out run, once, for the final configurations: bge-base 768 float32, bge-base 768 binary rescored with float32, bge-small 384 float32 (baseline), and the fine-tuned bge-small. Fresh 30 as the headline; Fine Print's spent 20 separately, labelled as previously used. Then the README with charts, then the Fine Print integration branch.
