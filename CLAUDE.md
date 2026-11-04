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
make check        # lint, typecheck and test, as CI does
make env          # hardware, MPS, Ollama, and what's in the shared cache
make data         # symlink Fine Print's data, write the manifest, fetch SciFact
make data-verify  # check Fine Print hasn't changed
make baseline     # measure bge-small and compare it with Fine Print's published scores
make benchmark    # measure every model, one process each; a rerun re-measures timings only
make prefixes     # score each model with and without its prefixes
make matryoshka   # score shorter vectors against storage, from cached vectors
uv run pytest tests/test_x.py::test_y   # a single test
```

Tests use no network, models, Ollama or real data, only tiny fakes. They do need `../fine-print`, because the chunker is imported from it. CI clones Fine Print at a pinned commit and installs CPU-only torch on Linux.

## Status

- **Done:** setup, data, fresh held-out set frozen, baseline, model benchmark, prefix ablation. bge-small reproduces Fine Print's published tuning scores, and SciFact nDCG@10 is within 0.011 of MTEB's for all six models. Results are in `results/models/`. Quality scores are final, but timings are provisional because Chrome was open.
- **Decisions:** EmbeddingGemma was replaced by `intfloat/e5-base-v2`, because there is no Hugging Face login on this machine. Nomic runs without `trust_remote_code`, which current transformers doesn't need; the README must say so. The unseen-companies fine-tuning test is in, labelled as a rough signal of about 9 questions.
- **Granite finding:** a hand read of the misses (`results/misses/hand_read.md`, not blind) shows the keyword test undercounts every model by 3 to 5 questions, and doesn't explain Granite's gap. Credited hit@5: bge-base 27/31, bge-small 25/31, Granite 16/31. Reported scores still use the keyword test.
- **Prefix finding:** keep prefixes as on each card. Only Arctic depends on its prefix, and no combination beat the card by more than noise. Results are in `results/prefixes/`.
- **Matryoshka finding:** results are in `results/matryoshka/`. bge-base, not trained for truncation, keeps 23/31 down to 384 dimensions and 22/31 at 256, but SciFact falls steadily (.740 to .695 at 256, 3 wins / 23 losses). Applied to lengths, the selection rule picks bge-base at 256; keep full-length bge-base and bge-small as comparisons in the final configurations.
- **Next:** quantization. `make benchmark` must be rerun with Chrome closed before any timing is reported as final; embeddings come from the cache.
- **Synthetic generation:** time 20 chunks first and report an estimate. The full run must be resumable, because the user will run it overnight.
