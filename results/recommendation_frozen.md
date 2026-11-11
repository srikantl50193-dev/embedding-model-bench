# Frozen recommendation

Written and committed before any held-out question was scored. It uses tuning-set results only
(n=31), plus SciFact (n=300) as an out-of-domain check and the measured storage and latency. It
will not be changed because of the held-out results. If they contradict it, the README will say
so.

## Recommendation

**Use `BAAI/bge-base-en-v1.5` (768 dimensions, with its query prefix), storing vectors as binary
codes in memory and rescoring the top 40 candidates with float32 vectors kept on disk. Don't
fine-tune.**

At Fine Print's current size (1,581 chunks), plain float32 storage gives the same results and is
simpler. The binary-plus-rescoring setup is what to deploy once the corpus grows towards a
million chunks.

## Why

| Configuration | Tuning hit@5 | MRR | SciFact nDCG@10 | RAM at 1M chunks | Query p95 at 1M, GPU / CPU |
| --- | --- | --- | --- | --- | --- |
| bge-base 768d, binary + float32 rescoring | 23/31 | 0.596 | 0.732 | 96 MB (+3.1 GB on disk) | 17 / 38 ms |
| bge-base 768d, float32 | 23/31 | 0.596 | 0.740 | 3.1 GB | 75 / 96 ms |
| bge-base 256d, float32 | 22/31 | 0.539 | 0.695 | 1.0 GB | 34 / 55 ms |
| bge-small, fine-tuned, float32 | 22/31 | 0.603 | 0.685 | 1.5 GB | 40 / 42 ms (est.) |
| bge-small, float32 (current baseline) | 20/31 | 0.618 | 0.713 | 1.5 GB | 40 / 42 ms |

- **Quality.** bge-base has the best tuning hit@5 of all six models tested (23/31, against 20/31
  for bge-small; 3 wins, 0 losses). It also leads on SciFact (0.740 against 0.713; 22 wins, 7
  losses). Binary storage with float32 rescoring keeps every tuning question (0 wins, 0 losses
  against float32) and loses little on SciFact (2 wins, 3 losses).
- **Storage and latency.** At a million chunks, it needs 1/32 of float32's memory and searches
  in 3.7 ms instead of 62 ms (exact search, one thread).
- **Fine-tuning isn't worth it yet.** On synthetic validation questions it helped a lot (nDCG@10
  0.535 to 0.661), but on the human-written tuning questions it gained two questions (3 wins, 1
  loss, p = 0.63, which is noise) while MRR fell slightly and SciFact fell significantly (5 wins,
  16 losses, p = 0.03). It showed no gain on companies it wasn't trained on (5/9 before and
  after), and it adds a training pipeline to maintain.

## The selection rule, applied literally

The rule fixed before any results says the highest tuning hit@5 wins, and among configurations
within two questions of it, the smaller and faster one. Four configurations are within two
questions of the best (23/31): bge-base at 768d (float32 and binary with rescoring), bge-base at
256d, and fine-tuned bge-small.

The rule's code breaks that tie by model weights, then by dimensions, which picks **fine-tuned
bge-small** (127 MB of weights). That proxy was written when only whole models were compared. On
the measured figures the rule actually names, smaller and faster, binary bge-base with rescoring
wins at scale: the least memory for vectors and the lowest end-to-end query latency at a million
chunks, at the highest tuning score. The proxy also ignores fine-tuning's significant SciFact
loss and its maintenance cost. This recommendation therefore departs from the rule's mechanical
tie-break, and says so here, before the held-out run. Fine-tuned bge-small is in the held-out
run, so the literal pick is tested too.

## Held-out configurations

All five are scored once, against bge-small as the baseline:

1. bge-base 768d, float32
2. bge-base 768d, binary with float32 rescoring (recommended)
3. bge-base 256d, float32 (included: it was one question behind 768d on tuning, not more than
   two)
4. bge-small, fine-tuned on all companies, float32
5. bge-small, float32 (baseline)

The headline set is `eval/fresh_holdout/` (30 questions, written by Claude and frozen before any
benchmark). Fine Print's `eval/holdout` (20 questions) is reported separately and labelled as
previously used.
