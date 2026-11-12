## Fresh held-out set

Headline. 30 questions written by Claude from randomly sampled chunks, frozen (hash-checked) before any benchmark ran, and never scored before this run.

| Configuration | hit@1 | hit@5 | hit@10 | MRR | Wins / losses vs baseline (hit@5) | Sign test p |
| --- | --- | --- | --- | --- | --- | --- |
| bge-base 768d float32 | 7/30 | 20/30 | 23/30 | 0.418 | 1 / 4 | 0.375 |
| bge-base 768d binary + float32 rescoring (recommended) | 7/30 | 20/30 | 23/30 | 0.418 | 1 / 4 | 0.375 |
| bge-base 256d float32 | 6/30 | 19/30 | 23/30 | 0.388 | 1 / 5 | 0.219 |
| bge-small 384d float32, fine-tuned | 15/30 | 26/30 | 28/30 | 0.653 | 3 / 0 | 0.250 |
| bge-small 384d float32 (baseline) | 12/30 | 23/30 | 25/30 | 0.557 | 0 / 0 | 1.000 |

n=30: one question moves a score by 3.3 points, so gaps up to 6.7 points are likely noise.

## Fine Print's held-out set (previously used)

Previously used: Fine Print scored these 20 questions for its v2 and v3, so this is not a clean estimate. Reported separately, as agreed.

| Configuration | hit@1 | hit@5 | hit@10 | MRR | Wins / losses vs baseline (hit@5) | Sign test p |
| --- | --- | --- | --- | --- | --- | --- |
| bge-base 768d float32 | 10/20 | 15/20 | 17/20 | 0.610 | 0 / 2 | 0.500 |
| bge-base 768d binary + float32 rescoring (recommended) | 10/20 | 15/20 | 17/20 | 0.611 | 0 / 2 | 0.500 |
| bge-base 256d float32 | 9/20 | 17/20 | 17/20 | 0.574 | 1 / 1 | 1.000 |
| bge-small 384d float32, fine-tuned | 10/20 | 16/20 | 19/20 | 0.625 | 2 / 3 | 1.000 |
| bge-small 384d float32 (baseline) | 12/20 | 17/20 | 18/20 | 0.692 | 0 / 0 | 1.000 |

n=20: one question moves a score by 5.0 points, so gaps up to 10.0 points are likely noise.
