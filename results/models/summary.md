| Model | Tuning hit@1 | hit@5 | hit@10 | MRR | nDCG@10 | SciFact recall@5 | recall@10 | nDCG@10 | Chunks/s | p95 query ms | Params (M) | Weights MB | Peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-small-en-v1.5 | 0.581 | 0.645 | 0.677 | 0.618 | 0.618 | 0.759 | 0.836 | 0.713 | 125 (mps) | 8.8 | 33 | 127 | 2108 |
| granite-small-en-r2 | 0.258 | 0.419 | 0.581 | 0.360 | 0.414 | 0.813 | 0.867 | 0.744 | 150 (mps) | 9.0 | 48 | 91 | 1871 |
| bge-base-en-v1.5 | 0.516 | 0.742 | 0.742 | 0.596 | 0.607 | 0.815 | 0.874 | 0.740 | 49 (mps) | 13.1 | 109 | 418 | 3262 |
| arctic-embed-m-v1.5 | 0.516 | 0.645 | 0.774 | 0.587 | 0.614 | 0.782 | 0.846 | 0.716 | 48 (mps) | 13.2 | 109 | 415 | 2470 |
| e5-base-v2 | 0.548 | 0.613 | 0.710 | 0.587 | 0.586 | 0.795 | 0.846 | 0.719 | 48 (mps) | 14.2 | 109 | 418 | 2826 |
| nomic-embed-text-v1.5 | 0.452 | 0.581 | 0.742 | 0.525 | 0.558 | 0.778 | 0.844 | 0.703 | 48 (mps) | 16.1 | 137 | 522 | 2775 |

n=31: one question moves a score by 3.2 points, so gaps up to 6.5 points are likely noise.

Compared with bge-small-en-v1.5, by question. Wins are questions found only by that model; losses are questions found only by the reference.

| Model | Tuning hit@5 | Wins / losses (n=31) | Sign test p | Within noise? | SciFact wins / losses (n=300) | SciFact nDCG@10 | MTEB nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bge-small-en-v1.5 | 20/31 | 0 / 0 | 1.000 | yes | 0 / 0 | 0.713 | 0.713 |
| granite-small-en-r2 | 13/31 | 0 / 7 | 0.016 | no | 24 / 9 | 0.744 | 0.755 |
| bge-base-en-v1.5 | 23/31 | 3 / 0 | 0.250 | no | 22 / 7 | 0.740 | 0.743 |
| arctic-embed-m-v1.5 | 20/31 | 1 / 1 | 1.000 | yes | 19 / 13 | 0.716 | 0.718 |
| e5-base-v2 | 19/31 | 1 / 2 | 1.000 | yes | 19 / 10 | 0.719 | 0.719 |
| nomic-embed-text-v1.5 | 18/31 | 0 / 2 | 0.500 | yes | 17 / 12 | 0.703 | 0.703 |

Selection rule, on these models only: bge-base-en-v1.5.
