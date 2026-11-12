| Model | Prefixes | Tuning hit@5 | Wins / losses | p | MRR | SciFact nDCG@10 | SciFact wins / losses | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-small-en-v1.5 | as on the card | 0.645 | 0 / 0 | 1.000 | 0.618 | 0.713 | 0 / 0 | 1.000 |
| bge-small-en-v1.5 | no prefixes | 0.645 | 0 / 0 | 1.000 | 0.595 | 0.720 | 7 / 6 | 1.000 |
| bge-base-en-v1.5 | as on the card | 0.742 | 0 / 0 | 1.000 | 0.596 | 0.740 | 0 / 0 | 1.000 |
| bge-base-en-v1.5 | no prefixes | 0.742 | 0 / 0 | 1.000 | 0.618 | 0.738 | 7 / 5 | 0.774 |
| arctic-embed-m-v1.5 | as on the card | 0.645 | 0 / 0 | 1.000 | 0.587 | 0.716 | 0 / 0 | 1.000 |
| arctic-embed-m-v1.5 | no prefixes | 0.484 | 0 / 5 | 0.062 | 0.372 | 0.365 | 0 / 109 | 0.000 |
| e5-base-v2 | as on the card | 0.613 | 0 / 0 | 1.000 | 0.587 | 0.719 | 0 / 0 | 1.000 |
| e5-base-v2 | no prefixes | 0.613 | 0 / 0 | 1.000 | 0.520 | 0.710 | 2 / 5 | 0.453 |
| e5-base-v2 | query prefix only | 0.581 | 0 / 1 | 1.000 | 0.540 | 0.725 | 1 / 2 | 1.000 |
| e5-base-v2 | passage prefix only | 0.613 | 0 / 0 | 1.000 | 0.547 | 0.706 | 4 / 3 | 1.000 |
| nomic-embed-text-v1.5 | as on the card | 0.581 | 0 / 0 | 1.000 | 0.525 | 0.703 | 0 / 0 | 1.000 |
| nomic-embed-text-v1.5 | no prefixes | 0.581 | 1 / 1 | 1.000 | 0.485 | 0.702 | 7 / 13 | 0.263 |
| nomic-embed-text-v1.5 | query prefix only | 0.613 | 2 / 1 | 1.000 | 0.556 | 0.704 | 7 / 6 | 1.000 |
| nomic-embed-text-v1.5 | passage prefix only | 0.581 | 0 / 0 | 1.000 | 0.503 | 0.706 | 2 / 7 | 0.180 |

n=31: one question moves a score by 3.2 points, so gaps up to 6.5 points are likely noise.
