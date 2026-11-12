| Model | Trained on | Tuning hit@5 | Wins / losses | p | MRR | SciFact nDCG@10 | SciFact wins / losses | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-small-en-v1.5 (baseline) | - | 20/31 | - | - | 0.618 | 0.713 | - | - |
| bge-small-en-v1.5-ft-all | all companies | 22/31 | 3 / 1 | 0.625 | 0.603 | 0.685 | 5 / 16 | 0.027 |
| bge-small-en-v1.5-ft-unseen | all but discord, spotify, zoom | 22/31 | 3 / 1 | 0.625 | 0.603 | 0.686 | 5 / 9 | 0.424 |

By company group (tuning hit@5; wins and losses against the baseline):

| Model | Group | Companies | n | Baseline | Fine-tuned | Wins / losses | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bge-small-en-v1.5-ft-all | unseen companies | discord, spotify, zoom | 9 | 5 | 5 | 1 / 1 | 1.000 |
| bge-small-en-v1.5-ft-all | other companies | coursera, lyft, netflix, paypal, pinterest, reddit, strava | 22 | 15 | 17 | 2 / 0 | 0.500 |
| bge-small-en-v1.5-ft-unseen | unseen companies | discord, spotify, zoom | 9 | 5 | 5 | 1 / 1 | 1.000 |
| bge-small-en-v1.5-ft-unseen | other companies | coursera, lyft, netflix, paypal, pinterest, reddit, strava | 22 | 15 | 17 | 2 / 0 | 0.500 |

n=31: one question moves a score by 3.2 points, so gaps up to 6.5 points are likely noise.
