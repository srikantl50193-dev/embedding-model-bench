# embedding-model-bench

Which embedding model should a client use for their documents, and is fine-tuning worth it?

Work in progress: the recommendation will be published here once the experiments have run.

## Setup

This project imports its chunker and reads its data from [Fine Print](https://github.com/srikantl50193-dev/fine-print), which must be cloned next to this repo.

```bash
git clone https://github.com/srikantl50193-dev/fine-print ../fine-print
make setup
make data
```

## Credits

- Corpus: policies collected through Fine Print from [ToS;DR](https://tosdr.org) and [Open Terms Archive](https://opentermsarchive.org).
- Out-of-domain benchmark: SciFact from [BEIR](https://github.com/beir-cellar/beir). Thakur, Reimers, Rücklé, Srivastava and Gurevych, "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", NeurIPS 2021 Datasets and Benchmarks Track.
