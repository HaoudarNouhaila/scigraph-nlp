# Progress

Status of the project as of 2026-10-16. Only work that was actually executed is listed as
done; every number refers to a file under `reports/`.

## Implemented

| Area | State | Verified by |
|---|---|---|
| SciERC download, checksum, validation, sentence-level conversion | done | `scripts/prepare_data.py` run; statistics in `reports/metrics/dataset_statistics.json` |
| Data exploration notebook | done, executed | `notebooks/01_data_exploration.ipynb` |
| Rule-based spaCy baseline (NER and RE) | done, evaluated | `reports/metrics/baseline.json` |
| Span-based SciBERT NER (nested spans) | implemented, training on CPU | `outputs/ner/train_log.jsonl` |
| Typed-marker SciBERT relation classifier | implemented, queued after NER | `outputs/re/` |
| Resumable training (checkpoint, RNG state, mid-epoch resume) | done | resume tested on a small encoder |
| Gradient accumulation, mixed precision (CUDA) | done | accumulation tested on a small encoder; AMP not run (no GPU) |
| Experiment A / B evaluation | implemented | runs at the end of RE training |
| Notebooks 02-04 (Kaggle/Colab/local) | written | not executed on a GPU |
| Pipeline on new text and PDF (PyMuPDF, pages) | done | `tests/test_pipeline.py` |
| Normalisation, acronym resolution, type-aware deduplication | done | unit tests |
| Neo4j schema, batched idempotent ingestion, review status | done | integration tests against Neo4j 5.26.12 |
| Whitelisted parameterised queries, multi-hop paths | done | `reports/metrics/query_evaluation.json` |
| Rule-based `:INFERRED` layer, link hypotheses | done | integration test, gold graph |
| NetworkX analytics | done | unit tests |
| Ontology, Turtle export, SPARQL, SHACL | done | unit tests |
| FastAPI service | done | `tests/test_api.py` (unit and Neo4j integration) |
| Streamlit app, six pages | done | checked in a browser with the rule extractor |
| CLI | done | used to ingest the gold test graph |
| Docker, docker-compose | written | not built locally (Docker not installed); built in CI |
| GitHub Actions (Ruff, pytest with Neo4j service, Docker build) | written | runs once the repository is pushed |

## Experiments executed

1. Dataset statistics on the official splits.
2. Rule baseline on dev and test (NER, RE with gold and predicted entities).
3. Experiment 5 (query correctness and latency) on the gold test graph in Neo4j:
   all 250 reference cases passed.
4. SciBERT NER training (CPU configuration): in progress.

## Difficulties

* No local GPU, and remote GPU training was declined: models are trained on CPU with a
  reduced configuration (see `PROJECT_GUIDE.md`, section Compute actually used).
* `curl` fails TLS verification on this machine (intercepting proxy); downloads use
  PowerShell or Python, without disabling verification.
* SciERC has no stated licence, so the data is not redistributed.
* Docker is not installed locally, so the image is only built in CI.
* With 98 % negative spans, the NER model predicts no entity after the first epoch
  (dev F1 0.0) and starts extracting from the second epoch (dev F1 44.8).

## Remaining

* Finish NER and RE training; run `scripts/make_report.py`, `scripts/predict_scierc.py`,
  `scripts/evaluate_graph.py`, `scripts/benchmark.py`.
* Fill the Results section of the README and the CV bullets from the measured numbers.
* Screenshots of the application with the trained models.
* Publication on GitHub after explicit authorisation (account and visibility to confirm).
