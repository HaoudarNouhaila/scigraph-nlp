# Progress

Status as of 2026-10-17. Published at https://github.com/HaoudarNouhaila/scigraph-nlp. Only work that was actually executed is listed as done; every
number refers to a file under `reports/`.

## Implemented and verified

| Area | Verified by |
|---|---|
| SciERC download, checksum, validation, sentence-level conversion | `scripts/prepare_data.py`; `reports/metrics/dataset_statistics.json` |
| Data exploration | `notebooks/01_data_exploration.ipynb` (executed) |
| Rule-based spaCy baseline (NER and RE) | `reports/metrics/baseline.json` |
| Span-based SciBERT NER, nested spans | trained (CPU, 10 epochs); `reports/metrics/ner_metrics.json` |
| Typed-marker SciBERT relation classifier | trained (CPU, 5 epochs); `reports/metrics/re_metrics.json` |
| Experiments A and B | `re_metrics.json`; reproduced by `scripts/predict_scierc.py` |
| NER decoding ablation (nested vs flat) | `reports/metrics/ner_decoding.json` |
| Resumable training, gradient accumulation | resume and accumulation tested on a small encoder |
| Pipeline on new text and PDF (pages, offsets) | `tests/test_pipeline.py`; Streamlit with trained models |
| Normalisation, acronyms, type-aware entity resolution | unit tests; `reports/metrics/graph_quality.json` |
| Neo4j schema, batched idempotent ingestion, review status | integration tests against Neo4j 5.26.12 |
| Named parameterised queries, multi-hop paths | `reports/metrics/query_evaluation.json` (250/250) |
| Rule-based `:INFERRED` layer, link hypotheses | integration test; demonstration graph (121 inferred edges) |
| Ontology, Turtle export, SPARQL, SHACL | unit tests; graph quality experiment |
| FastAPI service | `tests/test_api.py` (unit and Neo4j integration); manual check with uvicorn |
| Streamlit app, six pages | screenshots in `docs/images/` captured from the running app |
| Evaluation notebook | `notebooks/04_evaluation.ipynb` (executed) |
| Performance measurements | `reports/metrics/performance.json` |
| Continuous integration (Ruff, 53 tests with a Neo4j service, Docker build) | GitHub Actions, passing on `main` since 2026-10-17 |

## Written but not executed

* Notebooks 02 and 03 on a GPU (Kaggle or Colab): remote GPU use was declined; the same
  training code was run locally through `scripts/train_cpu.sh`.
* GPU configurations `configs/ner.yaml` and `configs/re.yaml`.
* `docker-compose.yml`: Docker is not installed on the development machine. The image itself
  is built by CI, but no container has been started from it.

## Results (test split, single run)

| | Micro-F1 |
|---|---:|
| NER, SciBERT span model (baseline 26.94) | 67.04 |
| RE with gold entities (baseline 26.57) | 70.26 |
| RE with predicted entities, boundaries / strict | 47.67 / 34.92 |
| Graph facts against gold graph, typed / untyped | 34.98 / 46.74 |

## Difficulties

* No local GPU and remote GPU declined: reduced CPU configuration (no cross-sentence context,
  fewer epochs, negative sampling for RE). NER took 1 h 54 min, RE 6 h 15 min.
* RE training slowed down by memory pressure (about 500 MB free, heavy paging); stopping the
  local Streamlit and Neo4j processes during training freed about 2.5 GB.
* With 98 % negative spans, NER predicts no entity after the first epoch (dev F1 0.0) and
  starts extracting from the second (dev F1 44.8).
* Nested entities are representable but rarely recovered (3 of 20 test pairs).
* `curl` fails TLS verification on this machine (intercepting proxy); downloads use
  PowerShell or Python without disabling verification.
* SciERC has no stated licence, so the data is not redistributed.
* The first two CI runs failed on undeclared dependencies (`python-multipart`, `scipy`) that
  were installed transitively on the development machine; both are now declared.
* The in-app browser pane does not render when the window is hidden; screenshots are taken
  with headless Chrome through the DevTools protocol instead.

## Remaining

* Optional: run notebooks 02 and 03 on a GPU with the full configuration and a few seeds.
* Optional: publish the trained weights (about 420 MB each) as a release or model hub
  entry; they are not committed.
