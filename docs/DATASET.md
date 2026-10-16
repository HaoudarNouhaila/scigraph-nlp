# SciERC: dataset card for this project

## Source

SciERC was released with Luan, He, Ostendorf and Hajishirzi, *Multi-Task Identification of
Entities, Relations, and Coreference for Scientific Knowledge Graph Construction*, EMNLP 2018.
Official page: http://nlp.cs.washington.edu/sciIE/ (checked on 2026-10-16).

The corpus contains 500 abstracts from AI conference and workshop proceedings (taken from the
Semantic Scholar corpus), annotated with scientific entities, relations and coreference
clusters. The project page links three artefacts:

| Artefact | URL | Used here |
|---|---|---|
| Raw brat annotations (`sciERC_raw.tar.gz`, 3.4 MB) | `.../sciIE/data/sciERC_raw.tar.gz` | inspected only; no split information |
| Processed release (`sciERC_processed.tar.gz`, 695 MB, includes ELMo embeddings) | `.../sciIE/data/sciERC_processed.tar.gz` | yes, JSON splits only |
| Annotation guideline | `.../sciIE/annotation_guideline.pdf` | definitions of types |

`scripts/download_scierc.py` downloads the processed archive, keeps only
`processed_data/json/{train,dev,test}.json` and records SHA-256 checksums. The archive used
for all experiments has SHA-256
`bce752fb7ebe4acf570937d76ffb27c239cfc907e477a69075dc7082d0e72e9b`.

## Licence and redistribution

The official page and the archives do not state a licence. The abstracts come from
published papers indexed by Semantic Scholar. Because the redistribution terms are not
stated, **this repository does not redistribute SciERC**: `data/raw/` and
`data/processed/` are git-ignored, and the data is fetched from the official URL by the
download script. Users should cite the original paper.

## Format

One JSON document per line. Offsets are document-level token indices with inclusive ends:

```json
{"doc_key": "...", "sentences": [["tok", ...], ...],
 "ner": [[[start, end, "Method"], ...], ...],
 "relations": [[[s1, e1, s2, e2, "USED-FOR"], ...], ...],
 "clusters": [[[start, end], ...], ...]}
```

`scripts/prepare_data.py` converts this to sentence-level JSONL with sentence-local
offsets. The conversion fails loudly if a span crosses a sentence boundary, a label is
unknown, or a relation argument is not an annotated entity; none of these occur in the
release. Tokens follow Penn Treebank conventions (`-LRB-`, `-RRB-` for brackets).

## Label inventory

Entity types: Task, Method, Metric, Material, OtherScientificTerm, Generic.

Relation types: USED-FOR, FEATURE-OF, HYPONYM-OF, PART-OF, EVALUATE-FOR (directional) and
COMPARE, CONJUNCTION (symmetric). Every relation in the processed release links two entities
of the same sentence. Coreference clusters are available but not used by this project.

## Official splits and statistics

Computed by `scripts/prepare_data.py` (file `reports/metrics/dataset_statistics.json`).

| | train | dev | test |
|---|---:|---:|---:|
| Documents | 350 | 50 | 100 |
| Sentences | 1,861 | 275 | 551 |
| Tokens | 45,412 | 6,521 | 13,401 |
| Mean sentence length (tokens) | 24.40 | 23.71 | 24.32 |
| Entities | 5,598 | 811 | 1,685 |
| Relations | 3,219 | 455 | 974 |
| Coreference clusters | 720 | 100 | 195 |
| Candidate ordered entity pairs | 16,872 | 2,488 | 5,062 |
| Pairs with a relation (%) | 19.08 | 18.29 | 19.24 |

Entity types (train): OtherScientificTerm 1,536 (27.4 %), Method 1,445 (25.8 %),
Generic 950 (17.0 %), Task 894 (16.0 %), Material 542 (9.7 %), Metric 231 (4.1 %).

Relation types (train): USED-FOR 1,690 (52.5 %), CONJUNCTION 400 (12.4 %),
EVALUATE-FOR 313 (9.7 %), HYPONYM-OF 298 (9.3 %), PART-OF 179 (5.6 %),
FEATURE-OF 173 (5.4 %), COMPARE 166 (5.2 %).

No document appears in more than one split (checked by document key).

## Span properties relevant to modelling

* Entity length: maximum 13 tokens; 99.27 % of training entities have at most 8 tokens.
  The span model enumerates spans up to 8 tokens, so 41 training entities (0.73 %) are
  unreachable by construction; the corresponding test entities count as misses.
* Overlap: 61 pairs of overlapping entities in train (51 sentences), 7 in dev, 20 in test.
  All of them are *nested* (one span inside another); there are no crossing spans and no
  span with two labels. A BIO tagger cannot represent nested spans, which is why NER is
  formulated as span classification.
* Direction: in 1,408 of the 3,219 training relations (43.7 %) the head appears after the
  tail in the sentence, so argument order cannot be inferred from surface order.
