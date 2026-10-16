# Interview preparation

Technical notes to explain and defend SciGraph-NLP. Each answer is short enough to say in
about a minute and points to the code that implements it. Results quoted here are the
measured ones in `reports/metrics/`; see the last section.

## 1. What is a knowledge graph?

A knowledge graph represents facts as typed nodes (entities) connected by typed edges
(relations), with a schema or ontology that fixes what the types mean. Here, nodes are
scientific entities (Task, Method, Metric, Material, OtherScientificTerm, Generic) and
edges are the seven SciERC relations, for example
`(CRF) -[USED_FOR]-> (named entity recognition)`. What makes it a *knowledge* graph rather
than just a graph is (a) the shared semantics of the types, (b) entity identity across
sources (two documents mentioning the same method point to the same node), and
(c) provenance, so every fact can be traced to the sentence it came from.

## 2. Knowledge graph versus graph database

A knowledge graph is a *data model and a body of knowledge*; a graph database is a *storage
and query engine*. Neo4j is a graph database implementing the labelled property graph model
and the Cypher language. The same knowledge graph can be stored in Neo4j, serialised as RDF
and queried with SPARQL, or loaded into NetworkX for analytics; this project does all three
(`graph/neo4j_store.py`, `graph/rdf_export.py`, `graph/analytics.py`).

## 3. Why SciBERT rather than BERT?

SciBERT (Beltagy et al., 2019) has the BERT-base architecture but was pre-trained on 1.14 M
Semantic Scholar papers (82 % biomedical, 18 % computer science) with its own WordPiece
vocabulary (scivocab). Scientific terms are therefore split into fewer, more meaningful
sub-words, and the representations are tuned to scientific text. Its authors report gains
over BERT-base on SciERC NER and RE. The checkpoint used is
`allenai/scibert_scivocab_uncased`. I did not run a BERT-base comparison in this project, so
I would not quote a number for the difference on my setup.

## 4. How does fine-tuning work?

All encoder weights are initialised from the pre-trained checkpoint and a small task head is
initialised randomly. Both are trained end to end on labelled data with AdamW, a linear
warm-up (10 % of steps) followed by linear decay, gradient clipping at 1.0 and weight decay
0.01. The encoder receives a small learning rate (1e-5 to 3e-5) so that pre-trained knowledge
is not destroyed; the head receives a larger one (5e-4 for NER, 1e-4 for RE) because it starts
from random weights (`training/common.py: grouped_parameters`). The best epoch is selected on
the dev split, and the test split is evaluated once.

## 5. How does NER work here?

As span classification (`models/span_ner.py`). All spans of up to 8 tokens are enumerated;
each is represented by the encoder state of its first sub-token, the state of its last
sub-token and an embedding of its width; an MLP predicts one of six types or `O`. This handles
the nested entities of SciERC (for example "machine translation" inside "statistical machine
translation system"), which a BIO tagger cannot represent because it gives one label per token.
The number of candidate spans grows linearly with sentence length for a fixed maximum width (8 here).

## 6. How are sub-tokens handled?

Words are tokenised individually into WordPiece sub-tokens, and the index of the first and
last sub-token of each word is stored (`models/encoding.py`). Span representations use the
first sub-token of the first word and the last sub-token of the last word. Labels live at the
span level, so no label has to be copied to or removed from sub-tokens, and evaluation is done
on word-level spans, never on sub-tokens.

## 7. How does relation extraction work?

For every ordered pair of entities in a sentence, typed markers are inserted around the two
arguments (`<S:Method> ... </S:Method>`, `<O:Task> ... </O:Task>`) and the sentence is encoded
again. The states of the two opening markers are concatenated and classified into eight
classes: the seven relations and `NO-RELATION` (`models/relation.py`). Markers encode role and
type, so the model is directional and type-aware. Symmetric relations (COMPARE, CONJUNCTION)
are trained in both directions and merged at inference.

## 8. How are triples generated?

Each predicted relation gives a mention-level triple (head mention, relation, tail mention).
Mentions are resolved to entities (section 17), and triples are aggregated per
(head entity, relation, tail entity): the support counts the documents where the fact was
found, the score keeps the maximum confidence, and each evidence keeps the document,
sentence, page and model version (`pipeline/graph_builder.py`).

## 9. What loss function is used?

Cross-entropy. For NER it is averaged over all enumerated spans of the batch (padding spans
are ignored with index -100); about 98 % of spans are `O`, which is why the model predicts
nothing during the first epoch and only starts to output entities once the head has learnt
to separate the rare positive spans. For RE it is averaged over candidate pairs, of which 81 %
are `NO-RELATION`; the CPU configuration samples 1.5 negatives per positive each epoch to
shorten training, which tends to raise recall and lower precision.

## 10. Why precision, recall and F1?

Extraction is a retrieval problem over a very large candidate space dominated by negatives.
Accuracy would be near 98 % for a model that predicts nothing. Precision measures how many
extracted items are correct (graph cleanliness), recall how many gold items are found
(graph coverage), and F1 is their harmonic mean. Matching is exact: an entity counts only
with exact boundaries and type; a relation counts only with exact argument boundaries,
label and direction (boundaries criterion), and additionally correct argument types
(strict criterion).

## 11. Micro-F1 versus macro-F1

Micro-F1 pools true positives, predictions and gold items over all classes, so frequent
classes dominate (USED-FOR is 52 % of relations). Macro-F1 averages per-class F1 scores, so
each class weighs the same and rare classes (COMPARE, Metric) matter as much as frequent
ones. Micro-F1 is the standard SciERC number; macro-F1 shows whether rare classes are
handled. Both are computed in `evaluation/metrics.py`.

## 12. How do NER errors affect relation extraction?

A relation can only be correct if both of its arguments were found with exact boundaries.
With predicted entities, a missed or mis-bounded entity removes every relation that
involves it, and spurious entities create extra candidate pairs that can produce false
positives. The drop from Experiment A (gold entities) to Experiment B (predicted entities)
quantifies this propagation; the error analysis in notebook 04 counts how many missed gold
relations had an argument that NER did not find.

## 13. How is the graph built in Neo4j?

Uniqueness constraints on entity, document and mention identifiers and indexes on entity
key, type and status are created first. Rows are written with `UNWIND $rows ... MERGE` in
batches of 500 inside managed write transactions, so writes are idempotent and memory is
bounded. Relationship types are per relation (`:USED_FOR`, ...), with properties for score,
support, evidence, status, source and model version (`graph/neo4j_store.py`).

## 14. Why Cypher?

Cypher expresses graph patterns declaratively: `MATCH (m:Entity {type:'Method'})-[:USED_FOR]->
(t:Entity {type:'Task'})` reads like the question it answers. Variable-length patterns and
`allShortestPaths` make multi-hop exploration a one-line query, and parameters (`$task`)
separate the query text from user input. In this project, users never send Cypher; they pick a
named query and its validated parameters (`graph/queries.py`).

## 15. Neo4j versus RDF

| | Neo4j (labelled property graph) | RDF |
|---|---|---|
| Unit | nodes and relationships with properties | subject-predicate-object triples |
| Edge metadata | properties on the relationship | needs reification or RDF-star |
| Identity | internal ids or chosen keys | global IRIs |
| Schema | optional constraints | RDFS/OWL ontologies with formal semantics |
| Query | Cypher | SPARQL |
| Strength | traversals, application storage | interoperability, standards, reasoning |

The project stores in Neo4j and exports to RDF, with each fact reified as an `sg:Assertion`
so that confidence and provenance survive the export.

## 16. What is an ontology?

A formal specification of the concepts of a domain and of their relations: classes,
properties, domains, ranges and axioms. The light ontology here
(`resources/scigraph.ttl`) declares the six entity classes under `sg:ScientificEntity`, the
seven relation properties, COMPARE and CONJUNCTION as symmetric properties, and disjointness
between Task, Method, Metric and Material. SHACL shapes (`resources/shapes.ttl`) check the
structure of exported data; the ontology describes meaning, SHACL validates instances.

## 17. How are duplicates detected?

Two mentions refer to the same entity only if their types are equal and their normalised
keys are equal (case, determiners, plural, hyphen spacing and acronym definitions removed).
Acronyms defined in a document ("conditional random field (CRF)") resolve to the long form in
that document. Generic mentions are never merged, and bare head nouns ("method") are only
merged inside a document. Same-type entities with high token overlap are reported as
candidate duplicates instead of being merged automatically. Two identical strings with
different types stay separate.

## 18. How does multi-hop search work?

`allShortestPaths((a)-[:USED_FOR|...*..k]-(b))` with `k` between 1 and 5, restricted to the
seven relation types and to the requested knowledge layers. Paths are returned with the
direction of each edge. A path is an indirect association; it is presented as such and never
stored as a fact. Experiment 5 checks that Neo4j path lengths equal NetworkX shortest-path
lengths on sampled pairs.

## 19. Limitations

* Trained on 350 AI abstracts; performance on other fields or on full papers (methods
  sections, tables) is not measured.
* Models were trained on CPU with a reduced configuration (no cross-sentence context, fewer
  epochs, negative sampling); the GPU configuration was not run.
* Relations are intra-sentence only; no coreference resolution, so "this method" is not
  linked to what it refers to.
* Entity resolution is string-based within types; synonyms with different surface forms
  ("SVM" without a definition, "support vector machine") are not merged across documents.
* PDF extraction needs a text layer; equations, tables and figures are not interpreted.
* Graph-level precision is measured against SciERC gold only; there is no manual
  annotation of graph quality on new documents.

## 20. Possible improvements

* Train the GPU configuration (cross-sentence context, all negatives, more epochs) and run a
  seed study; add a BERT-base ablation.
* Joint or pipelined models with shared encoders and coreference (as in SciIE / DyGIE++)
  to resolve Generic mentions and enable cross-sentence relations.
* Entity linking to an external resource (for example Papers with Code or Wikidata) for
  canonical identifiers.
* Active learning loop on the review queue: verified and rejected edges become training data.
* Calibrated confidence (temperature scaling) so that thresholds have a probabilistic meaning.

## Trained models and measured results

Full tables are in `README.md` (section Results); the files are in `reports/metrics/`.
The points worth being able to explain without notes:

* **What was trained.** Span NER: SciBERT-base, span width up to 8, width embedding 150,
  two-layer MLP, 10 epochs, batch 16, learning rates 3e-5 (encoder) and 5e-4 (head),
  sentence-level input without context, 1 h 54 min on CPU. Relation classifier:
  SciBERT-base with 24 typed marker tokens, 5 epochs, batch 32, 1.5 sampled negatives per
  positive, learning rates 3e-5 and 1e-4, 6 h 15 min on CPU. One run each, seed 42.
* **Headline numbers (test).** NER micro-F1 67.04 (baseline 26.94). Relations with gold
  entities 70.26 (baseline 26.57). End to end 47.67 (boundaries) and 34.92 (strict).
* **Why the end-to-end score drops.** 311 of the 974 gold relations are lost because NER
  misses an argument boundary; 108 because the pair is classified as no relation; 40 wrong
  label; 25 wrong direction. Strict is 12.75 points below boundaries because argument types
  are wrong (OtherScientificTerm vs Method is the most frequent confusion).
* **Nested entities.** The model can represent them, but only 3 of 20 nested test pairs
  are fully recovered, and a non-overlapping decoder is slightly better (+0.3 dev F1). The
  honest conclusion is that the formulation supports nesting but this training budget does
  not exploit it.
* **Hardest relations.** FEATURE-OF and PART-OF (F1 about 44 with gold entities): they are
  rare and semantically close to USED-FOR and HYPONYM-OF.
* **Graph.** The predicted test graph has 1,517 entities and 1,073 triples; fact-level F1
  against the gold graph is 34.98 with types, consistent with the strict score; no SHACL
  violation; every triple has an evidence sentence.
* **Queries.** 250 of 250 reference checks against gold annotations and NetworkX passed;
  median latency 6 to 8 ms on a local instance.
* **What I would say about compute.** Training ran on CPU because no GPU was available; the
  configuration was reduced accordingly (no cross-sentence context, fewer epochs, negative
  sampling). I would not compare these numbers with published GPU results.
