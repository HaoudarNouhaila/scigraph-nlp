# CV description

Three bullet points for a one-page CV. Every number is a measured result on the official
SciERC test split (see README, section Results), and every technology listed was used and
run. Docker is not listed: the image is built in CI but no container was run from it.

## English

**SciGraph-NLP: automated scientific knowledge graph construction** (personal project, 2026)
Python, PyTorch, Hugging Face Transformers, SciBERT, spaCy, Neo4j, Cypher, RDFLib, NetworkX,
FastAPI, Streamlit, pytest, GitHub Actions

- Fine-tuned SciBERT for span-based named entity recognition (nested spans) and for
  directional relation extraction with typed entity markers on SciERC; reached 67.0 micro-F1
  on entities and 70.3 on relations with gold entities, against 26.9 and 26.6 for a spaCy
  rule baseline, and measured error propagation end to end (47.7 F1).
- Built the pipeline from text and PDF to a Neo4j knowledge graph: type-aware entity
  resolution, provenance for every relation (sentence, page, confidence, model version),
  separation of predicted, verified and rule-inferred knowledge, RDFS/OWL ontology with
  Turtle export, SPARQL queries and SHACL validation.
- Exposed the graph through whitelisted parameterised Cypher queries, multi-hop path search
  and NetworkX analytics in a FastAPI service and a Streamlit interface; 250 of 250
  reference query checks passed, with unit and integration tests run in CI.

## Français

**SciGraph-NLP : construction automatique d'un graphe de connaissances scientifiques**
(projet personnel, 2026)
Python, PyTorch, Hugging Face Transformers, SciBERT, spaCy, Neo4j, Cypher, RDFLib, NetworkX,
FastAPI, Streamlit, pytest, GitHub Actions

- Fine-tuning de SciBERT pour la reconnaissance d'entités par classification de spans
  (entités imbriquées) et pour l'extraction de relations orientées avec marqueurs d'entités
  typés sur SciERC : 67,0 de micro-F1 sur les entités et 70,3 sur les relations à entités de
  référence, contre 26,9 et 26,6 pour une baseline à règles spaCy ; mesure de la propagation
  d'erreurs de bout en bout (47,7 de F1).
- Pipeline du texte et du PDF jusqu'à un graphe Neo4j : résolution d'entités tenant compte
  des types, provenance de chaque relation (phrase, page, confiance, version du modèle),
  séparation des connaissances prédites, vérifiées et inférées par règles, ontologie
  RDFS/OWL avec export Turtle, requêtes SPARQL et validation SHACL.
- Exploration du graphe par requêtes Cypher paramétrées en liste blanche, recherche de
  chemins multi-sauts et analyses NetworkX via une API FastAPI et une interface Streamlit ;
  250 cas de référence de requêtes validés sur 250, avec tests unitaires et d'intégration exécutés en CI.
