"""FastAPI service exposing extraction, ingestion and graph exploration.

Run with::

    uvicorn scigraph.api.main:app --host 0.0.0.0 --port 8000

Configuration comes from environment variables (see ``.env.example``):
``SCIGRAPH_NER_DIR``, ``SCIGRAPH_RE_DIR``, ``SCIGRAPH_EXTRACTOR`` (auto|neural|rules),
``NEO4J_URI``, ``NEO4J_USER``, ``NEO4J_PASSWORD``, ``NEO4J_DATABASE``.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from scigraph import __version__
from scigraph.api.schemas import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    IngestRequest,
    QueryRequest,
    ReviewRequest,
)
from scigraph.graph import analytics
from scigraph.graph.queries import (
    QUERIES,
    QueryValidationError,
    find_paths,
    khop_neighbourhood,
    run_named_query,
)
from scigraph.pipeline.extractor import load_extractor, process_pdf, process_text, triples_table

MAX_PDF_BYTES = 15 * 1024 * 1024

app = FastAPI(
    title="SciGraph-NLP API",
    version=__version__,
    description="Scientific entity and relation extraction and knowledge graph exploration.",
)


@lru_cache(maxsize=1)
def get_extractor():
    mode = os.getenv("SCIGRAPH_EXTRACTOR", "auto")
    return load_extractor(prefer={"neural": "neural", "rules": "rules"}.get(mode, "auto"))


_store = None


def get_store():
    """Lazily connect to Neo4j; endpoints that need it return 503 when unavailable."""
    global _store
    if _store is None:
        from scigraph.graph.neo4j_store import Neo4jStore

        try:
            store = Neo4jStore()
            store.verify_connectivity()
            store.create_schema()
        except Exception as exc:  # noqa: BLE001 - surface any driver error as 503
            raise HTTPException(503, f"Neo4j unavailable: {type(exc).__name__}") from exc
        _store = store
    return _store


def _response(result: dict) -> ExtractResponse:
    kg = result["graph"]
    return ExtractResponse(
        doc_id=result["doc_id"],
        extractor=result["extractor"],
        model_version=result.get("model_version"),
        n_sentences=result["n_sentences"],
        sentences=[
            {k: s[k] for k in ("index", "text", "entities", "relations")}
            for s in result["sentences"]
        ],
        triples=triples_table(kg),
        n_entities=len(kg.entities),
        candidate_duplicates=kg.candidate_duplicates,
        seconds=result["seconds"],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        name = get_extractor().name
    except Exception:  # noqa: BLE001
        name = None
    try:
        get_store()
        neo4j_ok = True
    except HTTPException:
        neo4j_ok = False
    return HealthResponse(status="ok", extractor=name, neo4j=neo4j_ok, version=__version__)


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest, extractor=Depends(get_extractor)) -> ExtractResponse:
    return _response(process_text(req.text, extractor, title=req.title, source="api"))


@app.post("/extract/pdf", response_model=ExtractResponse)
async def extract_pdf(file: UploadFile = File(...), extractor=Depends(get_extractor)):
    data = await file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF larger than 15 MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "File is not a PDF")
    try:
        result = process_pdf(data, extractor, source=f"pdf:{file.filename or 'upload'}"[:100])
    except Exception as exc:  # noqa: BLE001 - corrupt PDFs raise various errors
        raise HTTPException(422, f"Could not read PDF: {type(exc).__name__}") from exc
    return _response(result)


@app.post("/graph/ingest")
def ingest(req: IngestRequest, extractor=Depends(get_extractor), store=Depends(get_store)):
    result = process_text(req.text, extractor, title=req.title, source=req.source)
    counts = store.ingest(result["graph"], status="predicted", source=extractor.name)
    return {"doc_id": result["doc_id"], "extractor": extractor.name, "written": counts}


@app.get("/graph/queries")
def list_queries():
    return [
        {
            "name": q.name,
            "description": q.description,
            "params": {
                k: {
                    "type": p.kind.__name__,
                    "required": p.required,
                    "default": p.default,
                    "choices": list(p.choices) if p.choices else None,
                }
                for k, p in q.params.items()
            },
        }
        for q in QUERIES.values()
    ]


@app.post("/graph/query")
def graph_query(req: QueryRequest, store=Depends(get_store)):
    try:
        return {"name": req.name, "rows": run_named_query(store, req.name, req.params)}
    except QueryValidationError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/graph/paths")
def graph_paths(
    source_id: str = Query(..., max_length=64),
    target_id: str = Query(..., max_length=64),
    max_hops: int = Query(3, ge=1, le=5),
    limit: int = Query(10, ge=1, le=50),
    verified_only: bool = False,
    store=Depends(get_store),
):
    statuses = ("verified",) if verified_only else ("predicted", "verified")
    return {"paths": find_paths(store, source_id, target_id, max_hops, limit, statuses)}


@app.get("/graph/neighbourhood/{entity_id}")
def neighbourhood(
    entity_id: str,
    hops: int = Query(1, ge=1, le=3),
    limit: int = Query(200, ge=1, le=1000),
    store=Depends(get_store),
):
    return {"edges": khop_neighbourhood(store, entity_id, hops, limit)}


@app.post("/graph/relations/{relation_id}/review")
def review(relation_id: str, req: ReviewRequest, store=Depends(get_store)):
    rec = store.set_relation_status(relation_id, req.status, req.reviewer, req.note)
    if rec is None:
        raise HTTPException(404, "Relation not found")
    return rec


@app.get("/graph/analytics")
def graph_analytics(top_k: int = Query(10, ge=1, le=100), store=Depends(get_store)):
    g = analytics.build_graph(store.export_triples())
    return {
        "summary": analytics.summary(g),
        "centrality": analytics.centrality(g, top_k),
        "communities": analytics.communities(g, top_k=top_k),
    }


@app.post("/graph/infer")
def graph_infer(include_predicted: bool = False, store=Depends(get_store)):
    """Apply the rule set and store conclusions as separate :INFERRED edges."""
    from scigraph.graph import reasoning

    statuses = ("predicted", "verified") if include_predicted else ("verified",)
    return {"rules": reasoning.materialize(store, statuses)}


@app.get("/graph/inferred")
def graph_inferred(limit: int = Query(50, ge=1, le=500), store=Depends(get_store)):
    from scigraph.graph import reasoning

    return {"relations": reasoning.list_inferred(store, limit)}


@app.get("/graph/hypotheses")
def graph_hypotheses(top_k: int = Query(20, ge=1, le=200), store=Depends(get_store)):
    """Unlinked entity pairs ranked by shared neighbours. Hypotheses, not facts."""
    g = analytics.build_graph(store.export_triples())
    return {"hypotheses": analytics.candidate_links(g, top_k)}


@app.get("/graph/stats")
def graph_stats(store=Depends(get_store)):
    return store.counts()


@app.get("/graph/export.ttl", response_class=PlainTextResponse)
def export_turtle(verified_only: bool = False, store=Depends(get_store)):
    from scigraph.graph.rdf_export import triples_to_rdf

    statuses = ("verified",) if verified_only else ("predicted", "verified")
    g = triples_to_rdf(store.export_triples(statuses))
    return PlainTextResponse(g.serialize(format="turtle"), media_type="text/turtle")
