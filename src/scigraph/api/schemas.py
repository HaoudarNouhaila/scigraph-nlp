"""Request and response models with input validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_CHARS = 20_000


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    title: str = Field("", max_length=300)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v


class IngestRequest(ExtractRequest):
    source: str = Field("api", max_length=100, pattern=r"^[A-Za-z0-9_.:\-]+$")


class EntityOut(BaseModel):
    start: int
    end: int
    label: str
    score: float
    text: str


class RelationOut(BaseModel):
    head: int
    tail: int
    label: str
    score: float


class SentenceOut(BaseModel):
    index: int
    text: str
    entities: list[EntityOut]
    relations: list[RelationOut]


class TripleOut(BaseModel):
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    score: float
    support: int
    evidence: str


class ExtractResponse(BaseModel):
    doc_id: str
    extractor: str
    model_version: dict | None = None
    n_sentences: int
    sentences: list[SentenceOut]
    triples: list[TripleOut]
    n_entities: int
    candidate_duplicates: list[dict]
    seconds: float


class QueryRequest(BaseModel):
    name: str = Field(..., max_length=64, pattern=r"^[a-z_]+$")
    params: dict[str, Any] = Field(default_factory=dict)


class ReviewRequest(BaseModel):
    status: Literal["verified", "rejected", "predicted"]
    reviewer: str = Field(..., min_length=1, max_length=100)
    note: str = Field("", max_length=1000)


class HealthResponse(BaseModel):
    status: str
    extractor: str | None
    neo4j: bool
    version: str
