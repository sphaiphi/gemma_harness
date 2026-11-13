"""EmbeddingGemma client that applies the model's task-instruction prefixes.

EmbeddingGemma is trained with short task prefixes on text inputs:
- Asymmetric tasks (retrieval): queries are "task: <task> | query: <text>" and corpus items are
  "title: <title or none> | text: <content>".
- Symmetric tasks (classification, clustering, similarity): every compared input is
  "task: <task> | query: <text>".
Callers pass raw text. This module is the only place that formats embedding prompts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from .common import ModelOutputError, digest

if TYPE_CHECKING:
    from .models.transport import OllamaTransport

log = logging.getLogger(__name__)

# Bump when the templates below change. It is part of every collection name, so a change forces a re-index.
PROMPT_FORMAT_VERSION = "1"

SEARCH = "search result"
QUESTION_ANSWERING = "question answering"
FACT_CHECKING = "fact checking"
CODE_RETRIEVAL = "code retrieval"
CLASSIFICATION = "classification"
CLUSTERING = "clustering"
SENTENCE_SIMILARITY = "sentence similarity"

ASYMMETRIC_TASKS = frozenset({SEARCH, QUESTION_ANSWERING, FACT_CHECKING, CODE_RETRIEVAL})
SYMMETRIC_TASKS = frozenset({CLASSIFICATION, CLUSTERING, SENTENCE_SIMILARITY})

Vector = list[float]


def require_task(task: str, allowed: frozenset[str]) -> str:
    if task not in allowed:
        raise ValueError(f"unknown EmbeddingGemma task {task!r}; expected one of {sorted(allowed)}")
    return task


def _one_line(text: str) -> str:
    return " ".join(text.split())


def query_prompt(task: str, text: str) -> str:
    return f"task: {task} | query: {_one_line(text)}"


def document_prompt(text: str, title: str | None = None) -> str:
    """Corpus item. Content keeps its line breaks (code needs them); a missing title becomes `none`."""
    t = _one_line(title or "")
    return f"title: {t or 'none'} | text: {text}"


@dataclass(frozen=True)
class Doc:
    text: str
    title: str | None = None


class GemmaEmbedder:
    """Text-only EmbeddingGemma client on the shared transport (same retries and error types as chat models)."""

    def __init__(self, model: str, transport: "OllamaTransport", *, lane: str = "embed", engine: str = "",
                 batch_size: int = 32, max_query_chars: int = 2000, max_doc_chars: int = 4000):
        self.model = model
        self.transport = transport
        self.lane = lane
        self.engine_name = engine
        self.batch_size = batch_size
        self.max_query_chars = max_query_chars
        self.max_doc_chars = max_doc_chars

    @property
    def engine(self) -> str:
        return self.engine_name or self.model

    @property
    def space_id(self) -> str:
        """Identifies the vector space. It changes when the model or the document/query format changes."""
        return digest(self.model, PROMPT_FORMAT_VERSION)[:8]

    def _embed(self, inputs: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            data = self.transport.post_sync("/api/embed", {"model": self.model, "input": batch, "truncate": True},
                                            self.lane)
            vecs = data.get("embeddings") or []
            if len(vecs) != len(batch):
                raise ModelOutputError(f"{self.engine} returned {len(vecs)} vectors for {len(batch)} inputs")
            out += vecs
        return out

    def embed_documents(self, docs: Sequence[Doc]) -> list[Vector]:
        if not docs:
            return []
        return self._embed([document_prompt(d.text[:self.max_doc_chars], d.title) for d in docs])

    def embed_query(self, task: str, text: str) -> Vector:
        require_task(task, ASYMMETRIC_TASKS)
        return self._embed([query_prompt(task, text[:self.max_query_chars])])[0]

    def embed_symmetric(self, task: str, texts: Sequence[str]) -> list[Vector]:
        require_task(task, SYMMETRIC_TASKS)
        if not texts:
            return []
        return self._embed([query_prompt(task, t[:self.max_query_chars]) for t in texts])


def open_collection(chroma: Any, base: str, embedder: GemmaEmbedder) -> Any:
    """Open `<base>-<space_id>`, dropping collections of the same base built in another vector space."""
    name = f"{base}-{embedder.space_id}"
    for c in chroma.list_collections():
        cname = getattr(c, "name", c)  # Chroma 1.x returns Collection objects, 0.6 returns names
        if cname != name and (cname == base or cname.startswith(f"{base}-")):
            chroma.delete_collection(cname)
            log.info("[index] dropped %s: built with a different model or embedding prompt format", cname)
    return chroma.get_or_create_collection(name, embedding_function=None, metadata={"hnsw:space": "cosine"})