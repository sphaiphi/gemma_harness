"""Semantic index of workspace Python code, chunked by top-level functions, classes and methods.

Chunks are embedded as EmbeddingGemma documents ("title: <path> :: <symbol> | text: <code>") and searched with
code-retrieval queries ("task: code retrieval | query: ...").
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import digest, iter_py_files
from .embeddings import ASYMMETRIC_TASKS, CODE_RETRIEVAL, Doc, GemmaEmbedder, open_collection, require_task


def node_span(node: ast.AST) -> tuple[int, int]:
    return min([node.lineno] + [d.lineno for d in node.decorator_list]), node.end_lineno


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    symbol: str | None = None  # "func", "Class" or "Class.method"; None for a whole-file chunk


class RepoIndex:
    def __init__(self, root: Path, chroma: Any, embedder: GemmaEmbedder, skip_dirs: Iterable[str],
                 task: str = CODE_RETRIEVAL):
        self.root = root
        self.skip_dirs = set(skip_dirs)
        self.embedder = embedder
        self.task = require_task(task, ASYMMETRIC_TASKS)
        self.collection = open_collection(chroma, "repo_chunks", embedder)

    @staticmethod
    def chunk(text: str) -> list[Chunk]:
        whole = [Chunk(1, max(1, len(text.splitlines())))]
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return whole
        chunks: list[Chunk] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunks.append(Chunk(*node_span(node), node.name))
                if isinstance(node, ast.ClassDef):
                    chunks += [Chunk(*node_span(n), f"{node.name}.{n.name}") for n in node.body
                               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return chunks or whole

    @staticmethod
    def title(rel: str, c: Chunk) -> str:
        return f"{rel} :: {c.symbol}" if c.symbol else rel

    def on_file_changed(self, rel: str) -> None:
        """Workspace listener: re-index Python files after every write or restore."""
        if rel.endswith(".py"):
            self.index_file(rel)

    def _indexed_sha(self, rel: str) -> str | None:
        got = self.collection.get(where={"path": rel}, limit=1, include=["metadatas"])
        metas = got.get("metadatas") or []
        return (metas[0] or {}).get("sha") if metas else None

    def index_file(self, rel: str, *, force: bool = False) -> None:
        try:
            text = (self.root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self.collection.delete(where={"path": rel})  # deleted or unreadable: drop its chunks
            return
        sha = digest(text)
        if not force and self._indexed_sha(rel) == sha:
            return  # unchanged since it was last embedded
        self.collection.delete(where={"path": rel})
        lines = text.splitlines()
        chunks = list(dict.fromkeys(self.chunk(text)))
        bodies = ["\n".join(lines[c.start - 1:c.end])[:self.embedder.max_doc_chars] for c in chunks]
        vectors = self.embedder.embed_documents(
            [Doc(text=b, title=self.title(rel, c)) for b, c in zip(bodies, chunks)])
        self.collection.upsert(
            ids=[f"{rel}:{c.start}-{c.end}" for c in chunks],
            embeddings=vectors,
            documents=bodies,
            metadatas=[{"path": rel, "start": c.start, "end": c.end, "symbol": c.symbol or "", "sha": sha}
                       for c in chunks],
        )

    def index_all(self) -> None:
        seen: set[str] = set()
        for p in iter_py_files(self.root, self.skip_dirs):
            rel = p.relative_to(self.root).as_posix()
            seen.add(rel)
            self.index_file(rel)
        known = {m["path"] for m in (self.collection.get(include=["metadatas"]).get("metadatas") or []) if m}
        for rel in known - seen:
            self.collection.delete(where={"path": rel})  # files removed outside the harness

    def search(self, query: str, n: int = 4, path: str | None = None) -> list[dict]:
        count = self.collection.count()
        if count == 0 or not query.strip():
            return []
        vec = self.embedder.embed_query(self.task, query)
        res = self.collection.query(query_embeddings=[vec], n_results=min(n, count),
                                    where={"path": path} if path else None,
                                    include=["metadatas", "documents"])
        metas, docs = (res.get("metadatas") or [[]])[0], (res.get("documents") or [[]])[0]
        return [{**m, "text": d} for m, d in zip(metas, docs)]


def regen_start(source: str, start: int, end: int) -> int | None:
    """First body line (after any docstring) of the function spanning exactly start-end, else None."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node_span(node) == (start, end):
            body = node.body
            has_doc = (len(body) > 1 and isinstance(body[0], ast.Expr)
                       and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str))
            first = body[1] if has_doc else body[0]
            return first.lineno if first.lineno > node.lineno else None
    return None