"""Semantic index of workspace Python code, chunked by top-level functions, classes and methods."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

from chromadb.api.types import EmbeddingFunction

from .common import iter_py_files


def node_span(node: ast.AST) -> tuple[int, int]:
    return min([node.lineno] + [d.lineno for d in node.decorator_list]), node.end_lineno


class RepoIndex:
    def __init__(self, root: Path, chroma: Any, embedding_fn: EmbeddingFunction, skip_dirs: Iterable[str]):
        self.root = root
        self.skip_dirs = set(skip_dirs)
        self.collection = chroma.get_or_create_collection("repo_chunks", embedding_function=embedding_fn)

    @staticmethod
    def chunk(text: str) -> list[tuple[int, int]]:
        whole = [(1, max(1, len(text.splitlines())))]
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return whole
        spans: list[tuple[int, int]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                spans.append(node_span(node))
                if isinstance(node, ast.ClassDef):
                    spans += [node_span(n) for n in node.body
                              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return spans or whole

    def on_file_changed(self, rel: str) -> None:
        """Workspace listener: re-index Python files after every write or restore."""
        if rel.endswith(".py"):
            self.index_file(rel)

    def index_file(self, rel: str) -> None:
        self.collection.delete(where={"path": rel})
        try:
            text = (self.root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        lines = text.splitlines()
        spans = list(dict.fromkeys(self.chunk(text)))
        self.collection.upsert(
            ids=[f"{rel}:{s}-{e}" for s, e in spans],
            documents=[(f"# {rel}\n" + "\n".join(lines[s - 1:e]))[:4000] for s, e in spans],
            metadatas=[{"path": rel, "start": s, "end": e} for s, e in spans],
        )

    def index_all(self) -> None:
        for p in iter_py_files(self.root, self.skip_dirs):
            self.index_file(p.relative_to(self.root).as_posix())

    def search(self, query: str, n: int = 4, path: str | None = None) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(n, count),
                                    where={"path": path} if path else None)
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