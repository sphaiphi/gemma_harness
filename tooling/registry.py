"""Tool schemas, handlers and the embedding collection used to pick relevant tools.

Tools are embedded as EmbeddingGemma documents ("title: <tool name> | text: <description + arguments>") and
matched with search queries ("task: search result | query: <subgoal>").
"""
from __future__ import annotations

import json
from typing import Any, Callable

import jsonschema

from ..common import digest
from ..embeddings import ASYMMETRIC_TASKS, SEARCH, Doc, GemmaEmbedder, open_collection, require_task

Handler = Callable[..., Any]


class ToolRegistry:
    def __init__(self, chroma: Any, embedder: GemmaEmbedder, task: str = SEARCH):
        self.embedder = embedder
        self.task = require_task(task, ASYMMETRIC_TASKS)
        self.collection = open_collection(chroma, "tools_registry", embedder)
        self._handlers: dict[str, Handler] = {}
        self._schemas: dict[str, dict] = {}

    @staticmethod
    def document(name: str, description: str, parameters: dict) -> Doc:
        props = parameters.get("properties") or {}
        args = ", ".join(f"{k}: {v.get('type', 'any')}" if isinstance(v, dict) else str(k)
                         for k, v in props.items())
        text = f"{' '.join(description.split())}\nArguments: {args or 'none'}"
        return Doc(text=text, title=name.replace("_", " "))

    def register(self, name: str, description: str, parameters: dict, handler: Handler) -> None:
        self._handlers[name] = handler
        self._schemas[name] = {"name": name, "description": description, "parameters": parameters}
        doc = self.document(name, description, parameters)
        sha = digest(doc.title or "", doc.text)
        metas = self.collection.get(ids=[name], include=["metadatas"]).get("metadatas") or []
        if metas and (metas[0] or {}).get("sha") == sha:
            return  # same description already embedded
        self.collection.upsert(ids=[name], embeddings=self.embedder.embed_documents([doc]),
                               documents=[doc.text], metadatas=[{"name": name, "sha": sha}])

    def has(self, name: str) -> bool:
        return name in self._schemas

    def names(self) -> list[str]:
        return list(self._schemas)

    def catalog(self) -> str:
        return "\n".join(f"- {n}: {s['description']}" for n, s in self._schemas.items())

    def prune(self) -> None:
        """Drop embeddings of tools that are no longer registered (e.g. deleted or failing synth tools)."""
        ids = self.collection.get(include=[]).get("ids", [])
        if stale := [i for i in ids if i not in self._schemas]:
            self.collection.delete(ids=stale)

    def relevant(self, query: str, n: int = 3) -> list[dict]:
        count = self.collection.count()
        if count == 0 or not query.strip():
            return []
        vec = self.embedder.embed_query(self.task, query)
        res = self.collection.query(query_embeddings=[vec], n_results=min(n, count), include=["metadatas"])
        names = [m["name"] for m in (res.get("metadatas") or [[]])[0]]
        return [{"type": "function", "function": self._schemas[x]} for x in names if x in self._schemas]

    def validate(self, name: str | None, args: dict) -> str | None:
        if name not in self._schemas:
            return f"unknown tool {name!r}"
        try:
            jsonschema.validate(args, self._schemas[name]["parameters"])
        except jsonschema.ValidationError as e:
            return f"invalid arguments for {name}: {e.message}"
        return None

    def execute(self, name: str, args: dict) -> tuple[bool, str]:
        if err := self.validate(name, args):
            return False, f"Error: {err}"
        try:
            out = self._handlers[name](**args)
        except Exception as e:
            return False, f"Error: {type(e).__name__}: {e}"
        out = out if isinstance(out, str) else json.dumps(out)
        return not out.startswith("Error"), out