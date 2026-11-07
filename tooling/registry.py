"""Tool schemas, handlers and the embedding collection used to pick relevant tools."""
from __future__ import annotations

import json
from typing import Any, Callable

import jsonschema
from chromadb.api.types import EmbeddingFunction

Handler = Callable[..., Any]


class ToolRegistry:
    def __init__(self, chroma: Any, embedding_fn: EmbeddingFunction):
        self.collection = chroma.get_or_create_collection("tools_registry", embedding_function=embedding_fn)
        self._handlers: dict[str, Handler] = {}
        self._schemas: dict[str, dict] = {}

    def register(self, name: str, description: str, parameters: dict, handler: Handler) -> None:
        self._handlers[name] = handler
        self._schemas[name] = {"name": name, "description": description, "parameters": parameters}
        self.collection.upsert(ids=[name], documents=[description], metadatas=[{"name": name}])

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
        if count == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(n, count))
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