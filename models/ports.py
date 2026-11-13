"""Capabilities the harness needs from models. Components depend on these, never on a concrete model.

Contracts every implementation keeps, so any model with the capability can replace another:
- Raise only ModelOutputError (unusable answer) or ModelUnavailable (server unreachable), never transport errors.
- JsonReasoner.ask returns a validated object that passed `check`.
- FimCompleter.complete returns only the middle code, with no special tokens, and never an empty string.
- ToolCaller.call returns a ToolCall whose args is a dict and which passed `check`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, TypeVar

from pydantic import BaseModel

from ..embeddings import Doc

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


class JsonReasoner(Protocol):
    @property
    def engine(self) -> str: ...

    async def ask(self, system: str, user: str, schema: type[T],
                  check: Callable[[T], list[str]] | None = None) -> T: ...


class FimCompleter(Protocol):
    @property
    def engine(self) -> str: ...

    async def complete(self, prefix: str, suffix: str) -> str: ...


class ToolCaller(Protocol):
    @property
    def engine(self) -> str: ...

    async def call(self, system: str, user: str, tools: list[dict],
                   check: Callable[[ToolCall], str | None] | None = None) -> ToolCall: ...


class TextNormalizer(Protocol):
    @property
    def engine(self) -> str: ...

    async def normalize(self, text: str) -> str: ...


class Embedder(Protocol):
    max_doc_chars: int

    @property
    def engine(self) -> str: ...

    @property
    def space_id(self) -> str: ...

    def embed_documents(self, docs: Sequence[Doc]) -> list[list[float]]: ...

    def embed_query(self, task: str, text: str) -> list[float]: ...