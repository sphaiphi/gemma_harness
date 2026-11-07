"""Worker registry. To add a subgoal kind, write a Worker subclass and add it here."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from .base import Worker, blocked, worker_context
from .edit import EditWorker
from .reason import ReasonWorker
from .synth import SynthWorker
from .tool import ToolWorker

if TYPE_CHECKING:
    from ..services import Services

WORKER_TYPES: dict[str, type[Worker]] = {w.kind: w for w in (EditWorker, ToolWorker, SynthWorker, ReasonWorker)}
FALLBACK_KIND = "reason"


def build_workers(s: "Services", types: Mapping[str, type[Worker]] | None = None) -> dict[str, Worker]:
    workers = {kind: cls(s) for kind, cls in (types or WORKER_TYPES).items()}
    if FALLBACK_KIND not in workers:
        raise ValueError(f"a {FALLBACK_KIND!r} worker is required as the fallback")
    return workers


__all__ = ["Worker", "WORKER_TYPES", "FALLBACK_KIND", "build_workers", "blocked", "worker_context",
           "EditWorker", "ToolWorker", "SynthWorker", "ReasonWorker"]