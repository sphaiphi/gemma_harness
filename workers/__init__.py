"""Worker registry. To add a subgoal kind, write a Worker subclass and add it here (or pass worker_types=)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from .base import VerifyPolicy, Worker, blocked, failed, worker_context
from .edit import EditWorker
from .reason import ReasonWorker
from .synth import SynthWorker
from .tool import ToolWorker

if TYPE_CHECKING:
    from ..services import Services

# Order sets the order of kinds in the Thinker prompt.
WORKER_TYPES: dict[str, type[Worker]] = {w.kind: w for w in (EditWorker, ToolWorker, ReasonWorker, SynthWorker)}
FALLBACK_KIND = "reason"


def build_workers(s: "Services", types: Mapping[str, type[Worker]] | None = None) -> dict[str, Worker]:
    workers = {kind: cls(s) for kind, cls in (types or WORKER_TYPES).items()}
    if FALLBACK_KIND not in workers:
        raise ValueError(f"a {FALLBACK_KIND!r} worker is required as the fallback")
    if missing := sorted(k for k, w in workers.items() if not getattr(w, "planner_hint", "").strip()):
        raise ValueError(f"workers without a planner_hint: {missing}")
    return workers


def worker_policies(workers: Mapping[str, Worker]) -> dict[str, VerifyPolicy]:
    return {kind: w.policy for kind, w in workers.items()}


def planner_hints(workers: Mapping[str, Worker]) -> dict[str, str]:
    return {kind: w.planner_hint for kind, w in workers.items()}


def policy_for(policies: Mapping[str, VerifyPolicy], kind: str) -> VerifyPolicy:
    return policies.get(kind) or policies[FALLBACK_KIND]


__all__ = ["Worker", "VerifyPolicy", "WORKER_TYPES", "FALLBACK_KIND", "build_workers", "worker_policies",
           "planner_hints", "policy_for", "blocked", "failed", "worker_context",
           "EditWorker", "ToolWorker", "SynthWorker", "ReasonWorker"]