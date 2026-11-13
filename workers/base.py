"""Worker contract, per-kind verification policy and helpers shared by all workers."""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Mapping

from ..common import CURRENT_THREAD, elide
from ..schemas import HarnessState, Route, Subgoal, WorkerOutput, WorkMeta

if TYPE_CHECKING:
    from ..services import Services

WorkResult = tuple[WorkerOutput, WorkMeta]


@dataclass(frozen=True)
class VerifyPolicy:
    """How verify and gate treat a kind of subgoal. Declared by each worker, so neither branches on kind."""
    run_suite: bool = False                              # run the full pytest suite after it (coding tasks)
    review: Literal["always", "final_only"] = "always"   # when the Gemma 3 12B verifier judges it
    collectors: tuple[str, ...] = ()                     # extra evidence collectors, by name
    final_test_route: Route = "thinker"                  # where a failing final test run goes


class Worker(ABC):
    kind: ClassVar[str]
    planner_hint: ClassVar[str]               # this kind's line in the Thinker prompt
    policy: ClassVar[VerifyPolicy] = VerifyPolicy()

    def __init__(self, services: "Services"):
        self.s = services

    @abstractmethod
    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        """Execute one subgoal. Raise PermissionError to report it as blocked."""

    async def recorded_changes(self, known: Mapping[str, str | None] | None = None) -> dict[str, str | None]:
        """Files changed in this attempt, read from the thread's journal: one rollback source for every worker.
        `known` covers runs without a thread id, such as unit tests that call a worker directly."""
        tid = CURRENT_THREAD.get()
        changes = await asyncio.to_thread(self.s.journal.load, tid) if tid else {}
        for rel, original in (known or {}).items():
            changes.setdefault(rel, original)
        return changes


def blocked(sg: Subgoal, reason: str) -> WorkResult:
    return WorkerOutput(subgoal_id=sg.id, status="blocked", blocked_reason=reason), WorkMeta()


def failed(sg: Subgoal, error: str) -> WorkResult:
    return WorkerOutput(subgoal_id=sg.id, status="complete"), WorkMeta(worker_error=error)


def worker_context(state: HarnessState, sg: Subgoal) -> str:
    parts = [f"Subgoal {sg.id}: {sg.description}"]
    if sg.notes:
        parts.append(f"Planner notes: {sg.notes}")
    if state["accepted"]:
        deps = set(sg.depends_on)
        parts.append("Accepted results of earlier subgoals (* = direct dependency):\n" + "\n".join(
            f"{'*' if k in deps else '-'} {k}: {elide(str(v), 1500)}" for k, v in state["accepted"].items()))
    if state.get("verifier_feedback"):
        parts.append(f"RETRY. The verifier rejected your previous output:\n{state['verifier_feedback']}")
        if prev := state.get("previous_output"):
            parts.append("Your rejected output:\n" + elide(json.dumps(prev), 3000))
    return "\n\n".join(parts)