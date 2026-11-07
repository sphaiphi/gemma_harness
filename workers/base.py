"""Worker contract and helpers shared by all workers."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from ..common import elide
from ..schemas import HarnessState, Subgoal, WorkerOutput

if TYPE_CHECKING:
    from ..services import Services

# (output shown to the verifier, meta: JSON-serializable facts for verify/gate such as written files)
WorkResult = tuple[WorkerOutput, dict]


class Worker(ABC):
    kind: ClassVar[str]

    def __init__(self, services: "Services"):
        self.s = services

    @abstractmethod
    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        """Execute one subgoal. Raise PermissionError to report it as blocked."""


def blocked(sg: Subgoal, reason: str) -> WorkResult:
    return WorkerOutput(subgoal_id=sg.id, status="blocked", blocked_reason=reason), {}


def failed(sg: Subgoal, error: str, meta: dict | None = None) -> WorkResult:
    return WorkerOutput(subgoal_id=sg.id, status="complete"), {**(meta or {}), "worker_error": error}


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