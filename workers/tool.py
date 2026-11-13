"""Tool worker: the tool-router model picks one allowed tool and its arguments; the harness runs it."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..common import ModelOutputError, elide
from ..prompts import TOOL_PROMPT
from ..schemas import HarnessState, Subgoal, WorkerOutput, WorkMeta
from .base import VerifyPolicy, Worker, WorkResult, blocked

if TYPE_CHECKING:
    from ..models import ToolCaller
    from ..services import Services


class ToolWorker(Worker):
    kind = "tool"
    planner_hint = ("exactly one call to an available tool. Never use a tool subgoal to author code; "
                    "use edit for that.")
    policy = VerifyPolicy(review="final_only")
    CANDIDATES = 3

    def __init__(self, services: "Services", router: "ToolCaller | None" = None):
        super().__init__(services)
        self.router = router or services.models.tool_router()

    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        s = self.s
        task_type = (state.get("plan") or {}).get("task_type", "other")
        scopes = s.cfg.allowed_scopes(task_type)
        query = f"{sg.description}\n{sg.notes}".strip()

        tools = await asyncio.to_thread(s.tools.relevant, query, self.CANDIDATES, scopes)
        if not tools:
            return blocked(sg, f"no registered tool is allowed for {task_type} tasks "
                               f"(allowed access: {', '.join(sorted(scopes)) or 'none'})")

        feedback = state.get("verifier_feedback")
        user = query + (f"\n\nPrevious attempt rejected: {feedback}" if feedback else "")
        try:
            call = await self.router.call(TOOL_PROMPT, user, tools,
                                          check=lambda c: s.tools.validate(c.name, c.args, scopes))
        except ModelOutputError as e:
            return (WorkerOutput(subgoal_id=sg.id, status="complete", result=f"NO TOOL RAN: {e}"),
                    WorkMeta(tool_ok=False))

        async with s.write_lock:
            res = await asyncio.to_thread(s.tools.execute, call.name, call.args, scopes)
        # Every workspace write, by a built-in or a synthesized tool, is in the thread's journal.
        changes = await self.recorded_changes()

        shown = ", ".join(f"{k}={str(v)[:80]!r}" for k, v in call.args.items())
        out = WorkerOutput(subgoal_id=sg.id, status="complete",
                           result=f"{call.name}({shown}) -> {'ok' if res.ok else 'ERROR'}\n{elide(res.output)}",
                           reasoning=f"{self.router.engine} selected {call.name}.")
        return out, WorkMeta(tool_ok=res.ok, changes=changes, extra={"tool": call.name})