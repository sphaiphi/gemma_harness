"""Tool worker: FunctionGemma picks one registered tool and its arguments; the harness runs it."""
from __future__ import annotations

import asyncio
import json

from ..common import ModelOutputError, elide
from ..prompts import TOOL_PROMPT
from ..schemas import HarnessState, Subgoal, WorkerOutput, WorkMeta
from .base import VerifyPolicy, Worker, WorkResult


class ToolWorker(Worker):
    kind = "tool"
    planner_hint = ("exactly one call to an available tool. Never use a tool subgoal to author code; "
                    "use edit for that.")
    policy = VerifyPolicy(review="final_only")
    CANDIDATES = 3
    ROUTE_ATTEMPTS = 2

    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        try:
            name, args, err = await self._route(sg, state.get("verifier_feedback"))
        except ModelOutputError as e:
            name, args, err = None, {}, str(e)
        if err:
            return (WorkerOutput(subgoal_id=sg.id, status="complete", result=f"NO TOOL RAN: {err}"),
                    WorkMeta(tool_ok=False))

        async with self.s.write_lock:
            res = await asyncio.to_thread(self.s.tools.execute, name, args)
        # Every workspace write, by a built-in or a synthesized tool, is in the thread's journal.
        changes = await self.recorded_changes()

        shown = ", ".join(f"{k}={str(v)[:80]!r}" for k, v in args.items())
        out = WorkerOutput(subgoal_id=sg.id, status="complete",
                           result=f"{name}({shown}) -> {'ok' if res.ok else 'ERROR'}\n{elide(res.output)}",
                           reasoning=f"FunctionGemma selected {name}.")
        return out, WorkMeta(tool_ok=res.ok, changes=changes, extra={"tool": name})

    async def _route(self, sg: Subgoal, feedback: str | None) -> tuple[str | None, dict, str | None]:
        query = f"{sg.description}\n{sg.notes}".strip()
        tools = await asyncio.to_thread(self.s.tools.relevant, query, self.CANDIDATES)
        if not tools:
            return None, {}, "no tools are registered"
        messages = [{"role": "system", "content": TOOL_PROMPT},
                    {"role": "user",
                     "content": query + (f"\n\nPrevious attempt rejected: {feedback}" if feedback else "")}]
        err: str | None = "no tool call"
        for _ in range(self.ROUTE_ATTEMPTS):
            msg = await self.s.llm.chat("function", messages, tools=tools, options={"temperature": 0.0})
            if calls := msg.get("tool_calls") or []:
                fn = calls[0].get("function") or {}
                name, args = fn.get("name"), fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = None
                err = ("arguments are not a JSON object" if not isinstance(args, dict)
                       else self.s.tools.validate(name, args))
                if err is None:
                    return name, args, None
            messages += [{"role": "assistant", "content": msg.get("content", "")},
                         {"role": "user",
                          "content": f"Not a valid tool call ({err}). Call one tool with valid arguments."}]
        return None, {}, err