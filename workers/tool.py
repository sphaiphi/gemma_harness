"""Tool worker: FunctionGemma picks one registered tool and its arguments; the harness runs it."""
from __future__ import annotations

import asyncio
import json

from ..common import CURRENT_THREAD, ModelOutputError, elide
from ..prompts import TOOL_PROMPT
from ..schemas import HarnessState, Subgoal, WorkerOutput
from .base import Worker, WorkResult


class ToolWorker(Worker):
    kind = "tool"
    CANDIDATES = 3
    ROUTE_ATTEMPTS = 2

    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        s = self.s
        try:
            name, args, err = await self._route(sg, state.get("verifier_feedback"))
        except ModelOutputError as e:
            name, args, err = None, {}, str(e)
        if err:
            return WorkerOutput(subgoal_id=sg.id, status="complete", result=f"NO TOOL RAN: {err}"), {"tool_ok": False}

        meta: dict = {"tool": name, "tool_ok": False}
        if name == "write_file" and args.get("path"):
            try:
                p = s.workspace.safe_path(args["path"])
                meta.update(path=s.workspace.rel(p), original=s.workspace.snapshot(p))
            except PermissionError:
                pass
        async with s.write_lock:
            ok, output = await asyncio.to_thread(s.tools.execute, name, args)
        meta["tool_ok"] = ok
        if ok and "path" in meta:
            meta["written"] = True

        # Synthesized tools log their writes through tools._sdk; fold them into the rollback set.
        tid = CURRENT_THREAD.get()
        sdk = await asyncio.to_thread(s.journal.read_sdk_log, tid) if tid else {}
        if sdk:
            files = {meta["path"]: meta.get("original")} if meta.get("written") and meta.get("path") else {}
            for rel, original in sdk.items():
                files.setdefault(rel, original)
            meta = {**meta, "files": files, "written": True}

        shown = ", ".join(f"{k}={str(v)[:80]!r}" for k, v in args.items())
        return WorkerOutput(subgoal_id=sg.id, status="complete",
                            result=f"{name}({shown}) -> {'ok' if ok else 'ERROR'}\n{elide(output)}",
                            reasoning=f"FunctionGemma selected {name}."), meta

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