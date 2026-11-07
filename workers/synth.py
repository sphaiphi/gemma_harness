"""Synth worker: Gemma 3 12B drafts a new tool and its tests; static checks run before anything is written."""
from __future__ import annotations

import asyncio

from ..common import ModelOutputError, digest, make_diff, strip_fences
from ..prompts import SYNTH_PROMPT
from ..schemas import HarnessState, Subgoal, ToolDraft, WorkerOutput
from ..tooling.checks import compose_tool_source, draft_errors, spec_of
from .base import Worker, WorkResult, failed, worker_context


class SynthWorker(Worker):
    kind = "synth"

    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        s = self.s
        ctx = worker_context(state, sg) + "\n\nExisting tools:\n" + s.tools.catalog()
        try:
            d = await s.llm.chat_json("orchestrator", SYNTH_PROMPT, ctx, ToolDraft, check=draft_errors)
        except ModelOutputError as e:
            return failed(sg, str(e))

        tool_src, test_src = compose_tool_source(d), strip_fences(d.tests).strip() + "\n"
        tool_rel, test_rel = f"tools/{d.name}.py", s.synth.test_rel(d.name)
        files: dict[str, str | None] = {}
        async with s.write_lock:
            for rel, text in ((tool_rel, tool_src), (test_rel, test_src)):
                p = s.workspace.safe_path(rel)
                files[rel] = s.workspace.snapshot(p)
                await asyncio.to_thread(s.workspace.write, p, text, managed=True)

        result = make_diff(tool_rel, files[tool_rel], tool_src) + "\n" + make_diff(test_rel, files[test_rel], test_src)
        meta = {"files": files, "written": True, "synth": spec_of(d), "digest": digest(tool_src, test_src)}
        return WorkerOutput(subgoal_id=sg.id, status="complete", result=result,
                            reasoning=f"Gemma 3 12B drafted tool {d.name}; passed static checks."), meta