"""Synth worker: Gemma 3 12B drafts a new tool and its tests; static checks run before anything is written."""
from __future__ import annotations

import asyncio

from ..common import ModelOutputError, digest, make_diff, strip_fences
from ..prompts import SYNTH_PROMPT
from ..schemas import REGISTER_SYNTH, HarnessState, Subgoal, ToolDraft, WorkerOutput, WorkMeta
from ..tooling.checks import compose_tool_source, draft_errors, spec_of
from ..tooling.layout import test_rel, tool_rel
from .base import VerifyPolicy, Worker, WorkResult, failed, worker_context


class SynthWorker(Worker):
    kind = "synth"
    planner_hint = ("create ONE new reusable tool, only when no available tool fits and the task needs one. Start "
                    'the description with "Create tool <snake_case_name>:". Follow it with a tool subgoal that '
                    "calls the new tool.\n"
                    "Files under tools/ and tests/test_tool_*.py are managed: change them only with a synth subgoal.")
    policy = VerifyPolicy(run_suite=True, collectors=("synth_checks",))

    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        s = self.s
        ctx = worker_context(state, sg) + "\n\nExisting tools:\n" + s.tools.catalog()
        try:
            d = await s.llm.chat_json("orchestrator", SYNTH_PROMPT, ctx, ToolDraft, check=draft_errors)
        except ModelOutputError as e:
            return failed(sg, str(e))

        tool_src, test_src = compose_tool_source(d), strip_fences(d.tests).strip() + "\n"
        t_rel, x_rel = tool_rel(d.name), test_rel(d.name)
        files: dict[str, str | None] = {}
        async with s.write_lock:
            for rel, text in ((t_rel, tool_src), (x_rel, test_src)):
                p = s.workspace.safe_path(rel)
                files[rel] = s.workspace.snapshot(p)
                await asyncio.to_thread(s.workspace.write, p, text, managed=True)

        result = make_diff(t_rel, files[t_rel], tool_src) + "\n" + make_diff(x_rel, files[x_rel], test_src)
        meta = WorkMeta(changes=await self.recorded_changes(files), on_accept=[REGISTER_SYNTH],
                        extra={"synth": spec_of(d), "digest": digest(tool_src, test_src)})
        return WorkerOutput(subgoal_id=sg.id, status="complete", result=result,
                            reasoning=f"Gemma 3 12B drafted tool {d.name}; passed static checks."), meta