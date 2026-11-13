"""Reason worker: a structured written answer from Gemma 3 12B, with no tools. Also the fallback worker."""
from __future__ import annotations

from ..common import ModelOutputError
from ..prompts import WORKER_PROMPT
from ..schemas import HarnessState, ReasonOutput, Subgoal, WorkerOutput, WorkMeta
from .base import Worker, WorkResult, failed, worker_context


class ReasonWorker(Worker):
    kind = "reason"
    planner_hint = "analysis, research or a written answer that needs no tool."

    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        try:
            out = await self.s.llm.chat_json("orchestrator", WORKER_PROMPT, worker_context(state, sg), ReasonOutput)
        except ModelOutputError as e:
            return failed(sg, str(e))
        return WorkerOutput(subgoal_id=sg.id, **out.model_dump()), WorkMeta()