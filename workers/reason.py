"""Reason worker: a structured written answer from the json_worker model, with no tools. Also the fallback."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..common import ModelOutputError
from ..models import JSON_WORKER
from ..prompts import WORKER_PROMPT
from ..schemas import HarnessState, ReasonOutput, Subgoal, WorkerOutput, WorkMeta
from .base import Worker, WorkResult, failed

if TYPE_CHECKING:
    from ..models import JsonReasoner
    from ..services import Services


class ReasonWorker(Worker):
    kind = "reason"
    planner_hint = "analysis, research or a written answer that needs no tool."

    def __init__(self, services: "Services", reasoner: "JsonReasoner | None" = None):
        super().__init__(services)
        self.reasoner = reasoner or services.models.json_worker()

    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        ctx = await self.context(JSON_WORKER, state, sg)
        try:
            out = await self.reasoner.ask(WORKER_PROMPT, ctx, ReasonOutput)
        except ModelOutputError as e:
            return failed(sg, str(e))
        return WorkerOutput(subgoal_id=sg.id, **out.model_dump()), WorkMeta()