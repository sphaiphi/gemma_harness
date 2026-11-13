"""Edit worker: locate the target, pick an edit strategy, lint, then write."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Sequence

from ..common import ModelOutputError, make_diff
from ..schemas import HarnessState, Subgoal, WorkerOutput, WorkMeta
from ..security import lint
from .base import VerifyPolicy, Worker, WorkResult, blocked, failed
from .edit_strategies import EditStrategy, FimStrategy, RewriteStrategy
from .locate import TargetLocator

if TYPE_CHECKING:
    from ..services import Services


class EditWorker(Worker):
    kind = "edit"
    requires_scope = "write"
    planner_hint = ("change one function, class or small region of ONE Python file, or create one new file.\n"
                    'Start the description with the file path, e.g. "In src/math_utils.py: ...".')
    policy = VerifyPolicy(run_suite=True, final_test_route="worker")

    def __init__(self, services: "Services", strategies: Sequence[EditStrategy] | None = None):
        super().__init__(services)
        self.locator = TargetLocator(services.workspace, services.repo)
        self.strategies = list(strategies or (FimStrategy(services, services.models.fim()),
                                              RewriteStrategy(services, services.models.json_worker())))

    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        try:
            t = await self.locator.locate(sg)
        except (LookupError, PermissionError) as e:
            return blocked(sg, str(e))
        if t.size > self.s.cfg.max_rewrite_lines:
            return blocked(sg, f"{t.rel} is too large to rewrite whole")

        strategy = next((st for st in self.strategies if st.applies(t, attempt)), None)
        if strategy is None:
            return failed(sg, "no edit strategy applies to this target")
        try:
            new_text = await strategy.edit(state, sg, t)
        except ModelOutputError as e:
            return failed(sg, str(e))

        out = WorkerOutput(subgoal_id=sg.id, status="complete", result=make_diff(t.rel, t.original, new_text),
                           reasoning=f"{strategy.engine} rewrote {t.span_label} of {t.rel}.")
        errors = lint(new_text) if t.rel.endswith(".py") else []
        if new_text == (t.original or ""):
            errors.append("edit changed nothing")
        if errors:
            return out, WorkMeta(static_errors=errors)

        async with self.s.write_lock:
            await asyncio.to_thread(self.s.workspace.write, t.path, new_text)
        return out, WorkMeta(changes=await self.recorded_changes({t.rel: t.original}))