"""sanitize, think and work nodes."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from ..common import CURRENT_THREAD, ModelError, ModelOutputError
from ..context import ContextBuilder, ContextRequest
from ..prompts import thinker_prompt
from ..schemas import HarnessState, Plan, plan_model, validate_plan
from ..scopes import access_rules, kind_scope_errors
from ..workers import FALLBACK_KIND, Worker, blocked, kind_scopes, planner_hints

if TYPE_CHECKING:
    from ..config import HarnessConfig
    from ..models import JsonReasoner, TextNormalizer


class Nodes:
    def __init__(self, cfg: "HarnessConfig", workers: dict[str, Worker], normalizer: "TextNormalizer",
                 planner: "JsonReasoner", planner_context: ContextBuilder):
        self.cfg = cfg
        self.workers = workers
        self.normalizer = normalizer
        self.planner = planner
        self.context = planner_context
        self.kinds = list(workers)
        self.kind_scopes = kind_scopes(workers)
        # Built from the registered workers and config, so a new kind needs no prompt or schema edits.
        self.plan_schema = plan_model(self.kinds)
        self.thinker_system = thinker_prompt(cfg.max_subgoals, planner_hints(workers),
                                             access_rules(cfg.task_scopes, self.kind_scopes))

    async def sanitize(self, state: HarnessState) -> dict:
        if state.get("structured_ast"):
            return {}
        try:
            text = await self.normalizer.normalize(state["raw_spec"])
        except ModelError:
            text = ""
        return {"structured_ast": text.strip() or state["raw_spec"]}

    def _plan_check(self, accepted: dict, prev: dict[str, dict]) -> Callable[[Plan], list[str]]:
        def check(p: Plan) -> list[str]:
            return (validate_plan(p, accepted, prev, self.cfg.max_subgoals, self.kinds)
                    + kind_scope_errors(p, self.kind_scopes, self.cfg.allowed_scopes(p.task_type)))
        return check

    async def think(self, state: HarnessState) -> dict:
        prior = state.get("prior_context")
        accepted = prior["accepted_results"] if prior else {}
        prev = {sg["id"]: sg for sg in prior["previous_plan"]["subgoals"]} if prior else {}

        user = await self.context.build(ContextRequest(state=state))
        try:
            plan: Plan = await self.planner.ask(self.thinker_system, user, self.plan_schema,
                                                check=self._plan_check(accepted, prev))
        except ModelOutputError as e:
            return {"status": "failed", "failure_reason": f"Thinker failed: {e}", "next_node": END}

        plan.revision_of = prior["previous_plan"]["plan_id"] if prior else None
        kept = {sg.id: accepted[sg.id] for sg in plan.subgoals if sg.id in accepted}
        return {"plan": plan.model_dump(), "plan_id": uuid.uuid4().hex[:8], "accepted": kept, "cursor": len(kept),
                "retries": {k: v for k, v in state["retries"].items() if k in kept}, "prior_context": None,
                "verifier_feedback": None, "previous_output": None, "next_node": "work"}

    async def work(self, state: HarnessState, config: RunnableConfig) -> dict:
        CURRENT_THREAD.set(config["configurable"]["thread_id"])
        sg = Plan.model_validate(state["plan"]).subgoals[state["cursor"]]
        attempt = state["retries"].get(sg.id, 0)
        worker = self.workers.get(sg.kind) or self.workers[FALLBACK_KIND]
        try:
            out, meta = await worker.run(state, sg, attempt)
        except PermissionError as e:
            out, meta = blocked(sg, str(e))
        return {"worker_output": out.model_dump(), "worker_meta": meta.model_dump()}