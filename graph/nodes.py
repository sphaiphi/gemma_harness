"""sanitize, think and work nodes."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from ..common import CURRENT_THREAD, ModelOutputError, elide
from ..prompts import thinker_prompt
from ..schemas import HarnessState, Plan, validate_plan
from ..workers import FALLBACK_KIND, Worker, blocked

if TYPE_CHECKING:
    from ..services import Services


class Nodes:
    def __init__(self, services: "Services", workers: dict[str, Worker]):
        self.s = services
        self.workers = workers

    async def sanitize(self, state: HarnessState) -> dict:
        if state.get("structured_ast"):
            return {}
        try:
            text = await self.s.llm.generate("transformer", f"Normalize task description:\n\n{state['raw_spec']}")
        except (httpx.HTTPError, ModelOutputError):
            text = ""
        return {"structured_ast": text.strip() or state["raw_spec"]}

    async def think(self, state: HarnessState) -> dict:
        s = self.s
        prior = state.get("prior_context")
        accepted = prior["accepted_results"] if prior else {}
        prev = {sg["id"]: sg for sg in prior["previous_plan"]["subgoals"]} if prior else {}

        hits = await asyncio.to_thread(s.repo.search, state["raw_spec"], 5)
        code = "\n".join(f"- {h['path']} lines {h['start']}-{h['end']}:\n{elide(h['text'], 600)}" for h in hits)
        parts = [f"Task: {state['raw_spec']}", f"Normalized: {state['structured_ast']}",
                 f"Available tools:\n{s.tools.catalog()}", f"Relevant code:\n{code or '(none)'}"]
        if prior:
            parts.append("REVISION prior_context:\n" + elide(json.dumps(prior, indent=1), 8000))

        try:
            plan = await s.llm.chat_json(
                "orchestrator", thinker_prompt(s.cfg.max_subgoals), "\n\n".join(parts), Plan,
                check=lambda p: validate_plan(p, accepted, prev, s.cfg.max_subgoals))
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
        return {"worker_output": out.model_dump(), "worker_meta": meta}