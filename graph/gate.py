"""Deterministic gate: final-test policy, commit or rollback, budgets and routing."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from ..common import elide
from ..git import commit_message
from ..schemas import HarnessState, Plan, Subgoal, Verdict, WorkerOutput
from ..workspace import changed_paths, files_of

if TYPE_CHECKING:
    from ..services import Services


class Gate:
    def __init__(self, services: "Services"):
        self.s = services

    async def decide(self, state: HarnessState, config: RunnableConfig) -> dict:
        tid = config["configurable"]["thread_id"]
        plan = Plan.model_validate(state["plan"])
        sg = plan.subgoals[state["cursor"]]
        wo = WorkerOutput.model_validate(state["worker_output"])
        meta = state.get("worker_meta") or {}
        is_final = state["cursor"] == len(plan.subgoals) - 1
        tests = (state.get("evidence") or {}).get("tests")

        v = self._apply_policy(Verdict.model_validate(state["verdict"]), plan, sg, is_final, tests)
        event = {"plan_id": state["plan_id"], "subgoal": sg.id, "kind": sg.kind,
                 "attempt": state["retries"].get(sg.id, 0) + 1, "verdict": v.verdict, "route_to": v.route_to,
                 "diagnosis": (v.diagnosis or "")[:300], "tests_passed": tests["passed"] if tests else None}
        # `event` is stored by reference, so _accept can still add the commit sha to it.
        base = {"events": [*state.get("events", []), event], "verdict": v.model_dump()}

        if v.verdict == "accept":
            return await self._accept(state, tid, sg, wo, meta, event, base, v.task_complete)
        return await self._reject(state, tid, sg, wo, v, meta, base)

    def _apply_policy(self, v: Verdict, plan: Plan, sg: Subgoal, is_final: bool, tests: dict | None) -> Verdict:
        if is_final and self.s.cfg.tests_required(plan.task_type) and v.verdict == "accept" \
                and not (tests and tests["passed"]):
            no_tests = tests is None or tests["no_tests"]
            v = Verdict(subgoal_id=sg.id, verdict="revise",
                        diagnosis="No passing tests collected." if no_tests
                        else f"Tests fail:\n{elide(tests['output'], 1500)}",
                        route_to="thinker" if no_tests or sg.kind != "edit" else "worker")
        if v.verdict == "revise" and v.route_to not in ("worker", "thinker"):
            v.route_to = "worker"
        v.subgoal_id = sg.id
        v.task_complete = is_final and v.verdict == "accept"
        return v

    async def _accept(self, state: HarnessState, tid: str, sg: Subgoal, wo: WorkerOutput, meta: dict,
                      event: dict, base: dict, complete: bool) -> dict:
        s = self.s
        accepted = {**state["accepted"], sg.id: wo.result}
        if meta.get("written"):
            sha = None
            async with s.write_lock:
                if spec := meta.get("synth"):
                    await asyncio.to_thread(s.synth.register, spec)
                    await asyncio.to_thread(s.synth.mark_verified, spec["name"], meta["digest"])
                if s.git.enabled:
                    subject, body = commit_message(tid, state["plan_id"], sg, event)
                    sha = await asyncio.to_thread(s.git.commit_paths, changed_paths(meta), subject, body)
            if sha:
                event["commit"] = sha
        await asyncio.to_thread(s.journal.clear, tid)
        if complete:
            return {**base, "accepted": accepted, "status": "complete", "next_node": END}
        return {**base, "accepted": accepted, "cursor": state["cursor"] + 1,
                "verifier_feedback": None, "previous_output": None, "next_node": "work"}

    async def _reject(self, state: HarnessState, tid: str, sg: Subgoal, wo: WorkerOutput, v: Verdict,
                      meta: dict, base: dict) -> dict:
        s, cfg = self.s, self.s.cfg
        if meta.get("written"):
            async with s.write_lock:
                await asyncio.to_thread(s.workspace.restore, files_of(meta))
        await asyncio.to_thread(s.journal.clear, tid)

        def fail(reason: str) -> dict:
            return {**base, "status": "failed", "failure_reason": reason, "next_node": END}

        if v.route_to == "worker":
            n = state["retries"].get(sg.id, 0)
            if n >= cfg.max_worker_retries:
                return fail(f"{sg.id} rejected after {cfg.max_worker_retries} retries: {v.diagnosis}")
            return {**base, "retries": {**state["retries"], sg.id: n + 1},
                    "verifier_feedback": v.diagnosis or "Rejected.",
                    "previous_output": {"result": wo.result, "reasoning": wo.reasoning}, "next_node": "work"}

        if state["replans"] >= cfg.max_replans:
            return fail(f"plan rejected after {cfg.max_replans} re-plans: {v.diagnosis}")
        return {**base, "replans": state["replans"] + 1, "next_node": "think",
                "prior_context": {"previous_plan": {**state["plan"], "plan_id": state["plan_id"]},
                                  "verifier_diagnosis": v.diagnosis, "accepted_results": state["accepted"]}}