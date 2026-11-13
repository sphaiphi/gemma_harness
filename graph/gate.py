"""Deterministic gate: final-test policy, budgets and routing. Side effects run as hooks."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from ..common import elide
from ..schemas import HarnessState, Plan, Subgoal, Verdict, WorkerOutput, WorkMeta
from ..workers import policy_for
from ..workers.base import VerifyPolicy
from .hooks import Hook, HookContext

if TYPE_CHECKING:
    from ..config import HarnessConfig


class Gate:
    def __init__(self, cfg: "HarnessConfig", policies: Mapping[str, VerifyPolicy],
                 accept_hooks: Sequence[Hook], reject_hooks: Sequence[Hook]):
        self.cfg = cfg
        self.policies = dict(policies)
        self.accept_hooks = list(accept_hooks)
        self.reject_hooks = list(reject_hooks)

    async def decide(self, state: HarnessState, config: RunnableConfig) -> dict:
        tid = config["configurable"]["thread_id"]
        plan = Plan.model_validate(state["plan"])
        sg = plan.subgoals[state["cursor"]]
        wo = WorkerOutput.model_validate(state["worker_output"])
        meta = WorkMeta.from_state(state.get("worker_meta"))
        is_final = state["cursor"] == len(plan.subgoals) - 1
        tests = (state.get("evidence") or {}).get("tests")
        policy = policy_for(self.policies, sg.kind)

        v = self._apply_policy(Verdict.model_validate(state["verdict"]), plan, sg, policy, is_final, tests)
        event = {"plan_id": state["plan_id"], "subgoal": sg.id, "kind": sg.kind,
                 "attempt": state["retries"].get(sg.id, 0) + 1, "verdict": v.verdict, "route_to": v.route_to,
                 "diagnosis": (v.diagnosis or "")[:300], "tests_passed": tests["passed"] if tests else None}
        # `event` is stored by reference, so hooks can still add to it (e.g. the commit sha).
        base = {"events": [*state.get("events", []), event], "verdict": v.model_dump()}
        ctx = HookContext(tid=tid, state=state, sg=sg, meta=meta, event=event)

        if v.verdict == "accept":
            for hook in self.accept_hooks:
                await hook(ctx)
            accepted = {**state["accepted"], sg.id: wo.result}
            if v.task_complete:
                return {**base, "accepted": accepted, "status": "complete", "next_node": END}
            return {**base, "accepted": accepted, "cursor": state["cursor"] + 1,
                    "verifier_feedback": None, "previous_output": None, "next_node": "work"}

        for hook in self.reject_hooks:
            await hook(ctx)
        return self._route_rejection(state, sg, wo, v, base)

    def _apply_policy(self, v: Verdict, plan: Plan, sg: Subgoal, policy: VerifyPolicy, is_final: bool,
                      tests: dict | None) -> Verdict:
        if is_final and self.cfg.tests_required(plan.task_type) and v.verdict == "accept" \
                and not (tests and tests["passed"]):
            no_tests = tests is None or tests["no_tests"]
            v = Verdict(subgoal_id=sg.id, verdict="revise",
                        diagnosis="No passing tests collected." if no_tests
                        else f"Tests fail:\n{elide(tests['output'], 1500)}",
                        route_to="thinker" if no_tests else policy.final_test_route)
        if v.verdict == "revise" and v.route_to not in ("worker", "thinker"):
            v.route_to = "worker"
        v.subgoal_id = sg.id
        v.task_complete = is_final and v.verdict == "accept"
        return v

    def _route_rejection(self, state: HarnessState, sg: Subgoal, wo: WorkerOutput, v: Verdict, base: dict) -> dict:
        cfg = self.cfg

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