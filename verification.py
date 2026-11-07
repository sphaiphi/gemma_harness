"""Two-tier verification. Tier 1 is a list of pluggable static checks; Tier 2 is the Gemma 3 12B judge."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from .common import ModelOutputError, elide
from .prompts import VERIFIER_PROMPT
from .schemas import HarnessState, Plan, Subgoal, Verdict, WorkerOutput

if TYPE_CHECKING:
    from .services import Services


@dataclass
class VerifyInput:
    state: HarnessState
    plan: Plan
    sg: Subgoal
    wo: WorkerOutput
    meta: dict
    is_final: bool
    tests_required: bool


# A Tier 1 check returns error strings. Any error rejects the subgoal back to the worker without an LLM call.
StaticCheck = Callable[[VerifyInput], list[str]]


def static_errors(vi: VerifyInput) -> list[str]:
    return list(vi.meta.get("static_errors", []))


def worker_error(vi: VerifyInput) -> list[str]:
    return [f"Worker error: {vi.meta['worker_error']}"] if vi.meta.get("worker_error") else []


def tool_failed(vi: VerifyInput) -> list[str]:
    if vi.sg.kind == "tool" and not vi.meta.get("tool_ok"):
        return [elide(vi.wo.result or "tool failed", 1500)]
    return []


DEFAULT_STATIC_CHECKS: tuple[StaticCheck, ...] = (static_errors, worker_error, tool_failed)


def build_verifier_context(vi: VerifyInput, evidence: dict) -> str:
    sg, plan, state = vi.sg, vi.plan, vi.state
    parts = [f"Subgoal {sg.id} ({sg.kind}): {sg.description}"]
    if sg.notes:
        parts.append(f"Planner notes: {sg.notes}")
    parts.append("Worker output:\n" + elide(json.dumps(vi.wo.model_dump(), indent=1), 6000))

    if tt := evidence.get("tool_tests"):
        mut = evidence.get("mutation") or {}
        parts.append("Synthesized tool checks (automated, real evidence):\n"
                     f"- tool tests: {'PASSED' if tt['passed'] else 'FAILED'}\n"
                     "- mutation check: " + ("PASSED (tests pass on the real tool in isolation and fail with "
                                             "NotImplementedError when its body is removed)"
                                             if mut.get("ok") else "not run"))

    if t := evidence.get("tests"):
        section = f"Test run (full suite): {'PASSED' if t['passed'] else 'FAILED'}\n{elide(t['output'], 3000)}"
        pending = plan.subgoals[state["cursor"] + 1:]
        if pending and not vi.is_final:
            section += ("\n\nLater subgoals are still pending:\n" +
                        "\n".join(f"- {s.id} ({s.kind}): {s.description}" for s in pending) +
                        "\nFailures in tests that only a pending subgoal will address are NOT this subgoal's "
                        "fault. Judge only what this subgoal is responsible for.")
        parts.append(section)

    if not vi.tests_required:
        parts.append(f"No tests are required for this task type ({plan.task_type}); "
                     "judge the result on its content.")

    if vi.is_final:
        results = {**state["accepted"], sg.id: vi.wo.result}
        rows = []
        for s in plan.subgoals:
            r = results.get(s.id)
            shown = elide(str(r), 500) if r is not None else "MISSING"
            rows.append(f"- {s.id} ({s.kind}): {s.description}\n  result: {shown}")
        parts.append("FINAL SUBGOAL check. Confirm every subgoal below has a result and that together they "
                     f"answer the query.\nQuery:\n{state['raw_spec']}\n\nAll subgoals and results:\n" +
                     "\n".join(rows))
    return "\n\n".join(parts)


class Verifier:
    def __init__(self, services: "Services", static_checks: Sequence[StaticCheck] | None = None):
        self.s = services
        self.static_checks = tuple(static_checks if static_checks is not None else DEFAULT_STATIC_CHECKS)

    def _input(self, state: HarnessState) -> VerifyInput:
        plan = Plan.model_validate(state["plan"])
        return VerifyInput(
            state=state, plan=plan, sg=plan.subgoals[state["cursor"]],
            wo=WorkerOutput.model_validate(state["worker_output"]), meta=state.get("worker_meta") or {},
            is_final=state["cursor"] == len(plan.subgoals) - 1,
            tests_required=self.s.cfg.tests_required(plan.task_type))

    async def verify(self, state: HarnessState) -> dict:
        vi = self._input(state)

        def verdict(v: str, diagnosis: str | None = None, route_to: str | None = None) -> dict:
            return Verdict(subgoal_id=vi.sg.id, verdict=v, diagnosis=diagnosis, route_to=route_to).model_dump()

        # Tier 1: no LLM.
        if vi.wo.status == "blocked":
            return {"verdict": verdict("revise", f"Worker blocked: {vi.wo.blocked_reason}", "thinker"),
                    "evidence": {}}
        if errors := [e for check in self.static_checks for e in check(vi)]:
            return {"verdict": verdict("revise", "Automated checks failed:\n" + "\n".join(f"- {e}" for e in errors),
                                       "worker"), "evidence": {}}

        evidence: dict = {}
        if vi.sg.kind == "synth" and vi.meta.get("synth"):
            problem, tool_tests = await self._check_synth(vi.meta["synth"]["name"])
            if problem:
                return {"verdict": verdict("revise", f"Tool checks failed: {problem}", "worker"),
                        "evidence": {"tool_tests": tool_tests}}
            evidence.update(tool_tests=tool_tests, mutation={"ok": True})
            vi.wo = vi.wo.model_copy(update={"reasoning": (vi.wo.reasoning + " Automated checks: tool tests "
                                                           "passed; mutation check passed.").strip()})

        if vi.tests_required and (vi.sg.kind in ("edit", "synth") or vi.is_final):
            async with self.s.write_lock:
                evidence["tests"] = await asyncio.to_thread(self.s.tests.run)

        out = {"evidence": evidence, "worker_output": vi.wo.model_dump()}
        if vi.sg.kind == "tool" and not vi.is_final:
            return {**out, "verdict": verdict("accept")}

        # Tier 2: Gemma 3 12B judgment.
        try:
            v = await self.s.llm.chat_json("orchestrator", VERIFIER_PROMPT,
                                           build_verifier_context(vi, evidence), Verdict)
        except ModelOutputError as e:
            return {**out, "verdict": verdict("revise", f"Verifier error: {e}", "worker")}
        return {**out, "verdict": v.model_dump()}

    async def _check_synth(self, name: str) -> tuple[str | None, dict]:
        s = self.s
        async with s.write_lock:
            tests = await asyncio.to_thread(s.tests.run, [s.synth.test_rel(name)])
            mutation = await asyncio.to_thread(s.synth.mutation_check, name) if tests["passed"] else None
        if not tests["passed"]:
            return f"tool tests fail:\n{elide(tests['output'], 2000)}", tests
        if not mutation["ok"]:
            return f"{mutation['why']}\n{elide(mutation['output'], 1500)}", tests
        return None, tests