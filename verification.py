"""Two-tier verification.

Tier 1: pluggable static checks, then evidence collectors chosen by the worker's VerifyPolicy. Any problem sends
the subgoal back to the worker without an LLM call.
Tier 2: the Gemma 3 12B judge, unless the policy says to review this kind only on the final subgoal.
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar, Mapping, Sequence

from .common import ModelOutputError, elide
from .prompts import VERIFIER_PROMPT
from .schemas import HarnessState, Plan, Subgoal, Verdict, WorkerOutput, WorkMeta
from .tooling.layout import test_rel
from .workers import policy_for
from .workers.base import VerifyPolicy

if TYPE_CHECKING:
    from .config import HarnessConfig
    from .llm import OllamaClient
    from .services import Services
    from .tooling import MutationChecker
    from .workspace import TestRunner

SUITE = "suite"  # name of the full-suite collector; runs for coding tasks after run_suite kinds and at the end


@dataclass
class VerifyInput:
    state: HarnessState
    plan: Plan
    sg: Subgoal
    wo: WorkerOutput
    meta: WorkMeta
    policy: VerifyPolicy
    is_final: bool
    tests_required: bool


# A static check returns error strings. Any error rejects the subgoal back to the worker.
StaticCheck = Callable[[VerifyInput], list[str]]


def static_errors(vi: VerifyInput) -> list[str]:
    return list(vi.meta.static_errors)


def worker_error(vi: VerifyInput) -> list[str]:
    return [f"Worker error: {vi.meta.worker_error}"] if vi.meta.worker_error else []


def tool_failed(vi: VerifyInput) -> list[str]:
    return [elide(vi.wo.result or "tool failed", 1500)] if vi.meta.tool_ok is False else []


DEFAULT_STATIC_CHECKS: tuple[StaticCheck, ...] = (static_errors, worker_error, tool_failed)


@dataclass
class Collected:
    evidence: dict = field(default_factory=dict)  # stored in state["evidence"]
    problem: str | None = None                    # Tier 1 failure: revise -> worker
    section: str | None = None                    # text added to the Tier 2 context
    note: str | None = None                       # appended to the worker's reasoning


class EvidenceCollector(ABC):
    name: ClassVar[str]

    @abstractmethod
    async def collect(self, vi: VerifyInput) -> Collected: ...


class SynthToolChecks(EvidenceCollector):
    name = "synth_checks"

    def __init__(self, tests: "TestRunner", mutation: "MutationChecker", write_lock: asyncio.Lock):
        self.tests = tests
        self.mutation = mutation
        self.lock = write_lock

    async def collect(self, vi: VerifyInput) -> Collected:
        spec = vi.meta.extra.get("synth")
        if not spec:
            return Collected()
        name = spec["name"]
        async with self.lock:
            tests = await asyncio.to_thread(self.tests.run, [test_rel(name)])
            mutation = await asyncio.to_thread(self.mutation.check, name) if tests["passed"] else None
        if not tests["passed"]:
            return Collected(evidence={"tool_tests": tests},
                             problem=f"Tool checks failed: tool tests fail:\n{elide(tests['output'], 2000)}")
        if not mutation["ok"]:
            return Collected(evidence={"tool_tests": tests},
                             problem=f"Tool checks failed: {mutation['why']}\n{elide(mutation['output'], 1500)}")
        return Collected(
            evidence={"tool_tests": tests, "mutation": {"ok": True}},
            section=("Synthesized tool checks (automated, real evidence):\n- tool tests: PASSED\n"
                     "- mutation check: PASSED (tests pass on the real tool in isolation and fail with "
                     "NotImplementedError when its body is removed)"),
            note="Automated checks: tool tests passed; mutation check passed.")


class FullSuite(EvidenceCollector):
    name = SUITE

    def __init__(self, tests: "TestRunner", write_lock: asyncio.Lock):
        self.tests = tests
        self.lock = write_lock

    async def collect(self, vi: VerifyInput) -> Collected:
        async with self.lock:
            t = await asyncio.to_thread(self.tests.run)
        section = f"Test run (full suite): {'PASSED' if t['passed'] else 'FAILED'}\n{elide(t['output'], 3000)}"
        pending = vi.plan.subgoals[vi.state["cursor"] + 1:]
        if pending and not vi.is_final:
            section += ("\n\nLater subgoals are still pending:\n" +
                        "\n".join(f"- {s.id} ({s.kind}): {s.description}" for s in pending) +
                        "\nFailures in tests that only a pending subgoal will address are NOT this subgoal's "
                        "fault. Judge only what this subgoal is responsible for.")
        return Collected(evidence={"tests": t}, section=section)


def default_collectors(s: "Services") -> list[EvidenceCollector]:
    return [SynthToolChecks(s.tests, s.mutation, s.write_lock), FullSuite(s.tests, s.write_lock)]


def build_verifier_context(vi: VerifyInput, sections: Sequence[str]) -> str:
    sg, plan, state = vi.sg, vi.plan, vi.state
    parts = [f"Subgoal {sg.id} ({sg.kind}): {sg.description}"]
    if sg.notes:
        parts.append(f"Planner notes: {sg.notes}")
    parts.append("Worker output:\n" + elide(json.dumps(vi.wo.model_dump(), indent=1), 6000))
    parts += sections
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
    def __init__(self, cfg: "HarnessConfig", llm: "OllamaClient", policies: Mapping[str, VerifyPolicy],
                 collectors: Sequence[EvidenceCollector], static_checks: Sequence[StaticCheck] | None = None):
        self.cfg = cfg
        self.llm = llm
        self.policies = dict(policies)
        self.collectors = {c.name: c for c in collectors}
        self.static_checks = tuple(static_checks if static_checks is not None else DEFAULT_STATIC_CHECKS)
        if SUITE not in self.collectors:
            raise ValueError(f"a {SUITE!r} evidence collector is required")
        if unknown := sorted({n for p in self.policies.values() for n in p.collectors} - set(self.collectors)):
            raise ValueError(f"worker policies name unknown evidence collectors: {unknown}")

    def _input(self, state: HarnessState) -> VerifyInput:
        plan = Plan.model_validate(state["plan"])
        sg = plan.subgoals[state["cursor"]]
        return VerifyInput(
            state=state, plan=plan, sg=sg, wo=WorkerOutput.model_validate(state["worker_output"]),
            meta=WorkMeta.from_state(state.get("worker_meta")), policy=policy_for(self.policies, sg.kind),
            is_final=state["cursor"] == len(plan.subgoals) - 1,
            tests_required=self.cfg.tests_required(plan.task_type))

    @staticmethod
    def _collector_names(vi: VerifyInput) -> list[str]:
        names = list(vi.policy.collectors)
        if vi.tests_required and (vi.policy.run_suite or vi.is_final):
            names.append(SUITE)
        return names

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
        sections: list[str] = []
        for name in self._collector_names(vi):
            c = await self.collectors[name].collect(vi)
            evidence.update(c.evidence)
            if c.problem:
                return {"verdict": verdict("revise", c.problem, "worker"), "evidence": evidence}
            if c.section:
                sections.append(c.section)
            if c.note:
                vi.wo = vi.wo.model_copy(update={"reasoning": f"{vi.wo.reasoning} {c.note}".strip()})

        out = {"evidence": evidence, "worker_output": vi.wo.model_dump()}
        if vi.policy.review == "final_only" and not vi.is_final:
            return {**out, "verdict": verdict("accept")}

        # Tier 2: Gemma 3 12B judgment.
        try:
            v = await self.llm.chat_json("orchestrator", VERIFIER_PROMPT, build_verifier_context(vi, sections),
                                         Verdict)
        except ModelOutputError as e:
            return {**out, "verdict": verdict("revise", f"Verifier error: {e}", "worker")}
        return {**out, "verdict": v.model_dump()}