"""Typed contracts between Thinker, Worker, Verifier and the graph state."""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

# To add a subgoal kind: extend this Literal, add a Worker subclass, register it in workers.WORKER_TYPES,
# and describe the kind in prompts.thinker_prompt().
SubgoalKind = Literal["edit", "tool", "reason", "synth"]
TaskType = Literal["coding", "math", "research", "reasoning", "other"]
Route = Literal["worker", "thinker"]


class Subgoal(BaseModel):
    id: str
    description: str
    kind: SubgoalKind
    suggested_role: Route = "worker"
    depends_on: list[str] = Field(default_factory=list)
    notes: str = ""


class Plan(BaseModel):
    task_type: TaskType
    summary: str
    subgoals: list[Subgoal] = Field(min_length=1)
    revision_of: str | None = None


class ReasonOutput(BaseModel):
    status: Literal["complete", "blocked"]
    result: str | None = None
    reasoning: str = ""
    blocked_reason: str | None = None


class WorkerOutput(ReasonOutput):
    subgoal_id: str


class EditCode(BaseModel):
    code: str


class ToolDraft(BaseModel):
    name: str = Field(description="snake_case tool name; also the function name")
    description: str = Field(description="One or two sentences: what the tool does, its inputs and its output.")
    parameters: dict[str, Any] = Field(description='JSON Schema with "type": "object" for the arguments')
    code: str = Field(description="Python module defining the function. No TOOL_SPEC.")
    tests: str = Field(description="pytest module for the tool")


class Verdict(BaseModel):
    subgoal_id: str
    verdict: Literal["accept", "revise"]
    diagnosis: str | None = None
    route_to: Route | None = None
    task_complete: bool = False


class HarnessState(TypedDict, total=False):
    raw_spec: str
    structured_ast: str
    plan: dict | None
    plan_id: str | None
    prior_context: dict | None
    cursor: int
    accepted: dict[str, Any]
    worker_output: dict | None
    worker_meta: dict
    verifier_feedback: str | None
    previous_output: dict | None
    evidence: dict
    verdict: dict | None
    retries: dict[str, int]
    replans: int
    status: str
    failure_reason: str | None
    next_node: str
    events: list[dict]


def initial_state(raw_spec: str) -> HarnessState:
    return HarnessState(
        raw_spec=raw_spec, structured_ast="", plan=None, plan_id=None, prior_context=None, cursor=0,
        accepted={}, worker_output=None, worker_meta={}, verifier_feedback=None, previous_output=None,
        evidence={}, verdict=None, retries={}, replans=0, status="running", failure_reason=None,
        next_node="think", events=[],
    )


def validate_plan(plan: Plan, accepted: dict, prev: dict[str, dict], max_subgoals: int) -> list[str]:
    errs: list[str] = []
    ids = [s.id for s in plan.subgoals]
    if len(ids) != len(set(ids)):
        errs.append("subgoal ids must be unique")
    if len(ids) > max_subgoals:
        errs.append(f"at most {max_subgoals} subgoals")
    seen: set[str] = set()
    for s in plan.subgoals:
        if bad := [d for d in s.depends_on if d not in seen]:
            errs.append(f"{s.id}.depends_on names {bad}, which are not earlier subgoals")
        seen.add(s.id)
    if prev:
        kept = [s for s in plan.subgoals if s.id in accepted]
        for s in kept:
            if s.description != prev[s.id]["description"]:
                errs.append(f"{s.id} was accepted; keep its description or give the changed subgoal a new id")
        if [s.id for s in plan.subgoals[:len(kept)]] != [s.id for s in kept]:
            errs.append("kept accepted subgoals must come first, in their original order")
        if len(kept) == len(plan.subgoals):
            errs.append("a revision must contain at least one new subgoal")
    return errs