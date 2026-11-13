"""Typed contracts between Thinker, Worker, Verifier and the graph state.

To add a subgoal kind, write a Worker subclass (kind, planner_hint, policy) and register it in
workers.WORKER_TYPES. Nothing in this module changes.
"""
from __future__ import annotations

from typing import Any, Collection, Literal, Sequence, TypedDict

from pydantic import BaseModel, Field, create_model

TaskType = Literal["coding", "math", "research", "reasoning", "other"]
Route = Literal["worker", "thinker"]

# Accept-hook trigger set by the synth worker: register the new tool once its subgoal is accepted.
REGISTER_SYNTH = "register_synth"


class Subgoal(BaseModel):
    id: str
    description: str
    kind: str  # one of the registered worker kinds; plan_model() constrains it for the Thinker
    suggested_role: Route = "worker"
    depends_on: list[str] = Field(default_factory=list)
    notes: str = ""


class Plan(BaseModel):
    task_type: TaskType
    summary: str
    subgoals: list[Subgoal] = Field(min_length=1)
    revision_of: str | None = None


def plan_model(kinds: Sequence[str]) -> type[Plan]:
    """A Plan schema whose subgoal kinds are limited to `kinds`. Used as the Thinker's structured output."""
    if not kinds:
        raise ValueError("at least one subgoal kind is required")
    kind_type = Literal[tuple(kinds)]  # type: ignore[valid-type]
    planned_subgoal = create_model("PlannedSubgoal", __base__=Subgoal, kind=(kind_type, ...))
    return create_model("PlannedPlan", __base__=Plan,
                        subgoals=(list[planned_subgoal], Field(min_length=1)))


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


class WorkMeta(BaseModel):
    """Facts a worker reports to verify and gate. Stored in the checkpoint, so it must stay JSON-serializable."""
    changes: dict[str, str | None] = Field(default_factory=dict)  # path -> original content; None = new file
    static_errors: list[str] = Field(default_factory=list)
    worker_error: str | None = None
    tool_ok: bool | None = None                                   # None when the subgoal ran no tool
    on_accept: list[str] = Field(default_factory=list)            # accept-hook triggers, e.g. REGISTER_SYNTH
    extra: dict[str, Any] = Field(default_factory=dict)           # worker-specific data, e.g. the synth spec

    @classmethod
    def from_state(cls, raw: dict | None) -> "WorkMeta":
        """Read worker_meta from graph state, converting the untyped dict saved by earlier versions."""
        raw = raw or {}
        if set(raw) <= set(cls.model_fields):
            return cls.model_validate(raw)
        changes: dict[str, str | None] = {}
        if raw.get("written"):
            if raw.get("files"):
                changes = dict(raw["files"])
            elif raw.get("path"):
                changes = {raw["path"]: raw.get("original")}
        return cls(changes=changes, static_errors=list(raw.get("static_errors") or []),
                   worker_error=raw.get("worker_error"), tool_ok=raw.get("tool_ok"),
                   on_accept=[REGISTER_SYNTH] if raw.get("synth") else [],
                   extra={k: raw[k] for k in ("synth", "digest", "tool") if k in raw})


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


def validate_plan(plan: Plan, accepted: dict, prev: dict[str, dict], max_subgoals: int,
                  kinds: Collection[str] | None = None) -> list[str]:
    errs: list[str] = []
    ids = [s.id for s in plan.subgoals]
    if len(ids) != len(set(ids)):
        errs.append("subgoal ids must be unique")
    if len(ids) > max_subgoals:
        errs.append(f"at most {max_subgoals} subgoals")
    if kinds is not None:
        for s in plan.subgoals:
            if s.kind not in kinds:
                errs.append(f"{s.id}.kind {s.kind!r} is not one of {sorted(kinds)}")
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