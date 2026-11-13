"""Scoped, budgeted model context.

Each role's prompt is built by a ContextBuilder from ContextProvider plug-ins. Each provider returns one section.
When the sections exceed the role's character budget, the builder trims the least important sections first.
Priority 0 (REQUIRED) sections are never trimmed, so the budget applies to everything else.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from .common import elide
from .index import node_span
from .models.registry import JSON_WORKER, PLANNER, VERIFIER
from .schemas import HarnessState, Subgoal

if TYPE_CHECKING:
    from .config import HarnessConfig
    from .index import RepoIndex
    from .tooling.registry import ToolRegistry

log = logging.getLogger(__name__)

REQUIRED = 0


@dataclass
class ContextRequest:
    state: HarnessState
    sg: Subgoal | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. target, task, vi, sections


class ContextProvider(ABC):
    name: ClassVar[str]
    priority: ClassVar[int] = 1  # 0 = required; higher numbers are trimmed first

    @abstractmethod
    async def provide(self, req: ContextRequest) -> str | None:
        """Return one section, or None to add nothing."""


class ContextBuilder:
    MIN_SECTION = 200  # a section that would be trimmed below this is dropped instead

    def __init__(self, role: str, budget: int, providers: Sequence[ContextProvider]):
        self.role = role
        self.budget = budget
        self.providers = list(providers)

    def add(self, provider: ContextProvider, *, before: str | None = None) -> None:
        """Add a provider at the end, or before the provider with the given name."""
        names = [p.name for p in self.providers]
        self.providers.insert(names.index(before) if before in names else len(self.providers), provider)

    async def build(self, req: ContextRequest) -> str:
        sections: list[list[Any]] = []
        for p in self.providers:
            text = await p.provide(req)
            if text and text.strip():
                sections.append([p, text])
        return "\n\n".join(self._fit(sections))

    def _fit(self, sections: list[list[Any]]) -> list[str]:
        over = sum(len(t) + 2 for _, t in sections) - self.budget
        if over > 0:
            trimmable = [(i, s) for i, s in enumerate(sections) if s[0].priority > REQUIRED]
            for _, item in sorted(trimmable, key=lambda x: (-x[1][0].priority, -x[0])):
                if over <= 0:
                    break
                p, text = item
                keep = len(text) - over
                item[1] = elide(text, keep) if keep >= self.MIN_SECTION else ""
                over -= len(text) - len(item[1])
                log.debug("[context] %s: trimmed %s from %d to %d chars", self.role, p.name, len(text), len(item[1]))
        return [t for _, t in sections if t]


def one_line(text: str, chars: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= chars else flat[:max(0, chars - 3)] + "..."


def file_outline(source: str, skip: tuple[int, int] | None = None) -> str:
    """Imports, top-level constants and def/class headers, with line numbers. Nodes inside `skip` are left out."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return elide(source, 3000)
    lines = source.splitlines()

    def inside(node: ast.AST) -> bool:
        s, e = node_span(node) if hasattr(node, "decorator_list") else (node.lineno, node.end_lineno)
        return skip is not None and skip[0] <= s and e <= skip[1]

    def numbered(a: int, b: int) -> list[str]:
        return [f"{i:>5}| {lines[i - 1]}" for i in range(a, b + 1)]

    def header(node: ast.AST) -> list[str]:
        s, _ = node_span(node)
        first_body = node.body[0].lineno if node.body else node.lineno
        return numbered(s, max(node.lineno, first_body - 1))

    out: list[str] = []
    for node in tree.body:
        if inside(node):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out += numbered(node.lineno, node.end_lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            out.append(f"{node.lineno:>5}| {lines[node.lineno - 1][:120]}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out += header(node)
            if isinstance(node, ast.ClassDef):
                for n in node.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not inside(n):
                        out += header(n)
    return "\n".join(out)


# ---------------------------------------------------------------- shared providers
class SubgoalSection(ContextProvider):
    name = "subgoal"
    priority = REQUIRED

    def __init__(self, show_kind: bool = False):
        self.show_kind = show_kind

    async def provide(self, req: ContextRequest) -> str | None:
        sg = req.sg
        if sg is None:
            return None
        kind = f" ({sg.kind})" if self.show_kind else ""
        return f"Subgoal {sg.id}{kind}: {sg.description}" + (f"\nPlanner notes: {sg.notes}" if sg.notes else "")


class Instruction(ContextProvider):
    """The caller's task text, passed as extra["task"]. Keep it last; it is never trimmed."""
    name = "instruction"
    priority = REQUIRED

    async def provide(self, req: ContextRequest) -> str | None:
        return req.extra.get("task")


# ---------------------------------------------------------------- worker providers
class DependencyResults(ContextProvider):
    name = "dependencies"
    priority = 1

    def __init__(self, per_result: int):
        self.per_result = per_result

    async def provide(self, req: ContextRequest) -> str | None:
        accepted = req.state.get("accepted") or {}
        deps = [d for d in req.sg.depends_on if d in accepted] if req.sg else []
        if not deps:
            return None
        return "Accepted results this subgoal depends on:\n" + "\n".join(
            f"* {k}: {elide(str(accepted[k]), self.per_result)}" for k in deps)


class OtherResultSummaries(ContextProvider):
    name = "other_results"
    priority = 3

    def __init__(self, chars: int):
        self.chars = chars

    async def provide(self, req: ContextRequest) -> str | None:
        accepted = req.state.get("accepted") or {}
        deps = set(req.sg.depends_on) if req.sg else set()
        others = [k for k in accepted if k not in deps]
        if not others:
            return None
        return "Other accepted subgoals (one-line summaries):\n" + "\n".join(
            f"- {k}: {one_line(accepted[k], self.chars)}" for k in others)


class RetryFeedback(ContextProvider):
    name = "retry"
    priority = REQUIRED

    async def provide(self, req: ContextRequest) -> str | None:
        feedback = req.state.get("verifier_feedback")
        if not feedback:
            return None
        text = f"RETRY. The verifier rejected your previous output:\n{feedback}"
        if prev := req.state.get("previous_output"):
            text += "\n\nYour rejected output:\n" + elide(json.dumps(prev), 3000)
        return text


class TargetFile(ContextProvider):
    """Edit rewrites: the target file's imports and signatures instead of the whole file."""
    name = "target_file"
    priority = 2

    async def provide(self, req: ContextRequest) -> str | None:
        t = req.extra.get("target")
        if t is None or t.is_new:
            return None
        outline = file_outline(t.original, (t.start, t.end))
        if not outline.strip():
            return None
        return (f"File {t.rel} outline (imports, constants and signatures; bodies omitted):\n"
                f"```python\n{outline}\n```")


# ---------------------------------------------------------------- planner providers
class TaskSection(ContextProvider):
    name = "task"
    priority = REQUIRED

    async def provide(self, req: ContextRequest) -> str | None:
        s = req.state
        return f"Task: {s['raw_spec']}\n\nNormalized: {s.get('structured_ast') or s['raw_spec']}"


class ToolCatalogSection(ContextProvider):
    name = "tools"
    priority = 2

    def __init__(self, tools: "ToolRegistry"):
        self.tools = tools

    async def provide(self, req: ContextRequest) -> str | None:
        return "Available tools (access level in brackets):\n" + (self.tools.catalog(show_scope=True) or "(none)")


class RelevantCode(ContextProvider):
    name = "code"
    priority = 3

    def __init__(self, repo: "RepoIndex", n: int = 5, per_hit: int = 600):
        self.repo = repo
        self.n = n
        self.per_hit = per_hit

    async def provide(self, req: ContextRequest) -> str | None:
        hits = await asyncio.to_thread(self.repo.search, req.state["raw_spec"], self.n)
        code = "\n".join(f"- {h['path']} lines {h['start']}-{h['end']}:\n{elide(h['text'], self.per_hit)}"
                         for h in hits)
        return f"Relevant code:\n{code or '(none)'}"


class RevisionSection(ContextProvider):
    name = "revision"
    priority = 1

    async def provide(self, req: ContextRequest) -> str | None:
        prior = req.state.get("prior_context")
        return "REVISION prior_context:\n" + json.dumps(prior, indent=1) if prior else None


# ---------------------------------------------------------------- verifier providers (extra["vi"]: VerifyInput)
class WorkerOutputSection(ContextProvider):
    name = "worker_output"
    priority = 1

    async def provide(self, req: ContextRequest) -> str | None:
        vi = req.extra["vi"]
        return "Worker output:\n" + elide(json.dumps(vi.wo.model_dump(), indent=1), 6000)


class EvidenceSection(ContextProvider):
    name = "evidence"
    priority = 2

    async def provide(self, req: ContextRequest) -> str | None:
        sections = req.extra.get("sections") or []
        return "\n\n".join(sections) or None


class NoTestsNote(ContextProvider):
    name = "no_tests"
    priority = REQUIRED

    async def provide(self, req: ContextRequest) -> str | None:
        vi = req.extra["vi"]
        if vi.tests_required:
            return None
        return f"No tests are required for this task type ({vi.plan.task_type}); judge the result on its content."


class FinalCheckSection(ContextProvider):
    name = "final_check"
    priority = REQUIRED

    async def provide(self, req: ContextRequest) -> str | None:
        vi = req.extra["vi"]
        if not vi.is_final:
            return None
        results = {**req.state["accepted"], vi.sg.id: vi.wo.result}
        rows = []
        for s in vi.plan.subgoals:
            r = results.get(s.id)
            rows.append(f"- {s.id} ({s.kind}): {s.description}\n  result: "
                        f"{elide(str(r), 500) if r is not None else 'MISSING'}")
        return ("FINAL SUBGOAL check. Confirm every subgoal below has a result and that together they answer the "
                f"query.\nQuery:\n{req.state['raw_spec']}\n\nAll subgoals and results:\n" + "\n".join(rows))


def default_context_builders(cfg: "HarnessConfig", repo: "RepoIndex",
                             tools: "ToolRegistry") -> dict[str, ContextBuilder]:
    budgets = cfg.context.budgets
    if missing := [r for r in (PLANNER, JSON_WORKER, VERIFIER) if r not in budgets]:
        raise ValueError(f"ContextConfig.budgets is missing {missing}")
    c = cfg.context
    return {
        PLANNER: ContextBuilder(PLANNER, budgets[PLANNER], [
            TaskSection(), ToolCatalogSection(tools), RelevantCode(repo), RevisionSection()]),
        JSON_WORKER: ContextBuilder(JSON_WORKER, budgets[JSON_WORKER], [
            SubgoalSection(), DependencyResults(c.dependency_result_chars), OtherResultSummaries(c.summary_chars),
            RetryFeedback(), TargetFile(), Instruction()]),
        VERIFIER: ContextBuilder(VERIFIER, budgets[VERIFIER], [
            SubgoalSection(show_kind=True), WorkerOutputSection(), EvidenceSection(), NoTestsNote(),
            FinalCheckSection()]),
    }