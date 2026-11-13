"""Access levels for tools and subgoal kinds, and which task types may use them."""
from __future__ import annotations

from typing import Any, Literal, Mapping

Scope = Literal["read", "write", "exec"]
SCOPES: tuple[str, ...] = ("read", "write", "exec")
ALL_SCOPES = frozenset(SCOPES)
READ_ONLY = frozenset({"read"})

# Task types not listed here get read-only access.
DEFAULT_TASK_SCOPES: dict[str, frozenset[str]] = {
    "coding": ALL_SCOPES,
    "math": ALL_SCOPES,
    "other": ALL_SCOPES,
    "research": READ_ONLY,
    "reasoning": READ_ONLY,
}


def check_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"unknown access scope {scope!r}; expected one of {list(SCOPES)}")
    return scope


def kind_scope_errors(plan: Any, kind_scopes: Mapping[str, str | None], allowed: frozenset[str]) -> list[str]:
    """Plan check: subgoal kinds that change files need an access level the plan's task type allows."""
    errs = []
    for sg in plan.subgoals:
        need = kind_scopes.get(sg.kind)
        if need and need not in allowed:
            errs.append(f"{sg.id}: {sg.kind} subgoals need {need} access, which {plan.task_type} tasks do not "
                        "have; use another kind or a task_type that allows it")
    return errs


def access_rules(task_scopes: Mapping[str, frozenset[str]], kind_scopes: Mapping[str, str | None]) -> str:
    """The access rules as one Thinker prompt bullet."""
    by_type = "; ".join(f"{t}: {', '.join(sorted(s)) or 'none'}" for t, s in task_scopes.items())
    needs = ", ".join(f"{k} needs {s}" for k, s in kind_scopes.items() if s) or "none"
    return (f"Access by task_type: {by_type}. Subgoal kinds that change files: {needs}. A tool subgoal can only "
            "use tools whose access level the task_type allows. Pick the task_type that matches the work.")