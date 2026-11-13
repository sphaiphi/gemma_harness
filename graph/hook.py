"""Side effects of a gate decision. The gate only routes; hooks register tools, commit, roll back and clean up."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..git import commit_message
from ..schemas import REGISTER_SYNTH, HarnessState, Subgoal, WorkMeta

if TYPE_CHECKING:
    from ..git import GitRepo
    from ..services import Services
    from ..tooling import ToolRuntime, VerifiedStore
    from ..workspace import Journal, Workspace


@dataclass
class HookContext:
    tid: str
    state: HarnessState
    sg: Subgoal
    meta: WorkMeta
    event: dict  # this decision's gate event; hooks may add fields such as "commit"


class Hook(ABC):
    @abstractmethod
    async def __call__(self, ctx: HookContext) -> None: ...


class RegisterSynthTool(Hook):
    def __init__(self, runtime: "ToolRuntime", store: "VerifiedStore", write_lock: asyncio.Lock):
        self.runtime = runtime
        self.store = store
        self.lock = write_lock

    async def __call__(self, ctx: HookContext) -> None:
        if REGISTER_SYNTH not in ctx.meta.on_accept:
            return
        spec = ctx.meta.extra["synth"]
        async with self.lock:
            await asyncio.to_thread(self.runtime.register, spec)
            await asyncio.to_thread(self.store.mark, spec["name"], ctx.meta.extra["digest"])


class GitCommit(Hook):
    def __init__(self, git: "GitRepo", write_lock: asyncio.Lock):
        self.git = git
        self.lock = write_lock

    async def __call__(self, ctx: HookContext) -> None:
        if not ctx.meta.changes or not self.git.enabled:
            return
        subject, body = commit_message(ctx.tid, ctx.state["plan_id"], ctx.sg, ctx.event)
        async with self.lock:
            sha = await asyncio.to_thread(self.git.commit_paths, list(ctx.meta.changes), subject, body)
        if sha:
            ctx.event["commit"] = sha


class Rollback(Hook):
    def __init__(self, workspace: "Workspace", write_lock: asyncio.Lock):
        self.workspace = workspace
        self.lock = write_lock

    async def __call__(self, ctx: HookContext) -> None:
        if ctx.meta.changes:
            async with self.lock:
                await asyncio.to_thread(self.workspace.restore, ctx.meta.changes)


class ClearJournal(Hook):
    def __init__(self, journal: "Journal"):
        self.journal = journal

    async def __call__(self, ctx: HookContext) -> None:
        await asyncio.to_thread(self.journal.clear, ctx.tid)


def default_accept_hooks(s: "Services") -> list[Hook]:
    return [RegisterSynthTool(s.runtime, s.verified, s.write_lock), GitCommit(s.git, s.write_lock),
            ClearJournal(s.journal)]


def default_reject_hooks(s: "Services") -> list[Hook]:
    return [Rollback(s.workspace, s.write_lock), ClearJournal(s.journal)]