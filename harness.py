"""Facade: builds the graph, runs startup and crash recovery, and drives concurrent threads."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .config import HarnessConfig
from .graph import Gate, Nodes, build_graph
from .schemas import initial_state
from .services import Services
from .verification import StaticCheck, Verifier
from .workers import Worker, build_workers

log = logging.getLogger(__name__)


class GemmaTrinityHarness:
    def __init__(self, workspace_dir: str | Path = ".", config: HarnessConfig | None = None, *,
                 worker_types: Mapping[str, type[Worker]] | None = None,
                 static_checks: Sequence[StaticCheck] | None = None,
                 services: Services | None = None):
        self.cfg = config or HarnessConfig()
        self.s = services or Services.create(workspace_dir, self.cfg)
        self.workers = build_workers(self.s, worker_types)
        self.nodes = Nodes(self.s, self.workers)
        self.verifier = Verifier(self.s, static_checks)
        self.gate = Gate(self.s)

    def build_graph(self, checkpointer: Any):
        return build_graph(self.nodes, self.verifier, self.gate, checkpointer)

    # ------------------------------------------------------------- startup
    async def _startup(self, app: Any, tasks: list[dict[str, str]]) -> None:
        s = self.s
        async with s.write_lock:
            if s.git.enabled:
                s.git.prepare()
            await self._recover(app, tasks)
            s.synth.install_runtime()
            s.synth.autoload()
            if s.git.enabled:
                s.git.baseline()
        await asyncio.to_thread(s.repo.index_all)

    async def _recover(self, app: Any, tasks: list[dict[str, str]]) -> None:
        s = self.s
        for t in tasks:
            tid = t["id"]
            files = await asyncio.to_thread(s.journal.load, tid)
            if not files:
                continue
            snap = await app.aget_state({"configurable": {"thread_id": tid}})
            if set(snap.next or ()) & {"verify", "gate"}:
                log.info("[%s] crash journal kept: %d file(s) pending verification on resume", tid, len(files))
                continue
            await asyncio.to_thread(s.workspace.restore, files)
            await asyncio.to_thread(s.journal.clear, tid)
            log.info("[%s] crash recovery: restored %d file(s): %s", tid, len(files), ", ".join(sorted(files)))

    # ------------------------------------------------------------- running
    async def run_thread(self, app: Any, raw_spec: str, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self.cfg.recursion_limit}
        snap = await app.aget_state(config)
        if snap.next:
            inp, verb = None, "Resuming"
        elif (done := snap.values.get("status")) in ("complete", "failed"):
            log.info("[%s] already finished (%s); the given spec was ignored. "
                     "Use a new thread id to run a new task.", thread_id, done)
            return snap.values
        else:
            inp, verb = initial_state(raw_spec), "Starting"
        log.info("[%s] %s thread...", thread_id, verb)

        async for update in app.astream(inp, config, stream_mode="updates"):
            for node, upd in update.items():
                v = (upd or {}).get("verdict") if node == "gate" else None
                if v:
                    log.info("[%s] gate: %s %s -> %s", thread_id, v["subgoal_id"], v["verdict"], v["route_to"])
                    if commit := ((upd.get("events") or [{}])[-1]).get("commit"):
                        log.info("[%s] committed %s", thread_id, commit)

        final = (await app.aget_state(config)).values
        status, reason = final.get("status"), final.get("failure_reason")
        log.info("[%s] finished: %s%s", thread_id, status, f" - {reason}" if reason else "")
        return final

    async def run_concurrent_workload(self, tasks: list[dict[str, str]]) -> list[Any]:
        db = str(self.s.workspace.root / self.cfg.checkpoint_db)
        async with self.s.llm.session(), AsyncSqliteSaver.from_conn_string(db) as cp:
            app = self.build_graph(cp)
            await self._startup(app, tasks)
            return await asyncio.gather(*(self.run_thread(app, t["spec"], t["id"]) for t in tasks),
                                        return_exceptions=True)


# Backward-compatible name for code that imported the single-file version.
GemmaTrinityUnifiedHarness = GemmaTrinityHarness