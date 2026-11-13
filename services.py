"""Service container and composition root. All wiring happens in Services.create()."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from .config import HarnessConfig
from .context import ContextBuilder, default_context_builders
from .git import GitRepo
from .index import RepoIndex
from .models import ModelRegistry, OllamaTransport
from .tooling import (MutationChecker, ToolAutoloader, ToolRegistry, ToolRuntime, VerifiedStore,
                      register_builtin_tools)
from .workspace import Journal, TestRunner, Workspace


@dataclass
class Services:
    cfg: HarnessConfig
    transport: OllamaTransport
    models: ModelRegistry
    workspace: Workspace
    journal: Journal
    repo: RepoIndex
    tools: ToolRegistry
    runtime: ToolRuntime
    verified: VerifiedStore
    autoloader: ToolAutoloader
    mutation: MutationChecker
    tests: TestRunner
    git: GitRepo
    contexts: dict[str, ContextBuilder]
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def create(cls, workspace_dir: str | Path, cfg: HarnessConfig) -> "Services":
        root = Path(workspace_dir).resolve()
        transport = OllamaTransport.from_config(cfg)
        models = ModelRegistry(cfg, transport)  # validates role bindings; raises on a bad config
        embedder = models.embedder()
        chroma = chromadb.PersistentClient(path=str(root / cfg.vector_db_dir))

        journal = Journal(root, cfg.journal_dir)
        workspace = Workspace(root, journal)
        repo = RepoIndex(root, chroma, embedder, cfg.skip_dirs, task=cfg.embedding.code_task)
        workspace.subscribe(repo.on_file_changed)

        tests = TestRunner(root, cfg)
        tools = ToolRegistry(chroma, embedder, task=cfg.embedding.tool_task)
        register_builtin_tools(tools, workspace, tests)
        runtime = ToolRuntime(root, tools, tests, journal, cfg.synth_tool_timeout_s)
        verified = VerifiedStore(root / cfg.verified_file)

        return cls(cfg=cfg, transport=transport, models=models, workspace=workspace, journal=journal,
                   repo=repo, tools=tools, runtime=runtime, verified=verified,
                   autoloader=ToolAutoloader(root, tools, runtime, verified, tests),
                   mutation=MutationChecker(root, tests), tests=tests, git=GitRepo(root, cfg.git),
                   contexts=default_context_builders(cfg, repo, tools))