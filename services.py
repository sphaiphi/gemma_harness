"""Service container and composition root. All wiring happens in Services.create()."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from .config import HarnessConfig
from .git import GitRepo
from .index import RepoIndex
from .llm import OllamaClient, OllamaEmbeddingFunction
from .tooling import ToolRegistry, ToolSynthesis, register_builtin_tools
from .workspace import Journal, TestRunner, Workspace


@dataclass
class Services:
    cfg: HarnessConfig
    llm: OllamaClient
    workspace: Workspace
    journal: Journal
    repo: RepoIndex
    tools: ToolRegistry
    synth: ToolSynthesis
    tests: TestRunner
    git: GitRepo
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def create(cls, workspace_dir: str | Path, cfg: HarnessConfig) -> "Services":
        root = Path(workspace_dir).resolve()
        embed = OllamaEmbeddingFunction(cfg.models.embedder, cfg.host)
        chroma = chromadb.PersistentClient(path=str(root / cfg.vector_db_dir))

        journal = Journal(root, cfg.journal_dir)
        workspace = Workspace(root, journal)
        repo = RepoIndex(root, chroma, embed, cfg.skip_dirs)
        workspace.subscribe(repo.on_file_changed)

        tests = TestRunner(root, cfg)
        tools = ToolRegistry(chroma, embed)
        register_builtin_tools(tools, workspace, tests)
        synth = ToolSynthesis(root, cfg, tools, tests, journal)

        return cls(cfg=cfg, llm=OllamaClient(cfg), workspace=workspace, journal=journal, repo=repo,
                   tools=tools, synth=synth, tests=tests, git=GitRepo(root, cfg.git))