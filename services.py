"""Service container and composition root. All wiring happens in Services.create()."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from .config import HarnessConfig
from .embeddings import GemmaEmbedder
from .git import GitRepo
from .index import RepoIndex
from .llm import OllamaClient
from .tooling import ToolRegistry, ToolSynthesis, register_builtin_tools
from .workspace import Journal, TestRunner, Workspace


@dataclass
class Services:
    cfg: HarnessConfig
    llm: OllamaClient
    embedder: GemmaEmbedder
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
        e = cfg.embedding
        embedder = GemmaEmbedder(cfg.models.embedder, cfg.host, batch_size=e.batch_size,
                                 max_query_chars=e.max_query_chars, max_doc_chars=e.max_doc_chars,
                                 timeout_s=e.timeout_s)
        chroma = chromadb.PersistentClient(path=str(root / cfg.vector_db_dir))

        journal = Journal(root, cfg.journal_dir)
        workspace = Workspace(root, journal)
        repo = RepoIndex(root, chroma, embedder, cfg.skip_dirs, task=e.code_task)
        workspace.subscribe(repo.on_file_changed)

        tests = TestRunner(root, cfg)
        tools = ToolRegistry(chroma, embedder, task=e.tool_task)
        register_builtin_tools(tools, workspace, tests)
        synth = ToolSynthesis(root, cfg, tools, tests, journal)

        return cls(cfg=cfg, llm=OllamaClient(cfg), embedder=embedder, workspace=workspace, journal=journal,
                   repo=repo, tools=tools, synth=synth, tests=tests, git=GitRepo(root, cfg.git))