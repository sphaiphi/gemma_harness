"""All tunables. Components receive a HarnessConfig instead of reading module constants."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    transformer: str = "t5gemma:2b-custom"
    embedder: str = "embedding-gemma:300m-custom"
    orchestrator: str = "gemma3:12b-custom"
    function: str = "functiongemma:270m-custom"
    coder: str = "codegemma:2b-custom"


@dataclass(frozen=True)
class EmbeddingConfig:
    # EmbeddingGemma query tasks. Changing a task affects queries only, so no re-index is needed.
    code_task: str = "code retrieval"   # repo chunks: Thinker context and edit-target lookup
    tool_task: str = "search result"    # tool descriptions: FunctionGemma candidate tools
    batch_size: int = 32
    max_query_chars: int = 2000
    max_doc_chars: int = 4000           # EmbeddingGemma context is 2048 tokens; Ollama truncates beyond it
    timeout_s: float = 60.0


@dataclass(frozen=True)
class GitConfig:
    enabled: bool = True
    baseline_commit: bool = True
    author: tuple[str, str] = ("Gemma Harness", "gemma-harness@localhost")
    ignore_lines: tuple[str, ...] = (
        ".gemma_vector_db/", ".gemma_checkpoints.db*", ".gemma_journal/",
        ".gemma_tools/", "__pycache__/", "*.pyc", ".pytest_cache/",
    )
    timeout_s: int = 120


@dataclass(frozen=True)
class HarnessConfig:
    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    models: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    git: GitConfig = field(default_factory=GitConfig)

    # Loop budgets
    max_worker_retries: int = 3
    max_replans: int = 3
    max_subgoals: int = 10

    # LLM transport
    max_concurrent_llm: int = 3
    llm_timeout_s: float = 180.0
    http_attempts: int = 3

    # Editing
    fim_max_lines: int = 60
    fim_attempts: int = 1
    max_rewrite_lines: int = 600

    # Tests and synthesized tools
    no_test_task_types: frozenset[str] = frozenset({"research", "reasoning"})
    test_timeout_s: int = 600
    synth_tool_timeout_s: int = 60
    test_env_passthrough: tuple[str, ...] = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
    skip_dirs: frozenset[str] = frozenset({"venv", "__pycache__", "node_modules", "build", "dist"})

    # Harness files inside the workspace
    vector_db_dir: str = ".gemma_vector_db"
    checkpoint_db: str = ".gemma_checkpoints.db"
    journal_dir: str = ".gemma_journal"
    verified_file: str = ".gemma_tools/verified.json"

    @property
    def recursion_limit(self) -> int:
        per_plan = 1 + self.max_subgoals * (self.max_worker_retries + 1) * 3
        return 2 + (self.max_replans + 1) * per_plan + 10

    def tests_required(self, task_type: str) -> bool:
        return task_type not in self.no_test_task_types