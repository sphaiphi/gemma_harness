"""All tunables. Components receive a HarnessConfig instead of reading module constants."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from .scopes import DEFAULT_TASK_SCOPES, READ_ONLY, SCOPES


@dataclass(frozen=True)
class ModelSpec:
    """One model the harness can call."""
    name: str                                                 # Ollama tag
    adapter: str                                              # key in models.registry.ADAPTERS
    lane: str = "large"                                       # concurrency lane, see ModelConfig.lanes
    options: Mapping[str, Any] = field(default_factory=dict)  # Ollama options, e.g. temperature
    engine: str = ""                                          # display name in logs and worker reasoning


# Role -> model id (a key of ModelConfig.resolved_specs()).
DEFAULT_ROLES: dict[str, str] = {
    "normalizer": "t5gemma",
    "planner": "gemma3-12b",
    "verifier": "gemma3-12b",   # rebind with roles={"verifier": "<model id>"}
    "json_worker": "gemma3-12b",
    "fim": "codegemma",
    "tool_router": "functiongemma",
    "embedder": "embeddinggemma",
}


@dataclass(frozen=True)
class ModelConfig:
    # Shortcuts (the fields earlier versions had): Ollama tags for the default models.
    transformer: str = "t5gemma:2b-custom"
    embedder: str = "embedding-gemma:300m-custom"
    orchestrator: str = "gemma3:12b-custom"
    function: str = "functiongemma:270m-custom"
    coder: str = "codegemma:2b-custom"
    # Full form. Entries here add to or replace the defaults built from the shortcuts.
    specs: Mapping[str, ModelSpec] = field(default_factory=dict)
    roles: Mapping[str, str] = field(default_factory=dict)
    # Concurrent requests per lane, so small models never wait behind a slow 12B call.
    lanes: Mapping[str, int] = field(default_factory=lambda: {"large": 1, "small": 4, "embed": 2})

    def resolved_specs(self) -> dict[str, ModelSpec]:
        defaults = {
            "t5gemma": ModelSpec(self.transformer, "t5gemma", "small", engine="T5Gemma 2B"),
            "embeddinggemma": ModelSpec(self.embedder, "embeddinggemma", "embed", engine="EmbeddingGemma 300M"),
            "gemma3-12b": ModelSpec(self.orchestrator, "gemma_json", "large", {"temperature": 0.2},
                                    engine="Gemma 3 12B"),
            "functiongemma": ModelSpec(self.function, "functiongemma", "small", {"temperature": 0.0},
                                       engine="FunctionGemma 270M"),
            "codegemma": ModelSpec(self.coder, "codegemma_fim", "small", {"temperature": 0.1, "num_predict": 1024},
                                   engine="CodeGemma 2B"),
        }
        return {**defaults, **self.specs}

    def resolved_roles(self) -> dict[str, str]:
        return {**DEFAULT_ROLES, **self.roles}


@dataclass(frozen=True)
class EmbeddingConfig:
    # EmbeddingGemma query tasks. Changing a task affects queries only, so no re-index is needed.
    code_task: str = "code retrieval"   # repo chunks: Thinker context and edit-target lookup
    tool_task: str = "search result"    # tool descriptions: candidate tools for the tool router
    batch_size: int = 32
    max_query_chars: int = 2000
    max_doc_chars: int = 4000           # EmbeddingGemma context is 2048 tokens; Ollama truncates beyond it
    timeout_s: float = 60.0


@dataclass(frozen=True)
class ContextConfig:
    # Character budgets per role. Required sections (task, subgoal, retry feedback, edit span) are never trimmed.
    budgets: Mapping[str, int] = field(default_factory=lambda: {"planner": 8000, "json_worker": 6000,
                                                                 "verifier": 10000})
    dependency_result_chars: int = 2000  # full result of each depends_on subgoal
    summary_chars: int = 160             # one-line summary of every other accepted subgoal


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
    context: ContextConfig = field(default_factory=ContextConfig)
    git: GitConfig = field(default_factory=GitConfig)

    # Loop budgets
    max_worker_retries: int = 3
    max_replans: int = 3
    max_subgoals: int = 10

    # Model transport (concurrency is set per lane in ModelConfig.lanes)
    llm_timeout_s: float = 180.0
    http_attempts: int = 3

    # Editing
    fim_max_lines: int = 60
    fim_attempts: int = 1
    max_rewrite_lines: int = 600

    # Access: which tool and subgoal scopes each task type may use
    task_scopes: Mapping[str, frozenset[str]] = field(default_factory=lambda: dict(DEFAULT_TASK_SCOPES))

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

    def __post_init__(self) -> None:
        for task_type, scopes in self.task_scopes.items():
            if bad := sorted(set(scopes) - set(SCOPES)):
                raise ValueError(f"task_scopes[{task_type!r}] has unknown scopes {bad}")

    @property
    def recursion_limit(self) -> int:
        per_plan = 1 + self.max_subgoals * (self.max_worker_retries + 1) * 3
        return 2 + (self.max_replans + 1) * per_plan + 10

    def tests_required(self, task_type: str) -> bool:
        return task_type not in self.no_test_task_types

    def allowed_scopes(self, task_type: str) -> frozenset[str]:
        return frozenset(self.task_scopes.get(task_type, READ_ONLY))