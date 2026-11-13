"""Gemma Trinity engineering-agent harness: Thinker / Worker / Verifier over five local Gemma models."""
from .config import EmbeddingConfig, GitConfig, HarnessConfig, ModelConfig
from .embeddings import Doc, GemmaEmbedder
from .graph import Gate, Hook, HookContext
from .harness import GemmaTrinityHarness, GemmaTrinityUnifiedHarness
from .schemas import WorkMeta
from .tooling import ToolResult
from .verification import DEFAULT_STATIC_CHECKS, Collected, EvidenceCollector, StaticCheck, VerifyInput
from .workers import WORKER_TYPES, VerifyPolicy, Worker

__all__ = [
    "GemmaTrinityHarness", "GemmaTrinityUnifiedHarness", "HarnessConfig", "ModelConfig", "GitConfig",
    "EmbeddingConfig", "GemmaEmbedder", "Doc",
    "Worker", "WORKER_TYPES", "VerifyPolicy", "WorkMeta", "ToolResult",
    "StaticCheck", "VerifyInput", "DEFAULT_STATIC_CHECKS", "EvidenceCollector", "Collected",
    "Gate", "Hook", "HookContext",
]