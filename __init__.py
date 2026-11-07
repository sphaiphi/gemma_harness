"""Gemma Trinity engineering-agent harness: Thinker / Worker / Verifier over five local Gemma models."""
from .config import EmbeddingConfig, GitConfig, HarnessConfig, ModelConfig
from .embeddings import Doc, GemmaEmbedder
from .harness import GemmaTrinityHarness, GemmaTrinityUnifiedHarness
from .verification import DEFAULT_STATIC_CHECKS, StaticCheck, VerifyInput
from .workers import WORKER_TYPES, Worker

__all__ = [
    "GemmaTrinityHarness", "GemmaTrinityUnifiedHarness", "HarnessConfig", "ModelConfig", "GitConfig",
    "EmbeddingConfig", "GemmaEmbedder", "Doc",
    "Worker", "WORKER_TYPES", "StaticCheck", "VerifyInput", "DEFAULT_STATIC_CHECKS",
]