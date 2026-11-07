"""Gemma Trinity engineering-agent harness: Thinker / Worker / Verifier over five local Gemma models."""
from .config import GitConfig, HarnessConfig, ModelConfig
from .harness import GemmaTrinityHarness, GemmaTrinityUnifiedHarness
from .verification import DEFAULT_STATIC_CHECKS, StaticCheck, VerifyInput
from .workers import WORKER_TYPES, Worker

__all__ = [
    "GemmaTrinityHarness", "GemmaTrinityUnifiedHarness", "HarnessConfig", "ModelConfig", "GitConfig",
    "Worker", "WORKER_TYPES", "StaticCheck", "VerifyInput", "DEFAULT_STATIC_CHECKS",
]