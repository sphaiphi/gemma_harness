"""Gemma Trinity engineering-agent harness: Thinker / Worker / Verifier over five local Gemma models."""
from .config import ContextConfig, EmbeddingConfig, GitConfig, HarnessConfig, ModelConfig, ModelSpec
from .context import ContextBuilder, ContextProvider, ContextRequest
from .embeddings import Doc, GemmaEmbedder
from .graph import Gate, Hook, HookContext
from .harness import GemmaTrinityHarness, GemmaTrinityUnifiedHarness
from .models import (Embedder, FimCompleter, JsonReasoner, ModelRegistry, OllamaTransport, TextNormalizer, ToolCall,
                     ToolCaller, register_adapter)
from .schemas import WorkMeta
from .scopes import SCOPES, Scope
from .tooling import ToolResult
from .verification import DEFAULT_STATIC_CHECKS, Collected, EvidenceCollector, StaticCheck, VerifyInput
from .workers import WORKER_TYPES, VerifyPolicy, Worker

__all__ = [
    "GemmaTrinityHarness", "GemmaTrinityUnifiedHarness",
    "HarnessConfig", "ModelConfig", "ModelSpec", "EmbeddingConfig", "ContextConfig", "GitConfig",
    "ModelRegistry", "OllamaTransport", "register_adapter",
    "JsonReasoner", "FimCompleter", "ToolCaller", "ToolCall", "TextNormalizer", "Embedder",
    "GemmaEmbedder", "Doc", "ContextBuilder", "ContextProvider", "ContextRequest", "Scope", "SCOPES",
    "Worker", "WORKER_TYPES", "VerifyPolicy", "WorkMeta", "ToolResult",
    "StaticCheck", "VerifyInput", "DEFAULT_STATIC_CHECKS", "EvidenceCollector", "Collected",
    "Gate", "Hook", "HookContext",
]