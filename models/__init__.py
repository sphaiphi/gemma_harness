"""Model layer: transport, capability ports, per-family adapters and role bindings."""
from .adapters import CodeGemmaFim, FunctionGemmaCaller, GemmaJsonReasoner, OllamaAdapter, T5GemmaNormalizer
from .ports import Embedder, FimCompleter, JsonReasoner, TextNormalizer, ToolCall, ToolCaller
from .registry import (ADAPTERS, EMBEDDER, FIM, JSON_WORKER, NORMALIZER, PLANNER, ROLE_CAPABILITY, TOOL_ROUTER,
                       VERIFIER, ModelRegistry, register_adapter)
from .transport import OllamaTransport

__all__ = [
    "OllamaTransport", "ModelRegistry", "register_adapter", "ADAPTERS", "ROLE_CAPABILITY",
    "NORMALIZER", "PLANNER", "VERIFIER", "JSON_WORKER", "FIM", "TOOL_ROUTER", "EMBEDDER",
    "JsonReasoner", "FimCompleter", "ToolCaller", "ToolCall", "TextNormalizer", "Embedder",
    "OllamaAdapter", "GemmaJsonReasoner", "CodeGemmaFim", "FunctionGemmaCaller", "T5GemmaNormalizer",
]