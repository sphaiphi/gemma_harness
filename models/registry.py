"""Binds harness roles to configured models and builds one adapter instance per model."""
from __future__ import annotations

from typing import Any, Callable, cast

from ..config import HarnessConfig, ModelSpec
from ..embeddings import GemmaEmbedder
from .adapters import CodeGemmaFim, FunctionGemmaCaller, GemmaJsonReasoner, T5GemmaNormalizer
from .ports import Embedder, FimCompleter, JsonReasoner, TextNormalizer, ToolCaller
from .transport import OllamaTransport

NORMALIZER, PLANNER, VERIFIER, JSON_WORKER = "normalizer", "planner", "verifier", "json_worker"
FIM, TOOL_ROUTER, EMBEDDER = "fim", "tool_router", "embedder"

# The capability each role needs. An adapter declares the capability it provides.
ROLE_CAPABILITY: dict[str, str] = {
    NORMALIZER: "normalize", PLANNER: "json", VERIFIER: "json", JSON_WORKER: "json",
    FIM: "fim", TOOL_ROUTER: "tool_call", EMBEDDER: "embed",
}

AdapterFactory = Callable[[ModelSpec, OllamaTransport, HarnessConfig], Any]
ADAPTERS: dict[str, tuple[str, AdapterFactory]] = {}


def register_adapter(name: str, capability: str, factory: AdapterFactory) -> None:
    """Add a model family. Roles can then bind models whose ModelSpec.adapter is `name`."""
    if capability not in set(ROLE_CAPABILITY.values()):
        raise ValueError(f"unknown capability {capability!r}")
    ADAPTERS[name] = (capability, factory)


def _embedding(spec: ModelSpec, t: OllamaTransport, cfg: HarnessConfig) -> GemmaEmbedder:
    e = cfg.embedding
    return GemmaEmbedder(spec.name, t, lane=spec.lane, engine=spec.engine, batch_size=e.batch_size,
                         max_query_chars=e.max_query_chars, max_doc_chars=e.max_doc_chars)


register_adapter("gemma_json", "json", lambda spec, t, cfg: GemmaJsonReasoner(spec, t))
register_adapter("codegemma_fim", "fim", lambda spec, t, cfg: CodeGemmaFim(spec, t))
register_adapter("functiongemma", "tool_call", lambda spec, t, cfg: FunctionGemmaCaller(spec, t))
register_adapter("t5gemma", "normalize", lambda spec, t, cfg: T5GemmaNormalizer(spec, t))
register_adapter("embeddinggemma", "embed", _embedding)


class ModelRegistry:
    def __init__(self, cfg: HarnessConfig, transport: OllamaTransport):
        self.cfg = cfg
        self.transport = transport
        self.specs = cfg.models.resolved_specs()
        self.roles = cfg.models.resolved_roles()
        self._instances: dict[str, Any] = {}
        if errs := self.validate():
            raise ValueError("invalid model configuration:\n- " + "\n- ".join(errs))

    def validate(self) -> list[str]:
        errs = [f"unknown role {r!r}" for r in self.roles if r not in ROLE_CAPABILITY]
        for role, cap in ROLE_CAPABILITY.items():
            mid = self.roles.get(role)
            spec = self.specs.get(mid) if mid else None
            if mid is None:
                errs.append(f"role {role!r} is not bound to a model")
            elif spec is None:
                errs.append(f"role {role!r} names unknown model {mid!r}")
            elif spec.adapter not in ADAPTERS:
                errs.append(f"model {mid!r} uses unknown adapter {spec.adapter!r}")
            elif ADAPTERS[spec.adapter][0] != cap:
                errs.append(f"role {role!r} needs a {cap!r} model, but {mid!r} provides "
                            f"{ADAPTERS[spec.adapter][0]!r}")
            if spec is not None and spec.lane not in self.cfg.models.lanes:
                errs.append(f"model {mid!r} uses unknown lane {spec.lane!r}")
        return errs

    def role(self, role: str) -> Any:
        mid = self.roles[role]
        if mid not in self._instances:
            spec = self.specs[mid]
            self._instances[mid] = ADAPTERS[spec.adapter][1](spec, self.transport, self.cfg)
        return self._instances[mid]

    def normalizer(self) -> TextNormalizer:
        return cast(TextNormalizer, self.role(NORMALIZER))

    def planner(self) -> JsonReasoner:
        return cast(JsonReasoner, self.role(PLANNER))

    def verifier(self) -> JsonReasoner:
        return cast(JsonReasoner, self.role(VERIFIER))

    def json_worker(self) -> JsonReasoner:
        return cast(JsonReasoner, self.role(JSON_WORKER))

    def fim(self) -> FimCompleter:
        return cast(FimCompleter, self.role(FIM))

    def tool_router(self) -> ToolCaller:
        return cast(ToolCaller, self.role(TOOL_ROUTER))

    def embedder(self) -> Embedder:
        return cast(Embedder, self.role(EMBEDDER))

    def describe(self) -> dict[str, str]:
        out = {}
        for role, mid in self.roles.items():
            spec = self.specs[mid]
            out[role] = f"{spec.engine or spec.name} ({spec.name}, lane {spec.lane})"
        return out