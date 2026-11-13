"""One adapter per model family. Each owns its model's request format and keeps its capability's contract."""
from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import ValidationError

from ..common import ModelOutputError
from ..config import ModelSpec
from .ports import T, ToolCall
from .transport import OllamaTransport


class OllamaAdapter:
    def __init__(self, spec: ModelSpec, transport: OllamaTransport):
        self.spec = spec
        self.transport = transport

    @property
    def engine(self) -> str:
        return self.spec.engine or self.spec.name

    def options(self, **overrides: Any) -> dict:
        return {**self.spec.options, **overrides}

    async def _chat(self, messages: list[dict], **extra: Any) -> dict:
        data = await self.transport.post("/api/chat", {"model": self.spec.name, "messages": messages,
                                                       "stream": False, **extra}, self.spec.lane)
        return data.get("message") or {}

    async def _generate(self, prompt: str, **extra: Any) -> str:
        data = await self.transport.post("/api/generate", {"model": self.spec.name, "prompt": prompt,
                                                           "stream": False, **extra}, self.spec.lane)
        return data.get("response") or ""


class GemmaJsonReasoner(OllamaAdapter):
    """Structured output with one repair round: the model sees why its JSON was rejected."""

    def __init__(self, spec: ModelSpec, transport: OllamaTransport, attempts: int = 2):
        super().__init__(spec, transport)
        self.attempts = attempts

    async def ask(self, system: str, user: str, schema: type[T],
                  check: Callable[[T], list[str]] | None = None) -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        err = "no output"
        for _ in range(self.attempts):
            msg = await self._chat(messages, format=schema.model_json_schema(), options=self.options())
            content = msg.get("content", "")
            try:
                obj = schema.model_validate_json(content)
                errs = check(obj) if check else []
                if not errs:
                    return obj
                err = "; ".join(errs)
            except ValidationError as e:
                err = str(e)[:1500]
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"Your JSON was rejected: {err}\nReturn corrected JSON only."}]
        raise ModelOutputError(err)


class CodeGemmaFim(OllamaAdapter):
    TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|file_separator|>")

    async def complete(self, prefix: str, suffix: str) -> str:
        out = await self._generate(f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>", raw=True,
                                   options=self.options(stop=list(self.TOKENS)))
        for tok in self.TOKENS:  # cut anything after a leaked special token
            out = out.split(tok)[0]
        if not out.strip():
            raise ModelOutputError(f"{self.engine} returned an empty completion")
        return out


class FunctionGemmaCaller(OllamaAdapter):
    """Native Ollama tool calling. `check` validates the call (e.g. against the tool registry)."""

    def __init__(self, spec: ModelSpec, transport: OllamaTransport, attempts: int = 2):
        super().__init__(spec, transport)
        self.attempts = attempts

    async def call(self, system: str, user: str, tools: list[dict],
                   check: Callable[[ToolCall], str | None] | None = None) -> ToolCall:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        err: str | None = "no tool call"
        for _ in range(self.attempts):
            msg = await self._chat(messages, tools=tools, options=self.options())
            call, err = self._parse(msg)
            if call is not None and check is not None:
                err = check(call)
            if call is not None and err is None:
                return call
            messages += [{"role": "assistant", "content": msg.get("content", "")},
                         {"role": "user", "content": f"Not a valid tool call ({err}). "
                                                     "Call one tool with valid arguments."}]
        raise ModelOutputError(err or "no valid tool call")

    @staticmethod
    def _parse(msg: dict) -> tuple[ToolCall | None, str | None]:
        calls = msg.get("tool_calls") or []
        if not calls:
            return None, "no tool call"
        fn = calls[0].get("function") or {}
        name, args = fn.get("name"), fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if not isinstance(name, str) or not name:
            return None, "tool call has no name"
        if not isinstance(args, dict):
            return None, "arguments are not a JSON object"
        return ToolCall(name, args), None


class T5GemmaNormalizer(OllamaAdapter):
    PROMPT = "Normalize task description:\n\n{text}"

    async def normalize(self, text: str) -> str:
        out = (await self._generate(self.PROMPT.format(text=text), options=self.options())).strip()
        if not out:
            raise ModelOutputError(f"{self.engine} returned empty text")
        return out