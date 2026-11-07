"""Ollama transport. Swap this module to target another backend; callers only use OllamaClient's methods."""
from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator, Callable

import httpx
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from pydantic import BaseModel, ValidationError

from .common import ModelOutputError
from .config import HarnessConfig


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str, host: str):
        self.model_name = model_name
        self.host = host
        self._client = httpx.Client(timeout=60.0)

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        r = self._client.post(f"{self.host}/api/embed", json={"model": self.model_name, "input": list(input)})
        r.raise_for_status()
        return r.json()["embeddings"]


class OllamaClient:
    FIM_STOP = ["<|file_separator|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.sem = asyncio.Semaphore(cfg.max_concurrent_llm)
        self._http: httpx.AsyncClient | None = None

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator["OllamaClient"]:
        async with httpx.AsyncClient(timeout=self.cfg.llm_timeout_s) as http:
            self._http = http
            try:
                yield self
            finally:
                self._http = None

    def model(self, key: str) -> str:
        return getattr(self.cfg.models, key)

    async def post(self, endpoint: str, payload: dict) -> dict:
        if self._http is None:
            raise RuntimeError("open a session first: async with client.session(): ...")
        attempts = self.cfg.http_attempts
        for attempt in range(attempts):
            try:
                async with self.sem:
                    r = await self._http.post(f"{self.cfg.host}{endpoint}", json=payload)
                if r.status_code >= 500 and attempt < attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if 400 <= r.status_code < 500:
                    # e.g. 404 model not pulled, 400 bad request: retrying or resuming will not help.
                    raise ModelOutputError(f"Ollama {r.status_code} on {endpoint}: {r.text[:300]}")
                r.raise_for_status()
                return r.json()
            except httpx.TransportError:
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    async def generate(self, model_key: str, prompt: str, **extra: Any) -> str:
        data = await self.post("/api/generate", {"model": self.model(model_key), "prompt": prompt,
                                                 "stream": False, **extra})
        return data.get("response", "")

    async def chat(self, model_key: str, messages: list[dict], **extra: Any) -> dict:
        data = await self.post("/api/chat", {"model": self.model(model_key), "messages": messages,
                                             "stream": False, **extra})
        return data.get("message") or {}

    async def chat_json(self, model_key: str, system: str, user: str, schema: type[BaseModel],
                        check: Callable[[Any], list[str]] | None = None, attempts: int = 2) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        err = "no output"
        for _ in range(attempts):
            msg = await self.chat(model_key, messages, format=schema.model_json_schema(),
                                  options={"temperature": 0.2})
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

    async def fim(self, prefix: str, suffix: str) -> str:
        return await self.generate(
            "coder", f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>", raw=True,
            options={"temperature": 0.1, "num_predict": 1024, "stop": self.FIM_STOP})