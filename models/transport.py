"""HTTP transport for every model call: retries, error mapping and concurrency lanes. Knows nothing about models."""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import TYPE_CHECKING, AsyncIterator, Mapping, TypeVar

import httpx

from ..common import ModelOutputError, ModelUnavailable

if TYPE_CHECKING:
    from ..config import HarnessConfig

L = TypeVar("L")


class OllamaTransport:
    def __init__(self, host: str, lanes: Mapping[str, int], *, timeout_s: float = 180.0, attempts: int = 3,
                 sync_timeout_s: float = 60.0):
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        if bad := sorted(k for k, n in lanes.items() if n < 1):
            raise ValueError(f"lane sizes must be at least 1: {bad}")
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.attempts = attempts
        self._async_lanes = {k: asyncio.Semaphore(n) for k, n in lanes.items()}
        self._sync_lanes = {k: threading.BoundedSemaphore(n) for k, n in lanes.items()}
        self._http: httpx.AsyncClient | None = None
        self._sync = httpx.Client(timeout=sync_timeout_s)  # embeddings run in worker threads

    @classmethod
    def from_config(cls, cfg: "HarnessConfig") -> "OllamaTransport":
        return cls(cfg.host, cfg.models.lanes, timeout_s=cfg.llm_timeout_s, attempts=cfg.http_attempts,
                   sync_timeout_s=cfg.embedding.timeout_s)

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator["OllamaTransport"]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as http:
            self._http = http
            try:
                yield self
            finally:
                self._http = None

    def close(self) -> None:
        self._sync.close()

    @staticmethod
    def _lane(lanes: Mapping[str, L], lane: str) -> L:
        try:
            return lanes[lane]
        except KeyError:
            raise ValueError(f"unknown concurrency lane {lane!r}; configure it in ModelConfig.lanes") from None

    @staticmethod
    def _accept(r: httpx.Response, endpoint: str) -> dict | None:
        """Return the JSON body, None to retry (5xx), or raise for an answer that retrying will not fix."""
        if r.status_code >= 500:
            return None
        if r.status_code >= 400:
            # e.g. 404 model not pulled, 400 bad request.
            raise ModelOutputError(f"Ollama {r.status_code} on {endpoint}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError as e:
            raise ModelOutputError(f"Ollama returned invalid JSON on {endpoint}") from e

    async def post(self, endpoint: str, payload: dict, lane: str) -> dict:
        if self._http is None:
            raise RuntimeError("open a session first: async with transport.session(): ...")
        sem = self._lane(self._async_lanes, lane)
        last = "no attempt"
        for attempt in range(self.attempts):
            try:
                async with sem:
                    r = await self._http.post(f"{self.host}{endpoint}", json=payload)
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if (data := self._accept(r, endpoint)) is not None:
                    return data
                last = f"HTTP {r.status_code}: {r.text[:300]}"
            if attempt < self.attempts - 1:
                await asyncio.sleep(2 ** attempt)
        raise ModelUnavailable(f"Ollama {endpoint} failed after {self.attempts} attempts: {last}")

    def post_sync(self, endpoint: str, payload: dict, lane: str) -> dict:
        sem = self._lane(self._sync_lanes, lane)
        last = "no attempt"
        for attempt in range(self.attempts):
            try:
                with sem:
                    r = self._sync.post(f"{self.host}{endpoint}", json=payload)
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if (data := self._accept(r, endpoint)) is not None:
                    return data
                last = f"HTTP {r.status_code}: {r.text[:300]}"
            if attempt < self.attempts - 1:
                time.sleep(2 ** attempt)
        raise ModelUnavailable(f"Ollama {endpoint} failed after {self.attempts} attempts: {last}")