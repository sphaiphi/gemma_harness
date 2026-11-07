"""Shared helpers with no harness dependencies."""
from __future__ import annotations

import contextvars
import difflib
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Iterator

OUTPUT_CAP = 4000

# Thread id of the graph run in progress. Set by the work node; asyncio.to_thread copies it into worker threads.
CURRENT_THREAD: contextvars.ContextVar[str | None] = contextvars.ContextVar("gemma_thread", default=None)


class ModelOutputError(Exception):
    """A model did not return usable output after its repair attempt."""


def elide(text: str, cap: int = OUTPUT_CAP) -> str:
    if len(text) <= cap:
        return text
    head = cap // 4
    return f"{text[:head]}\n[... {len(text) - cap} chars omitted ...]\n{text[-(cap - head):]}"


def make_diff(rel: str, before: str | None, after: str) -> str:
    diff = difflib.unified_diff((before or "").splitlines(True), after.splitlines(True), f"a/{rel}", f"b/{rel}")
    return "".join(diff) or "(no change)"


def strip_fences(code: str) -> str:
    m = re.match(r"^\s*```[\w-]*\n(.*?)\n?```\s*$", code, re.S)
    return m.group(1) if m else code


def iter_py_files(root: Path, skip_dirs: Iterable[str]) -> Iterator[Path]:
    skip = set(skip_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]