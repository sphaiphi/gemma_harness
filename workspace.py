"""Workspace file access, the crash journal and the pytest runner."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .common import CURRENT_THREAD, elide
from .config import HarnessConfig

Listener = Callable[[str], None]


def is_managed(rel: str) -> bool:
    """Files owned by tool synthesis; ordinary edits may not touch them."""
    return rel.startswith("tools/") or re.fullmatch(r"tests/test_tool_\w+\.py", rel) is not None


def files_of(meta: dict) -> dict[str, str | None]:
    """Normalize worker meta into {relative path: original content or None for a new file}."""
    if meta.get("files"):
        return dict(meta["files"])
    if meta.get("path"):
        return {meta["path"]: meta.get("original")}
    return {}


def changed_paths(meta: dict) -> list[str]:
    return list(files_of(meta))


class Journal:
    """Per-thread record of original file contents, so a crash mid-step can be rolled back on restart."""

    def __init__(self, root: Path, dirname: str):
        self.dir = root / dirname

    def paths(self, tid: str) -> tuple[Path, Path]:
        self.dir.mkdir(exist_ok=True)
        safe = re.sub(r"[^\w.-]", "_", tid)
        return self.dir / f"{safe}.json", self.dir / f"{safe}.sdk.jsonl"

    def sdk_log(self, tid: str) -> Path:
        return self.paths(tid)[1]

    def add(self, tid: str, rel: str, original: str | None) -> None:
        p, _ = self.paths(tid)
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        if rel in data:
            return
        data[rel] = original
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(p)

    def read_sdk_log(self, tid: str) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        s = self.sdk_log(tid)
        if s.is_file():
            for ln in s.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                out.setdefault(e["path"], e.get("original"))
        return out

    def load(self, tid: str) -> dict[str, str | None]:
        p, _ = self.paths(tid)
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        for rel, original in self.read_sdk_log(tid).items():
            data.setdefault(rel, original)
        return data

    def clear(self, tid: str) -> None:
        for p in self.paths(tid):
            p.unlink(missing_ok=True)


class Workspace:
    """Path-safe reads and writes. Subscribers are told about every changed file (e.g. the repo index)."""

    def __init__(self, root: Path, journal: Journal):
        self.root = root
        self.journal = journal
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _notify(self, rel: str) -> None:
        for fn in self._listeners:
            fn(rel)

    def safe_path(self, path: str) -> Path:
        target = (self.root / path.strip().lstrip("/")).resolve()
        if not target.is_relative_to(self.root):
            raise PermissionError(f"path escapes the workspace: {path}")
        if any(part.startswith(".") for part in target.relative_to(self.root).parts):
            raise PermissionError(f"hidden paths are off limits: {path}")
        return target

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    @staticmethod
    def snapshot(p: Path) -> str | None:
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def write(self, path: Path, text: str, *, managed: bool = False) -> None:
        rel = self.rel(path)
        if not managed and is_managed(rel):
            raise PermissionError(f"{rel} is managed by tool synthesis; use a synth subgoal to change tools")
        if tid := CURRENT_THREAD.get():
            self.journal.add(tid, rel, self.snapshot(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self._notify(rel)

    def restore(self, files: dict[str, str | None]) -> None:
        for rel, original in files.items():
            try:
                p = self.safe_path(rel)
            except PermissionError:
                continue
            if original is None:
                p.unlink(missing_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(original, encoding="utf-8")
            self._notify(rel)


class TestRunner:
    __test__ = False  # keep pytest from collecting this class

    def __init__(self, root: Path, cfg: HarnessConfig):
        self.root = root
        self.cfg = cfg

    def clean_env(self, root: Path | None = None) -> dict[str, str]:
        env = {k: os.environ[k] for k in self.cfg.test_env_passthrough if k in os.environ}
        env.update(PYTHONPATH=str(root or self.root), PYTHONDONTWRITEBYTECODE="1")
        return env

    def run(self, targets: list[str] | None = None, root: Path | None = None) -> dict:
        root = root or self.root
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *(targets or [])]
        try:
            p = subprocess.run(cmd, cwd=root, env=self.clean_env(root), capture_output=True, text=True,
                               timeout=self.cfg.test_timeout_s)
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "passed": False, "no_tests": False, "output": "pytest timed out"}
        return {"exit_code": p.returncode, "passed": p.returncode == 0, "no_tests": p.returncode == 5,
                "output": elide(p.stdout + p.stderr)}