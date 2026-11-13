"""Finds the file and line span an edit subgoal is about."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..schemas import Subgoal

if TYPE_CHECKING:
    from ..index import RepoIndex
    from ..workspace import Workspace

PATH_RE = re.compile(r"[\w./-]+\.py\b")


@dataclass
class Target:
    rel: str
    path: Path
    original: str | None          # None when the file does not exist yet
    lines: list[str]
    start: int | None
    end: int | None

    @property
    def is_new(self) -> bool:
        return self.original is None

    @property
    def size(self) -> int:
        return 0 if self.is_new else self.end - self.start + 1

    @property
    def span_label(self) -> str:
        return "new file" if self.is_new else f"lines {self.start}-{self.end}"


class TargetLocator:
    def __init__(self, workspace: "Workspace", repo: "RepoIndex"):
        self.ws = workspace
        self.repo = repo

    async def locate(self, sg: Subgoal) -> Target:
        """Raises LookupError when nothing matches and PermissionError for unsafe paths."""
        rel, start, end = await self._find(sg)
        path = self.ws.safe_path(rel)
        original = self.ws.snapshot(path)
        lines = (original or "").splitlines(keepends=True)
        if original is not None and start is None:
            start, end = 1, max(1, len(lines))
        return Target(rel, path, original, lines, start, end)

    async def _find(self, sg: Subgoal) -> tuple[str, int | None, int | None]:
        for m in PATH_RE.findall(f"{sg.description}\n{sg.notes}"):
            try:
                rel = self.ws.rel(self.ws.safe_path(m))
            except PermissionError:
                continue
            if (self.ws.root / rel).is_file():
                hits = await asyncio.to_thread(self.repo.search, sg.description, 1, rel)
                return (rel, hits[0]["start"], hits[0]["end"]) if hits else (rel, None, None)
            return rel, None, None
        hits = await asyncio.to_thread(self.repo.search, sg.description, 1)
        if hits:
            return hits[0]["path"], hits[0]["start"], hits[0]["end"]
        raise LookupError("no target file named in the subgoal and nothing relevant in repo index")