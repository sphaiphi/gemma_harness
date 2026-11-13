"""Digests of synthesized tools whose tests passed, so unchanged tools skip re-testing at startup."""
from __future__ import annotations

import json
from pathlib import Path


class VerifiedStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, str]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        except (OSError, ValueError):
            return {}

    def save(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def mark(self, name: str, dg: str) -> None:
        self.save({**self.load(), name: dg})