"""Mutation check: do a tool's tests fail when its body is removed?"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..workspace import TestRunner
from .checks import mutate
from .layout import RUNTIME_HELPERS, TESTS_DIR, TOOLS_DIR, test_rel, tool_rel


class MutationChecker:
    def __init__(self, root: Path, tests: TestRunner):
        self.root = root
        self.tests = tests

    def check(self, name: str) -> dict:
        """Control run on the real tool, then a run with its body removed, both in an isolated temp copy.
        Returns {"ok": bool, "why": str | None, "output": str}."""
        x_rel = test_rel(name)
        with tempfile.TemporaryDirectory(prefix="gemma-mutant-") as tmp:
            root = Path(tmp)
            (root / TOOLS_DIR).mkdir()
            (root / TESTS_DIR).mkdir()
            for helper in RUNTIME_HELPERS:
                src_helper = self.root / TOOLS_DIR / helper
                if src_helper.is_file():
                    shutil.copy(src_helper, root / TOOLS_DIR / helper)
            src = (self.root / tool_rel(name)).read_text(encoding="utf-8")
            tool_copy = root / tool_rel(name)
            tool_copy.write_text(src, encoding="utf-8")
            shutil.copy(self.root / x_rel, root / TESTS_DIR)

            control = self.tests.run([x_rel], root=root)
            if not control["passed"]:
                return {"ok": False, "output": control["output"],
                        "why": "tests do not pass in isolation (only the tool and its test file are present); "
                               "use tmp_path and monkeypatch tools._sdk.ROOT, not workspace files or conftest"}

            tool_copy.write_text(mutate(src, name), encoding="utf-8")
            mutant = self.tests.run([x_rel], root=root)
            caught = not mutant["passed"] and "NotImplementedError" in mutant["output"]
            return {"ok": caught, "output": mutant["output"],
                    "why": None if caught else
                    "tests do not detect a removed tool body (expected failures raising NotImplementedError)"}