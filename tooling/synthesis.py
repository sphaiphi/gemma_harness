"""Lifecycle of synthesized tools: runtime install, registration, verified store, auto-load, mutation check."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from ..common import digest, elide
from ..config import HarnessConfig
from ..workspace import Journal, TestRunner
from .checks import check_tool, mutate, read_tool_spec
from .registry import ToolRegistry
from .runtime import install_runtime, runner_handler

log = logging.getLogger(__name__)


class ToolSynthesis:
    def __init__(self, root: Path, cfg: HarnessConfig, registry: ToolRegistry, tests: TestRunner, journal: Journal):
        self.root = root
        self.cfg = cfg
        self.registry = registry
        self.tests = tests
        self.journal = journal

    @property
    def tools_dir(self) -> Path:
        return self.root / "tools"

    @staticmethod
    def test_rel(name: str) -> str:
        return f"tests/test_tool_{name}.py"

    def install_runtime(self) -> None:
        install_runtime(self.root)

    def register(self, spec: dict) -> None:
        handler = runner_handler(spec["name"], self.root, self.tests, self.journal, self.cfg.synth_tool_timeout_s)
        self.registry.register(spec["name"], spec["description"], spec["parameters"], handler)

    # ------------------------------------------------------------- verified store
    def load_verified(self) -> dict[str, str]:
        p = self.root / self.cfg.verified_file
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        except (OSError, ValueError):
            return {}

    def save_verified(self, data: dict[str, str]) -> None:
        p = self.root / self.cfg.verified_file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def mark_verified(self, name: str, dg: str) -> None:
        self.save_verified({**self.load_verified(), name: dg})

    # ------------------------------------------------------------- auto-loader
    def autoload(self) -> list[str]:
        """Register every valid tool on disk whose tests pass. Returns the loaded names."""
        verified, loaded = self.load_verified(), {}
        for f in sorted(self.tools_dir.glob("*.py")):
            name = f.stem
            if name.startswith("_"):
                continue
            test_f = self.root / self.test_rel(name)
            try:
                src = f.read_text(encoding="utf-8")
                spec = read_tool_spec(src)
                errs = check_tool(name, spec, src)
            except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as e:
                errs = [f"{type(e).__name__}: {e}"]
            if not errs and not test_f.is_file():
                errs = [f"missing {self.test_rel(name)}"]
            if errs:
                log.warning("[tools] skipped %s: %s", name, elide("; ".join(errs), 400))
                continue
            dg = digest(src, test_f.read_text(encoding="utf-8"))
            note = ""
            if verified.get(name) != dg:
                t = self.tests.run([self.test_rel(name)])
                if not t["passed"]:
                    log.warning("[tools] skipped %s: its tests fail\n%s", name, elide(t["output"], 800))
                    continue
                note = " (re-verified: source or tests changed)"
            loaded[name] = dg
            self.register(spec)
            log.info("[tools] auto-loaded %s%s", name, note)
        self.save_verified(loaded)
        self.registry.prune()
        return list(loaded)

    # ------------------------------------------------------------- mutation check
    def mutation_check(self, name: str) -> dict:
        """Control run on the real tool, then a run with its body removed, both in an isolated temp copy."""
        test_rel = self.test_rel(name)
        with tempfile.TemporaryDirectory(prefix="gemma-mutant-") as tmp:
            root = Path(tmp)
            (root / "tools").mkdir()
            (root / "tests").mkdir()
            for helper in ("__init__.py", "_sdk.py", "_runner.py"):
                if (self.tools_dir / helper).is_file():
                    shutil.copy(self.tools_dir / helper, root / "tools" / helper)
            src = (self.tools_dir / f"{name}.py").read_text(encoding="utf-8")
            tool_copy = root / "tools" / f"{name}.py"
            tool_copy.write_text(src, encoding="utf-8")
            shutil.copy(self.root / test_rel, root / "tests")

            control = self.tests.run([test_rel], root=root)
            if not control["passed"]:
                return {"ok": False, "output": control["output"],
                        "why": "tests do not pass in isolation (only the tool and its test file are present); "
                               "use tmp_path and monkeypatch tools._sdk.ROOT, not workspace files or conftest"}

            tool_copy.write_text(mutate(src, name), encoding="utf-8")
            mutant = self.tests.run([test_rel], root=root)
            caught = not mutant["passed"] and "NotImplementedError" in mutant["output"]
            return {"ok": caught, "output": mutant["output"],
                    "why": None if caught else
                    "tests do not detect a removed tool body (expected failures raising NotImplementedError)"}