"""Startup auto-loader: registers every valid synthesized tool on disk whose tests pass."""
from __future__ import annotations

import logging
from pathlib import Path

from ..common import digest, elide
from ..workspace import TestRunner
from .checks import check_tool, read_tool_spec, tool_scope
from .layout import TOOLS_DIR, test_rel
from .registry import ToolRegistry
from .runtime import ToolRuntime
from .verified import VerifiedStore

log = logging.getLogger(__name__)


class ToolAutoloader:
    def __init__(self, root: Path, registry: ToolRegistry, runtime: ToolRuntime, store: VerifiedStore,
                 tests: TestRunner):
        self.root = root
        self.registry = registry
        self.runtime = runtime
        self.store = store
        self.tests = tests

    def autoload(self) -> list[str]:
        """Returns the names it loaded. Tools whose digest is already verified are not re-tested."""
        verified, loaded = self.store.load(), {}
        for f in sorted((self.root / TOOLS_DIR).glob("*.py")):
            name = f.stem
            if name.startswith("_"):
                continue
            test_f = self.root / test_rel(name)
            try:
                src = f.read_text(encoding="utf-8")
                spec = read_tool_spec(src)
                errs = check_tool(name, spec, src)
            except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as e:
                errs = [f"{type(e).__name__}: {e}"]
            if not errs and not test_f.is_file():
                errs = [f"missing {test_rel(name)}"]
            if errs:
                log.warning("[tools] skipped %s: %s", name, elide("; ".join(errs), 400))
                continue
            dg = digest(src, test_f.read_text(encoding="utf-8"))
            note = ""
            if verified.get(name) != dg:
                t = self.tests.run([test_rel(name)])
                if not t["passed"]:
                    log.warning("[tools] skipped %s: its tests fail\n%s", name, elide(t["output"], 800))
                    continue
                note = " (re-verified: source or tests changed)"
            loaded[name] = dg
            self.runtime.register(spec, tool_scope(src))
            log.info("[tools] auto-loaded %s%s", name, note)
        self.store.save(loaded)
        self.registry.prune()
        return list(loaded)
