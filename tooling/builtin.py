"""Built-in tools. To add one, register it here and add its name to BUILTIN_TOOLS."""
from __future__ import annotations

import json

from ..security import lint
from ..workspace import TestRunner, Workspace
from .registry import ToolRegistry

BUILTIN_TOOLS = frozenset({"read_file", "write_file", "run_pytest"})


def register_builtin_tools(registry: ToolRegistry, ws: Workspace, tests: TestRunner) -> None:
    def read_file(path: str) -> str:
        p = ws.safe_path(path)
        return p.read_text(encoding="utf-8") if p.is_file() else f"Error: file {path} not found."

    def write_file(path: str, content: str) -> str:
        p = ws.safe_path(path)
        if p.suffix == ".py" and (errs := lint(content)):
            return "Error: blocked by security linter: " + "; ".join(errs)
        ws.write(p, content)
        return f"Wrote {len(content)} chars to {path}"

    def run_pytest() -> str:
        return json.dumps(tests.run())

    registry.register("read_file", "Read a workspace file.",
                      {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                      read_file)
    registry.register("write_file", "Write content to a workspace file.",
                      {"type": "object",
                       "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}, write_file)
    registry.register("run_pytest", "Run workspace pytest suite.", {"type": "object", "properties": {}}, run_pytest)