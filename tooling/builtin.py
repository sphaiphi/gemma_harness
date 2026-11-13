"""Built-in tools. To add one, register it here with a scope and add its name to BUILTIN_TOOLS."""
from __future__ import annotations

from ..security import lint
from ..workspace import TestRunner, Workspace
from .registry import ToolRegistry, ToolResult

BUILTIN_TOOLS = frozenset({"read_file", "write_file", "run_pytest"})


def register_builtin_tools(registry: ToolRegistry, ws: Workspace, tests: TestRunner) -> None:
    def read_file(path: str) -> ToolResult:
        p = ws.safe_path(path)
        if not p.is_file():
            return ToolResult.failure(f"file {path} not found.")
        return ToolResult.success(p.read_text(encoding="utf-8"))

    def write_file(path: str, content: str) -> ToolResult:
        p = ws.safe_path(path)
        if p.suffix == ".py" and (errs := lint(content)):
            return ToolResult.failure("blocked by security linter: " + "; ".join(errs))
        ws.write(p, content)
        return ToolResult.success(f"Wrote {len(content)} chars to {path}")

    def run_pytest() -> ToolResult:
        return ToolResult.success(tests.run())  # the call succeeded; pass/fail is in the JSON

    registry.register("read_file", "Read a workspace file.",
                      {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                      read_file, scope="read")
    registry.register("write_file", "Write content to a workspace file.",
                      {"type": "object",
                       "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}, write_file, scope="write")
    registry.register("run_pytest", "Run workspace pytest suite.", {"type": "object", "properties": {}},
                      run_pytest, scope="exec")