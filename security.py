"""AST linters. They filter obvious mistakes; they are not a sandbox."""
from __future__ import annotations

import ast
from typing import Literal

Policy = Literal["strict", "project"]


class SecurityLinter(ast.NodeVisitor):
    STRICT_ALLOWED = {"pathlib", "json", "re", "typing", "collections", "datetime", "math"}
    PROJECT_DENIED = {"subprocess", "socket", "ctypes", "pty"}
    FORBIDDEN_FUNCS = {"eval", "exec", "compile", "globals", "locals", "__import__", "breakpoint"}
    FORBIDDEN_ATTRS = {"system", "popen", "rmtree"}

    def __init__(self, policy: Policy = "project"):
        self.policy = policy
        self.errors: list[str] = []

    def analyze(self, code: str) -> list[str]:
        try:
            self.visit(ast.parse(code))
        except SyntaxError as e:
            return [f"SyntaxError: {e}"]
        return self.errors

    def _module(self, name: str) -> None:
        root = name.split(".")[0]
        if (self.policy == "strict" and root not in self.STRICT_ALLOWED) or \
                (self.policy == "project" and root in self.PROJECT_DENIED):
            self.errors.append(f"unauthorized import: {name!r}")

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self._module(a.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._module(node.module)
        for a in node.names:
            if a.name in self.FORBIDDEN_ATTRS | self.FORBIDDEN_FUNCS:
                self.errors.append(f"forbidden import: {a.name!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name) and f.id in self.FORBIDDEN_FUNCS:
            self.errors.append(f"forbidden call: {f.id}()")
        elif isinstance(f, ast.Attribute) and f.attr in self.FORBIDDEN_ATTRS:
            self.errors.append(f"forbidden call: .{f.attr}()")
        self.generic_visit(node)


def lint(code: str, policy: Policy = "project") -> list[str]:
    return SecurityLinter(policy).analyze(code)


class ToolLinter(SecurityLinter):
    # `string` is deliberately absent: string.Formatter().get_field() reaches module globals as objects.
    ALLOWED = {
        "json", "re", "math", "typing", "collections", "datetime", "itertools",
        "functools", "statistics", "textwrap", "difflib", "fnmatch",
        "dataclasses", "decimal", "fractions", "csv", "hashlib", "base64",
    }
    SDK = "tools._sdk"
    SDK_NAMES = {"ws_read", "ws_write", "ws_list"}
    FORBIDDEN_FUNCS = SecurityLinter.FORBIDDEN_FUNCS | {"open", "getattr", "setattr", "delattr", "vars", "input", "dir"}
    FORBIDDEN_ATTRS = SecurityLinter.FORBIDDEN_ATTRS | {"format_map", "get_field", "vformat"}

    def __init__(self):
        super().__init__("strict")

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name == self.SDK:
                self.errors.append("use `from tools._sdk import ws_read, ws_write, ws_list`")
            elif a.name.split(".")[0] not in self.ALLOWED:
                self.errors.append(f"unauthorized import: {a.name!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        is_sdk = (node.level == 0 and node.module == self.SDK) or (node.level == 1 and node.module == "_sdk")
        if is_sdk:
            if bad := [a.name for a in node.names if a.name not in self.SDK_NAMES]:
                self.errors.append(f"tools._sdk exposes only {sorted(self.SDK_NAMES)}, not {bad}")
        elif node.level:
            self.errors.append("relative imports are not allowed")
        elif (node.module or "").split(".")[0] not in self.ALLOWED:
            self.errors.append(f"unauthorized import from: {node.module!r}")
        for a in node.names:
            if a.name in self.FORBIDDEN_ATTRS | self.FORBIDDEN_FUNCS or a.name.startswith("__"):
                self.errors.append(f"forbidden import: {a.name!r}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self.errors.append(f"dunder names are not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.errors.append(f"dunder attributes are not allowed: .{node.attr}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        v = node.value
        if (isinstance(v, str) and "__" in v) or (isinstance(v, bytes) and b"__" in v):
            self.errors.append("string literals containing '__' are not allowed")