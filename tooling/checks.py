"""Static checks for synthesized tools and their tests (no execution)."""
from __future__ import annotations

import ast
import pprint
import re

import jsonschema

from ..common import strip_fences
from ..schemas import ToolDraft
from ..security import ToolLinter, lint
from .builtin import BUILTIN_TOOLS

NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,40}")


def name_errors(name: str) -> list[str]:
    if not NAME_RE.fullmatch(name or ""):
        return [f"tool name {name!r} must match {NAME_RE.pattern}"]
    if name in BUILTIN_TOOLS:
        return [f"{name!r} is a built-in tool; choose another name"]
    return []


def check_signature(tree: ast.Module, name: str, params: dict) -> list[str]:
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
              None)
    if fn is None:
        return [f"no top-level function named {name}"]
    if isinstance(fn, ast.AsyncFunctionDef):
        return [f"{name} must be a plain def, not async"]
    a, errs = fn.args, []
    if a.vararg or a.kwarg or a.posonlyargs:
        errs.append(f"{name} may not use *args, **kwargs or positional-only parameters")
    args = [x.arg for x in a.args + a.kwonlyargs]
    props = set(params.get("properties") or {})
    if set(args) != props:
        errs.append(f"function parameters {sorted(args)} must match schema properties {sorted(props)}")
    defaulted = {x.arg for x in a.args[len(a.args) - len(a.defaults):]} | \
                {x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None}
    required = set(params.get("required") or [])
    if missing := sorted(set(args) - defaulted - required):
        errs.append(f"parameters without defaults must be listed in required: {missing}")
    if unknown := sorted(required - props):
        errs.append(f"required names unknown properties: {unknown}")
    return errs


def check_tool(name: str, spec: dict, src: str) -> list[str]:
    if errs := name_errors(name):
        return errs
    errs = []
    if spec.get("name") != name:
        errs.append("TOOL_SPEC name must equal the file and function name")
    if not isinstance(spec.get("description"), str) or not spec["description"].strip():
        errs.append("description is empty")
    params = spec.get("parameters")
    if not isinstance(params, dict) or params.get("type") != "object" \
            or not isinstance(params.get("properties", {}), dict):
        errs.append('parameters must be a JSON Schema with "type": "object" and a "properties" object')
        params = None
    else:
        try:
            jsonschema.Draft202012Validator.check_schema(params)
        except jsonschema.SchemaError as e:
            errs.append(f"invalid parameters schema: {e.message}")
    lint_errs = ToolLinter().analyze(src)
    errs += lint_errs
    if params is not None and not any(e.startswith("SyntaxError") for e in lint_errs):
        errs += check_signature(ast.parse(src), name, params)
    return errs


def _imports_sdk(tree: ast.AST) -> bool:
    return any(isinstance(n, ast.ImportFrom) and ((n.level == 0 and n.module == ToolLinter.SDK)
                                                  or (n.level == 1 and n.module == "_sdk"))
               for n in ast.walk(tree))


def _patches_root(tree: ast.AST) -> bool:
    """True if the tests call monkeypatch.setattr("tools._sdk.ROOT", ...) or setattr(<sdk>, "ROOT", ...)."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "setattr" and n.args:
            a = n.args
            if isinstance(a[0], ast.Constant) and a[0].value == f"{ToolLinter.SDK}.ROOT":
                return True
            if len(a) >= 2 and isinstance(a[1], ast.Constant) and a[1].value == "ROOT":
                return True
    return False


def check_tests(name: str, src: str, tool_src: str | None = None) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"tests: SyntaxError: {e}"]
    errs = [f"tests: {e}" for e in lint(src)]
    n_tests = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
                  for n in ast.walk(tree))
    if n_tests < 2:
        errs.append("tests: write at least two test_ functions, including an edge case")
    if not any(isinstance(n, ast.ImportFrom) and n.module == f"tools.{name}" and any(a.name == name for a in n.names)
               for n in ast.walk(tree)):
        errs.append(f"tests must import the tool with `from tools.{name} import {name}`")
    touches_ws = _imports_sdk(tree) or any(isinstance(n, ast.Name) and n.id in ToolLinter.SDK_NAMES
                                           for n in ast.walk(tree))
    if tool_src:
        try:
            touches_ws = touches_ws or _imports_sdk(ast.parse(tool_src))
        except SyntaxError:
            pass
    if touches_ws and not _patches_root(tree):
        errs.append('tests: the tool touches the workspace; take the tmp_path and monkeypatch fixtures and call '
                    'monkeypatch.setattr("tools._sdk.ROOT", tmp_path) before using it')
    return errs


def read_tool_spec(src: str) -> dict:
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TOOL_SPEC" for t in n.targets):
            spec = ast.literal_eval(n.value)
            if isinstance(spec, dict):
                return spec
    raise ValueError("no literal TOOL_SPEC dict")


def mutate(src: str, name: str) -> str:
    """Replace the tool function's body with `raise NotImplementedError`."""
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            n.body = [ast.parse('raise NotImplementedError("mutation check")').body[0]]
    return ast.unparse(tree)


def spec_of(d: ToolDraft) -> dict:
    return {"name": d.name, "description": " ".join(d.description.split()), "parameters": d.parameters}


def compose_tool_source(d: ToolDraft) -> str:
    return (f'"""Synthesized tool `{d.name}`. Generated by the Gemma harness."""\n\n'
            f"TOOL_SPEC = {pprint.pformat(spec_of(d), sort_dicts=False, width=100)}\n\n\n"
            f"{strip_fences(d.code).strip()}\n")


def draft_errors(d: ToolDraft) -> list[str]:
    if errs := name_errors(d.name):
        return errs
    errs = ["do not define TOOL_SPEC; the harness adds it"] if "TOOL_SPEC" in d.code else []
    errs += check_tool(d.name, spec_of(d), compose_tool_source(d))
    errs += check_tests(d.name, strip_fences(d.tests), strip_fences(d.code))
    return errs