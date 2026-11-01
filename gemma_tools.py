"""
Gemma Trinity Harness, part 2: git commits per step, crash journal, tool synthesis and the tools/ auto-loader.
Builds on gemma_trinity_harness.py (apply its two small edits first: `synth` kind + Thinker prompt lines).

Git
  - Start: `git init` if needed, harness entries added to .gitignore, one baseline commit of uncommitted work.
  - Every ACCEPTED subgoal that changed files gets its own commit, with only its own paths and trailers.
    Rejected changes are rolled back and never committed.
  - Crash journal: originals are recorded per thread before any write. On restart, writes the checkpoint does not
    know about are rolled back before the thread resumes.

Tool synthesis (kind "synth")
  - Gemma 3 12B drafts name, description, JSON Schema, code and tests in one typed response.
  - Static checks before writing: name, schema, strict tool linter, signature == schema, test shape.
  - Verifier tier 1: the tool's tests pass AND fail on a mutant whose body is `raise NotImplementedError`.
    Tier 2: Gemma 3 12B judges fit. Accept -> registered, marked verified, committed. Reject -> restored.
  - Tools run in a child process (clean env, timeout). File access only via tools/_sdk.py.

Auto-loader
  - Start: each tools/<name>.py with a literal TOOL_SPEC is re-checked and loaded; tests re-run unless unchanged
    since they last passed. Stale ChromaDB entries are removed.
"""
from __future__ import annotations

import ast
import asyncio
import contextvars
import hashlib
import json
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import httpx
import jsonschema
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from gemma_trinity_harness import (
    LLM_TIMEOUT_S, TEST_ENV_PASSTHROUGH, TEST_TIMEOUT_S,
    GemmaTrinityHarness, ModelOutputError, Plan, SecurityLinter, Subgoal, Verdict, WorkerOutput,
    elide, lint, make_diff, strip_fences,
)

# =============================================================================
# Configuration
# =============================================================================

GIT_ENABLED = True
GIT_BASELINE_COMMIT = True          # commit uncommitted work before a run, so step commits stay clean
GIT_AUTHOR = ("Gemma Harness", "gemma-harness@localhost")
GITIGNORE_LINES = [".gemma_vector_db/", ".gemma_checkpoints.db*", ".gemma_journal/", ".gemma_tools/",
                   "__pycache__/", "*.pyc", ".pytest_cache/"]
JOURNAL_DIR = ".gemma_journal"
VERIFIED_FILE = ".gemma_tools/verified.json"
SYNTH_TOOL_TIMEOUT_S = 60
BUILTIN_TOOLS = {"read_file", "write_file", "run_pytest"}
NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,40}")

CURRENT_THREAD: contextvars.ContextVar[str | None] = contextvars.ContextVar("gemma_thread", default=None)

# =============================================================================
# Files the harness writes into tools/
# =============================================================================

INIT_SOURCE = '"""Synthesized tools. Managed by the Gemma harness."""\n'

SDK_SOURCE = '''"""Workspace access for synthesized tools. Written by the Gemma harness; do not edit."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_LIST = 1000


def _resolve(path):
    root = Path(ROOT).resolve()
    target = (root / str(path).strip().lstrip("/")).resolve()
    if not target.is_relative_to(root):
        raise PermissionError(f"path escapes the workspace: {path}")
    rel = target.relative_to(root)
    if any(part.startswith(".") for part in rel.parts):
        raise PermissionError(f"hidden paths are off limits: {path}")
    return target, rel.as_posix()


def ws_read(path):
    """Read a UTF-8 text file in the workspace."""
    return _resolve(path)[0].read_text(encoding="utf-8")


def ws_write(path, content):
    """Write a text file in the workspace. Python files, tools/ and tests/ are off limits."""
    target, rel = _resolve(path)
    if rel.endswith(".py") or rel.split("/")[0] in ("tools", "tests"):
        raise PermissionError(f"tools may not write code or tests: {path}")
    log = os.environ.get("GEMMA_WRITE_LOG")
    if log:  # crash journal + commit list for the harness, written BEFORE the change
        original = target.read_text(encoding="utf-8") if target.is_file() else None
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"path": rel, "original": original}) + "\\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8")
    return f"wrote {len(str(content))} chars to {rel}"


def ws_list(pattern="**/*"):
    """List workspace files matching a glob pattern; hidden paths excluded."""
    root = Path(ROOT).resolve()
    out = []
    for p in sorted(root.glob(pattern)):
        try:
            rel = p.resolve().relative_to(root)
        except ValueError:
            continue
        if p.is_file() and not any(part.startswith(".") for part in rel.parts):
            out.append(rel.as_posix())
            if len(out) >= MAX_LIST:
                break
    return out
'''

RUNNER_SOURCE = '''"""Runs one synthesized tool in a child process. Written by the Gemma harness; do not edit."""
import contextlib
import importlib
import io
import json
import sys


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if not name.isidentifier() or name.startswith("_"):
        print(json.dumps({"ok": False, "error": f"bad tool name {name!r}"}))
        return
    args = json.loads(sys.stdin.read() or "{}")
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # stray prints must not corrupt the result line
            result = getattr(importlib.import_module(f"tools.{name}"), name)(**args)
        payload = {"ok": True, "result": result}
    except Exception as e:
        payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(payload, default=str))


if __name__ == "__main__":
    main()
'''

# =============================================================================
# Static checks for synthesized tools
# =============================================================================

class ToolLinter(SecurityLinter):
    """Stricter than the project linter: allowlisted imports, no file/OS access except through tools._sdk,
    no reflection (getattr, dunders). Not a sandbox; the child process and timeout are the second layer."""
    ALLOWED = {"json", "re", "math", "typing", "collections", "datetime", "itertools", "functools", "statistics",
               "string", "textwrap", "difflib", "fnmatch", "dataclasses", "decimal", "fractions", "csv",
               "hashlib", "base64"}
    SDK = "tools._sdk"
    SDK_NAMES = {"ws_read", "ws_write", "ws_list"}
    FORBIDDEN_FUNCS = SecurityLinter.FORBIDDEN_FUNCS | {"open", "getattr", "setattr", "delattr", "vars",
                                                        "input", "dir"}

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


def name_errors(name: str) -> list[str]:
    if not NAME_RE.fullmatch(name or ""):
        return [f"tool name {name!r} must match {NAME_RE.pattern}"]
    if name in BUILTIN_TOOLS:
        return [f"{name!r} is a built-in tool; choose another name"]
    return []


def check_signature(tree: ast.Module, name: str, params: dict) -> list[str]:
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
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


def check_tests(name: str, src: str) -> list[str]:
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
    return errs


def read_tool_spec(src: str) -> dict:
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TOOL_SPEC" for t in n.targets):
            spec = ast.literal_eval(n.value)
            if isinstance(spec, dict):
                return spec
    raise ValueError("no literal TOOL_SPEC dict")


def mutate(src: str, name: str) -> str:
    """The tool with its body replaced by `raise NotImplementedError`: good tests must fail against it."""
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            n.body = [ast.parse('raise NotImplementedError("mutation check")').body[0]]
    return ast.unparse(tree)


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def is_managed(rel: str) -> bool:
    return rel.startswith("tools/") or re.fullmatch(r"tests/test_tool_\w+\.py", rel) is not None


def changed_paths(meta: dict) -> list[str]:
    if meta.get("files"):
        return list(meta["files"])
    return [meta["path"]] if meta.get("path") else []


# =============================================================================
# Synthesis schema + prompt
# =============================================================================

class ToolDraft(BaseModel):
    name: str = Field(description="snake_case tool name; also the function name")
    description: str = Field(description="One or two sentences: what the tool does, its inputs and its output.")
    parameters: dict[str, Any] = Field(description='JSON Schema with "type": "object" for the arguments')
    code: str = Field(description="Python module defining the function. No TOOL_SPEC.")
    tests: str = Field(description="pytest module for the tool")


def spec_of(d: ToolDraft) -> dict:
    return {"name": d.name, "description": " ".join(d.description.split()), "parameters": d.parameters}


def compose_tool_source(d: ToolDraft) -> str:
    return (f'"""Synthesized tool `{d.name}`. Generated by the Gemma harness; changes must pass the same checks."""\n\n'
            f"TOOL_SPEC = {pprint.pformat(spec_of(d), sort_dicts=False, width=100)}\n\n\n"
            f"{strip_fences(d.code).strip()}\n")


def draft_errors(d: ToolDraft) -> list[str]:
    if errs := name_errors(d.name):
        return errs
    errs = ["do not define TOOL_SPEC; the harness adds it"] if "TOOL_SPEC" in d.code else []
    errs += check_tool(d.name, spec_of(d), compose_tool_source(d))
    errs += check_tests(d.name, strip_fences(d.tests))
    return errs


SYNTH_PROMPT = f"""You are the Worker in a Thinker/Worker/Verifier pipeline, creating ONE new reusable tool.
Return JSON with name, description, parameters, code and tests.
- name: snake_case, 3-41 characters, also the function name. Use the name the subgoal gives, if any.
- description: one or two sentences on what the tool does, its inputs and what it returns. Tools are found by
  searching this text, so name the concrete job.
- parameters: a JSON Schema with "type": "object". Its properties must match the function parameters exactly;
  list every parameter without a default in "required".
- code: a Python module defining a plain `def <name>(...)` (not async, no *args/**kwargs) that returns a string
  or JSON-serializable data. Do not define TOOL_SPEC; the harness adds it.
  Allowed imports: {", ".join(sorted(ToolLinter.ALLOWED))}, plus `from tools._sdk import ws_read, ws_write, ws_list`
  for workspace files (paths relative to the workspace root; ws_write cannot write .py files, tools/ or tests/).
  Not allowed: open(), os, pathlib, subprocess, network access, eval/exec, getattr/setattr, any __dunder__ name.
- tests: a pytest module that imports `from tools.<name> import <name>`, has at least two test_ functions including
  an edge case, and would FAIL if the function body were removed. For file behavior, call
  monkeypatch.setattr("tools._sdk.ROOT", tmp_path) and create the files under tmp_path.
- On a retry, fix the specific flaw the verifier named.
Return JSON only."""


# =============================================================================
# Harness
# =============================================================================

class GemmaTrinityToolHarness(GemmaTrinityHarness):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.git_ok = GIT_ENABLED and shutil.which("git") is not None

    # ---------------------------------------------------------------- test runs
    def _clean_env(self, root: Path) -> dict[str, str]:
        import os
        env = {k: os.environ[k] for k in TEST_ENV_PASSTHROUGH if k in os.environ}  # no API keys
        env.update(PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
        return env

    def _pytest(self, targets: list[str], root: Path | None = None) -> dict:
        root = root or self.ws
        try:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets],
                               cwd=root, env=self._clean_env(root), capture_output=True, text=True,
                               timeout=TEST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "passed": False, "no_tests": False, "output": "pytest timed out"}
        return {"exit_code": p.returncode, "passed": p.returncode == 0, "no_tests": p.returncode == 5,
                "output": elide(p.stdout + p.stderr)}

    # ---------------------------------------------------------------- git
    def _git(self, *args: str) -> subprocess.CompletedProcess:
        name, email = GIT_AUTHOR
        try:
            return subprocess.run(["git", "-c", f"user.name={name}", "-c", f"user.email={email}",
                                   "-c", "commit.gpgsign=false", *args],
                                  cwd=self.ws, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            return subprocess.CompletedProcess(list(args), 127, "", str(e))

    def _has_head(self) -> bool:
        return self._git("rev-parse", "--verify", "-q", "HEAD").returncode == 0

    def _git_prepare(self) -> None:
        if self._git("rev-parse", "--is-inside-work-tree").returncode != 0:
            r = self._git("init", "-q")
            if r.returncode != 0:
                self.git_ok = False
                print(f"[git] disabled: git init failed: {(r.stderr or r.stdout).strip()[:200]}")
                return
        gi = self.ws / ".gitignore"
        lines = gi.read_text(encoding="utf-8").splitlines() if gi.is_file() else []
        if missing := [x for x in GITIGNORE_LINES if x not in lines]:
            gi.write_text("\n".join([*lines, "# Gemma harness", *missing]) + "\n", encoding="utf-8")

    def _git_baseline(self) -> None:
        if not self._git("status", "--porcelain", "--", ".").stdout.strip():
            return
        self._git("add", "-A", "--", ".")
        cmd = ["commit", "-q", "-m", "gemma: baseline snapshot before run"]
        if self._has_head():
            cmd += ["--", "."]
        r = self._git(*cmd)
        if r.returncode == 0:
            print(f"[git] baseline commit {self._git('rev-parse', '--short', 'HEAD').stdout.strip()}")
        else:
            print(f"[git] baseline commit failed: {(r.stderr or r.stdout).strip()[:200]}")

    def _commit_paths(self, paths: list[str], subject: str, body: str) -> str | None:
        """Commit ONLY these paths, so another thread's in-flight changes never ride along."""
        paths = list(dict.fromkeys(paths))
        if not paths:
            return None
        for p in paths:  # one by one: a single unmatched pathspec would abort a combined `git add`
            self._git("add", "-A", "--", p)
        if self._git("diff", "--cached", "--quiet", "--", *paths).returncode == 0:
            return None  # nothing changed
        cmd = ["commit", "-q", "-m", subject, "-m", body]
        if self._has_head():
            cmd += ["--", *paths]
        r = self._git(*cmd)
        if r.returncode != 0:
            print(f"[git] commit failed: {(r.stderr or r.stdout).strip()[:300]}")
            return None
        return self._git("rev-parse", "--short", "HEAD").stdout.strip()

    @staticmethod
    def _commit_message(tid: str, state: dict, sg: Subgoal, event: dict) -> tuple[str, str]:
        tests = {True: "passed", False: "failed", None: "not run"}[event.get("tests_passed")]
        subject = f"gemma({tid}): {sg.id} [{sg.kind}] " + textwrap.shorten(sg.description, 50, placeholder="...")
        body = "\n".join([textwrap.fill(sg.description, 72), "",
                          f"Gemma-Thread: {tid}", f"Gemma-Plan: {state['plan_id']}", f"Gemma-Subgoal: {sg.id}",
                          f"Gemma-Kind: {sg.kind}", f"Gemma-Attempt: {event['attempt']}", f"Gemma-Tests: {tests}"])
        return subject, body

    # ---------------------------------------------------------------- crash journal
    def _journal_files(self, tid: str) -> tuple[Path, Path]:
        d = self.ws / JOURNAL_DIR
        d.mkdir(exist_ok=True)
        safe = re.sub(r"[^\w.-]", "_", tid)
        return d / f"{safe}.json", d / f"{safe}.sdk.jsonl"

    def _journal_add(self, tid: str, rel: str, original: str | None) -> None:
        p, _ = self._journal_files(tid)
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        if rel in data:  # the first original is the one to restore
            return
        data[rel] = original
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(p)

    def _read_sdk_log(self, tid: str) -> dict[str, str | None]:
        _, s = self._journal_files(tid)
        out: dict[str, str | None] = {}
        if s.is_file():
            for ln in s.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(ln)
                except json.JSONDecodeError:
                    continue  # a line cut off by a crash
                out.setdefault(e["path"], e.get("original"))
        return out

    def _journal_load(self, tid: str) -> dict[str, str | None]:
        p, _ = self._journal_files(tid)
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        for rel, original in self._read_sdk_log(tid).items():
            data.setdefault(rel, original)
        return data

    def _journal_clear(self, tid: str) -> None:
        for p in self._journal_files(tid):
            p.unlink(missing_ok=True)

    async def _recover(self, app: Any, tasks: list[dict[str, str]]) -> None:
        for t in tasks:
            tid = t["id"]
            files = await asyncio.to_thread(self._journal_load, tid)
            if not files:
                continue
            snap = await app.aget_state({"configurable": {"thread_id": tid}})
            if set(snap.next or ()) & {"verify", "gate"}:
                continue  # the checkpointed worker_meta owns these writes; the gate will settle them
            await asyncio.to_thread(self._restore, {"files": files})
            await asyncio.to_thread(self._journal_clear, tid)
            print(f"[{tid}] rolled back {len(files)} uncheckpointed write(s): {', '.join(sorted(files))}")

    # ---------------------------------------------------------------- overrides: writes and rollback
    def _write_and_index(self, path: Path, text: str, *, managed: bool = False) -> None:
        rel = self.rel(path)
        if not managed and is_managed(rel):
            raise PermissionError(f"{rel} is managed by tool synthesis; use a synth subgoal to change tools")
        if tid := CURRENT_THREAD.get():
            self._journal_add(tid, rel, path.read_text(encoding="utf-8") if path.is_file() else None)
        super()._write_and_index(path, text)

    def _restore(self, meta: dict) -> None:
        files = meta.get("files")
        if not files:
            return super()._restore(meta)
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
            if p.suffix == ".py":
                self.repo.index_file(rel)

    # ---------------------------------------------------------------- tool runtime + auto-loader
    def _install_tool_runtime(self) -> None:
        d = self.ws / "tools"
        d.mkdir(exist_ok=True)
        if not (d / "__init__.py").exists():
            (d / "__init__.py").write_text(INIT_SOURCE, encoding="utf-8")
        for fname, src in (("_sdk.py", SDK_SOURCE), ("_runner.py", RUNNER_SOURCE)):
            p = d / fname
            if not p.is_file() or p.read_text(encoding="utf-8") != src:
                p.write_text(src, encoding="utf-8")

    def _runner_handler(self, name: str):
        def handler(**kwargs: Any) -> str:
            env = self._clean_env(self.ws)
            if tid := CURRENT_THREAD.get():
                env["GEMMA_WRITE_LOG"] = str(self._journal_files(tid)[1])
            try:
                p = subprocess.run([sys.executable, "-m", "tools._runner", name], input=json.dumps(kwargs),
                                   cwd=self.ws, env=env, capture_output=True, text=True,
                                   timeout=SYNTH_TOOL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return f"Error: tool {name} timed out after {SYNTH_TOOL_TIMEOUT_S}s"
            lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
            try:
                payload = json.loads(lines[-1])
            except (IndexError, json.JSONDecodeError):
                return f"Error: tool {name} crashed:\n{elide(p.stderr, 1500)}"
            if not payload.get("ok"):
                return f"Error: {payload.get('error')}"
            r = payload.get("result")
            return r if isinstance(r, str) else json.dumps(r)
        return handler

    def _register_synth(self, spec: dict) -> None:
        self.tools.register(spec["name"], spec["description"], spec["parameters"], self._runner_handler(spec["name"]))

    def _load_verified(self) -> dict[str, str]:
        p = self.ws / VERIFIED_FILE
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        except (OSError, ValueError):
            return {}

    def _save_verified(self, data: dict[str, str]) -> None:
        p = self.ws / VERIFIED_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _mark_verified(self, name: str, dg: str) -> None:
        self._save_verified({**self._load_verified(), name: dg})

    def _prune_registry(self) -> None:
        ids = self.tools.collection.get(include=[]).get("ids", [])
        if stale := [i for i in ids if i not in self.tools._schemas]:
            self.tools.collection.delete(ids=stale)

    def _autoload_tools(self) -> list[str]:
        log, verified, loaded = [], self._load_verified(), {}
        for f in sorted((self.ws / "tools").glob("*.py")):
            name = f.stem
            if name.startswith("_"):
                continue
            test_f = self.ws / "tests" / f"test_tool_{name}.py"
            try:
                src = f.read_text(encoding="utf-8")
                spec = read_tool_spec(src)
                errs = check_tool(name, spec, src)
            except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as e:
                errs = [f"{type(e).__name__}: {e}"]
            if not errs and not test_f.is_file():
                errs = [f"missing tests/{test_f.name}"]
            if errs:
                log.append(f"[tools] skipped {name}: {'; '.join(errs)[:300]}")
                continue
            dg = digest(src, test_f.read_text(encoding="utf-8"))
            if verified.get(name) != dg:  # changed since the last green run: test again
                t = self._pytest([f"tests/{test_f.name}"])
                if not t["passed"]:
                    log.append(f"[tools] skipped {name}: its tests fail")
                    continue
            loaded[name] = dg
            self._register_synth(spec)
            log.append(f"[tools] loaded {name}")
        self._save_verified(loaded)
        self._prune_registry()
        return log

    def _mutation_check(self, name: str) -> dict:
        """Run the tool's tests against a mutant in a temporary copy; the real files are never touched."""
        tools_dir = self.ws / "tools"
        with tempfile.TemporaryDirectory(prefix="gemma-mutant-") as tmp:
            root = Path(tmp)
            (root / "tools").mkdir()
            (root / "tests").mkdir()
            for helper in ("__init__.py", "_sdk.py", "_runner.py"):
                if (tools_dir / helper).is_file():
                    shutil.copy(tools_dir / helper, root / "tools" / helper)
            src = (tools_dir / f"{name}.py").read_text(encoding="utf-8")
            (root / "tools" / f"{name}.py").write_text(mutate(src, name), encoding="utf-8")
            shutil.copy(self.ws / "tests" / f"test_tool_{name}.py", root / "tests")
            return self._pytest([f"tests/test_tool_{name}.py"], root=root)

    # ---------------------------------------------------------------- Workers
    async def _work_synth(self, state: dict, sg: Subgoal) -> tuple[WorkerOutput, dict]:
        ctx = self._worker_context(state, sg) + "\n\nExisting tools (do not duplicate one):\n" + self.tools.catalog()
        try:
            d = await self.chat_json("orchestrator", SYNTH_PROMPT, ctx, ToolDraft, check=draft_errors)
        except ModelOutputError as e:
            return WorkerOutput(subgoal_id=sg.id, status="complete"), {"worker_error": str(e)}
        tool_src, test_src = compose_tool_source(d), strip_fences(d.tests).strip() + "\n"
        tool_rel, test_rel = f"tools/{d.name}.py", f"tests/test_tool_{d.name}.py"
        files: dict[str, str | None] = {}
        async with self.write_lock:
            for rel, text in ((tool_rel, tool_src), (test_rel, test_src)):
                p = self.safe_path(rel)
                files[rel] = p.read_text(encoding="utf-8") if p.is_file() else None
                await asyncio.to_thread(self._write_and_index, p, text, managed=True)
        result = make_diff(tool_rel, files[tool_rel], tool_src) + "\n" + make_diff(test_rel, files[test_rel], test_src)
        meta = {"files": files, "written": True, "synth": spec_of(d), "digest": digest(tool_src, test_src)}
        return WorkerOutput(subgoal_id=sg.id, status="complete", result=result,
                            reasoning=f"Gemma 3 12B drafted tool {d.name}; spec, signature, linter and test "
                                      "structure passed static checks."), meta

    async def _work_tool(self, state: dict, sg: Subgoal) -> tuple[WorkerOutput, dict]:
        out, meta = await super()._work_tool(state, sg)
        tid = CURRENT_THREAD.get()
        sdk = await asyncio.to_thread(self._read_sdk_log, tid) if tid else {}
        if sdk:  # files a synthesized tool wrote through ws_write: track them for rollback and commit
            files = {meta["path"]: meta.get("original")} if meta.get("written") and meta.get("path") else {}
            for rel, original in sdk.items():
                files.setdefault(rel, original)
            meta = {**meta, "files": files, "written": True}
        return out, meta

    # ---------------------------------------------------------------- graph nodes
    async def node_work(self, state: dict, config: RunnableConfig) -> dict:
        CURRENT_THREAD.set(config["configurable"]["thread_id"])  # copied into every to_thread call below
        sg = Plan.model_validate(state["plan"]).subgoals[state["cursor"]]
        try:
            if sg.kind != "synth":
                return await super().node_work(state)
            out, meta = await self._work_synth(state, sg)
        except PermissionError as e:
            out, meta = WorkerOutput(subgoal_id=sg.id, status="blocked", blocked_reason=str(e)), {}
        return {"worker_output": out.model_dump(), "worker_meta": meta}

    async def node_verify(self, state: dict) -> dict:
        sg = Plan.model_validate(state["plan"]).subgoals[state["cursor"]]
        meta, wo = state.get("worker_meta") or {}, state["worker_output"]
        if sg.kind != "synth" or wo["status"] == "blocked" or not meta.get("synth"):
            return await super().node_verify(state)  # includes worker_error handling
        name = meta["synth"]["name"]
        test_rel = f"tests/test_tool_{name}.py"
        async with self.write_lock:  # Tier 1 for tools: real test run + mutation check
            tests = await asyncio.to_thread(self._pytest, [test_rel])
            mutant = await asyncio.to_thread(self._mutation_check, name) if tests["passed"] else None
        problem = None
        if not tests["passed"]:
            problem = (f"no tests were collected from {test_rel}" if tests["no_tests"]
                       else f"the tool's own tests fail:\n{elide(tests['output'], 2000)}")
        elif mutant["passed"]:
            problem = ("the tests still pass when the tool body is replaced by `raise NotImplementedError`, so they "
                       "do not test the tool. Assert on real return values.")
        if problem:
            v = Verdict(subgoal_id=sg.id, verdict="revise", diagnosis=f"Automated tool checks failed: {problem}",
                        route_to="worker")
            return {"verdict": v.model_dump(), "evidence": {"tool_tests": tests}}
        shown = {**wo, "result": f"{wo['result']}\n\nTool tests (real output): PASSED\n{elide(tests['output'], 1500)}\n"
                                 "Mutation check: the tests fail when the tool body is removed."}
        res = await super().node_verify({**state, "worker_output": shown})  # Tier 2: Gemma 3 12B
        return {**res, "evidence": {**(res.get("evidence") or {}), "tool_tests": tests}}

    async def node_gate(self, state: dict, config: RunnableConfig) -> dict:
        tid = config["configurable"]["thread_id"]
        res = await super().node_gate(state)  # budgets, routing, test override, rollback of rejected writes
        meta = state.get("worker_meta") or {}
        if (res.get("verdict") or {}).get("verdict") == "accept" and meta.get("written"):
            sg = Plan.model_validate(state["plan"]).subgoals[state["cursor"]]
            event, sha = res["events"][-1], None
            async with self.write_lock:
                if spec := meta.get("synth"):
                    await asyncio.to_thread(self._register_synth, spec)
                    await asyncio.to_thread(self._mark_verified, spec["name"], meta["digest"])
                if self.git_ok:
                    subject, body = self._commit_message(tid, state, sg, event)
                    sha = await asyncio.to_thread(self._commit_paths, changed_paths(meta), subject, body)
            if sha:
                print(f"[{tid}] committed {sha}: {sg.id}")
                res["events"] = [*res["events"][:-1], {**event, "commit": sha}]
        await asyncio.to_thread(self._journal_clear, tid)  # settled: committed or rolled back
        return res

    # ---------------------------------------------------------------- runner
    async def run_concurrent_workload(self, tasks: list[dict[str, str]]) -> list[Any]:
        db = str(self.ws / ".gemma_checkpoints.db")
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_S) as http, AsyncSqliteSaver.from_conn_string(db) as cp:
            self.http = http
            try:
                app = self.build_graph(cp)
                async with self.write_lock:  # order matters: recover BEFORE the baseline commit
                    if self.git_ok:
                        await asyncio.to_thread(self._git_prepare)
                    await self._recover(app, tasks)
                    await asyncio.to_thread(self._install_tool_runtime)
                    for line in await asyncio.to_thread(self._autoload_tools):
                        print(line)
                    if self.git_ok and GIT_BASELINE_COMMIT:
                        await asyncio.to_thread(self._git_baseline)
                await asyncio.to_thread(self.repo.index_all)
                return await asyncio.gather(*(self.run_thread(app, t["spec"], t["id"]) for t in tasks),
                                            return_exceptions=True)
            finally:
                self.http = None


if __name__ == "__main__":
    harness = GemmaTrinityToolHarness(workspace_dir=".")
    workload = [{"id": "thread-alpha",
                 "spec": "Fix the division-by-zero bug in math_utils.py and verify via pytest."}]
    for result in asyncio.run(harness.run_concurrent_workload(workload)):
        if isinstance(result, Exception):
            print(f"Thread crashed (resumable): {type(result).__name__}: {result}")