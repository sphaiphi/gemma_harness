"""Files installed under <workspace>/tools/ and the subprocess handler that runs a synthesized tool."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ..common import CURRENT_THREAD, elide
from ..workspace import Journal, TestRunner

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
    return _resolve(path)[0].read_text(encoding="utf-8")


def ws_write(path, content):
    target, rel = _resolve(path)
    if rel.endswith(".py") or rel.split("/")[0] in ("tools", "tests"):
        raise PermissionError(f"tools may not write code or tests: {path}")
    log = os.environ.get("GEMMA_WRITE_LOG")
    if log:
        original = target.read_text(encoding="utf-8") if target.is_file() else None
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"path": rel, "original": original}) + "\\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8")
    return f"wrote {len(str(content))} chars to {rel}"


def ws_list(pattern="**/*"):
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
        with contextlib.redirect_stdout(io.StringIO()):
            result = getattr(importlib.import_module(f"tools.{name}"), name)(**args)
        payload = {"ok": True, "result": result}
    except Exception as e:
        payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(payload, default=str))


if __name__ == "__main__":
    main()
'''

# Started as `python -I -c RUNNER_BOOTSTRAP <workspace> <tool>`. Isolated mode ignores PYTHONPATH, user site and
# the cwd, so the workspace is appended LAST to sys.path: workspace files cannot shadow stdlib modules, and a
# workspace sitecustomize.py never runs. This filters mistakes; the real boundary is a container or nsjail.
RUNNER_BOOTSTRAP = (
    "import os, runpy, sys; "
    "ws = sys.argv.pop(1); "
    "sys.path[:] = [p for p in sys.path if p not in ('', os.getcwd(), ws)]; "
    "sys.path.append(ws); "
    "runpy.run_module('tools._runner', run_name='__main__', alter_sys=True)"
)

RUNTIME_FILES = {"_sdk.py": SDK_SOURCE, "_runner.py": RUNNER_SOURCE}


def install_runtime(root: Path) -> None:
    d = root / "tools"
    d.mkdir(exist_ok=True)
    if not (d / "__init__.py").exists():
        (d / "__init__.py").write_text(INIT_SOURCE, encoding="utf-8")
    for fname, src in RUNTIME_FILES.items():
        p = d / fname
        if not p.is_file() or p.read_text(encoding="utf-8") != src:
            p.write_text(src, encoding="utf-8")


def runner_handler(name: str, root: Path, tests: TestRunner, journal: Journal,
                   timeout_s: int) -> Callable[..., str]:
    def handler(**kwargs: Any) -> str:
        env = tests.clean_env(root)
        if tid := CURRENT_THREAD.get():
            env["GEMMA_WRITE_LOG"] = str(journal.sdk_log(tid))
        cmd = [sys.executable, "-I", "-c", RUNNER_BOOTSTRAP, str(root), name]
        try:
            p = subprocess.run(cmd, input=json.dumps(kwargs), cwd=root, env=env,
                               capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return f"Error: tool {name} timed out after {timeout_s}s"
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