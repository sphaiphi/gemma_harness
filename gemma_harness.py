"""
Gemma Trinity Harness: Thinker / Worker / Verifier on local Gemma models (Idea B).

  Thinker  = Gemma 3 12B. Typed plan; every subgoal has a `kind` (edit | tool | reason).
  Worker   = chosen by code from `kind`:
               edit   -> EmbeddingGemma locates the code; CodeGemma FIM rewrites a small function on the
                         first attempt; Gemma 3 12B rewrites on retries or for bigger regions
               tool   -> FunctionGemma picks from a ChromaDB-retrieved shortlist
               reason -> Gemma 3 12B
  Verifier = tier 1: code checks (blocked, syntax, linter, tool errors, pytest)
             tier 2: Gemma 3 12B, only when tier 1 passes and judgment is still needed
  Gate     = plain code: budgets, routing, test override, rollback. The router only READS `next_node`.

Graph: sanitize -> think -> work -> verify -> gate -> (work | think | END)
"""
from __future__ import annotations

import ast
import asyncio
import difflib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

import chromadb
import httpx
import jsonschema
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

# =============================================================================
# 0. Configuration
# =============================================================================

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODELS = {
    "transformer": "t5gemma:2b-custom",
    "embedder": "embedding-gemma:300m-custom",
    "orchestrator": "gemma3:12b-custom",
    "function": "functiongemma:270m-custom",
    "coder": "codegemma:2b-custom",
}

MAX_WORKER_RETRIES = 3            # per subgoal
MAX_REPLANS = 3                   # per run
MAX_SUBGOALS = 10
MAX_CONCURRENT_LLM = 3            # protects the 24 GB VRAM budget
NO_TEST_TASK_TYPES = {"research", "reasoning"}  # every other task type must end with a passing test run
FIM_MAX_LINES = 60                # larger regions go straight to the 12B
FIM_ATTEMPTS = 1                  # FIM cannot read feedback, so retries escalate to the 12B
MAX_REWRITE_LINES = 600
LLM_TIMEOUT_S = 180.0
HTTP_ATTEMPTS = 3
TEST_TIMEOUT_S = 600
OUTPUT_CAP = 4000
SKIP_DIRS = {"venv", "__pycache__", "node_modules", "build", "dist"}  # hidden dirs are always skipped
TEST_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
RECURSION_LIMIT = 2 + (MAX_REPLANS + 1) * (1 + MAX_SUBGOALS * (MAX_WORKER_RETRIES + 1) * 3) + 10
PATH_RE = re.compile(r"[\w./-]+\.py\b")


def elide(text: str, cap: int = OUTPUT_CAP) -> str:
    if len(text) <= cap:
        return text
    head = cap // 4  # test verdicts sit at the end, so the tail gets the room
    return f"{text[:head]}\n[... {len(text) - cap} chars omitted ...]\n{text[-(cap - head):]}"


def make_diff(rel: str, before: str | None, after: str) -> str:
    diff = difflib.unified_diff((before or "").splitlines(True), after.splitlines(True), f"a/{rel}", f"b/{rel}")
    return "".join(diff) or "(no change)"


def strip_fences(code: str) -> str:
    m = re.match(r"^\s*```[\w-]*\n(.*?)\n?```\s*$", code, re.S)
    return m.group(1) if m else code


def iter_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]  # prune while walking
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f


class ModelOutputError(Exception):
    """A model did not return usable output after its repair attempt."""


# =============================================================================
# A. FIXED INFRASTRUCTURE
# =============================================================================

class OllamaEmbeddingFunction(EmbeddingFunction):
    """Synchronous on purpose (ChromaDB calls it synchronously). Async code wraps Chroma calls in to_thread."""

    def __init__(self, model_name: str, host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.host = host
        self._client = httpx.Client(timeout=60.0)

    @staticmethod
    def name() -> str:
        return "ollama-embeddinggemma"

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        r = self._client.post(f"{self.host}/api/embed", json={"model": self.model_name, "input": list(input)})
        r.raise_for_status()
        return r.json()["embeddings"]  # one batched request instead of one per text


class SecurityLinter(ast.NodeVisitor):
    """strict: allowlist (for synthesized tools). project: denylist (for normal project code)."""
    STRICT_ALLOWED = {"pathlib", "json", "re", "typing", "collections", "datetime", "math"}
    PROJECT_DENIED = {"subprocess", "socket", "ctypes", "pty"}
    FORBIDDEN_FUNCS = {"eval", "exec", "compile", "globals", "locals", "__import__", "breakpoint"}
    FORBIDDEN_ATTRS = {"system", "popen", "rmtree"}

    def __init__(self, policy: Literal["strict", "project"] = "project"):
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
        for a in node.names:  # catches `from os import system`
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


def lint(code: str, policy: Literal["strict", "project"] = "project") -> list[str]:
    return SecurityLinter(policy).analyze(code)  # fresh visitor per call: thread-safe


class ToolRegistry:
    def __init__(self, chroma: Any, embedding_fn: EmbeddingFunction):
        self.collection = chroma.get_or_create_collection("tools_registry", embedding_function=embedding_fn)
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict] = {}

    def register(self, name: str, description: str, parameters: dict, handler: Callable[..., Any]) -> None:
        self._handlers[name] = handler
        self._schemas[name] = {"name": name, "description": description, "parameters": parameters}
        self.collection.upsert(ids=[name], documents=[description], metadatas=[{"name": name}])

    def catalog(self) -> str:
        return "\n".join(f"- {n}: {s['description']}" for n, s in self._schemas.items())

    def relevant(self, query: str, n: int = 3) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(n, count))
        names = [m["name"] for m in (res.get("metadatas") or [[]])[0]]
        # Ollama's tool format, and only tools that still have a handler (the DB outlives the process).
        return [{"type": "function", "function": self._schemas[x]} for x in names if x in self._schemas]

    def validate(self, name: str | None, args: dict) -> str | None:
        if name not in self._schemas:
            return f"unknown tool {name!r}"
        try:
            jsonschema.validate(args, self._schemas[name]["parameters"])
        except jsonschema.ValidationError as e:
            return f"invalid arguments for {name}: {e.message}"
        return None

    def execute(self, name: str, args: dict) -> tuple[bool, str]:
        if err := self.validate(name, args):
            return False, f"Error: {err}"
        try:
            out = self._handlers[name](**args)
        except Exception as e:  # noqa: BLE001 - every failure becomes feedback, not a crash
            return False, f"Error: {type(e).__name__}: {e}"
        out = out if isinstance(out, str) else json.dumps(out)
        return not out.startswith("Error"), out


def _span(node: ast.AST) -> tuple[int, int]:
    return min([node.lineno] + [d.lineno for d in node.decorator_list]), node.end_lineno


class RepoIndex:
    """Phase 1 RAG: Python code chunked by function / class / method."""

    def __init__(self, workspace: Path, chroma: Any, embedding_fn: EmbeddingFunction):
        self.ws = workspace
        self.collection = chroma.get_or_create_collection("repo_chunks", embedding_function=embedding_fn)

    @staticmethod
    def chunk(text: str) -> list[tuple[int, int]]:
        whole = [(1, max(1, len(text.splitlines())))]
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return whole
        spans: list[tuple[int, int]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                spans.append(_span(node))
                if isinstance(node, ast.ClassDef):
                    spans += [_span(n) for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return spans or whole

    def index_file(self, rel: str) -> None:
        self.collection.delete(where={"path": rel})
        p = self.ws / rel
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        lines = text.splitlines()
        spans = list(dict.fromkeys(self.chunk(text)))
        self.collection.upsert(
            ids=[f"{rel}:{s}-{e}" for s, e in spans],
            documents=[(f"# {rel}\n" + "\n".join(lines[s - 1:e]))[:4000] for s, e in spans],
            metadatas=[{"path": rel, "start": s, "end": e} for s, e in spans],
        )

    def index_all(self) -> None:
        for p in iter_py_files(self.ws):
            self.index_file(p.relative_to(self.ws).as_posix())

    def search(self, query: str, n: int = 4, path: str | None = None) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(n, count),
                                    where={"path": path} if path else None)
        metas, docs = (res.get("metadatas") or [[]])[0], (res.get("documents") or [[]])[0]
        return [{**m, "text": d} for m, d in zip(metas, docs)]


def regen_start(source: str, start: int, end: int) -> int | None:
    """If lines start..end are exactly one function, return the line where its body (after any docstring)
    begins, so FIM can regenerate the body below a fixed signature. Otherwise None."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _span(node) == (start, end):
            body = node.body
            has_doc = (len(body) > 1 and isinstance(body[0], ast.Expr)
                       and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str))
            first = body[1] if has_doc else body[0]
            return first.lineno if first.lineno > node.lineno else None  # one-liners: no FIM
    return None


# =============================================================================
# B. TRINITY ROLES: schemas, prompts, state
# =============================================================================

class Subgoal(BaseModel):
    id: str
    description: str
    kind: Literal["edit", "tool", "reason"]
    suggested_role: Literal["worker", "thinker"] = "worker"
    depends_on: list[str] = Field(default_factory=list)
    notes: str = ""


class Plan(BaseModel):
    task_type: Literal["coding", "math", "research", "reasoning", "other"]
    summary: str
    subgoals: list[Subgoal] = Field(min_length=1)
    revision_of: str | None = None


class ReasonOutput(BaseModel):
    status: Literal["complete", "blocked"]
    result: str | None = None
    reasoning: str = ""
    blocked_reason: str | None = None


class WorkerOutput(ReasonOutput):
    subgoal_id: str


class EditCode(BaseModel):
    code: str


class Verdict(BaseModel):
    subgoal_id: str
    verdict: Literal["accept", "revise"]
    diagnosis: str | None = None
    route_to: Literal["worker", "thinker"] | None = None
    task_complete: bool = False


THINKER_PROMPT = f"""You are the Thinker in a Thinker/Worker/Verifier pipeline. Decompose the task into the smallest
ordered list of subgoals (at most {MAX_SUBGOALS}). You only plan: never solve, write code or verify.
- Each subgoal must be concrete enough for a Worker to do without questions. Ids are unique: sg1, sg2, ...
- depends_on may list only ids that appear EARLIER in the list.
- kind for each subgoal:
  - edit: change one function, class or small region of ONE Python file, or create one new file.
    Start the description with the file path, e.g. "In src/math_utils.py: ...".
  - tool: exactly one call to an available tool. Never use a tool subgoal to author code; use edit for that.
  - reason: analysis, research or a written answer that needs no tool.
- The last subgoal produces the final deliverable. Do not add verification subgoals.
- For coding tasks the run must end with a passing pytest suite: include a subgoal that writes or updates tests
  unless suitable tests already exist.
- Never ask questions. Write assumptions in notes. suggested_role is always "worker".
- On a revision: keep every subgoal from accepted_results you still need with the SAME id and description, at the
  start and in the same order. Revise only from the first failing subgoal on. If an accepted result is itself wrong,
  give the changed subgoal a new id (e.g. sg2b). The revised plan must contain at least one new subgoal.
Return JSON only."""

WORKER_PROMPT = """You are the Worker in a Thinker/Worker/Verifier pipeline. Execute exactly one subgoal.
- Use the accepted results of earlier subgoals as given inputs; do not redo them.
- On a retry, read the verifier's feedback and your rejected output and fix that specific flaw. If you believe the
  feedback is wrong, still return a corrected-or-justified result and explain why in reasoning.
- You have no tools. If the subgoal needs running code, reading files or browsing and that output is not in your
  inputs, set status "blocked" and explain in blocked_reason. Never invent tool output.
- reasoning: a short justification the verifier can spot-check.
Return JSON only."""

EDIT_PROMPT = """You are the Worker in a Thinker/Worker/Verifier pipeline, editing Python code.
Do exactly this subgoal and nothing else. Keep existing names, signatures and style unless the subgoal says otherwise.
On a retry, fix the specific flaw the verifier named; do not resubmit the same code.
Return JSON {"code": "..."} containing only the requested code: no markdown fences, no explanation."""

TOOL_PROMPT = """Call exactly one tool that performs the subgoal. Copy argument values (paths, names) exactly from
the text. Do not answer in prose."""

VERIFIER_PROMPT = """You are the Verifier in a Thinker/Worker/Verifier pipeline. Judge whether the worker's result
correctly and completely satisfies its subgoal. Never do the work yourself.
- Judge against the subgoal and the planner's notes, not your own preference. A valid alternative is not a defect.
- Treat the worker's reasoning as a claim. Check the result itself. Automated test output in your input is real
  evidence; a claim that code was run with no such evidence is unverified.
- On the final subgoal, also confirm every subgoal has a result and the combined results answer the original query.
- On revise, give a specific, actionable diagnosis. route_to "worker" when the subgoal was sound but the execution
  was flawed or unverified; "thinker" when the subgoal was wrong or insufficient, or the plan misses the real ask.
- task_complete is true only when you accept the final subgoal after the whole-task check.
Return JSON only."""


class HarnessState(TypedDict, total=False):
    raw_spec: str
    structured_ast: str
    plan: dict | None
    plan_id: str | None
    prior_context: dict | None
    cursor: int
    accepted: dict[str, Any]
    worker_output: dict | None
    worker_meta: dict
    verifier_feedback: str | None
    previous_output: dict | None
    evidence: dict
    verdict: dict | None
    retries: dict[str, int]
    replans: int
    status: str            # running | complete | failed
    failure_reason: str | None
    next_node: str         # written ONLY by nodes; the router just reads it
    events: list[dict]


def initial_state(raw_spec: str) -> HarnessState:
    return HarnessState(
        raw_spec=raw_spec, structured_ast="", plan=None, plan_id=None, prior_context=None, cursor=0,
        accepted={}, worker_output=None, worker_meta={}, verifier_feedback=None, previous_output=None,
        evidence={}, verdict=None, retries={}, replans=0, status="running", failure_reason=None,
        next_node="think", events=[],
    )


def route(state: HarnessState) -> str:
    return state["next_node"]


def validate_plan(plan: Plan, accepted: dict, prev: dict[str, dict]) -> list[str]:
    errs: list[str] = []
    ids = [s.id for s in plan.subgoals]
    if len(ids) != len(set(ids)):
        errs.append("subgoal ids must be unique")
    if len(ids) > MAX_SUBGOALS:
        errs.append(f"at most {MAX_SUBGOALS} subgoals")
    seen: set[str] = set()
    for s in plan.subgoals:
        if bad := [d for d in s.depends_on if d not in seen]:
            errs.append(f"{s.id}.depends_on names {bad}, which are not earlier subgoals")
        seen.add(s.id)
    if prev:
        kept = [s for s in plan.subgoals if s.id in accepted]
        for s in kept:
            if s.description != prev[s.id]["description"]:
                errs.append(f"{s.id} was accepted; keep its description or give the changed subgoal a new id")
        if [s.id for s in plan.subgoals[:len(kept)]] != [s.id for s in kept]:
            errs.append("kept accepted subgoals must come first, in their original order")
        if len(kept) == len(plan.subgoals):
            errs.append("a revision must contain at least one new subgoal")
    return errs


# =============================================================================
# Harness
# =============================================================================

class GemmaTrinityHarness:
    def __init__(self, workspace_dir: str = ".", host: str = OLLAMA_HOST,
                 max_concurrent_llm: int = MAX_CONCURRENT_LLM):
        self.ws = Path(workspace_dir).resolve()
        self.host = host
        self.sem = asyncio.Semaphore(max_concurrent_llm)
        self.write_lock = asyncio.Lock()  # serializes writes and test runs within this workspace
        self.http: httpx.AsyncClient | None = None
        embed = OllamaEmbeddingFunction(MODELS["embedder"], host)
        chroma = chromadb.PersistentClient(path=str(self.ws / ".gemma_vector_db"))
        self.tools = ToolRegistry(chroma, embed)
        self.repo = RepoIndex(self.ws, chroma, embed)
        self._register_default_tools()

    # ---------------------------------------------------------------- A: files, tools, tests
    def safe_path(self, path: str) -> Path:
        target = (self.ws / path.strip().lstrip("/")).resolve()
        if not target.is_relative_to(self.ws):
            raise PermissionError(f"path escapes the workspace: {path}")
        if any(part.startswith(".") for part in target.relative_to(self.ws).parts):
            raise PermissionError(f"hidden paths are off limits: {path}")
        return target

    def rel(self, p: Path) -> str:
        return p.relative_to(self.ws).as_posix()

    def _write_and_index(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if path.suffix == ".py":
            self.repo.index_file(self.rel(path))

    def _restore(self, meta: dict) -> None:
        p = self.safe_path(meta["path"])
        if meta.get("original") is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(meta["original"], encoding="utf-8")
        if p.suffix == ".py":
            self.repo.index_file(meta["path"])

    def run_tests_sync(self) -> dict:
        env = {k: os.environ[k] for k in TEST_ENV_PASSTHROUGH if k in os.environ}  # no API keys
        env.update(PYTHONPATH=str(self.ws), PYTHONDONTWRITEBYTECODE="1")
        try:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=self.ws,
                               env=env, capture_output=True, text=True, timeout=TEST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "passed": False, "no_tests": False, "output": "pytest timed out"}
        return {"exit_code": p.returncode, "passed": p.returncode == 0, "no_tests": p.returncode == 5,
                "output": elide(p.stdout + p.stderr)}

    def _register_default_tools(self) -> None:
        def read_file(path: str) -> str:
            p = self.safe_path(path)
            return p.read_text(encoding="utf-8") if p.is_file() else f"Error: file {path} not found."

        def write_file(path: str, content: str) -> str:
            p = self.safe_path(path)
            if p.suffix == ".py" and (errs := lint(content)):
                return "Error: blocked by the security linter: " + "; ".join(errs)
            self._write_and_index(p, content)
            return f"Wrote {len(content)} chars to {path}"

        def run_pytest() -> str:
            return json.dumps(self.run_tests_sync())

        self.tools.register("read_file", "Read a workspace file.",
                            {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                            read_file)
        self.tools.register("write_file", "Write content to a workspace file.",
                            {"type": "object", "properties": {"path": {"type": "string"},
                                                              "content": {"type": "string"}},
                             "required": ["path", "content"]}, write_file)
        self.tools.register("run_pytest", "Run the workspace pytest suite.",
                            {"type": "object", "properties": {}}, run_pytest)

    # ---------------------------------------------------------------- A: model calls
    async def _post(self, endpoint: str, payload: dict) -> dict:
        if self.http is None:
            raise RuntimeError("use run_concurrent_workload(); it opens the shared HTTP client")
        for attempt in range(HTTP_ATTEMPTS):
            try:
                async with self.sem:
                    r = await self.http.post(f"{self.host}{endpoint}", json=payload)
                if r.status_code >= 500 and attempt < HTTP_ATTEMPTS - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except httpx.TransportError:
                if attempt == HTTP_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    async def generate(self, model_key: str, prompt: str, **extra: Any) -> str:
        data = await self._post("/api/generate", {"model": MODELS[model_key], "prompt": prompt,
                                                  "stream": False, **extra})
        return data.get("response", "")

    async def chat_json(self, model_key: str, system: str, user: str, schema: type[BaseModel],
                        check: Callable[[Any], list[str]] | None = None, attempts: int = 2) -> Any:
        """Typed output: JSON schema in Ollama's `format`, Pydantic validation, one repair retry."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        err = "no output"
        for _ in range(attempts):
            data = await self._post("/api/chat", {"model": MODELS[model_key], "messages": messages,
                                                  "format": schema.model_json_schema(), "stream": False,
                                                  "options": {"temperature": 0.2}})
            content = (data.get("message") or {}).get("content", "")
            try:
                obj = schema.model_validate_json(content)
                errs = check(obj) if check else []
                if not errs:
                    return obj
                err = "; ".join(errs)
            except ValidationError as e:
                err = str(e)[:1500]
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"Your JSON was rejected: {err}\nReturn corrected JSON only."}]
        raise ModelOutputError(err)

    async def fim(self, prefix: str, suffix: str) -> str:
        return await self.generate(
            "coder", f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
            raw=True,  # without this Ollama wraps the FIM tokens in a chat template
            options={"temperature": 0.1, "num_predict": 1024,
                     "stop": ["<|file_separator|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]})

    async def route_tool(self, sg: Subgoal, feedback: str | None) -> tuple[str | None, dict, str | None]:
        query = f"{sg.description}\n{sg.notes}".strip()
        tools = await asyncio.to_thread(self.tools.relevant, query, 3)
        if not tools:
            return None, {}, "no tools are registered"
        messages = [{"role": "system", "content": TOOL_PROMPT},
                    {"role": "user", "content": query + (f"\n\nPrevious attempt rejected: {feedback}" if feedback else "")}]
        err: str | None = "no tool call"
        for _ in range(2):
            data = await self._post("/api/chat", {"model": MODELS["function"], "messages": messages, "tools": tools,
                                                  "stream": False, "options": {"temperature": 0.0}})
            msg = data.get("message") or {}
            if calls := msg.get("tool_calls") or []:
                fn = calls[0].get("function") or {}
                name, args = fn.get("name"), fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = None
                err = "arguments are not a JSON object" if not isinstance(args, dict) else self.tools.validate(name, args)
                if err is None:
                    return name, args, None
            messages += [{"role": "assistant", "content": msg.get("content", "")},
                         {"role": "user", "content": f"Not a valid tool call ({err}). Call one tool with valid arguments."}]
        return None, {}, err  # no silent fallback to run_pytest

    # ---------------------------------------------------------------- B: Worker helpers
    def _worker_context(self, state: HarnessState, sg: Subgoal) -> str:
        parts = [f"Subgoal {sg.id}: {sg.description}"]
        if sg.notes:
            parts.append(f"Planner notes: {sg.notes}")
        if state["accepted"]:  # ALL accepted results, direct dependencies starred
            deps = set(sg.depends_on)
            parts.append("Accepted results of earlier subgoals (* = direct dependency):\n" + "\n".join(
                f"{'*' if k in deps else '-'} {k}: {elide(str(v), 1500)}" for k, v in state["accepted"].items()))
        if state.get("verifier_feedback"):
            parts.append(f"RETRY. The verifier rejected your previous output:\n{state['verifier_feedback']}")
            if prev := state.get("previous_output"):
                parts.append("Your rejected output:\n" + elide(json.dumps(prev), 3000))
        return "\n\n".join(parts)

    async def _locate(self, sg: Subgoal) -> tuple[str, int | None, int | None]:
        for m in PATH_RE.findall(f"{sg.description}\n{sg.notes}"):
            try:
                rel = self.rel(self.safe_path(m))
            except PermissionError:
                continue
            if (self.ws / rel).is_file():
                hits = await asyncio.to_thread(self.repo.search, sg.description, 1, rel)
                return (rel, hits[0]["start"], hits[0]["end"]) if hits else (rel, None, None)
            return rel, None, None  # new file
        hits = await asyncio.to_thread(self.repo.search, sg.description, 1)
        if hits:
            return hits[0]["path"], hits[0]["start"], hits[0]["end"]
        raise LookupError("no target file named in the subgoal and nothing relevant in the repo index")

    async def _edit_fim(self, sg: Subgoal, lines: list[str], start: int, end: int, body_at: int) -> str:
        before, after = "".join(lines[:start - 1]), "".join(lines[end:])
        header = "".join(lines[start - 1:body_at - 1])  # decorators, signature, docstring stay fixed
        old_body = lines[body_at - 1:end]
        indent = re.match(r"[ \t]*", old_body[0]).group(0)
        guide = f"{indent}# TASK: {' '.join(sg.description.split())}\n" + "".join(
            f"{indent}# OLD| {ln.rstrip()}\n" for ln in old_body)  # prompt-only, never written
        middle = await self.fim(before + header + guide, after)
        middle = "".join(ln for ln in middle.splitlines(True) if not ln.lstrip().startswith(("# OLD|", "# TASK:")))
        if not middle.strip():
            raise ModelOutputError("CodeGemma returned an empty completion")
        return before + header + middle.rstrip("\n") + "\n" + after

    async def _edit_12b(self, state: HarnessState, sg: Subgoal, rel: str, original: str | None,
                        lines: list[str], start: int, end: int) -> str:
        if original is None:
            before = after = ""
            task = f"The file {rel} does not exist yet. Return its complete content."
        else:
            before, after = "".join(lines[:start - 1]), "".join(lines[end:])
            task = (f"File {rel} for context:\n```python\n{elide(original, 12000)}\n```\n\n"
                    f"Replace lines {start}-{end}:\n```python\n{''.join(lines[start - 1:end])}```\n"
                    "Return only the replacement for those lines, with their original indentation.")
        out = await self.chat_json("orchestrator", EDIT_PROMPT, self._worker_context(state, sg) + "\n\n" + task, EditCode)
        return before + strip_fences(out.code).rstrip("\n") + "\n" + after

    # ---------------------------------------------------------------- B: Workers by kind
    async def _work_edit(self, state: HarnessState, sg: Subgoal, attempt: int) -> tuple[WorkerOutput, dict]:
        try:
            rel, start, end = await self._locate(sg)
            path = self.safe_path(rel)
        except (LookupError, PermissionError) as e:
            return WorkerOutput(subgoal_id=sg.id, status="blocked", blocked_reason=str(e)), {}
        original = path.read_text(encoding="utf-8") if path.is_file() else None
        lines = (original or "").splitlines(keepends=True)
        if original is not None and start is None:
            start, end = 1, max(1, len(lines))
        meta: dict = {"path": rel, "original": original}
        if original is not None and end - start + 1 > MAX_REWRITE_LINES:
            return WorkerOutput(subgoal_id=sg.id, status="blocked",
                                blocked_reason=f"{rel} is too large to rewrite whole; name the function to change"), {}
        try:
            body_at = regen_start(original, start, end) if original is not None else None
            if body_at and end - start + 1 <= FIM_MAX_LINES and attempt < FIM_ATTEMPTS:
                new_text, engine = await self._edit_fim(sg, lines, start, end, body_at), "CodeGemma FIM"
            else:
                new_text, engine = await self._edit_12b(state, sg, rel, original, lines, start, end), "Gemma 3 12B"
        except ModelOutputError as e:
            return WorkerOutput(subgoal_id=sg.id, status="complete"), {**meta, "worker_error": str(e)}
        span = f"lines {start}-{end}" if original is not None else "a new file"
        out = WorkerOutput(subgoal_id=sg.id, status="complete", result=make_diff(rel, original, new_text),
                           reasoning=f"{engine} rewrote {span} of {rel}.")
        errors = lint(new_text) if rel.endswith(".py") else []
        if new_text == (original or ""):
            errors.append("the edit changed nothing")
        if errors:  # checked BEFORE writing: unsafe or broken code never reaches disk
            return out, {**meta, "static_errors": errors}
        async with self.write_lock:
            await asyncio.to_thread(self._write_and_index, path, new_text)
        return out, {**meta, "written": True}

    async def _work_tool(self, state: HarnessState, sg: Subgoal) -> tuple[WorkerOutput, dict]:
        name, args, err = await self.route_tool(sg, state.get("verifier_feedback"))
        if err:
            return WorkerOutput(subgoal_id=sg.id, status="complete", result=f"NO TOOL RAN: {err}"), {"tool_ok": False}
        meta: dict = {"tool": name, "tool_ok": False}
        if name == "write_file" and args.get("path"):
            try:  # remember the original so a rejected write can be rolled back
                p = self.safe_path(args["path"])
                meta.update(path=self.rel(p), original=p.read_text(encoding="utf-8") if p.is_file() else None)
            except PermissionError:
                pass
        async with self.write_lock:
            ok, output = await asyncio.to_thread(self.tools.execute, name, args)
        meta["tool_ok"] = ok
        if ok and "path" in meta:
            meta["written"] = True
        shown = ", ".join(f"{k}={str(v)[:80]!r}" for k, v in args.items())
        return WorkerOutput(subgoal_id=sg.id, status="complete",
                            result=f"{name}({shown}) -> {'ok' if ok else 'ERROR'}\n{elide(output)}",
                            reasoning=f"FunctionGemma selected {name}; the output above is the real tool output."), meta

    async def _work_reason(self, state: HarnessState, sg: Subgoal) -> tuple[WorkerOutput, dict]:
        try:
            out = await self.chat_json("orchestrator", WORKER_PROMPT, self._worker_context(state, sg), ReasonOutput)
        except ModelOutputError as e:
            return WorkerOutput(subgoal_id=sg.id, status="complete"), {"worker_error": str(e)}
        return WorkerOutput(subgoal_id=sg.id, **out.model_dump()), {}

    # ---------------------------------------------------------------- B: graph nodes
    async def node_sanitize(self, state: HarnessState) -> dict:
        if state.get("structured_ast"):
            return {}
        try:
            text = await self.generate("transformer", f"Normalize into a structured task description:\n\n{state['raw_spec']}")
        except httpx.HTTPError:
            text = ""  # T5Gemma unavailable: the Thinker still gets the raw spec
        return {"structured_ast": text.strip() or state["raw_spec"]}

    async def node_think(self, state: HarnessState) -> dict:
        prior = state.get("prior_context")
        accepted = prior["accepted_results"] if prior else {}
        prev = {s["id"]: s for s in prior["previous_plan"]["subgoals"]} if prior else {}
        hits = await asyncio.to_thread(self.repo.search, state["raw_spec"], 5)
        code = "\n".join(f"- {h['path']} lines {h['start']}-{h['end']}:\n{elide(h['text'], 600)}" for h in hits)
        parts = [f"Task as written:\n{state['raw_spec']}",
                 f"Normalized task:\n{state['structured_ast']}",
                 f"Available tools:\n{self.tools.catalog()}",
                 f"Relevant code in the workspace:\n{code or '(nothing indexed)'}"]
        if prior:
            parts.append("REVISION. prior_context:\n" + elide(json.dumps(prior, indent=1), 8000))
        try:
            plan = await self.chat_json("orchestrator", THINKER_PROMPT, "\n\n".join(parts), Plan,
                                        check=lambda p: validate_plan(p, accepted, prev))
        except ModelOutputError as e:
            return {"status": "failed", "failure_reason": f"Thinker produced no valid plan: {e}", "next_node": END}
        plan.revision_of = prior["previous_plan"]["plan_id"] if prior else None  # stamped by code
        kept = {s.id: accepted[s.id] for s in plan.subgoals if s.id in accepted}
        return {"plan": plan.model_dump(), "plan_id": uuid.uuid4().hex[:8], "accepted": kept, "cursor": len(kept),
                "retries": {k: v for k, v in state["retries"].items() if k in kept}, "prior_context": None,
                "verifier_feedback": None, "previous_output": None, "next_node": "work"}

    async def node_work(self, state: HarnessState) -> dict:
        sg = Plan.model_validate(state["plan"]).subgoals[state["cursor"]]
        attempt = state["retries"].get(sg.id, 0)
        if sg.kind == "edit":
            out, meta = await self._work_edit(state, sg, attempt)
        elif sg.kind == "tool":
            out, meta = await self._work_tool(state, sg)
        else:
            out, meta = await self._work_reason(state, sg)
        return {"worker_output": out.model_dump(), "worker_meta": meta}

    def _verifier_context(self, state: HarnessState, plan: Plan, sg: Subgoal, wo: WorkerOutput,
                          evidence: dict, is_final: bool, tests_required: bool) -> str:
        parts = [f"Subgoal {sg.id} ({sg.kind}): {sg.description}"]
        if sg.notes:
            parts.append(f"Planner notes (assumptions the worker was allowed to make): {sg.notes}")
        parts.append("Worker output:\n" + elide(json.dumps(wo.model_dump(), indent=1), 6000))
        if t := evidence.get("tests"):
            status = "PASSED" if t["passed"] else "NO TESTS COLLECTED" if t["no_tests"] else "FAILED"
            parts.append(f"Automated test run (real output): {status}, exit code {t['exit_code']}\n{elide(t['output'], 3000)}")
            if not is_final:
                parts.append("Later subgoals may still be pending. Do not blame this subgoal for failures "
                             "that only those later subgoals would fix.")
        elif not tests_required:
            parts.append(f"Task type {plan.task_type!r} does not require tests. Judge the content.")
        if is_final:
            results = {**state["accepted"], sg.id: wo.result}
            parts.append("FINAL SUBGOAL: also run the whole-task check.\n"
                         f"Original query:\n{state['raw_spec']}\n\nAll subgoals:\n"
                         + "\n".join(f"- {s.id}: {s.description}" for s in plan.subgoals)
                         + "\n\nAll results:\n" + "\n".join(f"- {k}: {elide(str(v), 1500)}" for k, v in results.items()))
        else:
            parts.append("This is not the final subgoal: task_complete must be false.")
        return "\n\n".join(parts)

    async def node_verify(self, state: HarnessState) -> dict:
        plan = Plan.model_validate(state["plan"])
        sg = plan.subgoals[state["cursor"]]
        wo = WorkerOutput.model_validate(state["worker_output"])
        meta = state.get("worker_meta") or {}
        is_final = state["cursor"] == len(plan.subgoals) - 1
        tests_required = plan.task_type not in NO_TEST_TASK_TYPES

        def verdict(v: str, diagnosis: str | None = None, route_to: str | None = None) -> dict:
            return Verdict(subgoal_id=sg.id, verdict=v, diagnosis=diagnosis, route_to=route_to).model_dump()

        if wo.status == "blocked":  # spec step 1
            return {"verdict": verdict("revise", f"Worker blocked: {wo.blocked_reason}", "thinker"), "evidence": {}}

        # Tier 1: code checks, no model call
        errors = list(meta.get("static_errors", []))
        if meta.get("worker_error"):
            errors.append(f"the worker produced no valid output: {meta['worker_error']}")
        if sg.kind == "tool" and not meta.get("tool_ok"):
            errors.append(elide(wo.result or "tool call failed", 1500))
        if errors:
            return {"verdict": verdict("revise", "Automated checks failed:\n" + "\n".join(f"- {e}" for e in errors),
                                       "worker"), "evidence": {}}
        evidence: dict = {}
        if tests_required and (sg.kind == "edit" or is_final):
            async with self.write_lock:
                evidence["tests"] = await asyncio.to_thread(self.run_tests_sync)
        if sg.kind == "tool" and not is_final:  # real tool output, no judgment needed
            return {"verdict": verdict("accept"), "evidence": evidence}

        # Tier 2: Gemma 3 12B judgment
        try:
            v = await self.chat_json("orchestrator", VERIFIER_PROMPT,
                                     self._verifier_context(state, plan, sg, wo, evidence, is_final, tests_required),
                                     Verdict)
        except ModelOutputError as e:  # fail closed
            return {"verdict": verdict("revise", f"The verifier returned no valid verdict ({e}).", "worker"),
                    "evidence": evidence}
        return {"verdict": v.model_dump(), "evidence": evidence}

    async def node_gate(self, state: HarnessState) -> dict:
        plan = Plan.model_validate(state["plan"])
        sg = plan.subgoals[state["cursor"]]
        wo = WorkerOutput.model_validate(state["worker_output"])
        v = Verdict.model_validate(state["verdict"])
        meta = state.get("worker_meta") or {}
        is_final = state["cursor"] == len(plan.subgoals) - 1
        tests = (state.get("evidence") or {}).get("tests")

        # A failing or missing test run cannot be overruled on the final subgoal.
        if is_final and plan.task_type not in NO_TEST_TASK_TYPES and v.verdict == "accept" \
                and not (tests and tests["passed"]):
            no_tests = tests is None or tests["no_tests"]
            v = Verdict(subgoal_id=sg.id, verdict="revise",
                        diagnosis=("No tests were collected. This task type needs a passing pytest suite; "
                                   "add a subgoal that writes tests." if no_tests
                                   else f"The test suite fails:\n{elide(tests['output'], 1500)}"),
                        route_to="thinker" if no_tests or sg.kind != "edit" else "worker")
        if v.verdict == "revise" and v.route_to not in ("worker", "thinker"):
            v.route_to = "worker"
        v.subgoal_id = sg.id
        v.task_complete = is_final and v.verdict == "accept"  # enforced by code, whatever the model said

        event = {"plan_id": state["plan_id"], "subgoal": sg.id, "kind": sg.kind,
                 "attempt": state["retries"].get(sg.id, 0) + 1, "verdict": v.verdict, "route_to": v.route_to,
                 "diagnosis": (v.diagnosis or "")[:300], "tests_passed": tests["passed"] if tests else None}
        base = {"events": [*state.get("events", []), event], "verdict": v.model_dump()}

        def fail(reason: str) -> dict:
            return {**base, "status": "failed", "failure_reason": reason, "next_node": END}

        if v.verdict == "accept":
            accepted = {**state["accepted"], sg.id: wo.result}
            if v.task_complete:
                return {**base, "accepted": accepted, "status": "complete", "next_node": END}
            return {**base, "accepted": accepted, "cursor": state["cursor"] + 1,
                    "verifier_feedback": None, "previous_output": None, "next_node": "work"}

        if meta.get("written"):  # roll back the rejected change; the retry starts from the original
            async with self.write_lock:
                await asyncio.to_thread(self._restore, meta)

        if v.route_to == "worker":
            n = state["retries"].get(sg.id, 0)
            if n >= MAX_WORKER_RETRIES:
                return fail(f"{sg.id} was rejected after {MAX_WORKER_RETRIES} retries: {v.diagnosis}")
            return {**base, "retries": {**state["retries"], sg.id: n + 1},
                    "verifier_feedback": v.diagnosis or "Rejected without a diagnosis.",
                    "previous_output": {"result": wo.result, "reasoning": wo.reasoning}, "next_node": "work"}

        if state["replans"] >= MAX_REPLANS:
            return fail(f"plan rejected after {MAX_REPLANS} re-plans: {v.diagnosis}")
        return {**base, "replans": state["replans"] + 1, "next_node": "think",
                "prior_context": {"previous_plan": {**state["plan"], "plan_id": state["plan_id"]},
                                  "verifier_diagnosis": v.diagnosis, "accepted_results": state["accepted"]}}

    # ---------------------------------------------------------------- graph + runners
    def build_graph(self, checkpointer: Any):
        g = StateGraph(HarnessState)
        g.add_node("sanitize", self.node_sanitize)
        g.add_node("think", self.node_think)
        g.add_node("work", self.node_work)
        g.add_node("verify", self.node_verify)
        g.add_node("gate", self.node_gate)
        g.set_entry_point("sanitize")
        g.add_edge("sanitize", "think")
        g.add_conditional_edges("think", route, {"work": "work", END: END})
        g.add_edge("work", "verify")
        g.add_edge("verify", "gate")
        g.add_conditional_edges("gate", route, {"work": "work", "think": "think", END: END})
        return g.compile(checkpointer=checkpointer)

    async def run_thread(self, app: Any, raw_spec: str, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        snap = await app.aget_state(config)
        if snap.next:
            inp, verb = None, "Resuming"            # crash recovery: continue where it stopped
        elif snap.values.get("status") in ("complete", "failed"):
            print(f"[{thread_id}] Already {snap.values['status']}.")
            return snap.values
        else:
            inp, verb = initial_state(raw_spec), "Starting"
        print(f"[{thread_id}] {verb} thread...")
        async for update in app.astream(inp, config, stream_mode="updates"):
            for node, upd in update.items():
                v = (upd or {}).get("verdict") if node == "gate" else None
                print(f"[{thread_id}] {node}" + (f": {v['subgoal_id']} {v['verdict']} -> {v['route_to']}" if v else ""))
        final = (await app.aget_state(config)).values
        print(f"[{thread_id}] {final.get('status')}" + (f": {final['failure_reason']}" if final.get("failure_reason") else ""))
        return final

    async def run_concurrent_workload(self, tasks: list[dict[str, str]]) -> list[Any]:
        db = str(self.ws / ".gemma_checkpoints.db")  # hidden, so tools cannot read or write it
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_S) as http, AsyncSqliteSaver.from_conn_string(db) as cp:
            self.http = http
            try:
                await asyncio.to_thread(self.repo.index_all)
                app = self.build_graph(cp)  # one checkpointer shared by every thread
                return await asyncio.gather(*(self.run_thread(app, t["spec"], t["id"]) for t in tasks),
                                            return_exceptions=True)
            finally:
                self.http = None


if __name__ == "__main__":
    harness = GemmaTrinityHarness(workspace_dir=".")
    workload = [{"id": "thread-alpha",
                 "spec": "Fix the division-by-zero bug in math_utils.py and verify via pytest."}]
    for result in asyncio.run(harness.run_concurrent_workload(workload)):
        if isinstance(result, Exception):
            print(f"Thread crashed (resumable): {type(result).__name__}: {result}")