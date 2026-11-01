import os
import json
import re
import ast
import subprocess
import asyncio
import httpx
import aiosqlite
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

OLLAMA_HOST = "http://localhost:11434"

# --- 1. Embedding Function ---
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "embedding-gemma:300m-custom", host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.host = host

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = httpx.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30.0
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

# --- 2. Vector-Indexed Tool Registry ---
class VectorIndexedToolRegistry:
    def __init__(self, workspace_dir: Path, embedding_fn: Any, host: str = OLLAMA_HOST):
        self.workspace_dir = workspace_dir
        self.embedding_fn = embedding_fn
        self.host = host
        self.client = chromadb.PersistentClient(path=str(workspace_dir / ".gemma_vector_db"))
        self.collection = self.client.get_or_create_collection(
            name="tools_registry",
            embedding_function=self.embedding_fn
        )
        self._tool_handlers = {}

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], handler_fn: Callable):
        self._tool_handlers[name] = handler_fn
        self.collection.upsert(
            documents=[description],
            metadatas=[{"name": name, "schema": json.dumps({"name": name, "description": description, "parameters": parameters})}],
            ids=[name]
        )

    def get_relevant_tools(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        relevant_schemas = []
        if results.get("metadatas") and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                relevant_schemas.append(json.loads(meta["schema"]))
        return relevant_schemas

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name in self._tool_handlers:
            return self._tool_handlers[name](**arguments)
        return f"Error: Tool handler '{name}' not found."

# --- 3. AST Security Linter ---
class ToolSecurityLinter(ast.NodeVisitor):
    def __init__(self):
        self.errors = []
        self.allowed_modules = {"pathlib", "json", "re", "typing", "collections", "datetime"}
        self.forbidden_functions = {"eval", "exec", "globals", "locals", "__import__", "breakpoint"}
        self.forbidden_attributes = {"system", "popen", "rmtree"}

    def analyze(self, code_string: str) -> Tuple[bool, List[str]]:
        self.errors = []
        try:
            tree = ast.parse(code_string)
            self.visit(tree)
        except SyntaxError as e:
            return False, [f"Syntax Error: {e}"]
        return len(self.errors) == 0, self.errors

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name.split('.')[0] not in self.allowed_modules:
                self.errors.append(f"Unauthorized module import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and node.module.split('.')[0] not in self.allowed_modules:
            self.errors.append(f"Unauthorized module import from: '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in self.forbidden_functions:
            self.errors.append(f"Forbidden function call: '{node.func.id}()'")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in self.forbidden_attributes:
            self.errors.append(f"Forbidden method call: '.{node.func.attr}()'")
        self.generic_visit(node)

# --- 4. Main Asynchronous Gemma Harness ---
class AsyncGemmaSpecHarness:
    def __init__(self, workspace_dir: str = ".", host: str = OLLAMA_HOST, max_concurrent_llm: int = 3):
        self.host = host
        self.workspace_dir = Path(workspace_dir).resolve()
        self.llm_semaphore = asyncio.Semaphore(max_concurrent_llm)
        
        self.models = {
            "transformer": "t5gemma:2b-custom",
            "embedder": "embedding-gemma:300m-custom",
            "orchestrator": "gemma3:12b-custom",
            "function": "functiongemma:270m-custom",
            "coder": "codegemma:2b-custom"
        }

        # Setup Vector Tool Registry
        self.embedding_fn = OllamaEmbeddingFunction(model_name=self.models["embedder"], host=self.host)
        self.tool_registry = VectorIndexedToolRegistry(self.workspace_dir, self.embedding_fn, self.host)
        
        # Register Built-in Tools
        self._register_default_tools()

    def _register_default_tools(self):
        def read_file(path: str) -> str:
            target = (self.workspace_dir / path).resolve()
            if not target.exists(): return f"Error: File {path} not found."
            return target.read_text(encoding="utf-8")

        def write_file(path: str, content: str) -> str:
            target = (self.workspace_dir / path).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {path}"

        def run_pytest() -> str:
            res = subprocess.run(["pytest", "-v"], capture_output=True, text=True)
            return json.dumps({"exit_code": res.returncode, "stdout": res.stdout, "stderr": res.stderr})

        self.tool_registry.register_tool("read_file", "Reads local workspace file content.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, read_file)
        self.tool_registry.register_tool("write_file", "Writes content to a workspace file.", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}, write_file)
        self.tool_registry.register_tool("run_pytest", "Runs local pytest test suite.", {"type": "object", "properties": {}}, run_pytest)

    async def query_model_async(self, model_key: str, prompt: str, system: str = "", format_json: bool = False) -> str:
        payload = {"model": self.models[model_key], "prompt": prompt, "stream": False}
        if system: payload["system"] = system
        if format_json: payload["format"] = "json"

        async with self.llm_semaphore:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{self.host}/api/generate", json=payload)
                res.raise_for_status()
                return res.json().get("response", "").strip()

    async def sanitize_spec_async(self, raw_spec: str) -> str:
        prompt = f"Normalize into structured task AST:\n\n{raw_spec}"
        return await self.query_model_async("transformer", prompt)

    async def route_tool_dynamic_async(self, plan: str) -> Dict[str, Any]:
        selected_tools = self.tool_registry.get_relevant_tools(query=plan, n_results=2)
        messages = [
            {"role": "system", "content": "Select the correct tool based on the plan."},
            {"role": "user", "content": f"Execution Plan: {plan}"}
        ]
        payload = {
            "model": self.models["function"],
            "messages": messages,
            "tools": selected_tools,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        async with self.llm_semaphore:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(f"{self.host}/api/chat", json=payload)
                res.raise_for_status()
                tool_calls = res.json().get("message", {}).get("tool_calls", [])
                if tool_calls:
                    fc = tool_calls[0].get("function", {})
                    return {"name": fc.get("name"), "arguments": fc.get("arguments", {})}
        return {"name": "run_pytest", "arguments": {}}

    async def complete_code_fim_async(self, prefix: str, suffix: str) -> str:
        fim_prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        payload = {
            "model": self.models["coder"],
            "prompt": fim_prompt,
            "stream": False,
            "options": {"temperature": 0.1, "stop": ["<|file_separator|>"]}
        }
        async with self.llm_semaphore:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(f"{self.host}/api/generate", json=payload)
                res.raise_for_status()
                return res.json().get("response", "").strip()

    async def build_async_workflow(self, checkpointer):
        workflow = StateGraph(dict)

        async def node_sanitize(state):
            ast = await self.sanitize_spec_async(state["raw_spec"])
            return {"structured_ast": ast}

        async def node_plan(state):
            plan = await self.query_model_async("orchestrator", f"Create execution plan for AST:\n{state['structured_ast']}")
            return {"execution_plan": plan}

        async def node_route(state):
            tool_call = await self.route_tool_dynamic_async(state["execution_plan"])
            return {"tool_call": tool_call}

        async def node_execute(state):
            tc = state["tool_call"]
            output = self.tool_registry.execute(tc.get("name"), tc.get("arguments", {}))
            return {"tool_output": output}

        workflow.add_node("sanitize", node_sanitize)
        workflow.add_node("plan", node_plan)
        workflow.add_node("route", node_route)
        workflow.add_node("execute", node_execute)

        workflow.set_entry_point("sanitize")
        workflow.add_edge("sanitize", "plan")
        workflow.add_edge("plan", "route")
        workflow.add_edge("route", "execute")
        workflow.add_edge("execute", END)

        return workflow.compile(checkpointer=checkpointer)

    async def run_thread(self, raw_spec: str, thread_id: str):
        db_path = str(self.workspace_dir / "gemma_checkpoints.db")
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            app = await self.build_async_workflow(checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            print(f"[{thread_id}] Starting persistent workflow thread...")
            async for event in app.astream({"raw_spec": raw_spec}, config):
                print(f"[{thread_id}] State update: {list(event.keys())}")
            print(f"[{thread_id}] Workflow completed successfully.")

    async def run_concurrent_workload(self, tasks: List[Dict[str, str]]):
        coros = [self.run_thread(t["spec"], t["id"]) for t in tasks]
        await asyncio.gather(*coros)

if __name__ == "__main__":
    harness = AsyncGemmaSpecHarness()
    workload = [
        {"id": "thread-alpha", "spec": "Fix division-by-zero bug in math_utils.py and verify via pytest."}
    ]
    asyncio.run(harness.run_concurrent_workload(workload))