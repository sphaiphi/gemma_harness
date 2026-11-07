**How to extend it**
1. Change a budget or model: pass a config. For example: GemmaTrinityHarness(".", HarnessConfig(max_replans=5, models=ModelConfig(orchestrator="gemma3:27b"))).
2. Add a Tier 1 check: write a function def no_todo(vi): return ["TODO left in code"] if "TODO" in (vi.wo.result or "") else []. Then pass static_checks=(*DEFAULT_STATIC_CHECKS, no_todo).
3. Add a built-in tool: register it in tooling/builtin.py and add its name to BUILTIN_TOOLS, so synthesized tools can't use that name.
4. Add a subgoal kind, for example "docs":
 - add it to SubgoalKind in schemas.py,
 - write class DocsWorker(Worker): kind = "docs",
 - add it to WORKER_TYPES, or pass it with worker_types={**WORKER_TYPES, "docs": DocsWorker},
 - describe it in thinker_prompt().
5. React to file changes: call services.workspace.subscribe(fn). That's how the repo index stays current, and you can use it for a formatter or file watcher too.
6. Swap the LLM backend: write a class with the same session, generate, chat, chat_json and fim methods and put it into Services.
7. Unit-test a part: build Services yourself with fakes, for example a fake OllamaClient that returns canned JSON. Then pass it with GemmaTrinityHarness(services=...).

**Behavior differences to know about**
1. Logging instead of print: messages now go through logging. Run with python -m gemma_harness, which sets this up, or call logging.basicConfig(level=logging.INFO) yourself. Otherwise you won't see any output.
2. One restore path: a single-file rollback now also creates missing parent folders. It also skips unsafe paths instead of raising PermissionError.
3. The CLI: it takes spec, --workspace and --thread-id instead of the hard-coded workload. It exits with code 1 if a thread crashed or didn't complete.
4. The auto-loader: autoload() logs directly and returns the names it loaded, instead of returning log lines.
5. Still not done:
 - review step 6: startup runs on the event loop, ChromaDB name() is missing, and the full repo is re-embedded on every start,
 - a git worktree per thread.

