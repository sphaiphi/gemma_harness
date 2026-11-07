Here is a step-by-step breakdown with text architecture diagrams explaining how the **Gemma Trinity Harness** and its orchestrator models work together.

---

### Step 1: Understand the 5 Specialized Models

The harness avoids having a single model do everything. It assigns distinct responsibilities across 5 Gemma models:

| Model | Size | Primary Role | Triggered At |
|---|---|---|---|
| **T5Gemma** | 2B | Spec normalizer: transforms messy input into canonical task AST | `sanitize` node |
| **EmbeddingGemma** | 300M | Vector embedder: powers ChromaDB semantic search over repo & tools | Tool matching & code indexing |
| **Gemma 4** | 12B | System orchestrator: Thinker (planning), Tier 2 Verifier, complex Worker | `think`, `verify`, fallback `work` |
| **FunctionGemma** | 270M | High-speed tool router: outputs strict JSON function calls | `work` (when `kind="tool"`) |
| **CodeGemma** | 2B | Surgical code editor: Fill-In-The-Middle (FIM) code patches | `work` (when `kind="edit"`) |

---

### Step 2: System Flow & Core State Loop

The overall LangGraph lifecycle follows a deterministic loop controlled by code, not by an LLM supervisor.

```text
       [ User Raw Spec ]
               │
               ▼
      1. node_sanitize  ───────────► T5Gemma 2B (Clean AST)
               │
               ▼
       2. node_think    ◄──────────┐ Gemma 4 / 12B (Produces Plan)
               │                   │
               ▼                   │ [Re-plan / Revise Plan]
       3. node_work     ◄─────┐    │ (Max 3 Re-plans)
               │              │    │
               ▼              │    │
      4. node_verify          │    │
               │              │    │
               ▼              │    │
        5. node_gate  ────────┴────┘
         (Deterministic Code)
               │
               ├──────► [Next Subgoal] ──► loop back to node_work
               │
               └──────► [Task Complete] ─► Git Commit ──► END
```

1. **`sanitize`:** Ingests raw user requirements and outputs a structured specification.
2. **`think`:** G0emma 3 12B creates an ordered list of subgoals (`sg1`, `sg2`, ...), tagging each with a specific `kind`.
3. **`work`:** Dispatches execution to the appropriate model or tool based on `kind`.
4. **`verify`:** Inspects the worker's output through automated code tests first, then LLM evaluation.
5. **`gate`:** A purely deterministic Python node checking loop budgets, rolling back rejected files, committing accepted steps to Git, and selecting the next state.

---

### Step 3: Subgoal Execution by Worker Specialists

When `node_work` runs a subgoal, Python inspects the `kind` field to dispatch work to the right specialist:

```text
                      [ Subgoal from Plan ]
                                │
         ┌──────────────────────┼──────────────────────┬──────────────────────┐
         ▼                      ▼                      ▼                      ▼
  kind = "edit"          kind = "tool"          kind = "synth"         kind = "reason"
         │                      │                      │                      │
  EmbeddingGemma         FunctionGemma          Gemma 4 12B            Gemma 4 12B
 (Finds code chunk)     (Picks tool schema)   (Writes tool + tests)  (Structured output)
         │                      │                      │                      │
   ≤ 60 lines?                  │                      │                      │
   ├── Yes ─► CodeGemma FIM     │                      │                      │
   └── No  ─► Gemma 4 12B       │                      │                      │
         │                      │                      │                      │
         ▼                      ▼                      ▼                      ▼
 [Pre-write Lint]      [Isolated Execution]   [Mutation Check]      [Complete / Blocked]
```

1. **`edit` (Code Changes):**
   * EmbeddingGemma searches ChromaDB to find the target file and line span.
   * If the function is $\le 60$ lines and on its first try, **CodeGemma 2B FIM** generates the replacement.
   * If the code is larger or retrying, **Gemma 4 12B** writes the update.
   * The file is linted against forbidden calls *before* being written to disk.
2. **`tool` (Tool Invocations):**
   * Top 3 relevant tools are retrieved from ChromaDB.
   * **FunctionGemma 270M** extracts parameters via native tool-calling schemas.
   * Handled by built-in Python handlers (`read_file`, `write_file`, `run_pytest`).
3. **`synth` (New Tool Generation):**
   * **Gemma 4 12B** designs a new tool with a strict JSON schema, Python code, and companion pytest suite.
   * The tool is saved under `tools/` and verified in isolation.
4. **`reason` (Direct Analysis):**
   * **Gemma 4 12B** outputs textual answers, calculations, or architectural explanations.

---

### Step 4: Two-Tier Verification Engine

Verification combines deterministic static checks with LLM judgment to eliminate hallucinations:

```text
                      [ Worker Output ]
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
         │         Tier 1: Deterministic Checks        │
         │   (No LLM used - Instant Failure Return)    │
         ├─────────────────────────────────────────────┤
         │ • Worker marked 'blocked'?                  │
         │ • Python AST syntax errors?                 │
         │ • SecurityLinter forbidden imports/calls?   │
         │ • Tool execution errors?                    │
         │ • Tool mutant check failed?                 │
         │ • pytest suite failing?                     │
         └──────────────────────┬──────────────────────┘
                                │
               Passes Tier 1?   │
               ├── No  ─────────┼───────────────────► Verdict: REVISE (route_to: worker)
               │                │
               └── Yes          ▼
         ┌─────────────────────────────────────────────┐
         │         Tier 2: Gemma 4 12B Judgment        │
         ├─────────────────────────────────────────────┤
         │ • Compares diff/result with subgoal & notes │
         │ • On final subgoal: runs whole-task check   │
         │ • Confirms real evidence supports reasoning │
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
               Verdict: ACCEPT or REVISE
```

1. **Tier 1 (Automated Gate):** Python executes static linters, syntax parsers, and `pytest`. Any failure rejects the subgoal immediately and routes feedback to the worker without spending LLM tokens.
2. **Tier 2 (Gemma 4 12B):** If Tier 1 passes, the 12B model verifies that the result satisfies the original specification and planner assumptions. On the final subgoal, it checks whether all results combined resolve the task.

---

### Step 5: Persistence, Crash Recovery & Git Commits

The harness manages safe state synchronization across tasks and interruptions:

```text
[ Write Event ] ────► Record original file in .gemma_journal/<thread_id>.json
                             │
                             ▼
[ Node Gate ]   ────► Verdict = ACCEPT?
                             │
              ├── YES ───────┼─────────────── NO ──────────────┐
              │                              │                 │
              ▼                              ▼                 ▼
   Commit changed paths only          Restore original   Clear journal
   git commit -m "gemma(...)"        files from journal
              │                              │                 │
              └──────────────────────────────┴─────────────────┘
                                     │
                             Checkpoints saved
                        (.gemma_checkpoints.db)
```

1. **Crash Journal:** Before touching any file, the harness logs its pre-write content in `.gemma_journal/`. If the process crashes mid-step, restarting reverts uncommitted files automatically.
2. **Scoped Git Commits:** When a subgoal is accepted, only the touched files are staged and committed with metadata tags (`Gemma-Thread`, `Gemma-Plan`, `Gemma-Subgoal`).
3. **SQLite Checkpointing:** LangGraph saves the conversation and thread variables to `.gemma_checkpoints.db` after every state transition, enabling resumption at any node.

