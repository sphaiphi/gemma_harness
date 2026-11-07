"""Edit worker: CodeGemma FIM for small function bodies on the first attempt, Gemma 3 12B otherwise."""
from __future__ import annotations

import asyncio
import re

from ..common import ModelOutputError, elide, make_diff, strip_fences
from ..index import regen_start
from ..prompts import EDIT_PROMPT
from ..schemas import EditCode, HarnessState, Subgoal, WorkerOutput
from ..security import lint
from .base import Worker, WorkResult, blocked, failed, worker_context

PATH_RE = re.compile(r"[\w./-]+\.py\b")


class EditWorker(Worker):
    kind = "edit"

    async def run(self, state: HarnessState, sg: Subgoal, attempt: int) -> WorkResult:
        ws, cfg = self.s.workspace, self.s.cfg
        try:
            rel, start, end = await self._locate(sg)
            path = ws.safe_path(rel)
        except (LookupError, PermissionError) as e:
            return blocked(sg, str(e))

        original = ws.snapshot(path)
        lines = (original or "").splitlines(keepends=True)
        if original is not None and start is None:
            start, end = 1, max(1, len(lines))
        meta: dict = {"path": rel, "original": original}
        if original is not None and end - start + 1 > cfg.max_rewrite_lines:
            return blocked(sg, f"{rel} is too large to rewrite whole")

        try:
            body_at = regen_start(original, start, end) if original is not None else None
            if body_at and end - start + 1 <= cfg.fim_max_lines and attempt < cfg.fim_attempts:
                new_text, engine = await self._fim(sg, lines, start, end, body_at)
            else:
                new_text, engine = await self._rewrite(state, sg, rel, original, lines, start, end)
        except ModelOutputError as e:
            return failed(sg, str(e), meta)

        span = f"lines {start}-{end}" if original is not None else "new file"
        out = WorkerOutput(subgoal_id=sg.id, status="complete", result=make_diff(rel, original, new_text),
                           reasoning=f"{engine} rewrote {span} of {rel}.")
        errors = lint(new_text) if rel.endswith(".py") else []
        if new_text == (original or ""):
            errors.append("edit changed nothing")
        if errors:
            return out, {**meta, "static_errors": errors}
        async with self.s.write_lock:
            await asyncio.to_thread(ws.write, path, new_text)
        return out, {**meta, "written": True}

    async def _locate(self, sg: Subgoal) -> tuple[str, int | None, int | None]:
        ws, repo = self.s.workspace, self.s.repo
        for m in PATH_RE.findall(f"{sg.description}\n{sg.notes}"):
            try:
                rel = ws.rel(ws.safe_path(m))
            except PermissionError:
                continue
            if (ws.root / rel).is_file():
                hits = await asyncio.to_thread(repo.search, sg.description, 1, rel)
                return (rel, hits[0]["start"], hits[0]["end"]) if hits else (rel, None, None)
            return rel, None, None
        hits = await asyncio.to_thread(repo.search, sg.description, 1)
        if hits:
            return hits[0]["path"], hits[0]["start"], hits[0]["end"]
        raise LookupError("no target file named in the subgoal and nothing relevant in repo index")

    async def _fim(self, sg: Subgoal, lines: list[str], start: int, end: int, body_at: int) -> tuple[str, str]:
        before, after = "".join(lines[:start - 1]), "".join(lines[end:])
        header = "".join(lines[start - 1:body_at - 1])
        old_body = lines[body_at - 1:end]
        indent = re.match(r"[ \t]*", old_body[0]).group(0)
        guide = f"{indent}# TASK: {' '.join(sg.description.split())}\n" + "".join(
            f"{indent}# OLD| {ln.rstrip()}\n" for ln in old_body)
        middle = await self.s.llm.fim(before + header + guide, after)
        middle = "".join(ln for ln in middle.splitlines(True)
                         if not ln.lstrip().startswith(("# OLD|", "# TASK:")))
        if not middle.strip():
            raise ModelOutputError("CodeGemma returned empty completion")
        return before + header + middle.rstrip("\n") + "\n" + after, "CodeGemma FIM"

    async def _rewrite(self, state: HarnessState, sg: Subgoal, rel: str, original: str | None,
                       lines: list[str], start: int | None, end: int | None) -> tuple[str, str]:
        if original is None:
            before = after = ""
            task = f"The file {rel} does not exist yet. Return its complete content."
        else:
            before, after = "".join(lines[:start - 1]), "".join(lines[end:])
            task = (f"File {rel} for context:\n```python\n{elide(original, 12000)}\n```\n\n"
                    f"Replace lines {start}-{end}:\n```python\n{''.join(lines[start - 1:end])}```\n"
                    "Return only the replacement for those lines, with their original indentation.")
        out = await self.s.llm.chat_json("orchestrator", EDIT_PROMPT,
                                         worker_context(state, sg) + "\n\n" + task, EditCode)
        return before + strip_fences(out.code).rstrip("\n") + "\n" + after, "Gemma 3 12B"