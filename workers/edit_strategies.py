"""Edit engines. EditWorker uses the first strategy whose applies() is true."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..common import ModelOutputError, strip_fences
from ..context import ContextRequest
from ..index import regen_start
from ..models import JSON_WORKER
from ..prompts import EDIT_PROMPT
from ..schemas import EditCode, HarnessState, Subgoal
from .locate import Target

if TYPE_CHECKING:
    from ..models import FimCompleter, JsonReasoner
    from ..services import Services


class EditStrategy(ABC):
    def __init__(self, services: "Services", model: Any):
        self.s = services
        self.model = model

    @property
    def engine(self) -> str:
        return self.model.engine

    @abstractmethod
    def applies(self, t: Target, attempt: int) -> bool: ...

    @abstractmethod
    async def edit(self, state: HarnessState, sg: Subgoal, t: Target) -> str:
        """Return the complete new file text. Raise ModelOutputError on unusable model output."""


class FimStrategy(EditStrategy):
    """Fill-in-the-middle for one small function body, first attempt only."""
    model: "FimCompleter"

    def applies(self, t: Target, attempt: int) -> bool:
        cfg = self.s.cfg
        return (not t.is_new and attempt < cfg.fim_attempts and t.size <= cfg.fim_max_lines
                and bool(regen_start(t.original, t.start, t.end)))

    async def edit(self, state: HarnessState, sg: Subgoal, t: Target) -> str:
        body_at = regen_start(t.original, t.start, t.end)
        before, after = "".join(t.lines[:t.start - 1]), "".join(t.lines[t.end:])
        header = "".join(t.lines[t.start - 1:body_at - 1])
        old_body = t.lines[body_at - 1:t.end]
        indent = re.match(r"[ \t]*", old_body[0]).group(0)
        guide = f"{indent}# TASK: {' '.join(sg.description.split())}\n" + "".join(
            f"{indent}# OLD| {ln.rstrip()}\n" for ln in old_body)
        middle = await self.model.complete(before + header + guide, after)
        middle = "".join(ln for ln in middle.splitlines(True)
                         if not ln.lstrip().startswith(("# OLD|", "# TASK:")))
        if not middle.strip():
            raise ModelOutputError(f"{self.engine} returned only guide comments")
        return before + header + middle.rstrip("\n") + "\n" + after


class RewriteStrategy(EditStrategy):
    """The json_worker model rewrites the span, or writes a new file. Always applies, so keep it last."""
    model: "JsonReasoner"

    def applies(self, t: Target, attempt: int) -> bool:
        return True

    async def edit(self, state: HarnessState, sg: Subgoal, t: Target) -> str:
        if t.is_new:
            before = after = ""
            task = f"The file {t.rel} does not exist yet. Return its complete content."
        else:
            before, after = "".join(t.lines[:t.start - 1]), "".join(t.lines[t.end:])
            task = (f"Replace lines {t.start}-{t.end} of {t.rel}:\n```python\n{''.join(t.lines[t.start - 1:t.end])}```\n"
                    "Return only the replacement for those lines, with their original indentation.")
        user = await self.s.contexts[JSON_WORKER].build(
            ContextRequest(state=state, sg=sg, extra={"target": t, "task": task}))
        out = await self.model.ask(EDIT_PROMPT, user, EditCode)
        return before + strip_fences(out.code).rstrip("\n") + "\n" + after