"""Edit engines. EditWorker uses the first strategy whose applies() is true."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from ..common import ModelOutputError, elide, strip_fences
from ..index import regen_start
from ..prompts import EDIT_PROMPT
from ..schemas import EditCode, HarnessState, Subgoal
from .base import worker_context
from .locate import Target

if TYPE_CHECKING:
    from ..services import Services


class EditStrategy(ABC):
    engine: ClassVar[str]

    def __init__(self, services: "Services"):
        self.s = services

    @abstractmethod
    def applies(self, t: Target, attempt: int) -> bool: ...

    @abstractmethod
    async def edit(self, state: HarnessState, sg: Subgoal, t: Target) -> str:
        """Return the complete new file text. Raise ModelOutputError on unusable model output."""


class FimStrategy(EditStrategy):
    """CodeGemma fill-in-the-middle for one small function body, first attempt only."""
    engine = "CodeGemma FIM"

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
        middle = await self.s.llm.fim(before + header + guide, after)
        middle = "".join(ln for ln in middle.splitlines(True)
                         if not ln.lstrip().startswith(("# OLD|", "# TASK:")))
        if not middle.strip():
            raise ModelOutputError("CodeGemma returned empty completion")
        return before + header + middle.rstrip("\n") + "\n" + after


class RewriteStrategy(EditStrategy):
    """Gemma 3 12B rewrites the span, or writes a new file. Always applies, so keep it last."""
    engine = "Gemma 3 12B"

    def applies(self, t: Target, attempt: int) -> bool:
        return True

    async def edit(self, state: HarnessState, sg: Subgoal, t: Target) -> str:
        if t.is_new:
            before = after = ""
            task = f"The file {t.rel} does not exist yet. Return its complete content."
        else:
            before, after = "".join(t.lines[:t.start - 1]), "".join(t.lines[t.end:])
            task = (f"File {t.rel} for context:\n```python\n{elide(t.original, 12000)}\n```\n\n"
                    f"Replace lines {t.start}-{t.end}:\n```python\n{''.join(t.lines[t.start - 1:t.end])}```\n"
                    "Return only the replacement for those lines, with their original indentation.")
        out = await self.s.llm.chat_json("orchestrator", EDIT_PROMPT,
                                         worker_context(state, sg) + "\n\n" + task, EditCode)
        return before + strip_fences(out.code).rstrip("\n") + "\n" + after