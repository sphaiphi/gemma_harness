"""Git integration: repo setup, baseline snapshot and one commit per accepted subgoal."""
from __future__ import annotations

import logging
import shutil
import subprocess
import textwrap
from pathlib import Path

from .config import GitConfig
from .schemas import Subgoal

log = logging.getLogger(__name__)


class GitRepo:
    def __init__(self, root: Path, cfg: GitConfig):
        self.root = root
        self.cfg = cfg
        self.enabled = cfg.enabled and shutil.which("git") is not None

    def run(self, *args: str) -> subprocess.CompletedProcess:
        name, email = self.cfg.author
        try:
            return subprocess.run(["git", "-c", f"user.name={name}", "-c", f"user.email={email}",
                                   "-c", "commit.gpgsign=false", *args],
                                  cwd=self.root, capture_output=True, text=True, timeout=self.cfg.timeout_s)
        except (OSError, subprocess.TimeoutExpired) as e:
            return subprocess.CompletedProcess(list(args), 127, "", str(e))

    def has_head(self) -> bool:
        return self.run("rev-parse", "--verify", "-q", "HEAD").returncode == 0

    def prepare(self) -> None:
        if self.run("rev-parse", "--is-inside-work-tree").returncode != 0:
            r = self.run("init", "-q")
            if r.returncode != 0:
                self.enabled = False
                log.warning("[git] git init failed; commits disabled: %s", r.stderr.strip()[:300])
                return
            log.info("[git] initialized repository in %s", self.root)
        gi = self.root / ".gitignore"
        lines = gi.read_text(encoding="utf-8").splitlines() if gi.is_file() else []
        if missing := [x for x in self.cfg.ignore_lines if x not in lines]:
            gi.write_text("\n".join([*lines, "# Gemma harness", *missing]) + "\n", encoding="utf-8")

    def baseline(self) -> None:
        if not self.cfg.baseline_commit or not self.run("status", "--porcelain", "--", ".").stdout.strip():
            return
        self.run("add", "-A", "--", ".")
        cmd = ["commit", "-q", "-m", "gemma: baseline snapshot before run"]
        if self.has_head():
            cmd += ["--", "."]
        r = self.run(*cmd)
        if r.returncode != 0:
            log.warning("[git] baseline commit failed: %s", (r.stderr or r.stdout).strip()[:300])
        else:
            log.info("[git] baseline snapshot committed")

    def commit_paths(self, paths: list[str], subject: str, body: str) -> str | None:
        paths = list(dict.fromkeys(paths))
        if not paths:
            return None
        for p in paths:
            r = self.run("add", "-A", "--", p)
            if r.returncode != 0:
                log.warning("[git] git add failed for %s: %s", p, r.stderr.strip()[:300])
        if self.run("diff", "--cached", "--quiet", "--", *paths).returncode == 0:
            return None
        cmd = ["commit", "-q", "-m", subject, "-m", body]
        if self.has_head():
            cmd += ["--", *paths]
        r = self.run(*cmd)
        if r.returncode != 0:
            log.warning("[git] commit failed for %s: %s", paths, (r.stderr or r.stdout).strip()[:300])
            return None
        return self.run("rev-parse", "--short", "HEAD").stdout.strip()


def commit_message(tid: str, plan_id: str | None, sg: Subgoal, event: dict) -> tuple[str, str]:
    tests = {True: "passed", False: "failed", None: "not run"}[event.get("tests_passed")]
    subject = f"gemma({tid}): {sg.id} [{sg.kind}] " + textwrap.shorten(sg.description, 50, placeholder="...")
    body = "\n".join([textwrap.fill(sg.description, 72), "",
                      f"Gemma-Thread: {tid}", f"Gemma-Plan: {plan_id}", f"Gemma-Subgoal: {sg.id}",
                      f"Gemma-Kind: {sg.kind}", f"Gemma-Attempt: {event['attempt']}", f"Gemma-Tests: {tests}"])
    return subject, body