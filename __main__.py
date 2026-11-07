"""CLI entry point: python -m gemma_harness "task spec" --workspace . --thread-id thread-alpha"""
from __future__ import annotations

import argparse
import asyncio
import logging

from .harness import GemmaTrinityHarness

DEFAULT_SPEC = "Fix the division-by-zero bug in math_utils.py and verify via pytest."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gemma_harness", description="Run the Gemma Trinity harness on one task.")
    ap.add_argument("spec", nargs="?", default=DEFAULT_SPEC, help="task description")
    ap.add_argument("--workspace", default=".", help="project directory to work in")
    ap.add_argument("--thread-id", default="thread-alpha", help="resumable thread id; use a new id for a new task")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    harness = GemmaTrinityHarness(args.workspace)
    results = asyncio.run(harness.run_concurrent_workload([{"id": args.thread_id, "spec": args.spec}]))
    code = 0
    for r in results:
        if isinstance(r, Exception):
            logging.getLogger("gemma_harness").error("Thread crashed (resumable): %s: %s", type(r).__name__, r)
            code = 1
        elif r.get("status") != "complete":
            code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())