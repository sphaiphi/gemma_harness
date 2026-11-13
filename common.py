"""Shared helpers with no harness dependencies."""
from __future__ import annotations

import contextvars
import difflib
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Iterator

OUTPUT_CAP = 4000

# Thread id of the graph run in progress. Set by the work node; asyncio.to_thread copies it into worker threads.
CURRENT_THREAD: contextvars.ContextVar[str | None] = contextvars.ContextVar("gemma_thread", default=None)


class ModelError(Exception):
    """Base class for model-layer failures. Adapters raise only its subclasses."""


class ModelOutputError(ModelError):
    """The model answered, but not usably: bad JSON, empty output, or a 4xx such as an unknown model."""


class ModelUnavailable(ModelError):
    """The model server could not be reached, or kept returning 5xx, after all retries."""

