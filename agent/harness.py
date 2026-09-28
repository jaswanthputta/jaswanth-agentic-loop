"""Retries and small guardrail utilities for model and tool calls."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError

T = TypeVar("T")


class RateLimitError(RuntimeError):
    """The remote model rejected a request due to rate limiting."""


class MalformedResponseError(ValueError):
    """The model response did not match the expected JSON contract."""


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (RateLimitError, TimeoutError, ConnectionError)):
        return True
    if isinstance(error, HTTPError):
        return error.code == 429 or 500 <= error.code < 600
    return isinstance(error, URLError) and isinstance(error.reason, TimeoutError)


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter: float = 0.2,
    **kwargs: Any,
) -> T:
    """Retry transient failures using exponential backoff with bounded jitter."""
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")
    if base_delay < 0 or jitter < 0:
        raise ValueError("base_delay and jitter cannot be negative.")

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay + random.uniform(0.0, jitter * delay))
    raise RuntimeError("Retry loop exited unexpectedly.")
