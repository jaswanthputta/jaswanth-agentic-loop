"""Append-only JSON Lines logging for agent steps."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _summary(value: Any, limit: int = 1200) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    return text[:limit] + ("..." if len(text) > limit else "")


class StepLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        iteration: int,
        step: str,
        input_data: Any,
        output_data: Any,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "step": step,
            "input_summary": _summary(input_data),
            "output_summary": _summary(output_data),
            "latency_ms": round(latency_ms, 2),
            "error": error,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
