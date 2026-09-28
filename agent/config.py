"""Runtime configuration loaded from config.yaml and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: float
    max_retries: int
    base_delay_seconds: float
    max_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class HarnessConfig:
    max_iterations: int
    token_budget: int


@dataclass(frozen=True)
class MemoryConfig:
    backend: str
    persist_directory: str
    collection: str


@dataclass(frozen=True)
class LoggingConfig:
    path: str


@dataclass(frozen=True)
class AppConfig:
    mode: str
    llm: LLMConfig
    harness: HarnessConfig
    memory: MemoryConfig
    logging: LoggingConfig


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must contain a YAML mapping.")
    return data


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load settings, with environment variables taking precedence."""
    data = _read_config(Path(path))
    llm_data = data.get("llm", {})
    harness_data = data.get("harness", {})
    memory_data = data.get("memory", {})
    logging_data = data.get("logging", {})

    return AppConfig(
        mode=os.getenv("AGENT_MODE", str(data.get("mode", "demo"))).lower(),
        llm=LLMConfig(
            model=os.getenv("LLM_MODEL", str(llm_data.get("model", "gpt-4o-mini"))),
            base_url=os.getenv(
                "LLM_BASE_URL",
                str(llm_data.get("base_url", "https://api.openai.com/v1")),
            ).rstrip("/"),
            api_key_env=os.getenv(
                "LLM_API_KEY_ENV", str(llm_data.get("api_key_env", "OPENAI_API_KEY"))
            ),
            timeout_seconds=float(
                os.getenv(
                    "LLM_TIMEOUT_SECONDS",
                    str(llm_data.get("timeout_seconds", 30)),
                )
            ),
            max_retries=int(
                os.getenv("LLM_MAX_RETRIES", str(llm_data.get("max_retries", 3)))
            ),
            base_delay_seconds=float(
                os.getenv(
                    "LLM_BASE_DELAY_SECONDS",
                    str(llm_data.get("base_delay_seconds", 0.5)),
                )
            ),
            max_tokens=int(
                os.getenv("LLM_MAX_TOKENS", str(llm_data.get("max_tokens", 1024)))
            ),
            temperature=float(
                os.getenv("LLM_TEMPERATURE", str(llm_data.get("temperature", 0.2)))
            ),
            top_p=float(os.getenv("LLM_TOP_P", str(llm_data.get("top_p", 0.7)))),
        ),
        harness=HarnessConfig(
            max_iterations=int(
                os.getenv(
                    "AGENT_MAX_ITERATIONS",
                    str(harness_data.get("max_iterations", 10)),
                )
            ),
            token_budget=int(
                os.getenv(
                    "AGENT_TOKEN_BUDGET",
                    str(harness_data.get("token_budget", 12000)),
                )
            ),
        ),
        memory=MemoryConfig(
            backend=os.getenv(
                "AGENT_MEMORY_BACKEND",
                str(memory_data.get("backend", "chroma")),
            ).lower(),
            persist_directory=os.getenv(
                "AGENT_MEMORY_PATH",
                str(memory_data.get("persist_directory", ".agent_memory")),
            ),
            collection=os.getenv(
                "AGENT_MEMORY_COLLECTION",
                str(memory_data.get("collection", "interview_question_agent")),
            ),
        ),
        logging=LoggingConfig(
            path=os.getenv(
                "AGENT_LOG_PATH",
                str(logging_data.get("path", "logs/agent.jsonl")),
            )
        ),
    )
