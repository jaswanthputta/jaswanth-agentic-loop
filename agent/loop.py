"""Explicit Perceive -> Reason -> Act -> Reflect agent loop."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from agent.harness import MalformedResponseError, call_with_retry
from agent.logger import StepLogger
from agent.prompts import (
    REASON_FALLBACK_SYSTEM,
    REASON_SYSTEM,
    REFLECT_FALLBACK_SYSTEM,
    REFLECT_SYSTEM,
    reason_user_prompt,
    reflection_user_prompt,
)
from agent.tools import TOOL_DEFINITIONS, JSONLLM

logger = logging.getLogger(__name__)


class Memory(Protocol):
    def recall(self, session_id: str, query: str, top_k: int = 3) -> list[str]: ...

    def save(self, session_id: str, content: str, metadata: dict[str, Any]) -> None: ...


def perceive(input_data: str) -> dict[str, Any]:
    """Parse raw job description or the JSON envelope passed between iterations."""
    try:
        parsed = json.loads(input_data)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict) and "job_description" in parsed:
        job_description = str(parsed["job_description"])
        previous_reflection = parsed.get("previous_reflection", {})
        previous_result = parsed.get("previous_result", {})
    else:
        job_description = input_data.strip()
        previous_reflection = {}
        previous_result = {}

    if not job_description:
        raise ValueError("A non-empty job description is required.")
    return {
        "intent": "Generate a balanced, job-relevant interview question set.",
        "job_description": job_description,
        "constraints": {
            "ground_questions_in_description": True,
            "include_evaluation_signals": True,
            "avoid_duplicates": True,
        },
        "previous_reflection": previous_reflection,
        "previous_result": previous_result,
    }


def _parse_plan(payload: dict[str, Any]) -> dict[str, Any]:
    action = payload.get("action")
    parameters = payload.get("parameters")
    if not isinstance(action, str) or not isinstance(parameters, dict):
        raise MalformedResponseError("Plan must include a string action and object parameters.")
    return {
        "action": action,
        "parameters": parameters,
        "reasoning": str(payload.get("reasoning", "")),
    }


def reason(
    observation: dict[str, Any],
    memory: list[str],
    llm: JSONLLM | None = None,
) -> dict[str, Any]:
    """Recall context (performed by the loop) and ask the LLM for the next action."""
    if llm is None:
        raise ValueError("reason requires an LLM client.")
    user_prompt = reason_user_prompt(observation, memory, TOOL_DEFINITIONS)
    payload = llm.complete_json(
        REASON_SYSTEM,
        user_prompt,
        fallback_system_prompt=REASON_FALLBACK_SYSTEM,
        fallback_user_prompt=(
            "Select one tool for this task. Observation:\n"
            + json.dumps(observation, ensure_ascii=False)
        ),
    )
    return _parse_plan(payload)


def act(
    plan: dict[str, Any],
    tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    *,
    max_retries: int = 0,
    base_delay: float = 0.5,
) -> dict[str, Any]:
    """Execute one registered tool, returning explicit success or error information."""
    action = plan.get("action")
    parameters = plan.get("parameters", {})
    if not isinstance(action, str) or action not in tools:
        return {"ok": False, "action": action, "error": f"Unknown tool action: {action!r}"}
    if not isinstance(parameters, dict):
        return {"ok": False, "action": action, "error": "Tool parameters must be an object."}
    try:
        result = call_with_retry(
            tools[action],
            parameters,
            max_retries=max_retries,
            base_delay=base_delay,
        )
    except Exception as error:
        logger.exception("Tool %s failed.", action)
        return {"ok": False, "action": action, "error": f"{type(error).__name__}: {error}"}
    return {"ok": True, "action": action, "result": result}


def reflect(
    result: dict[str, Any],
    observation: dict[str, Any],
    llm: JSONLLM | None = None,
) -> dict[str, Any]:
    """Evaluate progress and return a structured continuation instruction."""
    if llm is None:
        raise ValueError("reflect requires an LLM client.")
    payload = llm.complete_json(
        REFLECT_SYSTEM,
        reflection_user_prompt(result, observation),
        fallback_system_prompt=REFLECT_FALLBACK_SYSTEM,
        fallback_user_prompt=(
            "Assess whether reviewed interview questions meet the goal. Latest result:\n"
            + json.dumps(result, ensure_ascii=False)
        ),
    )
    is_done = payload.get("is_done")
    quality_score = payload.get("quality_score")
    if not isinstance(is_done, bool) or not isinstance(quality_score, (int, float)):
        raise MalformedResponseError("Reflection requires boolean is_done and numeric quality_score.")
    if not 0 <= quality_score <= 100:
        raise MalformedResponseError("Reflection quality_score must be between 0 and 100.")
    return {
        "is_done": is_done,
        "quality_score": int(quality_score),
        "next_instruction": str(payload.get("next_instruction", "")),
        "summary": str(payload.get("summary", "")),
    }


def _serialize_next_input(
    job_description: str,
    reflection: dict[str, Any],
    result: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "job_description": job_description,
            "previous_reflection": reflection,
            "previous_result": result,
        },
        ensure_ascii=False,
    )


def run_agent(
    job_description: str,
    *,
    llm: JSONLLM,
    memory: Memory,
    tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    max_iterations: int = 10,
    token_budget: int = 12000,
    max_retries: int = 3,
    retry_base_delay: float = 0.5,
    logger_instance: StepLogger | None = None,
    session_id: str = "default",
) -> dict[str, Any]:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    if token_budget < 1:
        raise ValueError("token_budget must be at least 1.")

    current_input = job_description
    previous_reflection: dict[str, Any] | None = None
    previous_result: dict[str, Any] = {}
    previous_reflection_signature: str | None = None
    repeated_reflections = 0
    latest_result: dict[str, Any] = {}
    latest_reflection: dict[str, Any] = {}
    memory_events: list[dict[str, Any]] = []
    iterations = 0
    start_token_count = int(getattr(llm, "total_tokens_used", 0))

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        step_start = time.perf_counter()
        observation = perceive(current_input)
        if previous_reflection is not None:
            observation["previous_reflection"] = previous_reflection
            observation["previous_result"] = previous_result
        if logger_instance:
            logger_instance.log(
                iteration=iteration,
                step="perceive",
                input_data=current_input,
                output_data=observation,
                latency_ms=(time.perf_counter() - step_start) * 1000,
            )

        recall_start = time.perf_counter()
        try:
            memories = memory.recall(
                session_id, observation["job_description"], top_k=3
            )
        except Exception as error:
            logger.warning("Memory recall failed; continuing without memory: %s", error)
            memories = []
        memory_events.append(
            {"iteration": iteration, "recalled_count": len(memories), "recalled": memories}
        )
        if logger_instance:
            logger_instance.log(
                iteration=iteration,
                step="reason",
                input_data={"observation": observation, "memory": memories},
                output_data="",
                latency_ms=(time.perf_counter() - recall_start) * 1000,
                error=None,
            )

        reason_start = time.perf_counter()
        reason_error: str | None = None
        try:
            plan = reason(observation, memories, llm)
        except Exception as error:
            reason_error = f"{type(error).__name__}: {error}"
            failure = {
                "status": "FAILED",
                "iterations": iteration,
                "error": f"Reasoning/LLM request failed: {reason_error}",
                "result": latest_result,
                "token_usage": int(getattr(llm, "total_tokens_used", 0))
                - start_token_count,
                "memory_events": memory_events,
            }
            if logger_instance:
                logger_instance.log(
                    iteration=iteration,
                    step="reason",
                    input_data={"observation": observation, "memory_count": len(memories)},
                    output_data=failure,
                    latency_ms=(time.perf_counter() - reason_start) * 1000,
                    error=reason_error,
                )
            return failure
        if logger_instance:
            logger_instance.log(
                iteration=iteration,
                step="reason",
                input_data={"observation": observation, "memory_count": len(memories)},
                output_data=plan,
                latency_ms=(time.perf_counter() - reason_start) * 1000,
                error=reason_error,
            )

        act_start = time.perf_counter()
        action_result = act(
            plan,
            tools,
            max_retries=max_retries,
            base_delay=retry_base_delay,
        )
        if logger_instance:
            logger_instance.log(
                iteration=iteration,
                step="act",
                input_data=plan,
                output_data=action_result,
                latency_ms=(time.perf_counter() - act_start) * 1000,
                error=action_result.get("error"),
            )

        reflect_start = time.perf_counter()
        reflection_error: str | None = None
        try:
            latest_reflection = reflect(action_result, observation, llm)
        except Exception as error:
            reflection_error = f"{type(error).__name__}: {error}"
            latest_reflection = {
                "is_done": False,
                "quality_score": 0,
                "next_instruction": "Recover from the reflection error and continue with a safe next action.",
                "summary": f"Reflection could not complete: {reflection_error}",
            }
        if logger_instance:
            logger_instance.log(
                iteration=iteration,
                step="reflect",
                input_data=action_result,
                output_data=latest_reflection,
                latency_ms=(time.perf_counter() - reflect_start) * 1000,
                error=reflection_error,
            )
        if reflection_error:
            return {
                "status": "FAILED",
                "iterations": iteration,
                "error": f"Reflection/LLM request failed: {reflection_error}",
                "result": action_result,
                "reflection": latest_reflection,
                "token_usage": int(getattr(llm, "total_tokens_used", 0))
                - start_token_count,
                "memory_events": memory_events,
            }

        latest_result = action_result
        try:
            memory.save(
                session_id,
                json.dumps(
                    {
                        "iteration": iteration,
                        "action": plan.get("action"),
                        "reflection": latest_reflection,
                        "result_summary": action_result,
                    },
                    ensure_ascii=False,
                ),
                {"iteration": iteration, "action": str(plan.get("action", ""))},
            )
        except Exception as error:
            logger.warning("Memory write failed; continuing without persistence: %s", error)

        signature = json.dumps(latest_reflection, sort_keys=True, ensure_ascii=False)
        if signature == previous_reflection_signature:
            repeated_reflections += 1
        else:
            repeated_reflections = 0
        current_tokens = int(getattr(llm, "total_tokens_used", 0)) - start_token_count
        if current_tokens > token_budget:
            logger.warning(
                "Cumulative token usage %s exceeded configured budget %s.",
                current_tokens,
                token_budget,
            )
        if repeated_reflections >= 1:
            return {
                "status": "STUCK",
                "iterations": iterations,
                "result": latest_result,
                "reflection": latest_reflection,
                "token_usage": int(getattr(llm, "total_tokens_used", 0)) - start_token_count,
                "memory_events": memory_events,
            }
        if latest_reflection["is_done"]:
            return {
                "status": "DONE",
                "iterations": iterations,
                "result": latest_result,
                "reflection": latest_reflection,
                "token_usage": int(getattr(llm, "total_tokens_used", 0)) - start_token_count,
                "memory_events": memory_events,
            }

        previous_reflection = latest_reflection
        previous_result = action_result
        current_input = _serialize_next_input(
            observation["job_description"], latest_reflection, action_result
        )

    return {
        "status": "PARTIAL",
        "iterations": iterations,
        "result": latest_result,
        "reflection": latest_reflection,
        "token_usage": int(getattr(llm, "total_tokens_used", 0)) - start_token_count,
        "memory_events": memory_events,
    }
