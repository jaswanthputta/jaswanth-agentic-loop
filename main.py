"""Run the job-description interview question agent in demo or live mode."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from agent.config import AppConfig, load_config
from agent.llm import OpenAICompatibleClient
from agent.logger import StepLogger
from agent.loop import run_agent
from agent.memory_manager import MemoryManager
from agent.prompts import REASON_SYSTEM, REFLECT_SYSTEM
from agent.tools import build_tool_handlers

SAMPLE_JOB_DESCRIPTION = """Senior Backend Engineer
We are seeking a Senior Backend Engineer with 5+ years of experience building
reliable Python services. Responsibilities include designing REST APIs, improving
PostgreSQL data models, deploying services on AWS, and mentoring engineers.
Required skills include Python, distributed systems, automated testing, and
observability. The engineer will partner with product and security teams.
"""


class DemoLLM:
    """Deterministic stand-in used to run the full loop without API credentials."""

    def __init__(self) -> None:
        self.total_tokens_used = 0

    @staticmethod
    def _extract_json_after(marker: str, text: str) -> dict[str, Any]:
        content = text.split(marker, 1)[1]
        for boundary in (
            "\n\nRecalled memory:",
            "\n\nAvailable tools",
            "\n\nLatest action result:",
        ):
            content = content.split(boundary, 1)[0]
        content = content.strip()
        return json.loads(content)

    @staticmethod
    def _memory_records(prompt: str) -> list[dict[str, Any]]:
        marker = "Recalled memory:\n"
        if marker not in prompt:
            return []

        section = prompt.split(marker, 1)[1].split(
            "\n\nAvailable tools", 1
        )[0]

        try:
            records = json.loads(section)
        except json.JSONDecodeError:
            return []

        parsed: list[dict[str, Any]] = []

        for record in records:
            try:
                parsed.append(json.loads(record))
            except (TypeError, json.JSONDecodeError):
                continue

        return parsed

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_system_prompt: str | None = None,
        fallback_user_prompt: str | None = None,
    ) -> dict[str, Any]:

        self.total_tokens_used += max(1, len(user_prompt.split()) // 2)

        if system_prompt.startswith("ROLE: REASON"):
            observation = self._extract_json_after(
                "Observation:\n",
                user_prompt,
            )

            records = self._memory_records(user_prompt)
            actions = [record.get("action") for record in records]

            previous_action = observation.get(
                "previous_result",
                {},
            ).get("action")

            if (
                "review_question_coverage" in actions
                or previous_action == "generate_interview_questions"
            ):
                analysis = self._find_result(
                    records,
                    "analyze_job_description",
                )

                generated = observation.get(
                    "previous_result",
                    {},
                ).get("result", {})

                questions = generated.get("questions", [])

                if not questions:
                    questions = self._find_result(
                        records,
                        "generate_interview_questions",
                    ).get("questions", [])

                return {
                    "action": "review_question_coverage",
                    "parameters": {
                        "requirements": analysis.get(
                            "requirements",
                            [],
                        ),
                        "questions": questions,
                    },
                    "reasoning": (
                        "Recalled the stored requirements and drafted "
                        "questions; now validating coverage."
                    ),
                }

            if (
                "analyze_job_description" in actions
                or previous_action == "analyze_job_description"
            ):
                analysis = self._find_result(
                    records,
                    "analyze_job_description",
                )

                return {
                    "action": "generate_interview_questions",
                    "parameters": {
                        "job_description": observation["job_description"],
                        "requirements": analysis.get(
                            "requirements",
                            [],
                        ),
                        "question_count": 8,
                    },
                    "reasoning": (
                        "Using the requirements extracted and persisted "
                        "in the previous iteration."
                    ),
                }

            return {
                "action": "analyze_job_description",
                "parameters": {
                    "job_description": observation["job_description"],
                },
                "reasoning": (
                    "Extracting role competencies before drafting questions."
                ),
            }

        if system_prompt.startswith("ROLE: GENERATE"):
            parameters = self._extract_json_after(
                "Generate the question set using this context:\n",
                user_prompt,
            )

            requirements = parameters.get(
                "requirements",
                [],
            )

            count = int(
                parameters.get(
                    "question_count",
                    8,
                )
            )

            categories = [
                "technical",
                "system design",
                "testing",
                "behavioral",
            ]

            questions = []

            for index in range(count):
                requirement = (
                    requirements[index % len(requirements)]
                    if requirements
                    else "the role"
                )

                questions.append(
                    {
                        "question": (
                            f"Describe how you have handled "
                            f"{requirement.rstrip('.')} "
                            f"in a production setting."
                        ),
                        "category": categories[index % len(categories)],
                        "difficulty": (
                            "senior"
                            if index % 2
                            else "intermediate"
                        ),
                        "competency_tested": requirement,
                        "strong_answer_signals": [
                            "Explains personal contribution",
                            "Describes a measurable outcome",
                        ],
                        "follow_up": (
                            "What trade-off did you make, "
                            "and what would you change next time?"
                        ),
                    }
                )

            return {
                "questions": questions,
            }

        if system_prompt.startswith("ROLE: REFLECT"):
            payload = self._extract_json_after(
                "Latest action result:\n",
                user_prompt,
            )

            action = payload.get("action")

            if action == "review_question_coverage":
                report = payload.get(
                    "result",
                    {},
                )

                passed = bool(
                    report.get("review_passed")
                )

                return {
                    "is_done": passed,
                    "quality_score": int(
                        report.get(
                            "coverage_score",
                            0,
                        )
                    ),
                    "next_instruction": (
                        "Present the reviewed question set."
                        if passed
                        else "Revise questions to close the coverage gaps."
                    ),
                    "summary": (
                        "Coverage review passed."
                        if passed
                        else "Coverage review found gaps."
                    ),
                }

            if action == "generate_interview_questions":
                return {
                    "is_done": False,
                    "quality_score": 65,
                    "next_instruction": (
                        "Review the draft for requirement coverage "
                        "and duplicates."
                    ),
                    "summary": (
                        "Draft questions generated; a coverage review "
                        "is still required."
                    ),
                }

            return {
                "is_done": False,
                "quality_score": 25,
                "next_instruction": (
                    "Generate questions using the extracted requirements."
                ),
                "summary": (
                    "Requirements extracted and stored "
                    "for the next iteration."
                ),
            }

        raise ValueError(
            f"Unsupported demo prompt role: {system_prompt[:40]!r}"
        )

    @staticmethod
    def _find_result(
        records: list[dict[str, Any]],
        action: str,
    ) -> dict[str, Any]:

        for record in records:
            if record.get("action") == action:
                result = record.get(
                    "result_summary",
                    {},
                ).get(
                    "result",
                    {},
                )

                if isinstance(result, dict):
                    return result

        return {}


def _read_job_description_from_terminal(
    input_fn: Any = input,
    output_fn: Any = print,
) -> str:

    output_fn(
        "Paste the job description below. "
        "Type END on its own line when finished:"
    )

    lines: list[str] = []

    while True:
        try:
            line = input_fn()
        except EOFError:
            break

        if line.strip().upper() == "END":
            break

        lines.append(line)

    description = "\n".join(lines).strip()

    if not description:
        raise ValueError(
            "No job description was entered."
        )

    return description


def _load_job_description(
    args: argparse.Namespace,
) -> str:

    if args.job_description:
        description = args.job_description

    elif args.jd_file:
        description = Path(
            args.jd_file
        ).read_text(
            encoding="utf-8"
        )

    elif sys.stdin.isatty():
        description = _read_job_description_from_terminal()

    else:
        raise ValueError(
            "Provide --job-description or --jd-file, "
            "or run interactively in a terminal."
        )

    if not description.strip():
        raise ValueError(
            "The job description must not be empty."
        )

    return description


def _ensure_api_key(
    api_key_env: str,
    *,
    live: bool,
    prompt_fn: Any = getpass.getpass,
) -> None:

    if not live or os.getenv(api_key_env, "").strip():
        return

    api_key = prompt_fn(
        f"Enter API key for {api_key_env} (input hidden): "
    ).strip()

    if not api_key:
        raise ValueError(
            f"No API key provided for {api_key_env}."
        )

    os.environ[api_key_env] = api_key


def print_readable_result(
    result: dict[str, Any],
    session_id: str,
) -> None:
    """Print the final agent result in a clean human-readable format."""

    print("\n" + "=" * 60)
    print("        AI INTERVIEW QUESTION GENERATOR")
    print("=" * 60)

    print(f"\nSession: {session_id}")
    print(
        f"Status: {result.get('status', 'UNKNOWN')}"
    )
    print(
        f"Iterations: {result.get('iterations', 0)}"
    )

    reflection = result.get(
        "reflection",
        {},
    )

    final_result = result.get(
        "result",
        {},
    )

    tool_result = (
        final_result.get("result", {})
        if isinstance(final_result, dict)
        else {}
    )

    print("\n" + "-" * 60)
    print("FINAL RESULT")
    print("-" * 60)

    if isinstance(reflection, dict):
        print(
            f"Quality Score: "
            f"{reflection.get('quality_score', 'N/A')}"
        )

    if isinstance(tool_result, dict):
        print(
            f"Coverage: "
            f"{tool_result.get('coverage_score', 'N/A')}%"
        )

        print(
            f"Questions: "
            f"{tool_result.get('question_count', 'N/A')}"
        )

        gaps = tool_result.get(
            "gaps",
            [],
        )

        print(
            f"Gaps: {len(gaps)}"
        )

        print(
            f"Duplicates: "
            f"{tool_result.get('duplicate_count', 'N/A')}"
        )

        questions = tool_result.get(
            "questions",
            [],
        )

        if questions:
            print("\n" + "-" * 60)
            print("INTERVIEW QUESTIONS")
            print("-" * 60)

            for index, question in enumerate(
                questions,
                start=1,
            ):
                if isinstance(question, dict):
                    print(
                        f"\n{index}. "
                        f"{question.get('question', '')}"
                    )

                    print(
                        f"   Category: "
                        f"{question.get('category', '')}"
                    )

                    print(
                        f"   Difficulty: "
                        f"{question.get('difficulty', '')}"
                    )

    if isinstance(reflection, dict):
        summary = reflection.get(
            "summary",
            "",
        )

        if summary:
            print("\n" + "-" * 60)
            print("AGENT REFLECTION")
            print("-" * 60)
            print(summary)

    print("\n" + "=" * 60)
    print(
        f"FINAL STATUS: "
        f"{result.get('status', 'UNKNOWN')}"
    )
    print("=" * 60 + "\n")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration.",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the configured OpenAI-compatible LLM.",
    )

    parser.add_argument(
        "--job-description",
        help="Job description text.",
    )

    parser.add_argument(
        "--jd-file",
        help="UTF-8 file containing the job description.",
    )

    parser.add_argument(
        "--session-id",
        default=None,
        help="Memory session ID; generated if omitted.",
    )

    args = parser.parse_args()

    config: AppConfig = load_config(
        args.config
    )

    args.live = (
        args.live
        or config.mode == "live"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        _ensure_api_key(
            config.llm.api_key_env,
            live=args.live,
        )

    except (
        EOFError,
        KeyboardInterrupt,
        ValueError,
    ) as error:

        parser.error(str(error))

    llm = (
        OpenAICompatibleClient(config.llm)
        if args.live
        else DemoLLM()
    )

    if args.live and not llm.api_key:
        parser.error(
            f"Live mode requires an API key in the "
            f"{config.llm.api_key_env} environment variable. "
            "Set it in this PowerShell window before "
            "starting the program."
        )

    try:
        description = _load_job_description(
            args
        )

    except (
        OSError,
        ValueError,
    ) as error:

        parser.error(str(error))

    memory = MemoryManager(
        config.memory.persist_directory,
        config.memory.collection,
    )

    logger_instance = StepLogger(
        config.logging.path
    )

    session_id = (
        args.session_id
        or str(uuid.uuid4())
    )

    result = run_agent(
        description,
        llm=llm,
        memory=memory,
        tools=build_tool_handlers(llm),
        max_iterations=config.harness.max_iterations,
        token_budget=config.harness.token_budget,
        max_retries=config.llm.max_retries,
        retry_base_delay=config.llm.base_delay_seconds,
        logger_instance=logger_instance,
        session_id=session_id,
    )

    print_readable_result(
        result,
        session_id,
    )

    if result["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()