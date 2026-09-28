"""Use-case tools: requirement extraction, question generation, and review."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from agent.harness import MalformedResponseError
from agent.prompts import (
    GENERATE_FALLBACK_SYSTEM,
    GENERATE_SYSTEM,
    generation_user_prompt,
)


class JSONLLM(Protocol):
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_system_prompt: str | None = None,
        fallback_user_prompt: str | None = None,
    ) -> dict[str, Any]: ...


TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "analyze_job_description": {
        "description": "Extract role context, responsibilities, and explicit skills from a job description.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_description": {"type": "string"},
            },
            "required": ["job_description"],
            "additionalProperties": False,
        },
    },
    "generate_interview_questions": {
        "description": "Generate or revise a structured interview question set grounded in the job description.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_description": {"type": "string"},
                "requirements": {"type": "array", "items": {"type": "string"}},
                "question_count": {"type": "integer", "minimum": 4, "maximum": 12},
                "previous_questions": {"type": "array"},
                "review_feedback": {"type": "object"},
            },
            "required": ["job_description", "requirements", "question_count"],
            "additionalProperties": True,
        },
    },
    "review_question_coverage": {
        "description": "Check relevance, duplication, and requirement coverage for generated questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "requirements": {"type": "array", "items": {"type": "string"}},
                "questions": {"type": "array"},
            },
            "required": ["requirements", "questions"],
            "additionalProperties": False,
        },
    },
}


def _sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [re.sub(r"^[\s*\-\u2022]+", "", item).strip() for item in pieces if item.strip()]


def analyze_job_description(parameters: dict[str, Any]) -> dict[str, Any]:
    description = str(parameters.get("job_description", "")).strip()
    if not description:
        raise ValueError("job_description must not be empty.")

    sentences = _sentences(description)
    cues = re.compile(
        r"\b(?:required|requirements?|qualifications?|responsibilit(?:y|ies)|"
        r"must|experience|proficien(?:t|cy)|skills?|knowledge|ability|"
        r"develop|design|lead|manage|build|support|years?)\b",
        re.IGNORECASE,
    )
    requirements = [sentence for sentence in sentences if cues.search(sentence)]
    if not requirements:
        requirements = sentences
    requirements = list(dict.fromkeys(requirements))[:15]

    years = re.findall(r"\b\d+\+?\s+years?\b", description, flags=re.IGNORECASE)
    title_candidates = [
        line.strip()
        for line in description.splitlines()
        if line.strip() and len(line.strip()) <= 90
    ]
    return {
        "role_context": title_candidates[0] if title_candidates else "Unspecified role",
        "requirements": requirements,
        "experience_signals": list(dict.fromkeys(years)),
        "source_sentence_count": len(sentences),
        "note": "Requirement extraction is a first-pass heuristic; validate against the source description.",
    }


def _validate_questions(payload: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise MalformedResponseError("Question generation response must contain a non-empty questions list.")
    normalized: list[dict[str, Any]] = []
    required_fields = {
        "question",
        "category",
        "difficulty",
        "competency_tested",
        "strong_answer_signals",
        "follow_up",
    }
    for index, item in enumerate(questions[:expected_count]):
        if not isinstance(item, dict) or not required_fields.issubset(item):
            raise MalformedResponseError(
                f"Question at index {index} is missing one or more required fields."
            )
        if not isinstance(item["strong_answer_signals"], list):
            raise MalformedResponseError(
                f"Question at index {index} has invalid strong_answer_signals."
            )
        normalized.append(item)
    return normalized


def build_tool_handlers(llm: JSONLLM) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    def generate(parameters: dict[str, Any]) -> dict[str, Any]:
        question_count = int(parameters.get("question_count", 8))
        if not 4 <= question_count <= 12:
            raise ValueError("question_count must be between 4 and 12.")
        payload = llm.complete_json(
            GENERATE_SYSTEM,
            generation_user_prompt(parameters),
            fallback_system_prompt=GENERATE_FALLBACK_SYSTEM,
            fallback_user_prompt=generation_user_prompt(
                {
                    "job_description": parameters["job_description"],
                    "requirements": parameters["requirements"],
                    "question_count": question_count,
                    "revision_feedback": parameters.get("review_feedback", {}),
                }
            ),
        )
        questions = _validate_questions(payload, question_count)
        return {"questions": questions, "question_count": len(questions)}

    def review(parameters: dict[str, Any]) -> dict[str, Any]:
        requirements = parameters["requirements"]
        questions = parameters["questions"]
        question_texts = [
            " ".join(
                str(question.get(key, ""))
                for key in ("question", "category", "competency_tested")
            ).lower()
            for question in questions
        ]
        covered: list[str] = []
        gaps: list[str] = []
        for requirement in requirements:
            terms = {
                word.lower()
                for word in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", requirement)
                if word.lower() not in {"the", "and", "for", "with", "from", "this", "that"}
            }
            is_covered = any(
                any(term in question for term in terms)
                for question in question_texts
            )
            (covered if is_covered else gaps).append(requirement)

        normalized_questions = [
            re.sub(r"\W+", " ", text).strip() for text in question_texts
        ]
        duplicate_count = len(normalized_questions) - len(set(normalized_questions))
        coverage = len(covered) / max(len(requirements), 1)
        score = max(0, round(100 * coverage - duplicate_count * 10))
        return {
            "coverage_score": score,
            "covered_requirements": covered,
            "gaps": gaps,
            "duplicate_count": duplicate_count,
            "question_count": len(questions),
            "questions": questions,
            "review_passed": score >= 70 and duplicate_count == 0 and not gaps,
        }

    return {
        "analyze_job_description": analyze_job_description,
        "generate_interview_questions": generate,
        "review_question_coverage": review,
    }
