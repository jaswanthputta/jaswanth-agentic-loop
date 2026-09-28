
"""Prompt templates for planning, question generation, and reflection."""

from __future__ import annotations

import json
from typing import Any


REASON_SYSTEM = """ROLE: REASON
You control a job-description interview-question agent. Select exactly one tool
action that advances the task. Return only JSON with keys: action, parameters,
reasoning. The reasoning value must be a brief decision summary, not hidden
step-by-step chain-of-thought.

Follow this workflow:

1. First, use analyze_job_description when requirements have not yet been
   extracted.

2. Next, use generate_interview_questions when requirements exist but questions
   have not yet been generated.

3. After questions have been generated, ALWAYS use review_question_coverage
   before the task can be considered complete.

4. If the review identifies gaps or duplicates, use
   generate_interview_questions again with the review feedback.

5. Only after a review confirms adequate coverage should reflection be allowed
   to mark the task complete.

Use prior reflection instructions and recalled memory. Analyze first, generate
questions after requirements are available, and review after questions exist.
Use the available tool schemas exactly. Never claim completion through the
reasoning step; reflection decides completion.
"""


REASON_FALLBACK_SYSTEM = """ROLE: REASON
Return one valid JSON object only:
{"action":"tool_name","parameters":{},"reasoning":"brief reason"}.

Choose exactly one listed tool that advances the job-description
interview-question task.

Follow this order:
1. Analyze the job description if requirements are not yet available.
2. Generate questions if requirements are available but questions do not exist.
3. Review question coverage after questions have been generated.
4. If review finds gaps or duplicates, revise the questions.
5. Do not consider the task complete until generated questions have been
   reviewed.
"""


GENERATE_SYSTEM = """ROLE: GENERATE
Generate interview questions grounded in the supplied job description and
extracted requirements.

Return only JSON:
{"questions":[...]}

Each question must have:
- question
- category
- difficulty
- competency_tested
- strong_answer_signals
- follow_up

strong_answer_signals must be a list of concise evaluation signals.

Avoid duplicate questions and do not invent requirements that are not supported
by the description.

If revision feedback is provided, address the identified gaps and duplicates.
"""


GENERATE_FALLBACK_SYSTEM = """ROLE: GENERATE
Return JSON only in this shape:

{
  "questions": [
    {
      "question": "...",
      "category": "...",
      "difficulty": "...",
      "competency_tested": "...",
      "strong_answer_signals": ["..."],
      "follow_up": "..."
    }
  ]
}

Use the supplied job requirements. Address any supplied review feedback.
Do not return markdown.
"""


REFLECT_SYSTEM = """ROLE: REFLECT
Assess progress toward producing a relevant, balanced interview question set
from the job description.

Return JSON only with these keys:
- is_done (boolean)
- quality_score (integer 0-100)
- next_instruction (string)
- summary (string)

IMPORTANT COMPLETION RULE:

Do NOT mark the task as done immediately after question generation.

The generated questions must first pass the review_question_coverage tool.

Mark is_done true only when:
1. interview questions have been generated,
2. review_question_coverage has been performed,
3. the review shows adequate requirement coverage,
4. there are no material gaps or unacceptable duplicates.

If questions have not yet been reviewed, return is_done false and instruct the
next iteration to perform coverage review.

The summary should briefly explain the assessment and must not reveal hidden
chain-of-thought.
"""


REFLECT_FALLBACK_SYSTEM = """ROLE: REFLECT
Return JSON only:

{
  "is_done": false,
  "quality_score": 0,
  "next_instruction": "...",
  "summary": "..."
}

Set is_done true only when generated questions have already been reviewed and
the review confirms adequate requirement coverage with no material gaps.
"""


def json_context(data: Any) -> str:
    """Serialize structured data for inclusion in prompts."""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def reason_user_prompt(
    observation: dict[str, Any],
    memory: list[str],
    tool_schemas: dict[str, Any],
) -> str:
    """Build the planner prompt with observation, memory, and tools."""
    return (
        "Choose the next tool action.\n\n"
        f"Observation:\n{json_context(observation)}\n\n"
        f"Recalled memory:\n{json_context(memory)}\n\n"
        f"Available tools and JSON schemas:\n{json_context(tool_schemas)}"
    )


def generation_user_prompt(parameters: dict[str, Any]) -> str:
    """Build the question-generation prompt."""
    return "Generate the question set using this context:\n" + json_context(parameters)


def reflection_user_prompt(
    result: dict[str, Any],
    observation: dict[str, Any],
) -> str:
    """Build the reflection prompt."""
    return (
        f"Goal and current observation:\n{json_context(observation)}\n\n"
        f"Latest action result:\n{json_context(result)}"
    )

