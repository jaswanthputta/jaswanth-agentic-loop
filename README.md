# Agentic Interview Question Generator

An agent that turns a job description into a structured set of interview questions with competencies, difficulty, follow-up prompts, and answer-evaluation signals.

It implements the requested **Perceive → Reason → Act → Reflect** loop directly, without an agent framework.

## Run the end-to-end demo

Requires Python 3.10 or newer.

```powershell
py -m pip install -r requirements.txt
py main.py
```

When launched in a terminal without `--job-description` or `--jd-file`, the program asks you to paste a job description.

Paste as many lines as needed, then enter `END` on its own line.

The default demo uses a deterministic LLM stand-in and a real persistent Chroma memory store, so it runs without API credentials.

It performs three iterations:

1. Requirement extraction
2. Interview question generation
3. Coverage review

The program also writes JSONL step logs to `logs/agent.jsonl` and prints the final status and generated questions.

Re-run with `--session-id <id>` to continue a previous memory session.

## Run with a live model

Live mode asks for the API key using hidden terminal input if the configured environment variable is not set.

Alternatively, set it in the current PowerShell window:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
$env:LLM_MODEL = "openai/gpt-oss-20b"
$env:LLM_MAX_TOKENS = "4096"
$env:LLM_TIMEOUT_SECONDS = "90"

py main.py --live
```

The live command prompts for the job description too.

Alternatively, pass a job-description file:

```powershell
py main.py --live --jd-file .\job-description.txt
```

Or pass the job description directly:

```powershell
py main.py --live --job-description "Python Backend Developer with experience in REST APIs, SQL, Docker and testing."
```

Use `LLM_BASE_URL` and `LLM_MODEL` for another OpenAI-compatible chat-completions provider.

`config.yaml` contains runtime defaults. The following environment variables can override them:

* `AGENT_MAX_ITERATIONS`
* `AGENT_TOKEN_BUDGET`
* `AGENT_MEMORY_BACKEND`
* `AGENT_MEMORY_PATH`
* `AGENT_LOG_PATH`
* `LLM_TIMEOUT_SECONDS`
* `LLM_MAX_RETRIES`
* `LLM_BASE_DELAY_SECONDS`
* `LLM_MAX_TOKENS`
* `LLM_TEMPERATURE`
* `LLM_TOP_P`

Use only environment variables for API keys. Do not put API keys in source files, configuration files, logs, or the repository.

The client sends JSON-only instructions in the prompt and validates the returned JSON locally. It does not depend on the optional `response_format` request field.

## Project map

* `agent/loop.py` — four explicit cognitive steps and bounded agent loop.
* `agent/tools.py` — JSON Schema tool definitions and handlers for analysis, question generation, and review.
* `agent/prompts.py` — prompt templates used by the agent.
* `agent/memory_manager.py` — persistent session-scoped Chroma vector memory.
* `agent/harness.py` — retry, fallback, and guardrail logic.
* `agent/logger.py` — structured JSONL logging.
* `agent/config.py` — runtime configuration and environment overrides.
* `PATTERNS.md` — research on ReAct, Reflexion, Chain-of-Thought, Tree of Thoughts, and LATS.
* `MEMORY.md` — memory architecture and behavior.
* `HARNESS.md` — reliability and harness engineering decisions.
* `tests/` — focused unit and integration tests.

## Agentic loop

The agent implements four explicit steps:

```python
def perceive(input_data: str) -> dict:
    ...

def reason(observation: dict, memory: list) -> dict:
    ...

def act(plan: dict, tools: dict) -> dict:
    ...

def reflect(result: dict, observation: dict) -> dict:
    ...
```

The loop follows:

```text
Job Description
      ↓
   Perceive
      ↓
    Reason ←── Persistent Memory
      ↓
     Act
      ↓
    Tool
      ↓
   Reflect
      ↓
 Save to Memory
      ↓
 Next Iteration
```

The loop terminates when reflection determines that the task is complete or when the configured maximum iteration limit is reached.

## Tools

The agent provides three tools:

### `analyze_job_description`

Extracts important requirements and signals from the job description.

### `generate_interview_questions`

Generates structured interview questions containing:

* Question
* Category
* Difficulty
* Competency tested
* Strong-answer signals
* Follow-up prompt

### `review_question_coverage`

Reviews the generated questions against the job requirements and identifies:

* Requirement coverage
* Missing requirements
* Duplicate questions
* Coverage score
* Review status

## Memory

The project uses **ChromaDB** as a persistent vector memory store.

Memory is read before each reasoning step and written after each reflection.

The memory manager provides:

```python
save(session_id, content, metadata)
recall(session_id, query, top_k=3)
clear(session_id)
```

A successful three-iteration run demonstrated:

```text
Iteration 1 → recalled_count: 0
Iteration 2 → recalled_count: 1
Iteration 3 → recalled_count: 2
```

The recalled reflection from an earlier iteration influenced the next action, demonstrating that memory is actively used by the agent rather than only being stored.

See `MEMORY.md` for details.

## Agentic patterns

The project researches:

* ReAct
* Reflexion
* Chain-of-Thought
* Tree of Thoughts
* LATS

The implemented loop primarily uses **ReAct-style iteration combined with Reflexion**.

ReAct fits the use case because the agent repeatedly reasons about the next action, executes a tool, observes its result, and continues.

Reflexion is used because the agent evaluates the result after each action and stores feedback that can influence subsequent iterations.

See `PATTERNS.md` for the detailed research and design decision.

## Harness and reliability

The project includes a reliability layer around the core loop.

### Retry logic

LLM and tool calls support:

* Exponential backoff
* Jitter
* Configurable retry limits
* Timeout handling
* Rate-limit handling

### Malformed JSON

If the LLM returns malformed JSON, the agent retries with a simplified prompt.

### Tool failure

A failed tool call is converted into a structured error observation so the agent can adapt instead of crashing immediately.

### Maximum iterations

The loop has a configurable maximum iteration limit.

If the task cannot be completed within the limit, the agent returns:

```text
PARTIAL
```

### Stuck-loop detection

If `reflect` returns identical output twice in a row, the loop terminates with:

```text
STUCK
```

### Memory failure

If memory retrieval fails, the agent continues without memory and records a warning.

See `HARNESS.md` for details.

## Observability

Each step is recorded as structured JSON.

Log entries contain:

```text
timestamp
iteration
step
input_summary
output_summary
latency_ms
error
```

Logs are written to:

```text
logs/agent.jsonl
```

## Configuration

Runtime configuration is stored in:

```text
config.yaml
```

Environment variables can override configuration values.

No API keys are stored in `config.yaml`.

## Testing

Run the complete test suite:

```powershell
py -m pytest -q
```

Current result:

```text
14 passed
```

The tests cover areas including:

* LLM failure handling
* Tool failure handling
* Retry behavior
* Malformed JSON responses
* Simplified-prompt fallback
* HTTP/API errors
* Agent loop behavior

## Example end-to-end result

A live demonstration completed three iterations:

```text
Iteration 1 → Analyze job description
Iteration 2 → Generate interview questions
Iteration 3 → Review question coverage
```

Final result:

```text
Status: DONE
Iterations: 3
Coverage score: 100
Question count: 10
Gaps: 0
Duplicates: 0
Review passed: True
Quality score: 95
```

## Challenge milestones

### Milestone 1 — Core Agentic Loop

Implemented:

* Perceive
* Reason
* Act
* Reflect
* Three relevant tools
* LLM prompts
* Agentic pattern research
* Multi-iteration execution

### Milestone 2 — Memory Integration

Implemented:

* Persistent ChromaDB memory
* Session-based memory
* Memory save
* Memory recall
* Memory clear
* Reflection storage
* Memory-informed subsequent iterations

### Milestone 3 — Harness Engineering

Implemented:

* Exponential backoff
* Jitter
* Retry handling
* Timeout handling
* Rate-limit handling
* Malformed JSON fallback
* Tool failure fallback
* Maximum iteration guardrail
* Token budget warning
* STUCK detection
* Structured JSON logging
* YAML/environment configuration

## Submission notes

The challenge requires:

1. A private code repository
2. A 3–5 minute demo video
3. A 4–8 page solution PDF

The repository is kept private as required by the challenge.

The solution PDF is prepared separately and reflects the candidate's own design decisions, reasoning, engineering trade-offs, and reflections.

**Demo video:** Add the final demo URL here before submission.
