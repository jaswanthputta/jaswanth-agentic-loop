# Harness decisions

- **Retry:** Model requests and tool execution use `call_with_retry` with exponential backoff and jitter for rate limits, timeouts, connection failures, and server-side HTTP errors. Non-transient errors are surfaced immediately.
- **Malformed model output:** JSON is parsed strictly. On malformed JSON, the client retries once with a simplified prompt; if it remains invalid, the failure is explicit and reflection receives a non-completion recovery state.
- **Tool failure:** A tool exception becomes an `ok: false` result containing the error. The agent can use that observation to choose a subsequent action rather than terminate with a misleading success.
- **Iteration cap:** `max_iterations` is configurable (default 10). Exhaustion returns `PARTIAL` with the latest result and reflection.
- **Stuck guard:** Two identical consecutive reflection objects end the run with `STUCK`.
- **Token budget:** Token usage reported by the LLM API is accumulated and warned on when it exceeds the configurable budget. The budget warns rather than silently truncating a useful result.
- **Observability:** Each loop step writes a JSON Lines record containing UTC timestamp, iteration, step, bounded input/output summaries, latency, and error.
- **Memory failure:** Recall/save failures emit warnings. Recall failure uses an empty context; save failure does not discard the in-progress result.
- **Configuration:** YAML settings can be overridden through environment variables. The live provider is OpenAI-compatible; API keys are read from the configured environment variable and are never written to the config or logs.
