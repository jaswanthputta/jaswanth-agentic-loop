"""Small OpenAI-compatible HTTP client with strict JSON parsing and fallback prompts."""

from __future__ import annotations

import json
import os
import re
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent.config import LLMConfig
from agent.harness import MalformedResponseError, RateLimitError, call_with_retry


def _http_error_detail(error: HTTPError) -> str:
    """Return a bounded provider error message without exposing request data."""
    try:
        body = error.read().decode("utf-8", errors="replace")
    except OSError:
        return "No error details were provided."
    if not body:
        return "No error details were provided."

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        detail = body
    else:
        error_payload = payload.get("error", payload) if isinstance(payload, dict) else payload
        if isinstance(error_payload, dict):
            detail = str(
                error_payload.get("message")
                or error_payload.get("detail")
                or error_payload
            )
        else:
            detail = str(error_payload)

    detail = " ".join(detail.split())
    if len(detail) > 1000:
        detail = detail[:1000] + "..."
    return detail


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = os.getenv(config.api_key_env)
        self.total_tokens_used = 0

    def _request(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except json.JSONDecodeError as error:
            raise MalformedResponseError("LLM API returned malformed JSON.") from error
        except HTTPError as error:
            detail = _http_error_detail(error)
            if error.code == 429:
                raise RateLimitError(
                    f"LLM API rate limit exceeded: {detail}"
                ) from error
            hint = ""
            if error.code == 400 and detail == "400 Bad Request":
                hint = (
                    " The provider returned no validation details. Confirm the API key "
                    "belongs to this provider, the model is enabled for that key, and "
                    "the endpoint/model pair is correct."
                )
            raise RuntimeError(
                f"LLM API returned HTTP {error.code} from "
                f"{self.config.base_url} (model {self.config.model}): {detail}{hint}"
            ) from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError("LLM API request timed out.") from error
            raise
        except socket.timeout as error:
            raise TimeoutError("LLM API request timed out.") from error

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise MalformedResponseError(
                "LLM response did not contain choices[0].message.content."
            ) from error
        usage = response_data.get("usage", {})
        if isinstance(usage, dict):
            self.total_tokens_used += int(usage.get("total_tokens", 0) or 0)
        if not isinstance(content, str):
            raise MalformedResponseError("LLM message content was not a string.")
        return {"content": content}

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise MalformedResponseError("LLM returned unparseable JSON.") from error
        if not isinstance(parsed, dict):
            raise MalformedResponseError("LLM response JSON must be an object.")
        return parsed

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_system_prompt: str | None = None,
        fallback_user_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Call once, simplify the prompt once on malformed JSON, then surface errors."""
        try:
            response = call_with_retry(
                self._request,
                system_prompt,
                user_prompt,
                max_retries=self.config.max_retries,
                base_delay=self.config.base_delay_seconds,
            )
            return self._parse_json(response["content"])
        except MalformedResponseError as original_error:
            if not fallback_system_prompt or not fallback_user_prompt:
                raise

            try:
                response = call_with_retry(
                    self._request,
                    fallback_system_prompt,
                    fallback_user_prompt,
                    max_retries=self.config.max_retries,
                    base_delay=self.config.base_delay_seconds,
                )
                return self._parse_json(response["content"])
            except MalformedResponseError as fallback_error:
                raise MalformedResponseError(
                    "LLM returned malformed JSON for both the original and simplified prompts."
                ) from fallback_error
