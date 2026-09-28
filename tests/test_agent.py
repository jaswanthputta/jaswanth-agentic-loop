from __future__ import annotations

import json
import argparse
import tempfile
import unittest
from pathlib import Path
from typing import Any
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError
from unittest.mock import MagicMock

from agent.config import LLMConfig
from agent.harness import MalformedResponseError, call_with_retry
from agent.llm import OpenAICompatibleClient
from agent.loop import act, run_agent
from agent.logger import StepLogger
from agent.memory_manager import MemoryManager
from agent.tools import build_tool_handlers
from main import (
    DemoLLM,
    _ensure_api_key,
    _load_job_description,
    _read_job_description_from_terminal,
    SAMPLE_JOB_DESCRIPTION,
)


class InMemoryStore:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def add(self, *, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]], **kwargs: Any) -> None:
        self.records.append((ids[0], documents[0], metadatas[0]))

    def query(self, *, where: dict[str, Any], n_results: int, **kwargs: Any) -> dict[str, Any]:
        matches = [
            content
            for _, content, metadata in self.records
            if metadata.get("session_id") == where["session_id"]
        ]
        return {"documents": [matches[-n_results:][::-1]]}

    def delete(self, *, where: dict[str, Any]) -> None:
        self.records = [
            row for row in self.records if row[2].get("session_id") != where["session_id"]
        ]


class AgentTests(unittest.TestCase):
    def test_live_mode_prompts_for_missing_key_and_sets_process_environment(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            _ensure_api_key(
                "TEST_LLM_KEY",
                live=True,
                prompt_fn=lambda _prompt: "test-only-key",
            )
            self.assertEqual(__import__("os").environ["TEST_LLM_KEY"], "test-only-key")

    def test_demo_mode_does_not_prompt_for_key(self) -> None:
        def should_not_prompt(_prompt: str) -> str:
            raise AssertionError("Demo mode must not prompt for an API key.")

        _ensure_api_key("TEST_LLM_KEY", live=False, prompt_fn=should_not_prompt)

    def test_terminal_input_reads_multiline_description_until_end(self) -> None:
        lines = iter(
            [
                "Senior Data Engineer",
                "Build reliable data pipelines.",
                "",
                "Required: Python and SQL.",
                "END",
            ]
        )

        description = _read_job_description_from_terminal(
            input_fn=lambda: next(lines),
            output_fn=lambda _message: None,
        )

        self.assertEqual(
            description,
            "Senior Data Engineer\nBuild reliable data pipelines.\n\nRequired: Python and SQL.",
        )

    def test_job_description_can_be_passed_as_command_argument(self) -> None:
        args = argparse.Namespace(
            job_description="Backend Engineer\nBuild APIs.",
            jd_file=None,
        )

        self.assertEqual(
            _load_job_description(args),
            "Backend Engineer\nBuild APIs.",
        )

    def test_llm_planning_failure_returns_failed_status_without_looping(self) -> None:
        class BrokenLLM:
            total_tokens_used = 0

            def complete_json(
                self,
                system_prompt: str,
                user_prompt: str,
                fallback_system_prompt: str | None = None,
                fallback_user_prompt: str | None = None,
            ) -> dict[str, Any]:
                raise RuntimeError("HTTP Error 401: Unauthorized")

        memory = MemoryManager("unused", collection=InMemoryStore())
        result = run_agent(
            "Backend Engineer. Required: Python.",
            llm=BrokenLLM(),
            memory=memory,
            tools={},
            max_iterations=10,
        )

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["iterations"], 1)
        self.assertIn("401", result["error"])

    def test_demo_completes_three_iterations_and_recalls_memory(self) -> None:
        memory = MemoryManager("unused", collection=InMemoryStore())
        llm = DemoLLM()
        with tempfile.TemporaryDirectory() as directory:
            result = run_agent(
                SAMPLE_JOB_DESCRIPTION,
                llm=llm,
                memory=memory,
                tools=build_tool_handlers(llm),
                max_iterations=5,
                logger_instance=StepLogger(Path(directory) / "steps.jsonl"),
                session_id="test-session",
            )
            entries = [
                json.loads(line)
                for line in (Path(directory) / "steps.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["status"], "DONE")
        self.assertEqual(result["iterations"], 3)
        self.assertGreater(result["memory_events"][1]["recalled_count"], 0)
        self.assertEqual(result["result"]["action"], "review_question_coverage")
        self.assertEqual(len(result["result"]["result"]["questions"]), 8)
        self.assertEqual(
            [entry["step"] for entry in entries if entry["step"] != "reason"],
            ["perceive", "act", "reflect"] * 3,
        )
        self.assertTrue(all("latency_ms" in entry for entry in entries))

    def test_tool_failure_is_returned_as_error_observation(self) -> None:
        outcome = act(
            {"action": "explode", "parameters": {}},
            {"explode": lambda _parameters: (_ for _ in ()).throw(ValueError("broken"))},
        )
        self.assertFalse(outcome["ok"])
        self.assertIn("ValueError: broken", outcome["error"])

    def test_retry_retries_timeout_then_succeeds(self) -> None:
        calls = 0

        def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError("temporary")
            return "ok"

        self.assertEqual(call_with_retry(flaky, max_retries=2, base_delay=0), "ok")
        self.assertEqual(calls, 3)

    def test_malformed_model_json_uses_simplified_prompt(self) -> None:
        config = LLMConfig(
            model="test",
            base_url="https://example.invalid/v1",
            api_key_env="UNUSED_API_KEY",
            timeout_seconds=1,
            max_retries=0,
            base_delay_seconds=0,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.7,
        )

        class ScriptedClient(OpenAICompatibleClient):
            def __init__(self) -> None:
                super().__init__(config)
                self.prompts: list[str] = []

            def _request(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
                self.prompts.append(system_prompt)
                if len(self.prompts) == 1:
                    return {"content": "not-json"}
                return {"content": '{"valid": true}'}

        client = ScriptedClient()
        response = client.complete_json(
            "complex prompt",
            "input",
            fallback_system_prompt="simplified prompt",
            fallback_user_prompt="simple input",
        )
        self.assertEqual(response, {"valid": True})
        self.assertEqual(client.prompts, ["complex prompt", "simplified prompt"])

    def test_memory_manager_scopes_and_clears_sessions(self) -> None:
        store = InMemoryStore()
        memory = MemoryManager("unused", collection=store)
        memory.save("session-a", "Python testing reflection", {"iteration": 1})
        memory.save("session-b", "Other session data", {"iteration": 1})
        self.assertEqual(memory.recall("session-a", "Python testing", top_k=3), ["Python testing reflection"])
        memory.clear("session-a")
        self.assertEqual(memory.recall("session-a", "Python testing"), [])
        self.assertEqual(memory.recall("session-b", "Other session"), ["Other session data"])

    def test_malformed_model_json_without_fallback_is_explicit(self) -> None:
        config = LLMConfig(
            model="test",
            base_url="https://example.invalid/v1",
            api_key_env="UNUSED_API_KEY",
            timeout_seconds=1,
            max_retries=0,
            base_delay_seconds=0,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.7,
        )

        class MalformedClient(OpenAICompatibleClient):
            def _request(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
                return {"content": "not-json"}

        with self.assertRaises(MalformedResponseError):
            MalformedClient(config).complete_json("prompt", "input")

    def test_http_error_includes_provider_message(self) -> None:
        config = LLMConfig(
            model="test",
            base_url="https://example.invalid/v1",
            api_key_env="UNUSED_API_KEY",
            timeout_seconds=1,
            max_retries=0,
            base_delay_seconds=0,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.7,
        )
        error_response = HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(b'{"error":{"message":"model is not available"}}'),
        )
        client = OpenAICompatibleClient(config)

        with patch("agent.llm.urlopen", side_effect=error_response):
            with self.assertRaisesRegex(
                RuntimeError, "HTTP 400.*model is not available"
            ):
                client._request("system", "user")

    def test_request_omits_unsupported_json_response_format(self) -> None:
        config = LLMConfig(
            model="meta/llama-3.1-8b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="UNUSED_API_KEY",
            timeout_seconds=1,
            max_retries=0,
            base_delay_seconds=0,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.7,
        )
        client = OpenAICompatibleClient(config)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": '{"action":"analyze_job_description"}'}}],
                "usage": {"total_tokens": 7},
            }
        ).encode("utf-8")

        with patch("agent.llm.urlopen", return_value=response) as mocked_urlopen:
            client._request("system", "user")

        request = mocked_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "meta/llama-3.1-8b-instruct")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["top_p"], 0.7)

    def test_request_sends_api_key_as_bearer_token(self) -> None:
        config = LLMConfig(
            model="test",
            base_url="https://example.invalid/v1",
            api_key_env="TEST_AGENT_API_KEY",
            timeout_seconds=1,
            max_retries=0,
            base_delay_seconds=0,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.7,
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "{}"}}]}
        ).encode("utf-8")

        with patch.dict("os.environ", {"TEST_AGENT_API_KEY": "test-only-key"}):
            client = OpenAICompatibleClient(config)
        with patch("agent.llm.urlopen", return_value=response) as mocked_urlopen:
            client._request("system", "user")

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only-key")


if __name__ == "__main__":
    unittest.main()
