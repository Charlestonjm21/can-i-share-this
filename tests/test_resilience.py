"""Tests for the failures seen on a real Groq free-tier run.

The error strings below are copied from that run, so these pin the behavior
against what the provider actually sends.
"""

import io
import json
import urllib.error

import pytest

from cish import llm

RATE_429 = json.dumps({"error": {
    "message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens "
               "per minute (TPM): Limit 8000, Used 7221, Requested 785. "
               "Please try again in 45ms.",
    "type": "tokens", "code": "rate_limit_exceeded"}})
RATE_429_SECONDS = RATE_429.replace("45ms", "3.2925s")
DAILY_429 = RATE_429.replace("tokens per minute (TPM)", "tokens per day (TPD)")
JSON_400 = json.dumps({"error": {
    "message": "Failed to generate JSON.", "type": "invalid_request_error",
    "code": "json_validate_failed",
    "failed_generation": "max completion tokens reached before generating a valid document"}})
OK_BODY = json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]})


class Resp:
    def __init__(self, body): self.body = body.encode()
    def read(self): return self.body
    def __enter__(self): return self
    def __exit__(self, *a): return False


def http_error(code, body):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))


@pytest.fixture(autouse=True)
def groq_env(monkeypatch):
    monkeypatch.setenv("CISH_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("CISH_MODEL", raising=False)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def script(monkeypatch, steps):
    """Each step is an exception to raise or a body string to return."""
    calls = []

    def fake(req, timeout=None):
        calls.append(json.loads(req.data))
        step = steps[min(len(calls) - 1, len(steps) - 1)]
        if isinstance(step, Exception):
            raise step
        return Resp(step)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    return calls


def test_429_is_retried_until_it_succeeds(monkeypatch):
    calls = script(monkeypatch, [http_error(429, RATE_429),
                                 http_error(429, RATE_429_SECONDS), OK_BODY])
    assert llm.complete("hi") == '{"ok": true}'
    assert len(calls) == 3


def test_retry_wait_parses_ms_and_seconds():
    assert llm._retry_after_seconds("try again in 45ms.", None) == pytest.approx(0.045)
    assert llm._retry_after_seconds("try again in 3.29s.", None) == pytest.approx(3.29)
    assert llm._retry_after_seconds("no hint here", None) == 5.0


def test_daily_limit_is_not_retried(monkeypatch):
    calls = script(monkeypatch, [http_error(429, DAILY_429)])
    with pytest.raises(RuntimeError, match="not recoverable"):
        llm.complete("hi")
    assert len(calls) == 1


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm, "MAX_RETRIES", 2)
    calls = script(monkeypatch, [http_error(429, RATE_429)])
    with pytest.raises(RuntimeError):
        llm.complete("hi")
    assert len(calls) == 3  # first try plus two retries


def test_json_validate_failure_is_retried_with_a_bigger_budget(monkeypatch):
    calls = script(monkeypatch, [http_error(400, JSON_400), OK_BODY])
    assert llm.complete("hi") == '{"ok": true}'
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]


def test_reasoning_model_gets_low_effort_and_a_real_budget(monkeypatch):
    calls = script(monkeypatch, [OK_BODY])
    llm.complete("hi", max_tokens=400)
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["max_tokens"] >= 700


def test_non_reasoning_model_does_not_get_reasoning_effort(monkeypatch):
    monkeypatch.setenv("CISH_MODEL", "llama-3.3-70b-versatile")
    calls = script(monkeypatch, [OK_BODY])
    llm.complete("hi")
    assert "reasoning_effort" not in calls[0]


def test_auth_errors_are_not_retried(monkeypatch):
    calls = script(monkeypatch, [http_error(403, '{"error":"forbidden"}')])
    with pytest.raises(RuntimeError, match="Authentication failed"):
        llm.complete("hi")
    assert len(calls) == 1
