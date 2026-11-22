"""Model provider abstraction.

One function, several backends, selected with CISH_PROVIDER:

    groq        open-weights models on Groq's free tier   (default)
    openai      OpenAI
    openrouter  OpenRouter (many models behind one key)
    together    Together AI
    ollama      a local model, no network egress at all
    anthropic   Claude via the Messages API
    custom      any OpenAI-compatible endpoint; set CISH_BASE_URL

Everything except ollama and anthropic speaks the OpenAI chat-completions
protocol, so they share one code path and differ only by base URL, default
model, and which environment variable holds the key. Switching providers is
an environment change, not a code change.

No backend registers a tool schema, so the model has no callable function in
any configuration. This is the capability constraint from Requirement 2 of
the Part 2 design, and it holds regardless of which provider is active.

Every backend uses stdlib urllib except anthropic, so the default install
needs no SDK.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

# provider -> (base_url, default model, env var holding the key)
OPENAI_COMPATIBLE = {
    "groq": (
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "GROQ_API_KEY",
    ),
    "openai": (
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "meta-llama/llama-3.3-70b-instruct",
        "OPENROUTER_API_KEY",
    ),
    "together": (
        "https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "TOGETHER_API_KEY",
    ),
    "custom": (None, None, "CISH_API_KEY"),
}

TIMEOUT = int(os.environ.get("CISH_TIMEOUT", "180"))


# Read at call time, not import time, so evaluate.py can switch providers
# inside one process to compare them.
def provider() -> str:
    return os.environ.get("CISH_PROVIDER", "groq").strip().lower()


def _openai_config(name: str):
    base, default_model, key_var = OPENAI_COMPATIBLE[name]
    base = os.environ.get("CISH_BASE_URL", base)
    model = os.environ.get("CISH_MODEL", default_model)
    key = os.environ.get(key_var) or os.environ.get("CISH_API_KEY")
    if not base:
        raise RuntimeError(
            "CISH_BASE_URL must be set when CISH_PROVIDER=custom."
        )
    if not model:
        raise RuntimeError("CISH_MODEL must be set for this provider.")
    if not key:
        raise RuntimeError(
            f"No API key found. Set {key_var} (or CISH_API_KEY) in your "
            f"environment."
        )
    return base, model, key


def anthropic_model() -> str:
    return os.environ.get("CISH_MODEL", "claude-sonnet-5-5")


def ollama_model() -> str:
    return os.environ.get("CISH_OLLAMA_MODEL", "llama3.1:8b")


def ollama_host() -> str:
    return os.environ.get("CISH_OLLAMA_HOST", "http://localhost:11434")


def active_model() -> str:
    """Identifier written to the audit log so a decision can be traced to
    the exact model that produced it."""
    name = provider()
    if name == "anthropic":
        return anthropic_model()
    if name == "ollama":
        return f"ollama/{ollama_model()}"
    base, model, _ = OPENAI_COMPATIBLE.get(name, (None, None, None))
    return f"{name}/{os.environ.get('CISH_MODEL', model)}"


MAX_RETRIES = int(os.environ.get("CISH_MAX_RETRIES", "8"))


def _retry_after_seconds(detail: str, headers) -> float:
    """How long the provider says to wait.

    Groq reports it in the error text ("Please try again in 3.29s" or
    "...in 45ms") and sometimes in a Retry-After header.
    """
    header = headers.get("Retry-After") if headers else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = re.search(r"try again in ([\d.]+)\s*(ms|s|m)\b", detail)
    if not m:
        return 5.0
    value, unit = float(m.group(1)), m.group(2)
    return value / 1000 if unit == "ms" else value * 60 if unit == "m" else value


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    """POST JSON, retrying rate limits.

    A 429 on a free tier is a request to slow down, not a failure. We wait
    the time the provider asks for and try again. A daily-limit 429 is the
    exception: waiting seconds will not help, so it fails with a clear message.
    """
    data = json.dumps(payload).encode("utf-8")
    all_headers = {
        "Content-Type": "application/json",
        # Cloudflare, which fronts Groq and others, rejects urllib's
        # default "Python-urllib" User-Agent with a 403 before the key
        # is even checked. Identifying the client avoids that.
        "User-Agent": "can-i-share-this/0.1",
        **headers,
    }
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=data, headers=all_headers)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            if exc.code == 429:
                daily = "per day" in detail or "(TPD)" in detail or "(RPD)" in detail
                if daily or attempt >= MAX_RETRIES:
                    raise RuntimeError(
                        f"Rate limited by the provider (429) and not "
                        f"recoverable by waiting. {detail}"
                    ) from exc
                time.sleep(min(_retry_after_seconds(detail, exc.headers) + 0.5, 60))
                attempt += 1
                continue
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"Authentication failed ({exc.code}). Check the API key "
                    f"for CISH_PROVIDER={provider()}. {detail}"
                ) from exc
            raise ProviderError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {url}: {exc}") from exc


class ProviderError(RuntimeError):
    def __init__(self, code: int, detail: str):
        super().__init__(f"Provider error {code}: {detail}")
        self.code = code
        self.detail = detail


def _is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model.lower()


def _complete_openai_compatible(name: str, prompt: str, max_tokens: int) -> str:
    """Chat-completions protocol, shared by Groq, OpenAI, OpenRouter, etc."""
    base, model, key = _openai_config(name)

    # Reasoning models spend hidden "thinking" tokens out of the same budget
    # as the visible answer. With a small max_tokens they can run out before
    # writing the JSON, which the API reports as json_validate_failed. So the
    # floor is higher, and effort is kept low because this task is
    # classification, not deep reasoning.
    budget = int(os.environ.get("CISH_MAX_TOKENS", max(max_tokens, 700)))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": budget,
        # Constrains decoding to valid JSON. Both call sites in agent.py
        # ask for a JSON object.
        "response_format": {"type": "json_object"},
    }
    effort = os.environ.get("CISH_REASONING_EFFORT")
    if effort or _is_reasoning_model(model):
        payload["reasoning_effort"] = effort or "low"

    headers = {"Authorization": f"Bearer {key}"}
    try:
        body = _post_json(f"{base}/chat/completions", payload, headers)
    except ProviderError as exc:
        # Ran out of tokens mid-answer: retry once with a bigger budget.
        if exc.code == 400 and "json_validate_failed" in exc.detail:
            payload["max_tokens"] = budget * 3
            body = _post_json(f"{base}/chat/completions", payload, headers)
        else:
            raise
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response shape: {body}") from exc


def _complete_anthropic(prompt: str, max_tokens: int) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)  # no tools passed
    resp = client.messages.create(
        model=anthropic_model(),
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in resp.content
        if getattr(block, "type", "") == "text"
    )


def _complete_ollama(prompt: str, max_tokens: int) -> str:
    payload = {
        "model": ollama_model(),
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        f"{ollama_host()}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "User-Agent": "can-i-share-this/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {ollama_host()}. Is `ollama serve` "
            f"running and `ollama pull {ollama_model()}` done? ({exc})"
        ) from exc
    return body.get("response", "")


def complete(prompt: str, max_tokens: int = 400) -> str:
    name = provider()
    if name in OPENAI_COMPATIBLE:
        return _complete_openai_compatible(name, prompt, max_tokens)
    if name == "anthropic":
        return _complete_anthropic(prompt, max_tokens)
    if name == "ollama":
        return _complete_ollama(prompt, max_tokens)
    raise RuntimeError(
        f"Unknown CISH_PROVIDER '{name}'. Known: "
        f"{', '.join(sorted(set(OPENAI_COMPATIBLE) | {'anthropic', 'ollama'}))}."
    )


def available() -> bool:
    """Cheap reachability check, used by the UI to show provider status."""
    name = provider()
    if name == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if name == "ollama":
        try:
            with urllib.request.urlopen(f"{ollama_host()}/api/tags", timeout=3):
                return True
        except Exception:
            return False
    try:
        _openai_config(name)
        return True
    except Exception:
        return False
