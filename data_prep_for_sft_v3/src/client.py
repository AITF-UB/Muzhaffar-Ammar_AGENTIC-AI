"""
OpenRouter API client with retry, rate-limit, and structured logging.
"""

from __future__ import annotations

import time

import httpx
from rich.console import Console
# pyrefly: ignore [missing-import]
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# pyrefly: ignore [missing-import]
from src.config import (
    MAX_RETRIES,
    MAX_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    REQUEST_DELAY,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
    TEMPERATURE,
)

console = Console()


class OpenRouterError(Exception):
    """Raised when the OpenRouter API returns a non-2xx response."""


class RateLimitError(OpenRouterError):
    """Raised on HTTP 429 — triggers retry."""


def _log_retry_before_sleep(rs) -> None:
    """Tenacity hook: log retry reason + backoff."""
    exc = None
    try:
        exc = rs.outcome.exception() if rs.outcome else None
    except Exception:
        exc = None

    exc_name = type(exc).__name__ if exc else "Exception"
    exc_msg = str(exc) if exc else ""

    console.print(
        f"[yellow]⏳ Retrying in {rs.next_action.sleep:.1f}s "
        f"(attempt {rs.attempt_number}/{MAX_RETRIES}) — {exc_name}: {exc_msg}[/]"
    )


@retry(
    retry=retry_if_exception_type((RateLimitError, httpx.RequestError)),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    reraise=True,
    before_sleep=_log_retry_before_sleep,
)
def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """
    Send a single chat-completion request to OpenRouter.

    Returns the assistant's response text.
    Raises OpenRouterError on non-retriable failures,
           RateLimitError on 429 (auto-retried).
    """
    if not OPENROUTER_API_KEY:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file."
        )

    payload = {
        "model": model or OPENROUTER_MODEL,
        "temperature": temperature if temperature is not None else TEMPERATURE,
        "max_tokens": max_tokens or MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "Sekolah Rakyat SFT Generator",
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(OPENROUTER_BASE_URL, json=payload, headers=headers)

    if resp.status_code == 429:
        raise RateLimitError(f"Rate-limited: {resp.text}")

    if resp.status_code >= 400:
        raise OpenRouterError(
            f"OpenRouter HTTP {resp.status_code}: {resp.text}"
        )

    data = resp.json()

    # Extract content from the response
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise OpenRouterError(
            f"Unexpected response structure: {data}"
        ) from exc

    return content


def call_with_delay(
    system_prompt: str,
    user_prompt: str,
    **kwargs,
) -> str:
    """Wrapper that adds a polite delay between requests."""
    result = call_openrouter(system_prompt, user_prompt, **kwargs)
    time.sleep(REQUEST_DELAY)
    return result
