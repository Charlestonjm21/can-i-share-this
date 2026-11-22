"""Usage limits for a public deployment.

The hosted demo spends one free-tier API key, so a crowd (or one curious
visitor) could use up the daily quota for everyone. Two caps protect it:

    CISH_MAX_QUESTIONS   per visitor session   (default 20, 0 = unlimited)
    CISH_DAILY_CAP       whole app, per day    (default 400, 0 = unlimited)
    CISH_MAX_CHARS       longest question      (default 600)

Each question costs up to two model calls, and Groq's free tier allows about
1,000 requests a day, so 400 questions a day stays safely under it.
"""

from __future__ import annotations

import os
import threading
from datetime import date


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def max_questions() -> int:
    return _int_env("CISH_MAX_QUESTIONS", 20)


def daily_cap() -> int:
    return _int_env("CISH_DAILY_CAP", 400)


def max_chars() -> int:
    return _int_env("CISH_MAX_CHARS", 600)


class DailyCounter:
    """Counts questions for the whole app and resets each calendar day."""

    def __init__(self):
        self._day = date.today()
        self._n = 0
        self._lock = threading.Lock()

    def try_take(self, cap: int, today: date | None = None) -> bool:
        """Reserve one question. False means the daily cap is used up."""
        today = today or date.today()
        with self._lock:
            if today != self._day:
                self._day, self._n = today, 0
            if cap and self._n >= cap:
                return False
            self._n += 1
            return True
