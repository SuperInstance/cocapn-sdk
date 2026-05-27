"""Error types and retry logic for the Cocapn SDK."""

from __future__ import annotations

import time
from typing import Optional


class CocapnError(Exception):
    """Base error for all Cocapn SDK errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, data: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.data = data or {}


class AuthenticationError(CocapnError):
    """Raised on 401 Unauthorized responses."""
    pass


class RateLimitError(CocapnError):
    """Raised on 429 Too Many Requests responses."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class NotFoundError(CocapnError):
    """Raised on 404 Not Found responses."""
    pass


class ValidationError(CocapnError):
    """Raised on 400 Bad Request responses."""
    pass


class ServerError(CocapnError):
    """Raised on 5xx server error responses."""
    pass


def classify_error(status_code: int, message: str, data: Optional[dict] = None) -> CocapnError:
    """Create the appropriate error type from an HTTP status code."""
    kwargs: dict = {"status_code": status_code, "data": data}

    if status_code == 400:
        return ValidationError(message, **kwargs)
    elif status_code == 401:
        return AuthenticationError(message, **kwargs)
    elif status_code == 404:
        return NotFoundError(message, **kwargs)
    elif status_code == 429:
        retry_after = None
        if data and "retry_after" in data:
            retry_after = float(data["retry_after"])
        return RateLimitError(message, retry_after=retry_after, **kwargs)
    elif status_code >= 500:
        return ServerError(message, **kwargs)
    else:
        return CocapnError(message, **kwargs)


def is_retryable(error: CocapnError) -> bool:
    """Check if an error is worth retrying."""
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, ServerError):
        return True
    return False


def backoff_delay(attempt: int, base: float = 0.5) -> float:
    """Exponential backoff with jitter."""
    import random
    delay = base * (2 ** attempt)
    jitter = random.uniform(0, delay * 0.5)
    return delay + jitter


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 0.5):
    """Decorator-ish retry logic. Call with a callable that may raise CocapnError."""
    last_error: Optional[CocapnError] = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except CocapnError as e:
            last_error = e
            if not is_retryable(e) or attempt == max_retries:
                raise
            delay = backoff_delay(attempt, base_delay)
            if isinstance(e, RateLimitError) and e.retry_after:
                delay = max(delay, e.retry_after)
            time.sleep(delay)
    raise last_error  # type: ignore[misc]
