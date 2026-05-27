"""SDK configuration with authentication and endpoint settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_BASE_URL = "https://cocapn.ai"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3


@dataclass
class SDKConfig:
    """Configuration for the Cocapn SDK client."""

    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    default_model: str = "deepseek-chat"
    default_max_tokens: int = 4096
    default_temperature: Optional[float] = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("COCAPN_API_KEY")

    @property
    def auth_header(self) -> dict[str, str]:
        """Build authorization headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if not self.api_key:
            raise ValueError(
                "API key is required. Set COCAPN_API_KEY env var or pass api_key to SDKConfig."
            )
        if not self.base_url:
            raise ValueError("base_url must not be empty.")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
