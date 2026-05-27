"""Cocapn SDK — One API key, any AI model, see what it costs."""

from .client import CocapnClient
from .config import SDKConfig
from .agent import AgentHandle
from .message import MessageBuilder
from .errors import (
    CocapnError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ServerError,
    ValidationError,
)

__version__ = "1.0.0"
__all__ = [
    "CocapnClient",
    "SDKConfig",
    "AgentHandle",
    "MessageBuilder",
    "CocapnError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "ServerError",
    "ValidationError",
]
