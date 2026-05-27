"""Message builder for constructing chat requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """A single chat message."""

    role: str  # 'system', 'user', 'assistant'
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class MessageBuilder:
    """Fluent builder for constructing chat request payloads."""

    model: str = "deepseek-chat"
    messages: list[Message] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: Optional[float] = None
    stream: bool = False

    def system(self, prompt: str) -> MessageBuilder:
        """Add a system message."""
        self.messages.append(Message(role="system", content=prompt))
        return self

    def user(self, content: str) -> MessageBuilder:
        """Add a user message."""
        self.messages.append(Message(role="user", content=content))
        return self

    def assistant(self, content: str) -> MessageBuilder:
        """Add an assistant message."""
        self.messages.append(Message(role="assistant", content=content))
        return self

    def history(self, messages: list[dict[str, str]]) -> MessageBuilder:
        """Append a list of prior messages (each with 'role' and 'content')."""
        for msg in messages:
            self.messages.append(Message(role=msg["role"], content=msg["content"]))
        return self

    def with_model(self, model: str) -> MessageBuilder:
        """Set the model."""
        self.model = model
        return self

    def with_max_tokens(self, max_tokens: int) -> MessageBuilder:
        """Set max tokens."""
        self.max_tokens = max_tokens
        return self

    def with_temperature(self, temperature: float) -> MessageBuilder:
        """Set temperature."""
        self.temperature = temperature
        return self

    def with_stream(self, stream: bool = True) -> MessageBuilder:
        """Enable streaming."""
        self.stream = stream
        return self

    def build(self) -> dict:
        """Build the request payload."""
        payload: dict = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.stream:
            payload["stream"] = True
        return payload

    def reset(self) -> MessageBuilder:
        """Clear all messages, keep settings."""
        self.messages.clear()
        return self
