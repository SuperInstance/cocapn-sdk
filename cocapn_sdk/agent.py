"""AgentHandle for interacting with remote Cocapn agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .message import MessageBuilder


@dataclass
class AgentHandle:
    """Represents a remote Cocapn agent that can be interacted with."""

    agent_id: str
    client: object  # CocapnClient, using object to avoid circular import
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    _conversation: list[dict[str, str]] = field(default_factory=list)

    def chat(self, message: str, **options) -> dict:
        """Send a message to this agent and get a response.

        Maintains conversation history within this handle.
        """
        builder = MessageBuilder(model=options.get("model", "deepseek-chat"))
        builder.max_tokens = options.get("max_tokens", 4096)

        if self.system_prompt:
            builder.system(self.system_prompt)

        # Include conversation history
        for msg in self._conversation:
            builder.messages.append(
                __import__("cocapn_sdk.message", fromlist=["Message"]).Message(
                    role=msg["role"], content=msg["content"]
                )
            )

        builder.user(message)

        if "temperature" in options:
            builder.with_temperature(options["temperature"])

        response = self.client._request("POST", "/v1/chat/completions", builder.build())

        # Track conversation
        self._conversation.append({"role": "user", "content": message})
        if "choices" in response:
            assistant_content = response["choices"][0]["message"]["content"]
            self._conversation.append({"role": "assistant", "content": assistant_content})

        return response

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation.clear()

    @property
    def conversation_length(self) -> int:
        """Number of messages in the current conversation."""
        return len(self._conversation)
