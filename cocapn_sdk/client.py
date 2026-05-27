"""CocapnClient — main client with connection management."""

from __future__ import annotations

import json
from typing import Any, Optional
from http.client import HTTPConnection, HTTPSConnection, HTTPResponse
from urllib.parse import urlparse

from .config import SDKConfig
from .message import MessageBuilder
from .agent import AgentHandle
from .errors import classify_error, retry_with_backoff, CocapnError


class ChatResponse:
    """Structured response from a chat completion."""

    __slots__ = ("text", "cost", "tokens_in", "tokens_out", "model", "provider", "raw")

    def __init__(
        self,
        text: str,
        cost: float,
        tokens_in: int,
        tokens_out: int,
        model: str,
        provider: str,
        raw: dict,
    ):
        self.text = text
        self.cost = cost
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.model = model
        self.provider = provider
        self.raw = raw

    def __repr__(self) -> str:
        return (
            f"ChatResponse(text={self.text!r:.80}, cost={self.cost}, "
            f"tokens={{in: {self.tokens_in}, out: {self.tokens_out}}}, "
            f"model={self.model!r})"
        )


class CocapnClient:
    """Main client for the Cocapn API.

    Usage:
        client = CocapnClient(api_key="cocapn_your_key")
        response = client.chat("Hello!", model="deepseek-chat")
        print(response.text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[SDKConfig] = None,
    ):
        if config is not None:
            self._config = config
        else:
            self._config = SDKConfig(
                api_key=api_key,
                base_url=base_url or SDKConfig.base_url,
            )

        self._config.validate()
        self._parsed = urlparse(self._config.base_url)
        self._conn: Optional[Any] = None

    # ─── Connection management ───

    def _get_connection(self):
        """Get or create an HTTP(S) connection."""
        if self._conn is not None:
            return self._conn

        host = self._parsed.hostname
        port = self._parsed.port
        if self._parsed.scheme == "https":
            self._conn = HTTPSConnection(host, port or 443, timeout=self._config.timeout)
        else:
            self._conn = HTTPConnection(host, port or 80, timeout=self._config.timeout)
        return self._conn

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ─── Core HTTP ───

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        """Make an HTTP request to the Cocapn API with retry logic."""
        headers = self._config.auth_header
        request_body = json.dumps(body).encode("utf-8") if body else None

        def do_request():
            conn = self._get_connection()
            try:
                conn.request(method, path, body=request_body, headers=headers)
                resp: HTTPResponse = conn.getresponse()
                data = resp.read().decode("utf-8")
            except ConnectionError as e:
                self.close()
                raise CocapnError(f"Connection error: {e}")

            status = resp.status

            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                parsed = {"raw": data}

            if status >= 400:
                msg = parsed.get("error", {}).get("message", f"HTTP {status}")
                raise classify_error(status, msg, parsed)

            return parsed

        return retry_with_backoff(do_request, max_retries=self._config.max_retries)

    # ─── High-level API ───

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ChatResponse:
        """Send a chat message and get a response."""
        builder = MessageBuilder(
            model=model or self._config.default_model,
            max_tokens=max_tokens or self._config.default_max_tokens,
            temperature=temperature or self._config.default_temperature,
        )

        if system:
            builder.system(system)
        if history:
            builder.history(history)

        builder.user(message)

        resp = self._request("POST", "/v1/chat/completions", builder.build())

        text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        cost = float(resp.get("cocapn_cost", 0))
        tokens_in = resp.get("usage", {}).get("prompt_tokens", 0)
        tokens_out = resp.get("usage", {}).get("completion_tokens", 0)
        model_name = resp.get("model", model or self._config.default_model)
        provider = resp.get("cocapn_provider", model_name.split("-")[0])

        return ChatResponse(
            text=text,
            cost=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model_name,
            provider=provider,
            raw=resp,
        )

    def models(self) -> list[dict]:
        """List available models."""
        resp = self._request("GET", "/v1/models")
        return [
            {
                "id": m.get("id"),
                "provider": m.get("owned_by"),
                "cost_in": m.get("cocapn_cost_in"),
                "cost_out": m.get("cocapn_cost_out"),
            }
            for m in resp.get("data", [])
        ]

    def usage(self, period: str = "day") -> dict:
        """Get usage statistics."""
        return self._request("GET", f"/v1/usage?period={period}")

    def agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AgentHandle:
        """Get an AgentHandle for interacting with a remote agent."""
        return AgentHandle(
            agent_id=agent_id,
            client=self,
            name=name,
            description=description,
            system_prompt=system_prompt,
        )
