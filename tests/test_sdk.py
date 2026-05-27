"""Tests for the Cocapn SDK."""

import json
import os
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest

from cocapn_sdk.config import SDKConfig, DEFAULT_BASE_URL
from cocapn_sdk.errors import (
    CocapnError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ServerError,
    ValidationError,
    classify_error,
    is_retryable,
    backoff_delay,
    retry_with_backoff,
)
from cocapn_sdk.message import Message, MessageBuilder
from cocapn_sdk.agent import AgentHandle
from cocapn_sdk.client import CocapnClient, ChatResponse


# ─── Config Tests ───


class TestSDKConfig:
    def test_defaults(self):
        os.environ.pop("COCAPN_API_KEY", None)
        cfg = SDKConfig(api_key="test-key")
        assert cfg.api_key == "test-key"
        assert cfg.base_url == DEFAULT_BASE_URL
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 3

    def test_env_api_key(self):
        with patch.dict(os.environ, {"COCAPN_API_KEY": "env-key"}):
            cfg = SDKConfig()
            assert cfg.api_key == "env-key"

    def test_auth_header(self):
        cfg = SDKConfig(api_key="my-key")
        headers = cfg.auth_header
        assert headers["Authorization"] == "Bearer my-key"
        assert headers["Content-Type"] == "application/json"

    def test_auth_header_no_key(self):
        os.environ.pop("COCAPN_API_KEY", None)
        cfg = SDKConfig(api_key=None)
        headers = cfg.auth_header
        assert "Authorization" not in headers

    def test_validate_no_key(self):
        os.environ.pop("COCAPN_API_KEY", None)
        cfg = SDKConfig(api_key=None)
        with pytest.raises(ValueError, match="API key"):
            cfg.validate()

    def test_validate_empty_url(self):
        cfg = SDKConfig(api_key="key", base_url="")
        with pytest.raises(ValueError, match="base_url"):
            cfg.validate()

    def test_validate_bad_timeout(self):
        cfg = SDKConfig(api_key="key", timeout=-1)
        with pytest.raises(ValueError, match="timeout"):
            cfg.validate()

    def test_validate_bad_retries(self):
        cfg = SDKConfig(api_key="key", max_retries=-1)
        with pytest.raises(ValueError, match="max_retries"):
            cfg.validate()

    def test_extra_headers(self):
        cfg = SDKConfig(api_key="key", extra_headers={"X-Custom": "yes"})
        assert cfg.auth_header["X-Custom"] == "yes"


# ─── Error Tests ───


class TestErrors:
    def test_base_error(self):
        err = CocapnError("test", status_code=500, data={"foo": "bar"})
        assert str(err) == "test"
        assert err.status_code == 500
        assert err.data == {"foo": "bar"}

    def test_classify_400(self):
        err = classify_error(400, "bad")
        assert isinstance(err, ValidationError)
        assert err.status_code == 400

    def test_classify_401(self):
        err = classify_error(401, "nope")
        assert isinstance(err, AuthenticationError)

    def test_classify_404(self):
        err = classify_error(404, "gone")
        assert isinstance(err, NotFoundError)

    def test_classify_429(self):
        err = classify_error(429, "slow down", data={"retry_after": 2.5})
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 2.5

    def test_classify_500(self):
        err = classify_error(500, "oops")
        assert isinstance(err, ServerError)

    def test_classify_unknown(self):
        err = classify_error(418, "teapot")
        assert isinstance(err, CocapnError)

    def test_is_retryable(self):
        assert is_retryable(RateLimitError("slow"))
        assert is_retryable(ServerError("x", status_code=500))
        assert not is_retryable(AuthenticationError("no"))
        assert not is_retryable(ValidationError("x"))

    def test_backoff_delay(self):
        # Test that delay is reasonable and increases with attempts
        delays = [backoff_delay(i, base=0.5) for i in range(5)]
        assert all(d >= 0 for d in delays)
        # Generally increasing (with jitter, not strict)
        assert delays[4] > delays[0]

    def test_retry_with_backoff_success(self):
        result = retry_with_backoff(lambda: 42, max_retries=3)
        assert result == 42

    def test_retry_with_backoff_retries(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ServerError("fail", status_code=500)
            return "ok"

        with patch("cocapn_sdk.errors.time"):
            result = retry_with_backoff(flaky, max_retries=3)
        assert result == "ok"
        assert calls["n"] == 3

    def test_retry_with_backoff_non_retryable(self):
        with pytest.raises(AuthenticationError):
            retry_with_backoff(lambda: (_ for _ in ()).throw(AuthenticationError("no")))

    def test_retry_exhausted(self):
        with patch("cocapn_sdk.errors.time"):
            with pytest.raises(ServerError):
                retry_with_backoff(lambda: (_ for _ in ()).throw(ServerError("x", status_code=500)), max_retries=2)


# ─── Message Tests ───


class TestMessage:
    def test_to_dict(self):
        m = Message(role="user", content="hello")
        assert m.to_dict() == {"role": "user", "content": "hello"}


class TestMessageBuilder:
    def test_basic_build(self):
        builder = MessageBuilder()
        builder.user("hello")
        payload = builder.build()
        assert payload["model"] == "deepseek-chat"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["max_tokens"] == 4096

    def test_full_build(self):
        payload = (
            MessageBuilder(model="gpt-4o")
            .system("You are helpful")
            .user("hi")
            .assistant("hello!")
            .user("how are you?")
            .with_temperature(0.7)
            .with_max_tokens(2048)
            .with_stream()
            .build()
        )
        assert payload["model"] == "gpt-4o"
        assert len(payload["messages"]) == 4
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 2048
        assert payload["stream"] is True

    def test_history(self):
        builder = MessageBuilder()
        builder.history([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ])
        builder.user("c")
        assert len(builder.messages) == 3

    def test_reset(self):
        builder = MessageBuilder().user("x").assistant("y")
        builder.reset()
        assert len(builder.messages) == 0
        assert builder.model == "deepseek-chat"  # settings preserved

    def test_fluent_chaining(self):
        result = MessageBuilder().system("s").user("u").with_model("m")
        assert isinstance(result, MessageBuilder)

    def test_no_temperature_when_none(self):
        payload = MessageBuilder().user("hi").build()
        assert "temperature" not in payload

    def test_no_stream_by_default(self):
        payload = MessageBuilder().user("hi").build()
        assert "stream" not in payload


# ─── AgentHandle Tests ───


class TestAgentHandle:
    def _make_client(self, response):
        client = MagicMock()
        client._request.return_value = response
        return client

    def test_basic_chat(self):
        response = {
            "choices": [{"message": {"content": "Hello!"}}],
            "cocapn_cost": "0.001",
        }
        client = self._make_client(response)
        agent = AgentHandle(agent_id="agent-1", client=client)
        result = agent.chat("hi")
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert agent.conversation_length == 2  # user + assistant

    def test_system_prompt(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client, system_prompt="Be nice")
        agent.chat("hi")
        # Check the builder included the system prompt
        call_args = client._request.call_args
        body = call_args[0][2]
        messages = body["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be nice"

    def test_conversation_history(self):
        response = {"choices": [{"message": {"content": "reply"}}]}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client)
        agent.chat("first")
        agent.chat("second")
        # Second call should have 3 messages: first user, first assistant, second user
        second_call = client._request.call_args_list[1]
        body = second_call[0][2]
        assert len(body["messages"]) == 3

    def test_reset_conversation(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client)
        agent.chat("hi")
        assert agent.conversation_length == 2
        agent.reset_conversation()
        assert agent.conversation_length == 0


# ─── ChatResponse Tests ───


class TestChatResponse:
    def test_repr(self):
        r = ChatResponse("hello world", 0.01, 10, 20, "gpt-4o", "openai", {})
        s = repr(r)
        assert "hello world" in s
        assert "0.01" in s

    def test_fields(self):
        r = ChatResponse("text", 0.5, 5, 10, "model", "provider", {"raw": True})
        assert r.text == "text"
        assert r.cost == 0.5
        assert r.tokens_in == 5
        assert r.tokens_out == 10
        assert r.raw == {"raw": True}


# ─── Client Tests (with mocked HTTP) ───


class TestCocapnClient:
    def _mock_response(self, status=200, data=None):
        resp = MagicMock()
        resp.status = status
        resp.read.return_value = json.dumps(data or {}).encode()
        return resp

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_chat(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "Hi there!"}}],
            "cocapn_cost": "0.002",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "deepseek-chat",
            "cocapn_provider": "deepseek",
        })

        client = CocapnClient(api_key="test-key")
        response = client.chat("Hello!")
        assert response.text == "Hi there!"
        assert response.cost == 0.002
        assert response.tokens_in == 10
        assert response.tokens_out == 5
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_chat_with_options(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "Summarized."}}],
            "model": "gpt-4o",
        })

        client = CocapnClient(api_key="key")
        resp = client.chat(
            "Summarize this",
            model="gpt-4o",
            system="Be concise",
            max_tokens=100,
            temperature=0.3,
        )
        assert resp.model == "gpt-4o"

        # Verify request body
        call_body = json.loads(mock_conn.request.call_args[1]["body"])
        assert call_body["model"] == "gpt-4o"
        assert call_body["temperature"] == 0.3
        assert call_body["max_tokens"] == 100
        assert call_body["messages"][0]["role"] == "system"
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_models(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "data": [
                {"id": "deepseek-chat", "owned_by": "deepseek", "cocapn_cost_in": 0.14, "cocapn_cost_out": 0.28},
                {"id": "gpt-4o", "owned_by": "openai", "cocapn_cost_in": 5.0, "cocapn_cost_out": 15.0},
            ]
        })

        client = CocapnClient(api_key="key")
        models = client.models()
        assert len(models) == 2
        assert models[0]["id"] == "deepseek-chat"
        assert models[1]["cost_out"] == 15.0
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_usage(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "totalCost": 0.42,
            "requests": 127,
        })

        client = CocapnClient(api_key="key")
        usage = client.usage("week")
        assert usage["totalCost"] == 0.42
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_context_manager(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "hi"}}],
            "model": "test",
        })

        with CocapnClient(api_key="key") as client:
            resp = client.chat("hello")
            assert resp.text == "hi"

    def test_no_api_key_raises(self):
        os.environ.pop("COCAPN_API_KEY", None)
        with pytest.raises(ValueError):
            CocapnClient()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_http_error(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(401, {
            "error": {"message": "Invalid API key"}
        })

        client = CocapnClient(api_key="bad-key")
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            client.chat("hello")
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_agent_handle(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "agent reply"}}],
            "model": "deepseek-chat",
        })

        client = CocapnClient(api_key="key")
        agent = client.agent("agent-123", name="Helper", system_prompt="Be helpful")
        assert isinstance(agent, AgentHandle)
        assert agent.agent_id == "agent-123"
        assert agent.name == "Helper"
        client.close()


# ─── Integration-style test (fully mocked) ───


class TestEndToEnd:
    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_conversation_flow(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn

        responses = [
            self._make_response({"choices": [{"message": {"content": "Hello!"}}], "model": "deepseek-chat"}),
            self._make_response({"choices": [{"message": {"content": "I'm doing well!"}}], "model": "deepseek-chat"}),
        ]
        mock_conn.getresponse.side_effect = responses

        client = CocapnClient(api_key="key")
        agent = client.agent("a1", system_prompt="Be friendly")

        r1 = agent.chat("Hi")
        assert r1["choices"][0]["message"]["content"] == "Hello!"

        r2 = agent.chat("How are you?")
        assert r2["choices"][0]["message"]["content"] == "I'm doing well!"

        # Verify history was sent
        second_body = json.loads(mock_conn.request.call_args_list[1][1]["body"])
        msgs = second_body["messages"]
        # system + first user + first assistant + second user
        assert len(msgs) == 4
        client.close()

    def _make_response(self, data, status=200):
        resp = MagicMock()
        resp.status = status
        resp.read.return_value = json.dumps(data).encode()
        return resp
