"""Additional tests for improved coverage of the Cocapn SDK."""

import json
import os
import time
from unittest.mock import MagicMock, patch, call

import pytest

from cocapn_sdk.config import SDKConfig, DEFAULT_BASE_URL, DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES
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


# ─── Config: additional coverage ───


class TestSDKConfigExtra:
    def test_validate_passes_with_valid_config(self):
        cfg = SDKConfig(api_key="key")
        cfg.validate()  # should not raise

    def test_custom_base_url(self):
        cfg = SDKConfig(api_key="key", base_url="https://custom.example.com")
        assert cfg.base_url == "https://custom.example.com"

    def test_custom_timeout_and_retries(self):
        cfg = SDKConfig(api_key="key", timeout=60.0, max_retries=5)
        assert cfg.timeout == 60.0
        assert cfg.max_retries == 5

    def test_default_model_and_tokens(self):
        cfg = SDKConfig(api_key="key")
        assert cfg.default_model == "deepseek-chat"
        assert cfg.default_max_tokens == 4096
        assert cfg.default_temperature is None

    def test_auth_header_includes_extra(self):
        cfg = SDKConfig(api_key="k", extra_headers={"X-Trace": "abc", "X-Other": "1"})
        h = cfg.auth_header
        assert h["X-Trace"] == "abc"
        assert h["X-Other"] == "1"
        assert h["Authorization"] == "Bearer k"

    def test_env_key_overridden_by_explicit(self):
        with patch.dict(os.environ, {"COCAPN_API_KEY": "env-key"}):
            cfg = SDKConfig(api_key="explicit-key")
            assert cfg.api_key == "explicit-key"

    def test_zero_timeout_invalid(self):
        cfg = SDKConfig(api_key="key", timeout=0)
        with pytest.raises(ValueError, match="timeout"):
            cfg.validate()

    def test_zero_max_retries_valid(self):
        cfg = SDKConfig(api_key="key", max_retries=0)
        cfg.validate()  # zero is non-negative, should pass

    def test_default_constants(self):
        assert DEFAULT_BASE_URL == "https://cocapn.ai"
        assert DEFAULT_TIMEOUT == 30.0
        assert DEFAULT_MAX_RETRIES == 3


# ─── Errors: additional coverage ───


class TestErrorsExtra:
    def test_cocapn_error_defaults(self):
        err = CocapnError("msg")
        assert err.status_code is None
        assert err.data == {}

    def test_cocapn_error_is_exception(self):
        with pytest.raises(CocapnError):
            raise CocapnError("boom")

    def test_rate_limit_error_default_message(self):
        err = RateLimitError()
        assert "Rate limit" in str(err)
        assert err.retry_after is None

    def test_rate_limit_error_custom_retry_after(self):
        err = RateLimitError("slow", retry_after=5.0)
        assert err.retry_after == 5.0

    def test_classify_429_without_retry_after(self):
        err = classify_error(429, "slow", data={})
        assert isinstance(err, RateLimitError)
        assert err.retry_after is None

    def test_classify_429_with_none_data(self):
        err = classify_error(429, "slow", data=None)
        assert isinstance(err, RateLimitError)
        assert err.retry_after is None

    def test_error_hierarchy(self):
        assert issubclass(AuthenticationError, CocapnError)
        assert issubclass(RateLimitError, CocapnError)
        assert issubclass(NotFoundError, CocapnError)
        assert issubclass(ServerError, CocapnError)
        assert issubclass(ValidationError, CocapnError)

    def test_classify_502(self):
        assert isinstance(classify_error(502, "bad gateway"), ServerError)

    def test_classify_503(self):
        assert isinstance(classify_error(503, "unavailable"), ServerError)

    def test_classify_422(self):
        err = classify_error(422, "unprocessable")
        assert isinstance(err, CocapnError)
        assert not isinstance(err, (ValidationError, AuthenticationError, NotFoundError, RateLimitError, ServerError))

    def test_is_retryable_not_found(self):
        assert not is_retryable(NotFoundError("gone"))

    def test_backoff_delay_increases(self):
        d0 = backoff_delay(0, base=1.0)
        d2 = backoff_delay(2, base=1.0)
        # d2 should be significantly larger than d0 on average
        assert d2 > d0

    def test_retry_with_backoff_respects_retry_after(self):
        """RateLimitError with retry_after should use max(delay, retry_after)."""
        calls = {"n": 0}
        sleep_times = []

        def fake_sleep(t):
            sleep_times.append(t)

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RateLimitError("slow", retry_after=10.0, status_code=429)
            return "ok"

        with patch("cocapn_sdk.errors.time.sleep", side_effect=fake_sleep):
            result = retry_with_backoff(flaky, max_retries=3, base_delay=0.5)

        assert result == "ok"
        assert len(sleep_times) == 1
        assert sleep_times[0] >= 10.0  # retry_after should dominate

    def test_retry_with_backoff_non_retryable_gives_up_immediately(self):
        """Non-retryable errors should not retry at all."""
        calls = {"n": 0}

        def fail():
            calls["n"] += 1
            raise AuthenticationError("nope")

        with pytest.raises(AuthenticationError):
            retry_with_backoff(fail, max_retries=5)

        assert calls["n"] == 1

    def test_retry_exhausted_raises_last_error(self):
        calls = {"n": 0}

        def always_fail():
            calls["n"] += 1
            raise ServerError("down", status_code=500)

        with patch("cocapn_sdk.errors.time.sleep"):
            with pytest.raises(ServerError) as exc_info:
                retry_with_backoff(always_fail, max_retries=2)

        # Should try max_retries + 1 times
        assert calls["n"] == 3


# ─── Message/MessageBuilder: additional coverage ───


class TestMessageBuilderExtra:
    def test_with_model(self):
        builder = MessageBuilder().with_model("gpt-4o")
        assert builder.model == "gpt-4o"
        payload = builder.user("hi").build()
        assert payload["model"] == "gpt-4o"

    def test_history_with_multiple_roles(self):
        builder = MessageBuilder()
        builder.history([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        assert len(builder.messages) == 3
        assert builder.messages[0].role == "system"
        assert builder.messages[1].role == "user"
        assert builder.messages[2].role == "assistant"

    def test_build_without_messages(self):
        payload = MessageBuilder().build()
        assert payload["messages"] == []

    def test_temperature_none_excluded(self):
        payload = MessageBuilder(temperature=None).user("x").build()
        assert "temperature" not in payload

    def test_temperature_set(self):
        payload = MessageBuilder(temperature=0.9).user("x").build()
        assert payload["temperature"] == 0.9

    def test_with_temperature_returns_builder(self):
        b = MessageBuilder().with_temperature(0.5)
        assert isinstance(b, MessageBuilder)
        assert b.temperature == 0.5

    def test_with_max_tokens_returns_builder(self):
        b = MessageBuilder().with_max_tokens(100)
        assert isinstance(b, MessageBuilder)
        assert b.max_tokens == 100

    def test_with_stream_true(self):
        payload = MessageBuilder().user("x").with_stream(True).build()
        assert payload["stream"] is True

    def test_with_stream_default_true(self):
        payload = MessageBuilder().user("x").with_stream().build()
        assert payload["stream"] is True

    def test_with_stream_false(self):
        payload = MessageBuilder().user("x").with_stream(False).build()
        assert "stream" not in payload

    def test_reset_preserves_settings(self):
        builder = MessageBuilder(model="gpt-4o", max_tokens=100, temperature=0.5)
        builder.user("x")
        builder.reset()
        assert builder.model == "gpt-4o"
        assert builder.max_tokens == 100
        assert builder.temperature == 0.5
        assert len(builder.messages) == 0

    def test_message_dataclass(self):
        m = Message(role="assistant", content="world")
        assert m.role == "assistant"
        assert m.content == "world"

    def test_builder_fluent_all_methods(self):
        """Every fluent method should return self for chaining."""
        builder = MessageBuilder()
        assert builder.system("s") is builder
        assert builder.user("u") is builder
        assert builder.assistant("a") is builder
        assert builder.history([]) is builder
        assert builder.with_model("m") is builder
        assert builder.with_max_tokens(1) is builder
        assert builder.with_temperature(0.1) is builder
        assert builder.with_stream() is builder
        assert builder.reset() is builder


# ─── AgentHandle: additional coverage ───


class TestAgentHandleExtra:
    def _make_client(self, response):
        client = MagicMock()
        client._request.return_value = response
        return client

    def test_agent_name_and_description(self):
        client = MagicMock()
        agent = AgentHandle(
            agent_id="a1", client=client, name="Bot", description="A test bot"
        )
        assert agent.name == "Bot"
        assert agent.description == "A test bot"
        assert agent.agent_id == "a1"

    def test_chat_with_temperature(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client)
        agent.chat("hi", temperature=0.3)

        call_args = client._request.call_args
        body = call_args[0][2]
        assert body["temperature"] == 0.3

    def test_chat_with_custom_model(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client)
        agent.chat("hi", model="gpt-4o")

        call_args = client._request.call_args
        body = call_args[0][2]
        assert body["model"] == "gpt-4o"

    def test_chat_with_max_tokens(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client)
        agent.chat("hi", max_tokens=1024)

        call_args = client._request.call_args
        body = call_args[0][2]
        assert body["max_tokens"] == 1024

    def test_chat_no_choices_in_response(self):
        """Response without choices should not crash."""
        response = {"cocapn_cost": "0.001"}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client)
        result = agent.chat("hi")
        # No assistant content tracked
        assert agent.conversation_length == 1  # only user message

    def test_conversation_length_initially_zero(self):
        agent = AgentHandle(agent_id="a", client=MagicMock())
        assert agent.conversation_length == 0

    def test_multiple_resets(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client)
        agent.chat("hi")
        agent.reset_conversation()
        agent.reset_conversation()  # double reset is fine
        assert agent.conversation_length == 0

    def test_chat_default_model_is_deepseek(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client)
        agent.chat("hi")

        body = client._request.call_args[0][2]
        assert body["model"] == "deepseek-chat"

    def test_chat_default_max_tokens(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        client = self._make_client(response)
        agent = AgentHandle(agent_id="a", client=client)
        agent.chat("hi")

        body = client._request.call_args[0][2]
        assert body["max_tokens"] == 4096


# ─── ChatResponse: additional coverage ───


class TestChatResponseExtra:
    def test_repr_truncates_long_text(self):
        long_text = "x" * 200
        r = ChatResponse(long_text, 0.01, 5, 10, "m", "p", {})
        s = repr(r)
        assert "ChatResponse(" in s
        # Should be truncated (not the full 200 chars)
        assert len(s) < 250

    def test_all_slots(self):
        r = ChatResponse("text", 1.0, 2, 3, "model", "provider", {"k": "v"})
        assert r.text == "text"
        assert r.cost == 1.0
        assert r.tokens_in == 2
        assert r.tokens_out == 3
        assert r.model == "model"
        assert r.provider == "provider"
        assert r.raw == {"k": "v"}


# ─── Client: additional coverage ───


class TestCocapnClientExtra:
    def _mock_response(self, status=200, data=None):
        resp = MagicMock()
        resp.status = status
        resp.read.return_value = json.dumps(data or {}).encode()
        return resp

    def test_client_with_config_object(self):
        config = SDKConfig(api_key="cfg-key", base_url="https://example.com")
        with patch("cocapn_sdk.client.HTTPSConnection"):
            client = CocapnClient(config=config)
            assert client._config.api_key == "cfg-key"
            assert client._config.base_url == "https://example.com"
            client.close()

    def test_close_idempotent(self):
        with patch("cocapn_sdk.client.HTTPSConnection"):
            client = CocapnClient(api_key="key")
            client.close()
            client.close()  # should not raise

    @patch("cocapn_sdk.client.HTTPConnection")
    def test_http_connection(self, mock_conn_cls):
        """Non-HTTPS base_url should use HTTPConnection."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "hi"}}],
            "model": "test",
        })

        client = CocapnClient(api_key="key", base_url="http://localhost:8080")
        resp = client.chat("hello")
        assert resp.text == "hi"
        mock_conn_cls.assert_called_once()
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_chat_with_history(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "Casey"}}],
            "model": "deepseek-chat",
        })

        client = CocapnClient(api_key="key")
        history = [
            {"role": "user", "content": "My name is Casey"},
            {"role": "assistant", "content": "Hello Casey!"},
        ]
        resp = client.chat("What is my name?", history=history)

        body = json.loads(mock_conn.request.call_args[1]["body"])
        msgs = body["messages"]
        # history[0] + history[1] + user message
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "My name is Casey"
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_chat_provider_fallback(self, mock_conn_cls):
        """If no cocapn_provider in response, derive from model name."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "ok"}}],
            "model": "deepseek-chat",
        })

        client = CocapnClient(api_key="key")
        resp = client.chat("hi")
        assert resp.provider == "deepseek"
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_connection_error_handling(self, mock_conn_cls):
        """ConnectionError during request should be wrapped in CocapnError."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.request.side_effect = ConnectionError("refused")

        client = CocapnClient(api_key="key", config=SDKConfig(api_key="key", max_retries=0))
        with pytest.raises(CocapnError, match="Connection error"):
            client.chat("hello")
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_json_decode_error_handling(self, mock_conn_cls):
        """Non-JSON response should be handled gracefully."""
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b"not json at all"

        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = resp

        client = CocapnClient(api_key="key")
        result = client.models()  # should not crash
        assert isinstance(result, list)
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_400_error(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(400, {
            "error": {"message": "bad request"}
        })

        client = CocapnClient(api_key="key", config=SDKConfig(api_key="key", max_retries=0))
        with pytest.raises(ValidationError, match="bad request"):
            client.chat("hello")
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_404_error(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(404, {
            "error": {"message": "not found"}
        })

        client = CocapnClient(api_key="key", config=SDKConfig(api_key="key", max_retries=0))
        with pytest.raises(NotFoundError, match="not found"):
            client.chat("hello")
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_429_error(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(429, {
            "error": {"message": "rate limited"},
            "retry_after": 1.0,
        })

        client = CocapnClient(api_key="key", config=SDKConfig(api_key="key", max_retries=0))
        with pytest.raises(RateLimitError):
            client.chat("hello")
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_500_error(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(500, {
            "error": {"message": "internal error"}
        })

        client = CocapnClient(api_key="key", config=SDKConfig(api_key="key", max_retries=0))
        with pytest.raises(ServerError, match="internal error"):
            client.chat("hello")
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_error_without_message_key(self, mock_conn_cls):
        """Error response without structured error.message should still work."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(401, {
            "something": "else"
        })

        client = CocapnClient(api_key="key", config=SDKConfig(api_key="key", max_retries=0))
        with pytest.raises(AuthenticationError, match="HTTP 401"):
            client.chat("hello")
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_agent_method(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn

        client = CocapnClient(api_key="key")
        agent = client.agent(
            "agent-42",
            name="TestAgent",
            description="Testing",
            system_prompt="You test things",
        )
        assert isinstance(agent, AgentHandle)
        assert agent.agent_id == "agent-42"
        assert agent.name == "TestAgent"
        assert agent.description == "Testing"
        assert agent.system_prompt == "You test things"
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_usage_default_period(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {"requests": 5})

        client = CocapnClient(api_key="key")
        usage = client.usage()
        assert usage["requests"] == 5

        # Verify default period is 'day'
        call_args = mock_conn.request.call_args
        assert "/v1/usage?period=day" in call_args[0][1] or "period=day" in str(call_args)
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_models_empty_data(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {})

        client = CocapnClient(api_key="key")
        models = client.models()
        assert models == []
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_chat_response_defaults(self, mock_conn_cls):
        """Chat response with minimal data should use defaults."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "hi"}}],
        })

        client = CocapnClient(api_key="key")
        resp = client.chat("hello")
        assert resp.text == "hi"
        assert resp.cost == 0.0
        assert resp.tokens_in == 0
        assert resp.tokens_out == 0
        client.close()

    @patch("cocapn_sdk.client.HTTPSConnection")
    def test_chat_with_system_prompt(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.getresponse.return_value = self._mock_response(200, {
            "choices": [{"message": {"content": "ok"}}],
            "model": "test",
        })

        client = CocapnClient(api_key="key")
        client.chat("hi", system="Be concise")

        body = json.loads(mock_conn.request.call_args[1]["body"])
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "Be concise"
        client.close()


# ─── Package-level imports ───


class TestPackageImports:
    def test_version(self):
        import cocapn_sdk
        assert cocapn_sdk.__version__ == "1.0.0"

    def test_all_exports(self):
        import cocapn_sdk
        expected = [
            "CocapnClient", "SDKConfig", "AgentHandle", "MessageBuilder",
            "CocapnError", "AuthenticationError", "RateLimitError",
            "NotFoundError", "ServerError", "ValidationError",
        ]
        for name in expected:
            assert hasattr(cocapn_sdk, name), f"Missing export: {name}"

    def test_all_list(self):
        import cocapn_sdk
        assert len(cocapn_sdk.__all__) == 10
