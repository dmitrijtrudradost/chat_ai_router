"""Проверки разбора стрима, агента и выбора провайдера. Сеть не нужна."""

from types import SimpleNamespace

import pytest

from aichat.agents import TextAgent, VoiceAgent
from aichat.console import format_request_error
from aichat.config import Config, ConfigError
from aichat.providers import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    StreamPiece,
    build_provider,
    iter_openai_chunks,
)


def _config(**overrides) -> Config:
    data = dict(
        provider="proxy",
        proxy_api_key="proxy-key",
        proxy_base_url="https://api.proxyapi.ru/v1",
        proxy_model="openai/gpt-5-mini",
        reasoning_effort="medium",
        openai_api_key="openai-key",
        openai_model="gpt-5-mini",
        anthropic_api_key="anthropic-key",
        anthropic_model="claude-haiku-4-5",
        anthropic_thinking_budget=2048,
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
    )
    data.update(overrides)
    return Config(**data)


def _delta(**fields):
    delta = SimpleNamespace(model_extra={}, **fields)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_stream_skips_usage_chunk_and_splits_thinking():
    chunks = [
        _delta(content=None, reasoning_content="шаг "),
        _delta(content="ответ", reasoning_content=None),
        SimpleNamespace(choices=[]),
    ]
    pieces = list(iter_openai_chunks(chunks))
    assert pieces == [
        StreamPiece("thinking", "шаг "),
        StreamPiece("answer", "ответ"),
    ]


def test_reasoning_can_arrive_in_model_extra():
    delta = SimpleNamespace(content=None, reasoning_content=None, reasoning=None, model_extra={"reasoning": "думаю"})
    chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]
    pieces = list(iter_openai_chunks(chunks))
    assert pieces == [StreamPiece("thinking", "думаю")]


def test_empty_delta_yields_nothing():
    chunks = [_delta(content=None, reasoning_content=None), SimpleNamespace(choices=[SimpleNamespace(delta=None)])]
    assert list(iter_openai_chunks(chunks)) == []


class FakeProvider:
    name = "fake"

    def __init__(self, pieces=None, error=None):
        self.pieces = pieces or []
        self.error = error
        self.seen = None

    def stream(self, messages):
        self.seen = list(messages)
        if self.error:
            raise self.error
        yield from self.pieces


def test_text_agent_stores_answer_without_thinking():
    provider = FakeProvider(
        [
            StreamPiece("thinking", "черновик"),
            StreamPiece("answer", "Привет"),
            StreamPiece("answer", "!"),
        ]
    )
    agent = TextAgent(provider)
    assert list(agent.reply("Как дела?")) == provider.pieces
    assert agent.history == [
        {"role": "user", "content": "Как дела?"},
        {"role": "assistant", "content": "Привет!"},
    ]
    assert provider.seen[0]["content"] == "Как дела?"


def test_text_agent_drops_user_message_when_stream_fails():
    agent = TextAgent(FakeProvider(error=RuntimeError("сеть")))
    with pytest.raises(RuntimeError, match="сеть"):
        list(agent.reply("вопрос"))
    assert agent.history == []


def test_voice_agent_uses_text_core():
    agent = VoiceAgent(FakeProvider([StreamPiece("answer", "да")]))
    assert "".join(piece.text for piece in agent.reply("слышно?")) == "да"
    assert agent.history[-1]["content"] == "да"


def test_build_proxy_provider():
    provider = build_provider(_config())
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "proxy"
    assert provider.base_url == "https://api.proxyapi.ru/v1"
    assert provider.model == "openai/gpt-5-mini"
    assert provider.reasoning_effort == "medium"


def test_build_direct_providers():
    openai_provider = build_provider(_config(provider="openai"))
    assert openai_provider.base_url is None
    assert openai_provider.model == "gpt-5-mini"

    deepseek = build_provider(_config(provider="deepseek"))
    assert deepseek.base_url == "https://api.deepseek.com"
    assert deepseek.model == "deepseek-chat"

    anthropic = build_provider(_config(provider="anthropic"))
    assert isinstance(anthropic, AnthropicProvider)
    assert anthropic.thinking_budget == 2048


def test_request_error_shows_api_detail():
    error = Exception("Error code: 400")
    error.body = {"detail": "Model not supported"}
    assert format_request_error(error) == "Ошибка запроса: Model not supported"


def test_missing_key_and_unknown_provider():
    with pytest.raises(ConfigError, match="PROXYAPI_API_KEY"):
        build_provider(_config(proxy_api_key=""))
    with pytest.raises(ConfigError, match="PROVIDER"):
        build_provider(_config(provider="gemini"))
