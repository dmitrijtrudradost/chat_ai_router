"""Провайдеры моделей.

ProxyAPI описан на https://proxyapi.ru/docs:
один адрес https://api.proxyapi.ru/v1, формат OpenAI, ключ в Authorization.
Модель указывается целиком: openai/gpt-5-mini, anthropic/claude-haiku-4-5.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from aichat.config import Config, ConfigError


@dataclass(frozen=True)
class StreamPiece:
    kind: str  # "thinking" или "answer"
    text: str


class ModelProvider(Protocol):
    name: str

    def stream(self, messages: list[dict]) -> Iterator[StreamPiece]:
        """Отдать куски думания и ответа по мере генерации."""


def _require(key: str, env_name: str) -> str:
    if not key:
        raise ConfigError(f"В .env пусто поле {env_name}.")
    return key


def _reasoning_text(delta) -> str:
    for field in ("reasoning_content", "reasoning"):
        value = getattr(delta, field, None)
        if isinstance(value, str) and value:
            return value
    extra = getattr(delta, "model_extra", None) or {}
    for field in ("reasoning_content", "reasoning"):
        value = extra.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def iter_openai_chunks(chunks) -> Iterator[StreamPiece]:
    """Разобрать поток /v1/chat/completions.

    Текст лежит в delta.content. Последний фрагмент ProxyAPI приходит
    с пустым choices и полем usage — его нужно пропустить.
    """
    for chunk in chunks:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        thinking = _reasoning_text(delta)
        if thinking:
            yield StreamPiece("thinking", thinking)
        content = getattr(delta, "content", None)
        if content:
            yield StreamPiece("answer", content)


class OpenAICompatibleProvider:
    """Чат в формате OpenAI: ProxyAPI, прямой OpenAI и прямой DeepSeek."""

    def __init__(self, name: str, api_key: str, model: str, base_url: str | None, reasoning_effort: str):
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort

    def stream(self, messages: list[dict]) -> Iterator[StreamPiece]:
        from openai import OpenAI

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        request = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if self.reasoning_effort:
            request["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        yield from iter_openai_chunks(client.chat.completions.create(**request))


class AnthropicProvider:
    """Прямой Anthropic Messages API. Thinking включается бюджетом токенов."""

    def __init__(self, api_key: str, model: str, thinking_budget: int | None):
        self.name = "anthropic"
        self.api_key = api_key
        self.model = model
        self.thinking_budget = thinking_budget

    def stream(self, messages: list[dict]) -> Iterator[StreamPiece]:
        import anthropic

        system_parts = [item["content"] for item in messages if item["role"] == "system"]
        dialog = [item for item in messages if item["role"] != "system"]
        request = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": dialog,
        }
        if system_parts:
            request["system"] = "\n".join(system_parts)
        if self.thinking_budget:
            request["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            request["max_tokens"] = max(request["max_tokens"], self.thinking_budget + 1024)

        client = anthropic.Anthropic(api_key=self.api_key)
        with client.messages.stream(**request) as stream:
            for event in stream:
                if getattr(event, "type", "") != "content_block_delta":
                    continue
                delta = event.delta
                delta_type = getattr(delta, "type", "")
                if delta_type == "thinking_delta":
                    text = getattr(delta, "thinking", "")
                    if text:
                        yield StreamPiece("thinking", text)
                elif delta_type == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        yield StreamPiece("answer", text)


def build_provider(config: Config) -> ModelProvider:
    name = config.provider
    if name == "proxy":
        return OpenAICompatibleProvider(
            name="proxy",
            api_key=_require(config.proxy_api_key, "PROXYAPI_API_KEY"),
            model=config.proxy_model,
            base_url=config.proxy_base_url,
            reasoning_effort=config.reasoning_effort,
        )
    if name == "openai":
        return OpenAICompatibleProvider(
            name="openai",
            api_key=_require(config.openai_api_key, "OPENAI_API_KEY"),
            model=config.openai_model,
            base_url=None,
            reasoning_effort=config.reasoning_effort,
        )
    if name == "deepseek":
        return OpenAICompatibleProvider(
            name="deepseek",
            api_key=_require(config.deepseek_api_key, "DEEPSEEK_API_KEY"),
            model=config.deepseek_model,
            base_url=config.deepseek_base_url,
            reasoning_effort=config.reasoning_effort,
        )
    if name == "anthropic":
        return AnthropicProvider(
            api_key=_require(config.anthropic_api_key, "ANTHROPIC_API_KEY"),
            model=config.anthropic_model,
            thinking_budget=config.anthropic_thinking_budget,
        )
    raise ConfigError(
        "PROVIDER должен быть одним из: proxy, openai, anthropic, deepseek. "
        f"Сейчас указано: {name!r}."
    )
