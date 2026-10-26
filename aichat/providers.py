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


def _plain(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    data = {}
    extra = getattr(obj, "model_extra", None) or {}
    if isinstance(extra, dict):
        data.update(extra)
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "input_tokens",
        "output_tokens",
        "completion_tokens_details",
        "output_tokens_details",
    ):
        if hasattr(obj, name):
            data[name] = getattr(obj, name)
    return data


def _number(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def format_usage(usage) -> str:
    """Собрать строку показателей. Пустая строка, если чисел нет."""
    data = _plain(usage)
    details = _plain(data.get("completion_tokens_details")) or _plain(data.get("output_tokens_details"))
    reasoning = _number(data.get("reasoning_tokens"))
    if reasoning is None:
        reasoning = _number(details.get("reasoning_tokens"))
    prompt = _number(data.get("prompt_tokens"))
    if prompt is None:
        prompt = _number(data.get("input_tokens"))
    completion = _number(data.get("completion_tokens"))
    if completion is None:
        completion = _number(data.get("output_tokens"))
    total = _number(data.get("total_tokens"))
    parts = []
    if reasoning is not None:
        parts.append(f"рассуждения {reasoning}")
    if prompt is not None:
        parts.append(f"промпт {prompt}")
    if completion is not None:
        parts.append(f"ответ {completion}")
    if total is not None:
        parts.append(f"всего {total}")
    if not parts:
        return ""
    return "Показатели: " + ", ".join(parts) + "."


def iter_openai_chunks(chunks, show_metrics: bool = False) -> Iterator[StreamPiece]:
    """Разобрать поток /v1/chat/completions.

    Текст лежит в delta.content. Последний фрагмент ProxyAPI приходит
    с пустым choices и полем usage: текст из него не берётся, а показатели
    рассуждений — только в режиме думающей модели.
    """
    for chunk in chunks:
        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                thinking = _reasoning_text(delta)
                if thinking:
                    yield StreamPiece("thinking", thinking)
                content = getattr(delta, "content", None)
                if content:
                    yield StreamPiece("answer", content)
        if not show_metrics:
            continue
        usage = getattr(chunk, "usage", None)
        if usage is None:
            extra = getattr(chunk, "model_extra", None) or {}
            if isinstance(extra, dict):
                usage = extra.get("usage")
        text = format_usage(usage)
        if text:
            yield StreamPiece("metrics", text)


class OpenAICompatibleProvider:
    """Чат в формате OpenAI: ProxyAPI, прямой OpenAI и прямой DeepSeek.

    Два режима: chat — обычная модель без reasoning_effort;
    think — думающая модель, стрим рассуждений и показатели usage.
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        chat_model: str,
        thinking_model: str,
        base_url: str | None,
        reasoning_effort: str,
        mode: str = "chat",
    ):
        self.name = name
        self.api_key = api_key
        self.chat_model = chat_model
        self.thinking_model = thinking_model or chat_model
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.mode = "chat"
        self.use(mode)

    def use(self, mode: str) -> None:
        if mode not in {"chat", "think"}:
            raise ConfigError("Режим должен быть chat или think.")
        self.mode = mode

    @property
    def model(self) -> str:
        if self.mode == "think":
            return self.thinking_model
        return self.chat_model

    def chat_request(self, messages: list[dict]) -> dict:
        request = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if self.mode == "think" and self.reasoning_effort:
            request["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        return request

    def stream(self, messages: list[dict]) -> Iterator[StreamPiece]:
        from openai import OpenAI

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        yield from iter_openai_chunks(
            client.chat.completions.create(**self.chat_request(messages)),
            show_metrics=self.mode == "think",
        )


class AnthropicProvider:
    """Прямой Anthropic Messages API. Thinking включается в режиме think."""

    def __init__(
        self,
        api_key: str,
        model: str,
        thinking_budget: int | None,
        thinking_model: str = "",
        mode: str = "chat",
    ):
        self.name = "anthropic"
        self.api_key = api_key
        self.chat_model = model
        self.thinking_model = thinking_model or model
        self.thinking_budget = thinking_budget
        self.mode = "chat"
        self.use(mode)

    def use(self, mode: str) -> None:
        if mode not in {"chat", "think"}:
            raise ConfigError("Режим должен быть chat или think.")
        self.mode = mode

    @property
    def model(self) -> str:
        if self.mode == "think":
            return self.thinking_model
        return self.chat_model

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
        if self.mode == "think" and self.thinking_budget:
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
            if self.mode == "think" and hasattr(stream, "get_final_message"):
                text = format_usage(stream.get_final_message().usage)
                if text:
                    yield StreamPiece("metrics", text)


def _pair(default_model: str, config: Config) -> tuple[str, str]:
    chat_model = config.chat_model or default_model
    return chat_model, config.thinking_model or chat_model


def build_provider(config: Config) -> ModelProvider:
    name = config.provider
    if name == "proxy":
        chat_model, thinking_model = _pair(config.proxy_model, config)
        return OpenAICompatibleProvider(
            name="proxy",
            api_key=_require(config.proxy_api_key, "PROXYAPI_API_KEY"),
            chat_model=chat_model,
            thinking_model=thinking_model,
            base_url=config.proxy_base_url,
            reasoning_effort=config.reasoning_effort,
            mode=config.chat_mode,
        )
    if name == "openai":
        chat_model, thinking_model = _pair(config.openai_model, config)
        return OpenAICompatibleProvider(
            name="openai",
            api_key=_require(config.openai_api_key, "OPENAI_API_KEY"),
            chat_model=chat_model,
            thinking_model=thinking_model,
            base_url=None,
            reasoning_effort=config.reasoning_effort,
            mode=config.chat_mode,
        )
    if name == "deepseek":
        chat_model, thinking_model = _pair(config.deepseek_model, config)
        return OpenAICompatibleProvider(
            name="deepseek",
            api_key=_require(config.deepseek_api_key, "DEEPSEEK_API_KEY"),
            chat_model=chat_model,
            thinking_model=thinking_model,
            base_url=config.deepseek_base_url,
            reasoning_effort=config.reasoning_effort,
            mode=config.chat_mode,
        )
    if name == "anthropic":
        chat_model, thinking_model = _pair(config.anthropic_model, config)
        return AnthropicProvider(
            api_key=_require(config.anthropic_api_key, "ANTHROPIC_API_KEY"),
            model=chat_model,
            thinking_model=thinking_model,
            thinking_budget=config.anthropic_thinking_budget,
            mode=config.chat_mode,
        )
    raise ConfigError(
        "PROVIDER должен быть одним из: proxy, openai, anthropic, deepseek. "
        f"Сейчас указано: {name!r}."
    )
