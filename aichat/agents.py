"""Агенты. Консоль и будущие боты вызывают один и тот же метод reply."""

import json
from collections.abc import Iterator
from pathlib import Path

from aichat.config import ConfigError
from aichat.providers import ModelProvider, StreamPiece

_ROLES = {"user", "assistant", "system"}


def load_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"Не удалось прочитать историю {path.name}: {error}") from error
    if not isinstance(data, list):
        raise ConfigError(f"Файл истории {path.name} должен быть списком сообщений.")
    messages = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in _ROLES and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def save_history(path: Path, messages: list[dict]) -> None:
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


class TextAgent:
    """Текстовый диалог: хранит историю и отправляет её с каждым запросом."""

    def __init__(self, provider: ModelProvider, history_path: Path | None = None):
        self.provider = provider
        self.history_path = history_path
        self.history = load_history(history_path) if history_path else []

    def _save(self) -> None:
        if self.history_path is not None:
            save_history(self.history_path, self.history)

    def clear(self) -> None:
        self.history = []
        self._save()

    def reply(self, user_text: str) -> Iterator[StreamPiece]:
        self.history.append({"role": "user", "content": user_text})
        answer: list[str] = []
        try:
            for piece in self.provider.stream(self.history):
                if piece.kind == "answer":
                    answer.append(piece.text)
                yield piece
        except Exception:
            self.history.pop()
            raise
        self.history.append({"role": "assistant", "content": "".join(answer)})
        self._save()


class VoiceAgent(TextAgent):
    """Голосовой агент.

    Сейчас делит текстовое ядро с TextAgent. Речь ProxyAPI живёт на
    /v1/audio/transcriptions и /v1/audio/speech того же адреса и ключа.
    Микрофон и синтез подключаются отдельным интерфейсом, не новым ядром.
    """
