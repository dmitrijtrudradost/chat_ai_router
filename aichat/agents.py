"""Агенты. Консоль и будущие боты вызывают один и тот же метод reply."""

from collections.abc import Iterator

from aichat.providers import ModelProvider, StreamPiece


class TextAgent:
    """Текстовый диалог: копит историю и стримит ответ модели."""

    def __init__(self, provider: ModelProvider):
        self.provider = provider
        self.history: list[dict] = []

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


class VoiceAgent(TextAgent):
    """Голосовой агент.

    Сейчас делит текстовое ядро с TextAgent. Речь ProxyAPI живёт на
    /v1/audio/transcriptions и /v1/audio/speech того же адреса и ключа.
    Микрофон и синтез подключаются отдельным интерфейсом, не новым ядром.
    """
