# Чат-маршрутизатор AI, версия 1.0

Консольный чат на Python. Маршрутизатор отправляет реплику в модели OpenAI, Anthropic и DeepSeek напрямую или через [ProxyAPI](https://proxyapi.ru/docs). Ответ и ход думания модели печатаются по мере генерации.

## Запуск

Нужен Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
python main.py
```

В `.env` впишите ключ выбранного провайдера. По умолчанию это `PROVIDER=proxy` и поле `PROXYAPI_API_KEY`. Выход из чата: `exit`.

## Провайдеры

Значение `PROVIDER` в `.env`:

| Значение | Куда идёт запрос | Ключ |
| --- | --- | --- |
| `proxy` | `https://api.proxyapi.ru/v1` | `PROXYAPI_API_KEY` |
| `openai` | API OpenAI | `OPENAI_API_KEY` |
| `anthropic` | API Anthropic | `ANTHROPIC_API_KEY` |
| `deepseek` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |

Для ProxyAPI модель указывается вместе с вендором, например `openai/gpt-5-mini`. Поле `REASONING_EFFORT` (`minimal`, `low`, `medium`, `high`) просит модель показать рассуждения. Если модель его не поддерживает и отвечает ошибкой 400, очистите поле.

Ключи хранятся только в `.env`. Этот файл не коммитится.

## Тесты

```powershell
python -m pytest tests -q
```

Тесты проверяют разбор стрима, историю диалога и выбор провайдера. К сети они не обращаются.

## Документы

- [architecture.md](architecture.md) — слои и поток сообщения.
- [tz/tz1.md](tz/tz1.md) — исходная постановка.
