"""Консольный чат: диалог и стрим думания через Colorama."""

from colorama import Fore, Style, init

from aichat import __version__
from aichat.agents import TextAgent
from aichat.config import ROOT, ConfigError, load_config
from aichat.providers import StreamPiece, build_provider


def _mode_title(mode: str) -> str:
    if mode == "think":
        return "думающая"
    return "обычная"


def help_text(agent: TextAgent) -> str:
    title = _mode_title(agent.provider.mode)
    chat_model = getattr(agent.provider, "chat_model", agent.provider.model)
    thinking_model = getattr(agent.provider, "thinking_model", agent.provider.model)
    return (
        "Команды пишутся отдельной строкой, без текста вопроса.\n"
        f"Сейчас: {title} модель ({agent.provider.model}). "
        f"В истории {len(agent.history)} сообщений.\n"
        "\n"
        "/chat\n"
        f"  Обычная модель {chat_model}. Ответ без хода думания.\n"
        "  Пример: /chat\n"
        "\n"
        "/think\n"
        f"  Думающая модель {thinking_model}. Печатает рассуждения и показатели токенов.\n"
        "  Пример: /think\n"
        "          Сколько будет 17 * 24?\n"
        "\n"
        "/mode\n"
        "  Показывает текущий режим и число сохранённых реплик.\n"
        "  Пример: /mode\n"
        "\n"
        "/clear\n"
        "  Стирает историю. Следующий вопрос пойдёт без прошлого диалога.\n"
        "  Пример: /clear\n"
        "\n"
        "/help\n"
        "  Это описание.\n"
        "\n"
        "exit\n"
        "  Выход. История остаётся в history.json и подгрузится при следующем запуске.\n"
        "  Пример: exit\n"
        "\n"
        "Любая другая строка уходит в модель как реплика.\n"
        "  Пример: Объясни, что такое API, в двух предложениях."
    )


def apply_command(agent: TextAgent, text: str) -> str | None:
    command = text.lower()
    if command in {"/help", "help", "/?", "?"}:
        return help_text(agent)
    if command in {"/chat", "/обычная"}:
        agent.provider.use("chat")
        return f"Режим: обычная модель ({agent.provider.model})."
    if command in {"/think", "/думающая"}:
        agent.provider.use("think")
        return f"Режим: думающая модель ({agent.provider.model})."
    if command in {"/clear", "/сброс"}:
        agent.clear()
        return "История очищена."
    if command == "/mode":
        title = _mode_title(agent.provider.mode)
        return f"Сейчас: {title} модель ({agent.provider.model}). Сообщений в истории: {len(agent.history)}."
    return None


def _print_piece(piece: StreamPiece, state: dict) -> None:
    if piece.kind == "thinking":
        if not state["thinking"]:
            print(Fore.MAGENTA + "Думает: " + Style.RESET_ALL, end="", flush=True)
            state["thinking"] = True
        print(Style.DIM + piece.text + Style.RESET_ALL, end="", flush=True)
        return
    if piece.kind == "metrics":
        if state["thinking"] or state["answer"]:
            print()
        print(Fore.CYAN + piece.text + Style.RESET_ALL)
        state["metrics"] = True
        return
    if state["thinking"] and not state["answer"]:
        print()
    if not state["answer"]:
        print(Fore.GREEN + "Ответ: " + Style.RESET_ALL, end="", flush=True)
        state["answer"] = True
    print(piece.text, end="", flush=True)


def format_request_error(error: Exception) -> str:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body
        if isinstance(detail, dict):
            detail = detail.get("message", detail)
        return f"Ошибка запроса: {detail}"
    return f"Ошибка запроса: {error}"


def run(agent: TextAgent) -> None:
    init(autoreset=True)
    title = _mode_title(agent.provider.mode)
    print(
        Fore.CYAN
        + (
            f"Чат-маршрутизатор AI, версия {__version__}. "
            f"Режим: {title} ({agent.provider.model}). "
            "Команды: /help."
        )
        + Style.RESET_ALL
    )
    if agent.history:
        print(Fore.CYAN + f"Загружена история: {len(agent.history)} сообщений." + Style.RESET_ALL)
    while True:
        try:
            user_text = input(Fore.YELLOW + "Вы: " + Style.RESET_ALL).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "/exit"}:
            break
        status = apply_command(agent, user_text)
        if status is not None:
            print(status)
            continue
        state = {"thinking": False, "answer": False}
        try:
            for piece in agent.reply(user_text):
                _print_piece(piece, state)
        except ConfigError as error:
            print(Fore.RED + str(error) + Style.RESET_ALL)
            continue
        except Exception as error:
            print(Fore.RED + format_request_error(error) + Style.RESET_ALL)
            continue
        if (state["answer"] or state["thinking"]) and not state.get("metrics"):
            print()


def main() -> None:
    try:
        config = load_config()
        provider = build_provider(config)
    except ConfigError as error:
        init(autoreset=True)
        print(Fore.RED + str(error) + Style.RESET_ALL)
        return
    run(TextAgent(provider, ROOT / "history.json"))
