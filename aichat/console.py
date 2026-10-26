"""Консольный чат: диалог и стрим думания через Colorama."""

from colorama import Fore, Style, init

from aichat import __version__
from aichat.agents import TextAgent
from aichat.config import ConfigError, load_config
from aichat.providers import StreamPiece, build_provider


def _print_piece(piece: StreamPiece, state: dict) -> None:
    if piece.kind == "thinking":
        if not state["thinking"]:
            print(Fore.MAGENTA + "Думает: " + Style.RESET_ALL, end="", flush=True)
            state["thinking"] = True
        print(Style.DIM + piece.text + Style.RESET_ALL, end="", flush=True)
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
    print(
        Fore.CYAN
        + f"Чат-маршрутизатор AI, версия {__version__}. Провайдер: {agent.provider.name}. Выход: exit."
        + Style.RESET_ALL
    )
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
        if state["answer"] or state["thinking"]:
            print()


def main() -> None:
    try:
        config = load_config()
        provider = build_provider(config)
    except ConfigError as error:
        init(autoreset=True)
        print(Fore.RED + str(error) + Style.RESET_ALL)
        return
    run(TextAgent(provider))
