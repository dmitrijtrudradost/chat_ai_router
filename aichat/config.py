"""Ключи и адреса из .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


class ConfigError(Exception):
    """В .env не хватает ключа или указан неизвестный провайдер."""


@dataclass(frozen=True)
class Config:
    provider: str
    proxy_api_key: str
    proxy_base_url: str
    proxy_model: str
    reasoning_effort: str
    openai_api_key: str
    openai_model: str
    anthropic_api_key: str
    anthropic_model: str
    anthropic_thinking_budget: int | None
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str


def load_config() -> Config:
    load_dotenv(ROOT / ".env")
    budget_raw = os.getenv("ANTHROPIC_THINKING_BUDGET", "").strip()
    budget = int(budget_raw) if budget_raw else None
    return Config(
        provider=os.getenv("PROVIDER", "proxy").strip().lower() or "proxy",
        proxy_api_key=os.getenv("PROXYAPI_API_KEY", "").strip(),
        proxy_base_url=os.getenv("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/v1").strip(),
        proxy_model=os.getenv("PROXYAPI_MODEL", "openai/gpt-5-mini").strip(),
        reasoning_effort=os.getenv("REASONING_EFFORT", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini").strip(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5").strip(),
        anthropic_thinking_budget=budget,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
    )
