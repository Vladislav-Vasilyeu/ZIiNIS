"""
Конфигурация NeuralTextStego.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Провайдеры ---
    openrouter_api_key: str = Field(default="")

    default_provider: Literal["openrouter", "ollama", "classic"] = "openrouter"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_num_ctx: int = 2048

    # --- Стеганография ---
    # Количество кандидатов на замену (2 → 1 бит, 4 → 2 бита)
    num_candidates: int = 4
    # Максимальная длина секретного сообщения (символы)
    max_secret_length: int = 2000
    # Температура запросов к LLM
    temperature: float = 0.2
    # Таймаут одного запроса к API (сек)
    api_timeout: float = 60.0
    # Количество повторных попыток
    max_retries: int = 3
    # Базовая задержка retry (сек)
    retry_base_delay: float = 1.0

    # --- Криптография ---
    enable_encryption_by_default: bool = False

    # --- Логирование ---
    log_level: str = "INFO"
    log_file: Path = Path("logs/neuraltextstego.log")

    # --- Пути ---
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = project_root / "data"
    experiments_dir: Path = project_root / "experiments" / "results"


settings = Settings()
