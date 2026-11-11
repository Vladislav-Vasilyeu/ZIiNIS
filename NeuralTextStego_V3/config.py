"""
Конфигурация NeuralTextStego.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(Path(__file__).resolve().with_name(".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Провайдеры нейросетевых методов ---
    # Retained for the preserved OpenRouter modules; the application never
    # selects this provider.
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = "openrouter/free"
    tooken_club_api_key: str = Field(default="", repr=False)
    tooken_club_model: str = "gpt-5.5"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_num_ctx: int = 2048

    # --- Метод по умолчанию ---
    default_method: Literal[
        "classic_synonyms",
        "classic_whitespace",
        "neural_rank",
        "neural_arithmetic",
        "neural_sampling",
    ] = "classic_synonyms"
    default_logits_provider: Literal["ollama", "tooken_club"] = "ollama"

    @field_validator("default_logits_provider", mode="before")
    @classmethod
    def migrate_legacy_logits_provider(cls, value: object) -> object:
        """Ignore legacy OpenRouter selection saved in a user's .env file."""
        return "ollama" if value == "openrouter" else value

    # --- Параметры кодирования ---
    num_candidates: int = 4
    rank_chunk_bits: int = 2
    top_logprobs: int = 20
    temperature: float = 0.8
    max_secret_length: int = 2000
    max_gen_tokens: int = 512
    use_marker_by_default: bool = True

    # --- Сеть ---
    api_timeout: float = 90.0
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # --- UI persistence ---
    ui_state_file: Path = Path("data/ui_state.json")

    # --- Логирование ---
    log_level: str = "INFO"
    log_file: Path = Path("logs/neuraltextstego.log")

    # --- Пути ---
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = project_root / "data"
    experiments_dir: Path = project_root / "experiments" / "results"


settings = Settings()
