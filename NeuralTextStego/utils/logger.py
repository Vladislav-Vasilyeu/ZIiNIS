"""
Настройка логирования (loguru).
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config import settings


def setup_logger() -> None:
    """Инициализировать логгер: консоль + файл."""
    logger.remove()

    # Консоль
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    )

    # Файл
    log_path = settings.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        level="DEBUG",
        rotation="5 MB",
        retention="10 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
