"""
Генерация стегоконтейнера:
- через LLM (Ollama / Tooken Club) — основной путь;
- локальный шаблон — запасной, если сеть/модель недоступны.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

_SENTENCES = [
    "Информационные технологии продолжают стремительно развиваться и влиять на все сферы жизни.",
    "Современные системы обработки данных позволяют эффективно решать сложные аналитические задачи.",
    "Текстовые документы остаются одним из наиболее распространённых способов передачи знаний.",
    "Стеганография скрывает сам факт наличия сообщения внутри безобидного контейнера.",
    "Криптография защищает содержимое, но указывает на наличие защищаемой информации.",
    "Языковые модели открывают новые возможности для лингвистической стеганографии.",
    "Практические эксперименты помогают оценить ёмкость и незаметность выбранного метода.",
    "Надёжность извлечения зависит от устойчивости контейнера к редактированию и копированию.",
    "Электронный документооборот широко применяется в организациях и научных сообществах.",
    "Исследование характеристик метода требует серии воспроизводимых опытов и метрик.",
]

_SYNONYM_FRIENDLY = (
    "метод способ приём подход алгоритм быстро современный большой текст "
    "передача информация данные система анализ исследование документ сообщение "
    "защита безопасность естественный значительный процесс результат пример "
    "задача возможность область проект пользователь проверка точность обмен "
    "охрана обработка хранение показатель"
)


def _bits_for_secret(secret: str) -> int:
    # +16 бит префикса длины для режима без маркера
    return max(8, len(secret.encode("utf-8")) * 8) + 16


def generate_cover_local(secret: str, method: str = "classic_synonyms", *, seed: int | None = None) -> str:
    """Локальный (без сети) генератор — запасной вариант."""
    rng = random.Random(seed)
    needed_bits = _bits_for_secret(secret)
    method = (method or "").lower().strip()

    if method == "classic_whitespace":
        spaces_needed = max(80, needed_bits * 2)
        lexicon = [
            "данные", "текст", "метод", "анализ", "система", "защита",
            "канал", "модель", "оценка", "параметр", "пример", "результат",
        ]
        words = [rng.choice(lexicon) for _ in range(spaces_needed + 1)]
        return " ".join(words) + "."

    if method == "classic_synonyms":
        positions = max(30, math.ceil(needed_bits / 2) * 3)
        pool = _SYNONYM_FRIENDLY.split()
        parts: list[str] = []
        for i in range(positions):
            parts.append(rng.choice(pool))
            if (i + 1) % 8 == 0:
                parts.append(rng.choice(_SENTENCES))
        return " ".join(parts)

    intro = (
        "Напиши развёрнутый связный текст на русском языке по теме "
        "информационной безопасности и текстовой стеганографии.\n\n"
    )
    body = " ".join(rng.choice(_SENTENCES) for _ in range(max(4, math.ceil(needed_bits / 64))))
    return intro + body


def build_llm_cover_prompt(secret: str, method: str) -> str:
    """Промпт для LLM-генерации контейнера под секрет и метод."""
    n_chars = max(400, len(secret) * 40)
    method = (method or "").lower()
    if method == "classic_whitespace":
        extra = (
            "В тексте должно быть много обычных пробелов между словами "
            "(не меньше 100 пробелов). Пиши связными предложениями."
        )
    elif method == "classic_synonyms":
        extra = (
            "Используй разнообразную лексику: метод, способ, алгоритм, данные, "
            "система, защита, анализ, исследование, документ, сообщение, "
            "современный, быстрый, естественный — в разных формах."
        )
    else:
        extra = (
            "Текст будет использован как начало (промпт) для генеративной стеганографии. "
            "Сделай естественное введение, которое удобно продолжать."
        )
    return (
        f"Напиши связный научно-популярный текст на русском языке объёмом примерно "
        f"{n_chars} символов по теме защиты информации и стеганографии. "
        f"{extra} "
        "Без списков, без заголовков, без кавычек вокруг всего текста, только абзацы."
    )


async def generate_cover_for_secret_async(
    secret: str,
    method: str = "classic_synonyms",
    *,
    provider_name: str = "ollama",
    model: str | None = None,
    allow_local_fallback: bool = True,
) -> str:
    """
    Сгенерировать контейнер через нейросеть (Ollama/Tooken Club).
    При ошибке — локальный шаблон.
    """
    if not secret or not secret.strip():
        raise ValueError("Секрет пуст")

    prompt = build_llm_cover_prompt(secret, method)
    try:
        from providers.factory import create_logits_provider

        provider = create_logits_provider(provider_name, model=model)
        try:
            if hasattr(provider, "complete_text"):
                text = await provider.complete_text(prompt, max_tokens=500)
            else:
                text = ""
        finally:
            await provider.close()
        text = (text or "").strip()
        if len(text) >= 80:
            return text
        if not allow_local_fallback:
            raise RuntimeError("Модель вернула слишком короткий текст для контейнера.")
    except Exception:
        if not allow_local_fallback:
            raise

    return generate_cover_local(secret, method)


# Синхронная обёртка для старых вызовов
def generate_cover_for_secret(secret: str, method: str = "classic_synonyms", *, seed: int | None = None) -> str:
    return generate_cover_local(secret, method, seed=seed)
