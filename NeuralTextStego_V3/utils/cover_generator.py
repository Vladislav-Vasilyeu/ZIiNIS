"""
Генерация стегоконтейнера:
- через LLM (Ollama / Tooken Club / Groq) — основной путь;
- локальный шаблон — запасной, если сеть/модель недоступны.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from utils.capacity import assess_capacity, payload_size, plan_cover


def ensure_cover_capacity(text: str, secret: str, method: str, *, password: str | None = None) -> str:
    """Add clearly templated prose until the actual codec can fit the payload."""
    capacity = assess_capacity(text, secret, method, password=password)
    if capacity is None or capacity.sufficient:
        return text
    sentence = (
        "Современная система помогает решать важные задачи. "
        "Пользователь выбирает метод, проверка определяет результат. "
        "Обработка информации позволяет создать документ."
    )
    unit = assess_capacity(sentence, secret, method, password=password).available
    repeats = math.ceil((capacity.required - capacity.available) / unit)
    return text + "\n\n" + "\n".join([sentence] * repeats)

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
    return max(8, payload_size(secret) * 8) + 16


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


def build_llm_cover_prompt(secret: str, method: str, *, password: str | None = None,
                          use_marker: bool = False) -> str:
    """Промпт для LLM-генерации контейнера под секрет и метод."""
    plan = plan_cover(secret, method, password=password, use_marker=use_marker)
    required = plan.required_bits
    method = (method or "").lower()
    if method == "classic_whitespace":
        extra = (
            "В тексте должно быть много обычных пробелов между словами "
            f"(не меньше {max(100, required)} пробелов). Пиши связными предложениями."
        )
    elif method == "classic_synonyms":
        extra = (
            f"Нужно не менее {math.ceil(required / 2)} вхождений слов из следующего набора. "
            "Используй лексику: метод, способ, алгоритм, данные, "
            "система, защита, анализ, исследование, документ, сообщение, "
            "современный, быстрый, естественный — в разных формах."
        )
    else:
        extra = (
            "Текст будет использован как начало (промпт) для генеративной стеганографии. "
            "Сделай естественное введение, которое удобно продолжать."
        )
    return (
        f"Напиши связный научно-популярный текст на русском языке объёмом не менее "
        f"{plan.target_words} слов по теме защиты информации и стеганографии. "
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
    password: str | None = None,
    return_report: bool = False,
    use_marker: bool = False,
    max_rounds: int = 3,
) -> str | tuple[str, str]:
    """
    Сгенерировать контейнер через нейросеть (Ollama/Tooken Club/Groq).
    При ошибке — локальный шаблон.
    """
    if not secret or not secret.strip():
        raise ValueError("Секрет пуст")

    method = method.lower().strip()
    plan = plan_cover(secret, method, password=password, use_marker=use_marker)
    prompt = build_llm_cover_prompt(secret, method, password=password, use_marker=use_marker)
    source = "нейросеть"
    try:
        from providers.factory import create_logits_provider

        provider = create_logits_provider(provider_name, model=model)
        text = ""
        try:
            for attempt in range(max(1, min(max_rounds, 5))):
                fragment = (await provider.complete_text(prompt, max_tokens=plan.max_tokens) or "").strip()
                if fragment:
                    text = (text + "\n\n" + fragment).strip()
                capacity = assess_capacity(text, secret, method, password=password, use_marker=use_marker)
                if len(text) >= 80 and (capacity is None or capacity.sufficient):
                    return (text, f"нейросеть: {attempt + 1} запрос(а)") if return_report else text
                deficit = capacity.required - capacity.available if capacity else 0
                prompt = (
                    build_llm_cover_prompt(secret, method, password=password, use_marker=use_marker)
                    + f" Уже подготовленного текста недостаточно: не хватает {max(0, deficit)} бит ёмкости. "
                    "Напиши дополнительные абзацы, не повторяй прежний текст. "
                    + "Последний фрагмент для согласования темы:\n" + text[-1200:]
                )
        finally:
            await provider.close()
        text = (text or "").strip()
        if len(text) >= 80:
            prepared = ensure_cover_capacity(text, secret, method, password=password)
            source = "нейросеть + локальное дополнение по ёмкости" if prepared != text else source
            return (prepared, source) if return_report else prepared
        if not allow_local_fallback:
            raise RuntimeError("Модель вернула слишком короткий текст для контейнера.")
    except Exception:
        if not allow_local_fallback:
            raise

    prepared = ensure_cover_capacity(generate_cover_local(secret, method), secret, method, password=password)
    return (prepared, "локальный шаблон: модель недоступна или ответ слишком короткий") if return_report else prepared


# Синхронная обёртка для старых вызовов
def generate_cover_for_secret(secret: str, method: str = "classic_synonyms", *, seed: int | None = None) -> str:
    return generate_cover_local(secret, method, seed=seed)
