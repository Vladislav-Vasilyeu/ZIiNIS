#!/usr/bin/env python3
"""
NeuralTextStego — точка входа.

Запуск GUI:
    python main.py

Запуск CLI (для отладки):
    python main.py --cli embed --cover data/sample.txt --secret "секрет" --out stego.txt
    python main.py --cli extract --stego stego.txt
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import setup_logger


def run_gui() -> None:
    from ui.app import run_app
    run_app()


async def run_cli(args: argparse.Namespace) -> None:
    from adapters.factory import create_adapter
    from core.stego_engine import StegoEngine
    from utils.file_handlers import load_text, save_text

    adapter = create_adapter(args.provider)
    engine = StegoEngine(adapter, num_candidates=args.candidates)

    try:
        if args.command == "embed":
            cover = load_text(args.cover)
            secret = args.secret
            if args.secret_file:
                secret = Path(args.secret_file).read_text(encoding="utf-8").strip()

            print(f"Внедрение... провайдер={args.provider}, кандидатов={args.candidates}")
            result = await engine.embed(
                cover,
                secret,
                password=args.password,
            )
            if not result.success:
                print(f"Ошибка: {result.message}")
                sys.exit(1)

            out = args.out or "stego_output.txt"
            save_text(out, result.text)
            print(f"Успех. {result.message}")
            print(
                f"В маркер сохранено бит: {result.bits_embedded}, "
                f"видимо закодировано бит: "
                f"{result.details.get('visible_bits_embedded', 0)}, "
                f"позиций заменено: {result.positions_used}"
            )
            print(f"Результат сохранён: {out}")

        elif args.command == "extract":
            stego = load_text(args.stego)
            print(f"Извлечение... провайдер={args.provider}")
            result = await engine.extract(
                stego,
                expected_bits=args.bits,
                password=args.password,
            )
            if not result.success:
                print(f"Ошибка: {result.message}")
                sys.exit(1)

            print(f"Извлечено бит: {result.bits_extracted}")
            print("--- Секретное сообщение ---")
            print(result.secret)
            print("---------------------------")
    except Exception as exc:
        print(f"Ошибка выполнения: {exc}")
        sys.exit(1)
    finally:
        await adapter.close()


def main() -> None:
    from config import settings

    setup_logger()

    parser = argparse.ArgumentParser(description="NeuralTextStego")
    parser.add_argument("--cli", action="store_true", help="Запуск в режиме CLI")
    sub = parser.add_subparsers(dest="command")

    # embed
    p_embed = sub.add_parser("embed", help="Внедрить сообщение")
    p_embed.add_argument("--cover", required=True, help="Путь к контейнеру (.txt/.docx)")
    p_embed.add_argument("--secret", default="", help="Секретное сообщение")
    p_embed.add_argument("--secret-file", help="Файл с секретным сообщением")
    p_embed.add_argument("--out", help="Куда сохранить стеготекст")
    p_embed.add_argument("--password", help="Пароль для шифрования")
    p_embed.add_argument(
        "--provider",
        default=settings.default_provider,
        choices=["openrouter", "ollama", "classic"],
        help="Метод: облачная модель, локальная Ollama или словарь синонимов",
    )
    p_embed.add_argument("--candidates", type=int, default=4, choices=[2, 4])

    # extract
    p_extract = sub.add_parser("extract", help="Извлечь сообщение")
    p_extract.add_argument("--stego", required=True, help="Путь к стеготексту")
    p_extract.add_argument("--bits", type=int, help="Ожидаемое количество бит")
    p_extract.add_argument("--password", help="Пароль для расшифровки")
    p_extract.add_argument(
        "--provider",
        default=settings.default_provider,
        choices=["openrouter", "ollama", "classic"],
        help="Провайдер создания стеготекста (извлечение использует маркер)",
    )
    p_extract.add_argument("--candidates", type=int, default=4, choices=[2, 4])

    args = parser.parse_args()

    if args.cli or args.command:
        if not args.command:
            parser.print_help()
            sys.exit(1)
        asyncio.run(run_cli(args))
    else:
        run_gui()


if __name__ == "__main__":
    main()
