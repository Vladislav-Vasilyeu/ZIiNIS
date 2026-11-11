#!/usr/bin/env python3
"""NeuralTextStego — точка входа (GUI / CLI)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from providers.factory import LOGITS_PROVIDERS
from utils.logger import setup_logger


def run_gui() -> None:
    from ui.app import run_app
    run_app()


async def run_cli(args: argparse.Namespace) -> None:
    from methods.registry import create_method
    from utils.file_handlers import load_text, save_text

    method = create_method(
        args.method,
        logits_provider=args.provider,
        model=args.model,
        num_candidates=args.candidates,
    )
    try:
        if args.command == "embed":
            cover = load_text(args.cover) if args.cover else ""
            secret = args.secret
            if args.secret_file:
                secret = Path(args.secret_file).read_text(encoding="utf-8").strip()
            prompt = args.prompt or cover
            print(f"Внедрение: method={args.method}, provider={args.provider}, marker={not args.no_marker}")
            result = await method.embed(
                secret=secret,
                cover_text=cover,
                prompt=prompt,
                password=args.password,
                use_marker=not args.no_marker,
            )
            if not result.success:
                print(f"Ошибка: {result.message}")
                sys.exit(1)
            out = args.out or "stego_output.txt"
            save_text(out, result.text)
            print(f"OK. бит={result.bits_embedded}, шагов={result.positions_used}, {result.elapsed_sec:.2f}с")
            print(f"Сохранено: {out}")
            print(result.message)
        elif args.command == "extract":
            stego = load_text(args.stego)
            result = await method.extract(
                stego_text=stego,
                prompt=args.prompt or "",
                password=args.password,
                expected_bits=args.bits,
                use_marker=not args.no_marker,
            )
            if not result.success:
                print(f"Ошибка: {result.message}")
                sys.exit(1)
            print(f"Бит: {result.bits_extracted}")
            print("--- Секрет ---")
            print(result.secret)
            print("--------------")
    finally:
        await method.close()


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser(description="NeuralTextStego")
    parser.add_argument("--cli", action="store_true")
    sub = parser.add_subparsers(dest="command")
    methods = [
        "classic_synonyms", "classic_whitespace",
        "neural_rank", "neural_arithmetic", "neural_sampling",
    ]
    p_embed = sub.add_parser("embed")
    p_embed.add_argument("--method", default="classic_synonyms", choices=methods)
    p_embed.add_argument("--provider", default=settings.default_logits_provider, choices=LOGITS_PROVIDERS)
    p_embed.add_argument("--model", default=None)
    p_embed.add_argument("--cover", help="Контейнер для классических методов")
    p_embed.add_argument("--prompt", help="Промпт для нейросетевых методов")
    p_embed.add_argument("--secret", default="")
    p_embed.add_argument("--secret-file")
    p_embed.add_argument("--out")
    p_embed.add_argument("--password")
    p_embed.add_argument("--no-marker", action="store_true")
    p_embed.add_argument("--candidates", type=int, default=4, choices=[2, 4])

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--method", default="classic_synonyms", choices=methods)
    p_extract.add_argument("--provider", default=settings.default_logits_provider, choices=LOGITS_PROVIDERS)
    p_extract.add_argument("--model", default=None)
    p_extract.add_argument("--stego", required=True)
    p_extract.add_argument("--prompt", default="")
    p_extract.add_argument("--bits", type=int)
    p_extract.add_argument("--password")
    p_extract.add_argument("--no-marker", action="store_true")
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
