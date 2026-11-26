"""Opt-in real Ollama verification: python scripts/verify_local.py."""
from __future__ import annotations

import asyncio
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from config import settings
from methods.registry import create_method, NEURAL_KINDS
from providers.factory import create_logits_provider
from utils.cover_generator import generate_cover_local
from utils.file_handlers import load_text, save_text


async def main() -> int:
    out = settings.project_root / "experiments/results/local_verification"
    out.mkdir(parents=True, exist_ok=True)
    report = {"python": platform.python_version(), "model": settings.ollama_model, "cases": []}
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=10) as client:
        report["server"] = (await client.get("/api/version")).json()
    provider = create_logits_provider("ollama")
    try:
        report["models"] = await provider.list_models()
        dist = await provider.next_token_distribution("Information can be stored in", top_k=20)
        report["candidate_count"] = len(dist)
    finally:
        await provider.close()

    async def case(kind, secret, marker, password=None, formats=False):
        method = create_method(kind, logits_provider="ollama")
        prompt = "Information can be stored in" if kind in NEURAL_KINDS else generate_cover_local(secret, kind, seed=42)
        if password and kind not in NEURAL_KINDS:
            from utils.cover_generator import ensure_cover_capacity
            prompt = ensure_cover_capacity(prompt, secret, kind, password=password)
        started = time.perf_counter()
        row = dict(method=kind, utf8_bytes=len(secret.encode("utf-8")), marker=marker, encrypted=bool(password))
        try:
            emb = await method.embed(secret=secret, cover_text=prompt, prompt=prompt,
                                     password=password, use_marker=marker)
            row.update(embed=emb.success, bits=emb.bits_embedded, steps=emb.positions_used,
                       message=emb.message, sample=emb.text[:250])
            if not emb.success:
                row["success"] = False
                return
            ext = await method.extract(stego_text=emb.text, prompt=prompt, password=password, use_marker=marker)
            row["success"] = ext.success and ext.secret == secret
            row["extract_message"] = ext.message
            if formats:
                row["formats"] = {}
                for suffix in ("txt", "docx"):
                    path = out / f"roundtrip_{kind}.{suffix}"
                    save_text(path, emb.text)
                    text = load_text(path)
                    restored = await method.extract(stego_text=text, prompt=prompt, password=password, use_marker=marker)
                    row["formats"][suffix] = restored.success and restored.secret == secret
                row["success"] = row["success"] and all(row["formats"].values())
            if password and marker:
                wrong = await method.extract(stego_text=emb.text, prompt=prompt, password="wrong password", use_marker=marker)
                row["wrong_password_rejected"] = not wrong.success
                row["success"] = row["success"] and not wrong.success
        except Exception as exc:
            row.update(success=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            await method.close()
            row["seconds"] = round(time.perf_counter() - started, 3)
            report["cases"].append(row)
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in row.items() if k != "sample"}, ensure_ascii=True), flush=True)

    for kind in ("classic_whitespace", "classic_synonyms"):
        for marker in (False, True):
            for password in (None, " пароль "):
                await case(kind, " Секретное Сообщение 🔑\n", marker, password, formats=True)
    for kind in sorted(NEURAL_KINDS):
        for secret in ("Hi", "КП", "🔑", "Секретное Сообщение"):
            await case(kind, secret, False, formats=secret == "Hi")
        await case(kind, "Hi", True)
        await case(kind, "Hi", False, password=" пароль ")
        await case(kind, "Hi", True, password=" пароль ")
    success = all(row["success"] for row in report["cases"])
    print(f"RESULT: {len(report['cases'])} cases; success={success}; report={out / 'report.json'}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
