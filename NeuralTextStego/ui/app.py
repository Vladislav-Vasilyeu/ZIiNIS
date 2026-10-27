"""
Клиентоориентированный GUI на CustomTkinter.

Требования ТЗ:
- асинхронные операции (не блокируют UI)
- прогресс-бар + возможность отмены
- понятные сообщения об ошибках
- разделы: Внедрение / Извлечение / Настройки
"""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from adapters.factory import create_adapter
from config import settings
from core.stego_engine import StegoEngine
from utils.file_handlers import load_text, save_text
from utils.logger import setup_logger

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

PROVIDER_LABELS = {
    "openrouter": "OpenRouter Free",
    "ollama": "Ollama (локальная модель)",
    "classic": "Классический словарь",
}


class NeuralTextStegoApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NeuralTextStego — нейросетевая текстовая стеганография")
        self.geometry("960x680")
        self.minsize(800, 560)

        self._cancel_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: threading.Thread | None = None

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        self.tab_embed = self.tabview.add("Внедрение")
        self.tab_extract = self.tabview.add("Извлечение")
        self.tab_settings = self.tabview.add("Настройки")

        self._build_embed_tab()
        self._build_extract_tab()
        self._build_settings_tab()

        # Статус-бар
        self.status = ctk.CTkLabel(self, text="Готово", anchor="w")
        self.status.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))

        self.progress = ctk.CTkProgressBar(self)
        self.progress.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 12))
        self.progress.set(0)

    def _build_embed_tab(self) -> None:
        tab = self.tab_embed
        tab.grid_columnconfigure(0, weight=1)

        # Контейнер
        ctk.CTkLabel(tab, text="Текст-контейнер (или загрузите файл):").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 0)
        )
        self.cover_text = ctk.CTkTextbox(tab, height=140)
        self.cover_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        tab.grid_rowconfigure(1, weight=1)

        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=8)
        ctk.CTkButton(btn_frame, text="Загрузить файл…", width=140, command=self._load_cover).pack(
            side="left", padx=(0, 8)
        )

        # Секрет
        ctk.CTkLabel(tab, text="Секретное сообщение:").grid(
            row=3, column=0, sticky="w", padx=8, pady=(12, 0)
        )
        self.secret_text = ctk.CTkTextbox(tab, height=80)
        self.secret_text.grid(row=4, column=0, sticky="ew", padx=8, pady=4)

        # Параметры
        opts = ctk.CTkFrame(tab, fg_color="transparent")
        opts.grid(row=5, column=0, sticky="ew", padx=8, pady=8)
        ctk.CTkLabel(opts, text="Пароль (опционально):").pack(side="left")
        self.embed_password = ctk.CTkEntry(opts, show="*", width=160)
        self.embed_password.pack(side="left", padx=8)

        # Кнопки действий
        action = ctk.CTkFrame(tab, fg_color="transparent")
        action.grid(row=6, column=0, sticky="ew", padx=8, pady=8)
        self.btn_embed = ctk.CTkButton(action, text="Внедрить", command=self._start_embed)
        self.btn_embed.pack(side="left", padx=(0, 8))
        self.btn_cancel_embed = ctk.CTkButton(
            action, text="Отмена", fg_color="gray", command=self._cancel_operation, state="disabled"
        )
        self.btn_cancel_embed.pack(side="left")

        # Результат
        ctk.CTkLabel(tab, text="Стеготекст:").grid(row=7, column=0, sticky="w", padx=8, pady=(8, 0))
        self.stego_result = ctk.CTkTextbox(tab, height=120)
        self.stego_result.grid(row=8, column=0, sticky="nsew", padx=8, pady=4)
        tab.grid_rowconfigure(8, weight=1)

        ctk.CTkButton(tab, text="Сохранить стеготекст…", command=self._save_stego).grid(
            row=9, column=0, sticky="w", padx=8, pady=8
        )

    def _build_extract_tab(self) -> None:
        tab = self.tab_extract
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="Стеготекст (или загрузите файл):").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 0)
        )
        self.extract_input = ctk.CTkTextbox(tab, height=180)
        self.extract_input.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        tab.grid_rowconfigure(1, weight=1)

        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=8)
        ctk.CTkButton(btn_frame, text="Загрузить файл…", width=140, command=self._load_stego).pack(
            side="left"
        )

        opts = ctk.CTkFrame(tab, fg_color="transparent")
        opts.grid(row=3, column=0, sticky="ew", padx=8, pady=8)
        ctk.CTkLabel(opts, text="Пароль (если использовался):").pack(side="left")
        self.extract_password = ctk.CTkEntry(opts, show="*", width=160)
        self.extract_password.pack(side="left", padx=8)
        ctk.CTkLabel(opts, text="Ожидаемо бит (0 = авто):").pack(side="left", padx=(16, 0))
        self.expected_bits = ctk.CTkEntry(opts, width=80)
        self.expected_bits.insert(0, "0")
        self.expected_bits.pack(side="left", padx=8)

        action = ctk.CTkFrame(tab, fg_color="transparent")
        action.grid(row=4, column=0, sticky="ew", padx=8, pady=8)
        self.btn_extract = ctk.CTkButton(action, text="Извлечь", command=self._start_extract)
        self.btn_extract.pack(side="left", padx=(0, 8))
        self.btn_cancel_extract = ctk.CTkButton(
            action, text="Отмена", fg_color="gray", command=self._cancel_operation, state="disabled"
        )
        self.btn_cancel_extract.pack(side="left")

        ctk.CTkLabel(tab, text="Извлечённое сообщение:").grid(
            row=5, column=0, sticky="w", padx=8, pady=(8, 0)
        )
        self.extract_result = ctk.CTkTextbox(tab, height=100)
        self.extract_result.grid(row=6, column=0, sticky="nsew", padx=8, pady=4)
        tab.grid_rowconfigure(6, weight=1)

    def _build_settings_tab(self) -> None:
        tab = self.tab_settings
        tab.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(tab, text="Метод подбора замен:").grid(
            row=0, column=0, sticky="w", padx=12, pady=12
        )
        self.provider_var = ctk.StringVar(value=PROVIDER_LABELS[settings.default_provider])
        self.provider_menu = ctk.CTkOptionMenu(
            tab,
            values=list(PROVIDER_LABELS.values()),
            variable=self.provider_var,
        )
        self.provider_menu.grid(row=0, column=1, sticky="w", padx=8, pady=12)

        ctk.CTkLabel(tab, text="Кандидатов на замену:").grid(
            row=1, column=0, sticky="w", padx=12, pady=8
        )
        self.candidates_var = ctk.StringVar(value=str(settings.num_candidates))
        self.candidates_menu = ctk.CTkOptionMenu(
            tab, values=["2 (1 бит)", "4 (2 бита)"], variable=self.candidates_var
        )
        self.candidates_menu.grid(row=1, column=1, sticky="w", padx=8, pady=8)

        info = (
            "Ключи API задаются в файле .env (см. .env.example).\n\n"
            "OpenRouter Free — бесплатный API с автоматическим выбором доступной модели. "
            "Нужен API-ключ; бесплатные модели могут ограничивать частоту запросов.\n"
            "Ollama работает локально; модель по умолчанию — qwen2.5:3b. "
            "Установите Ollama и скачайте модель отдельно.\n"
            "Классический режим использует небольшой встроенный словарь точных словоформ.\n"
            "Видимые замены слов создаются выбранным методом; извлечение новых стеготекстов "
            "выполняется по невидимому маркеру.\n\n"
            "Алгоритм: edit-based (модификация существующих слов).\n"
            "Шифрование данных в маркере: опциональное AES-256-GCM по паролю."
        )
        ctk.CTkLabel(tab, text=info, justify="left", wraplength=600).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=12, pady=20
        )

    # ------------------------------------------------------------------ Helpers
    def _num_candidates(self) -> int:
        val = self.candidates_var.get()
        return 2 if val.startswith("2") else 4

    def _selected_provider(self) -> str:
        return next(
            provider
            for provider, label in PROVIDER_LABELS.items()
            if label == self.provider_var.get()
        )

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _set_progress(self, value: float) -> None:
        self.progress.set(max(0.0, min(1.0, value)))

    def _load_cover(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Текстовые документы", "*.txt *.docx"), ("Все файлы", "*.*")]
        )
        if path:
            try:
                text = load_text(path)
                self.cover_text.delete("1.0", tk.END)
                self.cover_text.insert("1.0", text)
                self._set_status(f"Загружен контейнер: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

    def _load_stego(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Текстовые документы", "*.txt *.docx"), ("Все файлы", "*.*")]
        )
        if path:
            try:
                text = load_text(path)
                self.extract_input.delete("1.0", tk.END)
                self.extract_input.insert("1.0", text)
                self._set_status(f"Загружен стеготекст: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

    def _save_stego(self) -> None:
        text = self.stego_result.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Внимание", "Нет данных для сохранения")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текст", "*.txt"), ("Word", "*.docx")],
        )
        if path:
            try:
                save_text(path, text)
                self._set_status(f"Сохранено: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

    def _cancel_operation(self) -> None:
        if self._cancel_event:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._cancel_event.set)
            else:
                self._cancel_event.set()
            self._set_status("Отмена операции…")

    def _run_async(self, coro, on_done) -> None:
        """Запустить корутину в фоне, не блокируя UI."""

        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                result = loop.run_until_complete(coro)
                self.after(0, lambda: on_done(result, None))
            except Exception as exc:
                self.after(0, lambda error=exc: on_done(None, error))
            finally:
                loop.close()
                self._loop = None

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ Actions
    def _start_embed(self) -> None:
        cover = self.cover_text.get("1.0", tk.END).strip()
        secret = self.secret_text.get("1.0", tk.END).strip()
        if not cover or not secret:
            messagebox.showwarning("Внимание", "Укажите контейнер и секретное сообщение")
            return

        self.btn_embed.configure(state="disabled")
        self.btn_cancel_embed.configure(state="normal")
        self._cancel_event = asyncio.Event()
        self._set_status("Внедрение…")
        self._set_progress(0)

        password = self.embed_password.get().strip() or None
        provider = self._selected_provider()
        n_cand = self._num_candidates()

        async def task():
            adapter = create_adapter(provider)
            engine = StegoEngine(adapter, num_candidates=n_cand)

            async def progress(p: float, msg: str):
                self.after(0, lambda: (self._set_progress(p), self._set_status(msg)))

            try:
                result = await engine.embed(
                    cover,
                    secret,
                    password=password,
                    progress_callback=progress,
                    cancel_event=self._cancel_event,
                )
                return result
            finally:
                await adapter.close()

        def done(result, error):
            self.btn_embed.configure(state="normal")
            self.btn_cancel_embed.configure(state="disabled")
            self._cancel_event = None
            if error:
                messagebox.showerror("Ошибка", str(error))
                self._set_status("Ошибка")
                self._set_progress(0)
                return
            if not result.success:
                messagebox.showerror("Ошибка внедрения", result.message)
                self._set_status(result.message)
                self._set_progress(0)
                return
            self.stego_result.delete("1.0", tk.END)
            self.stego_result.insert("1.0", result.text)
            self._set_status(
                f"{result.message}; видимо закодировано "
                f"{result.details.get('visible_bits_embedded', 0)} бит, "
                f"замен {result.positions_used}"
            )
            self._set_progress(1.0)

        self._run_async(task(), done)

    def _start_extract(self) -> None:
        stego = self.extract_input.get("1.0", tk.END).strip()
        if not stego:
            messagebox.showwarning("Внимание", "Укажите стеготекст")
            return

        self.btn_extract.configure(state="disabled")
        self.btn_cancel_extract.configure(state="normal")
        self._cancel_event = asyncio.Event()
        self._set_status("Извлечение…")
        self._set_progress(0)

        password = self.extract_password.get().strip() or None
        try:
            bits = int(self.expected_bits.get().strip() or "0")
            if bits <= 0:
                bits = None
        except ValueError:
            bits = None

        provider = self._selected_provider()
        n_cand = self._num_candidates()

        async def task():
            adapter = create_adapter(provider)
            engine = StegoEngine(adapter, num_candidates=n_cand)

            async def progress(p: float, msg: str):
                self.after(0, lambda: (self._set_progress(p), self._set_status(msg)))

            try:
                result = await engine.extract(
                    stego,
                    expected_bits=bits,
                    password=password,
                    progress_callback=progress,
                    cancel_event=self._cancel_event,
                )
                return result
            finally:
                await adapter.close()

        def done(result, error):
            self.btn_extract.configure(state="normal")
            self.btn_cancel_extract.configure(state="disabled")
            self._cancel_event = None
            if error:
                messagebox.showerror("Ошибка", str(error))
                self._set_status("Ошибка")
                self._set_progress(0)
                return
            if not result.success:
                messagebox.showerror("Ошибка извлечения", result.message)
                self._set_status(result.message)
                self._set_progress(0)
                return
            self.extract_result.delete("1.0", tk.END)
            self.extract_result.insert("1.0", result.secret)
            self._set_status(f"Готово: извлечено {result.bits_extracted} бит")
            self._set_progress(1.0)

        self._run_async(task(), done)


def run_app() -> None:
    setup_logger()
    app = NeuralTextStegoApp()
    app.mainloop()
