"""GUI NeuralTextStego: 5 методов, облачные и локальные модели, маркер, метрики."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from config import settings
from methods.registry import METHOD_LABELS, NEURAL_KINDS, create_method
from providers.factory import LOGITS_PROVIDERS, create_logits_provider
from utils.cover_generator import generate_cover_for_secret_async, generate_cover_local
from utils.file_handlers import load_text, save_text
from utils.experiment_results import write_experiment_csv
from utils.logger import setup_logger

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

METHOD_KEYS = list(METHOD_LABELS.keys())
METHOD_VALUES = list(METHOD_LABELS.values())
LABEL_TO_KEY = {v: k for k, v in METHOD_LABELS.items()}
PROVIDER_BY_LABEL = {label: key for key, label in LOGITS_PROVIDERS.items()}

BG = "#0b1020"
SURFACE = "#121a2b"
CARD = "#1b2740"
ACCENT = "#6c63ff"
ACCENT_HOVER = "#554ce3"
MUTED = "#9aa8c7"
TEXT = "#f4f6ff"
DANGER = "#dc5f72"


class NeuralTextStegoApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NeuralTextStego — текстовая стеганография")
        self.geometry("1240x820")
        self.minsize(1080, 700)
        self.configure(fg_color=BG)
        self._cancel_event: asyncio.Event | None = None
        self._load_ui_state()
        self._build_ui()

    def _state_path(self) -> Path:
        p = settings.project_root / settings.ui_state_file
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_ui_state(self) -> None:
        self._ui_state = {
            "method": settings.default_method,
            "provider": settings.default_logits_provider,
            "models": {},
            "use_marker": settings.use_marker_by_default,
        }
        path = self._state_path()
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("provider") not in LOGITS_PROVIDERS:
                    saved.pop("provider", None)
                if not isinstance(saved.get("models"), dict):
                    saved.pop("models", None)
                self._ui_state.update(saved)
            except Exception:
                pass

    def _save_ui_state(self) -> None:
        self._ui_state["method"] = self._selected_method()
        self._ui_state["provider"] = self._selected_provider()
        self._ui_state["use_marker"] = bool(self.use_marker_var.get())
        self._ui_state["models"][self._ui_state["provider"]] = self._selected_model()
        try:
            self._state_path().write_text(
                json.dumps(self._ui_state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=64)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="NeuralTextStego",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(10, 0))
        ctk.CTkLabel(
            header,
            text="Классические и нейросетевые методы скрытия данных в тексте",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
        ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 10))

        self.provider_var = ctk.StringVar(value=LOGITS_PROVIDERS[self._ui_state["provider"]])
        self.model_catalogs: dict[str, list[str]] = {}
        ctk.CTkLabel(header, text="Провайдер", text_color=MUTED).grid(
            row=0, column=1, rowspan=2, padx=(8, 4)
        )
        ctk.CTkOptionMenu(
            header,
            values=list(PROVIDER_BY_LABEL),
            variable=self.provider_var,
            command=self._change_provider,
            width=120,
        ).grid(row=0, column=2, rowspan=2, padx=(4, 10))
        ctk.CTkLabel(header, text="Модель", text_color=MUTED).grid(
            row=0, column=3, rowspan=2, padx=(0, 4)
        )
        self.model_menu = ctk.CTkComboBox(
            header, width=180, command=lambda _: self._save_ui_state(),
        )
        self.model_menu.grid(row=0, column=4, rowspan=2, padx=4)
        self.model_menu.bind("<Return>", lambda _: self._save_ui_state())
        self.model_menu.bind("<FocusOut>", lambda _: self._save_ui_state())
        self.btn_models = ctk.CTkButton(
            header, text="Загрузить модели", width=128, command=self._refresh_models,
        )
        self.btn_models.grid(row=0, column=5, rowspan=2, padx=(4, 16))
        self._show_provider_model()

        self.tabview = ctk.CTkTabview(
            self,
            fg_color=SURFACE,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_HOVER,
            segmented_button_unselected_color=CARD,
        )
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=16, pady=(12, 8))
        self.tab_embed = self.tabview.add("Внедрение")
        self.tab_extract = self.tabview.add("Извлечение")
        self.tab_exp = self.tabview.add("Эксперименты")
        self.tab_settings = self.tabview.add("Справка")
        self._build_embed_tab()
        self._build_extract_tab()
        self._build_experiments_tab()
        self._build_settings_tab()

        footer = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(footer, text="Готово", anchor="w", text_color=TEXT)
        self.status.grid(row=0, column=0, sticky="ew", padx=16, pady=(8, 0))
        self.metrics = ctk.CTkLabel(footer, text="", anchor="w", text_color=MUTED)
        self.metrics.grid(row=1, column=0, sticky="ew", padx=16)
        self.progress = ctk.CTkProgressBar(
            footer, progress_color=ACCENT, fg_color=CARD, height=8
        )
        self.progress.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 12))
        self.progress.set(0)

    def _method_frame(self, parent, row: int, *, marker_var: ctk.BooleanVar | None = None):
        frame = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 6))
        ctk.CTkLabel(frame, text="Метод", text_color=MUTED).pack(side="left", padx=(12, 6), pady=10)
        default_label = METHOD_LABELS.get(
            self._ui_state.get("method", "classic_synonyms"),
            METHOD_VALUES[0],
        )
        var = ctk.StringVar(value=default_label)
        ctk.CTkOptionMenu(
            frame,
            values=METHOD_VALUES,
            variable=var,
            width=340,
            fg_color=ACCENT,
            button_color=ACCENT_HOVER,
            button_hover_color=ACCENT_HOVER,
        ).pack(side="left", padx=6)
        marker = marker_var or ctk.BooleanVar(value=self._ui_state.get("use_marker", True))
        ctk.CTkCheckBox(
            frame,
            text="Невидимый маркер",
            variable=marker,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
        ).pack(side="left", padx=12)
        return var, marker

    def _textbox(self, parent, height: int) -> ctk.CTkTextbox:
        return ctk.CTkTextbox(
            parent,
            height=height,
            fg_color=BG,
            border_width=1,
            border_color="#33415f",
            text_color=TEXT,
        )

    def _build_embed_tab(self) -> None:
        tab = self.tab_embed
        tab.configure(fg_color=SURFACE)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=2)
        tab.grid_rowconfigure(6, weight=2)
        self.embed_method_var, self.use_marker_var = self._method_frame(tab, 0)

        ctk.CTkLabel(tab, text="Контейнер / промпт", text_color=MUTED).grid(
            row=1, column=0, sticky="w", padx=12, pady=(4, 0)
        )
        ctk.CTkLabel(tab, text="Секретное сообщение", text_color=MUTED).grid(
            row=1, column=1, sticky="w", padx=12, pady=(4, 0)
        )
        self.cover_text = self._textbox(tab, 180)
        self.cover_text.grid(row=2, column=0, sticky="nsew", padx=(12, 6), pady=6)
        self.secret_text = self._textbox(tab, 180)
        self.secret_text.grid(row=2, column=1, sticky="nsew", padx=(6, 12), pady=6)

        tools = ctk.CTkFrame(tab, fg_color="transparent")
        tools.grid(row=3, column=0, sticky="ew", padx=12)
        ctk.CTkButton(
            tools, text="Загрузить файл", width=140, fg_color=CARD, hover_color="#273653",
            command=self._load_cover,
        ).pack(side="left")
        ctk.CTkButton(
            tools, text="Сгенерировать контейнер", width=200,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self._generate_cover,
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            tools, text="Копировать контейнер", width=170, fg_color=CARD, hover_color="#273653",
            command=lambda: self._copy_from(self.cover_text, "Контейнер скопирован"),
        ).pack(side="left")

        opts = ctk.CTkFrame(tab, fg_color="transparent")
        opts.grid(row=3, column=1, sticky="e", padx=12)
        ctk.CTkLabel(opts, text="Пароль (опц.)", text_color=MUTED).pack(side="left")
        self.embed_password = ctk.CTkEntry(opts, show="*", width=160, fg_color=BG)
        self.embed_password.pack(side="left", padx=8)
        self.btn_embed = ctk.CTkButton(
            opts, text="Внедрить", width=120, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._start_embed,
        )
        self.btn_embed.pack(side="left", padx=(8, 6))
        self.btn_cancel_embed = ctk.CTkButton(
            opts, text="Отмена", width=90, fg_color=DANGER, hover_color="#b84459",
            command=self._cancel_operation, state="disabled",
        )
        self.btn_cancel_embed.pack(side="left")

        result_bar = ctk.CTkFrame(tab, fg_color="transparent")
        result_bar.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 0))
        ctk.CTkLabel(result_bar, text="Стегоконтейнер (результат)", text_color=MUTED).pack(side="left")
        ctk.CTkButton(
            result_bar, text="Копировать", width=120, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._copy_stego,
        ).pack(side="right")
        ctk.CTkButton(
            result_bar, text="В извлечение", width=130, fg_color=CARD, hover_color="#273653",
            command=self._send_stego_to_extract,
        ).pack(side="right", padx=8)
        ctk.CTkButton(
            result_bar, text="Сохранить…", width=120, fg_color=CARD, hover_color="#273653",
            command=self._save_stego,
        ).pack(side="right")

        self.stego_result = self._textbox(tab, 150)
        self.stego_result.grid(row=6, column=0, columnspan=2, sticky="nsew", padx=12, pady=8)

    def _build_extract_tab(self) -> None:
        tab = self.tab_extract
        tab.configure(fg_color=SURFACE)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=2)
        tab.grid_rowconfigure(6, weight=1)
        self.extract_method_var, self.extract_marker_var = self._method_frame(tab, 0)

        ctk.CTkLabel(tab, text="Стегоконтейнер", text_color=MUTED).grid(
            row=1, column=0, sticky="w", padx=12, pady=(4, 0)
        )
        self.extract_input = self._textbox(tab, 200)
        self.extract_input.grid(row=2, column=0, sticky="nsew", padx=12, pady=6)

        tools = ctk.CTkFrame(tab, fg_color="transparent")
        tools.grid(row=3, column=0, sticky="ew", padx=12)
        ctk.CTkButton(
            tools, text="Загрузить файл", width=140, fg_color=CARD, hover_color="#273653",
            command=self._load_stego,
        ).pack(side="left")
        ctk.CTkButton(
            tools, text="Копировать", width=120, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=lambda: self._copy_from(self.extract_input, "Стегоконтейнер скопирован"),
        ).pack(side="left", padx=8)
        options = ctk.CTkFrame(tab, fg_color=CARD, corner_radius=10)
        options.grid(row=4, column=0, sticky="ew", padx=12, pady=(10, 2))
        ctk.CTkLabel(options, text="Пароль", text_color=MUTED).pack(side="left", padx=(12, 6), pady=10)
        self.extract_password = ctk.CTkEntry(options, show="*", width=150, fg_color=BG)
        self.extract_password.pack(side="left")
        ctk.CTkLabel(options, text="Промпт для нейросетевого метода", text_color=MUTED).pack(side="left", padx=(18, 6))
        self.extract_prompt = ctk.CTkEntry(options, width=360, fg_color=BG)
        self.extract_prompt.pack(side="left", padx=(0, 8))
        self.btn_extract = ctk.CTkButton(
            options, text="Извлечь", width=110, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._start_extract,
        )
        self.btn_extract.pack(side="left", padx=6)
        self.btn_cancel_extract = ctk.CTkButton(
            options, text="Отмена", width=90, fg_color=DANGER, hover_color="#b84459",
            command=self._cancel_operation, state="disabled",
        )
        self.btn_cancel_extract.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(tab, text="Извлечённое сообщение", text_color=MUTED).grid(
            row=5, column=0, sticky="w", padx=12, pady=(10, 0)
        )
        self.extract_result = self._textbox(tab, 120)
        self.extract_result.grid(row=6, column=0, sticky="nsew", padx=12, pady=8)

    def _build_experiments_tab(self) -> None:
        tab = self.tab_exp
        tab.configure(fg_color=SURFACE)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(4, weight=1)
        info = (
            "Серия внедрений и извлечений с проверкой совпадения секрета.\n"
            "CSV сохраняется в UTF-8 с BOM — кириллица корректно открывается в Excel."
        )
        ctk.CTkLabel(tab, text=info, justify="left", text_color=MUTED).grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 4)
        )
        self.exp_method_var, self.exp_marker_var = self._method_frame(tab, 1)
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=12, pady=8)
        ctk.CTkLabel(row, text="Повторов", text_color=MUTED).pack(side="left")
        self.exp_runs = ctk.CTkEntry(row, width=60, fg_color=BG)
        self.exp_runs.insert(0, "5")
        self.exp_runs.pack(side="left", padx=8)
        ctk.CTkLabel(row, text="Секрет", text_color=MUTED).pack(side="left", padx=(12, 0))
        self.exp_secret = ctk.CTkEntry(row, width=220, fg_color=BG)
        self.exp_secret.insert(0, "test")
        self.exp_secret.pack(side="left", padx=8)
        self.btn_exp = ctk.CTkButton(
            row, text="Запустить серию", fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._start_experiments,
        )
        self.btn_exp.pack(side="left", padx=12)
        self.exp_log = self._textbox(tab, 280)
        self.exp_log.grid(row=4, column=0, sticky="nsew", padx=12, pady=8)

    def _build_settings_tab(self) -> None:
        tab = self.tab_settings
        tab.configure(fg_color=SURFACE)
        info = (
            "Ключи и модели задаются в файле .env (см. .env.example).\n\n"
            "Классические методы меняют готовый текст-контейнер:\n"
            "  • синонимы — замена слов по словарю;\n"
            "  • пробелы — обычный пробел = 0, неразрывный (NBSP) = 1.\n\n"
            "Нейросетевые методы генерируют продолжение промпта, кодируя биты\n"
            "в выборе следующего токена по top-logprobs Ollama или Tooken Club.\n"
            "Нужна модель с поддержкой logprobs. Для генерации контейнера достаточно обычного чата.\n\n"
            "Маркер — невидимые символы с полным секретом и контрольной суммой.\n"
            "Если маркер выключен, секрет кодируется только в самом тексте:\n"
            "для нейросетевых методов при извлечении укажите тот же промпт."
        )
        ctk.CTkLabel(tab, text=info, justify="left", wraplength=820, text_color=TEXT).grid(
            row=0, column=0, sticky="w", padx=20, pady=20
        )

    def _selected_provider(self) -> str:
        return PROVIDER_BY_LABEL[self.provider_var.get()]

    def _selected_model(self) -> str | None:
        return self.model_menu.get().strip() or None

    def _show_provider_model(self) -> None:
        provider = self._selected_provider()
        default = create_logits_provider(provider).model
        saved = self._ui_state["models"].get(provider)
        model = saved if isinstance(saved, str) and saved else default
        self.model_menu.configure(values=list(dict.fromkeys(
            [default, model, *self.model_catalogs.get(provider, [])]
        )))
        self.model_menu.set(model)

    def _change_provider(self, _: str) -> None:
        previous = self._ui_state["provider"]
        self._ui_state["models"][previous] = self._selected_model()
        self._show_provider_model()
        self._save_ui_state()

    def _refresh_models(self) -> None:
        provider_name = self._selected_provider()
        label = LOGITS_PROVIDERS[provider_name]
        self.btn_models.configure(state="disabled")
        self._set_status(f"Загрузка списка моделей {label}…")

        async def task():
            provider = create_logits_provider(provider_name)
            try:
                return await provider.list_models()
            finally:
                await provider.close()

        def done(models, error):
            self.btn_models.configure(state="normal")
            if error:
                self._set_status(f"Ошибка загрузки моделей {label}")
                messagebox.showerror("Ошибка загрузки моделей", str(error))
                return
            self.model_catalogs[provider_name] = models
            if self._selected_provider() == provider_name:
                self._save_ui_state()
                self._show_provider_model()
            self._set_status(f"Модели {label}: {len(models)}")

        self._run_async(task(), done)

    def _selected_method(self) -> str:
        return LABEL_TO_KEY.get(self.embed_method_var.get(), "classic_synonyms")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _set_metrics(self, text: str) -> None:
        self.metrics.configure(text=text)

    def _set_progress(self, value: float) -> None:
        self.progress.set(max(0.0, min(1.0, value)))

    def _box_text(self, box: ctk.CTkTextbox, *, strip: bool = False) -> str:
        text = box.get("1.0", "end-1c")
        return text.strip() if strip else text

    def _copy_from(self, box: ctk.CTkTextbox, ok_message: str) -> None:
        text = self._box_text(box)
        if not text:
            messagebox.showwarning("Внимание", "Нет данных для копирования")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        self._set_status(ok_message)

    def _copy_stego(self) -> None:
        self._copy_from(self.stego_result, "Стегоконтейнер скопирован в буфер обмена")

    def _send_stego_to_extract(self) -> None:
        text = self._box_text(self.stego_result)
        if not text:
            messagebox.showwarning("Внимание", "Сначала выполните внедрение")
            return
        self.extract_input.delete("1.0", tk.END)
        self.extract_input.insert("1.0", text)
        self.extract_method_var.set(self.embed_method_var.get())
        self.extract_marker_var.set(self.use_marker_var.get())
        prompt = self._box_text(self.cover_text, strip=True)
        self.extract_prompt.delete(0, tk.END)
        self.extract_prompt.insert(0, prompt)
        self.tabview.set("Извлечение")
        self._set_status("Стегоконтейнер перенесён на вкладку извлечения")

    def _load_cover(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Текст", "*.txt *.docx"), ("Все", "*.*")])
        if path:
            try:
                self.cover_text.delete("1.0", tk.END)
                self.cover_text.insert("1.0", load_text(path))
                self._set_status(f"Загружен: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

    def _generate_cover(self) -> None:
        secret = self._box_text(self.secret_text, strip=True)
        if not secret:
            messagebox.showwarning(
                "Секрет не введён",
                "Невозможно выполнить генерацию стегоконтейнера.\n"
                "Сначала введите секретное сообщение.",
            )
            self._set_status("Введите секрет перед генерацией контейнера")
            return
        method_key = LABEL_TO_KEY.get(self.embed_method_var.get(), "classic_synonyms")
        provider = self._selected_provider()
        model = self._selected_model()
        self._set_status(f"Генерация контейнера через {LOGITS_PROVIDERS[provider]}…")
        self._set_progress(0.2)

        async def task():
            return await generate_cover_for_secret_async(
                secret, method=method_key, provider_name=provider, model=model,
                allow_local_fallback=provider != "tooken_club",
            )

        def done(cover, error):
            self._set_progress(0)
            if error:
                messagebox.showerror("Ошибка генерации", str(error))
                self._set_status("Ошибка генерации контейнера")
                return
            self.cover_text.delete("1.0", tk.END)
            self.cover_text.insert("1.0", cover)
            bits = len(secret.encode("utf-8")) * 8
            self._set_status(
                f"Сгенерирован стегоконтейнер ({len(secret)} симв. секрета, ~{bits} бит), "
                f"метод={method_key}, провайдер={provider}"
            )
            self._set_metrics(f"контейнер: {len(cover)} символов · секрет: {len(secret)} символов")
            self._set_progress(1.0)

        self._run_async(task(), done)

    def _load_stego(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Текст", "*.txt *.docx"), ("Все", "*.*")])
        if path:
            try:
                self.extract_input.delete("1.0", tk.END)
                self.extract_input.insert("1.0", load_text(path))
                self._set_status(f"Загружен: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

    def _save_stego(self) -> None:
        text = self._box_text(self.stego_result)
        if not text:
            messagebox.showwarning("Внимание", "Нет данных")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Текст", "*.txt"), ("Word", "*.docx")])
        if path:
            try:
                save_text(path, text)
                self._set_status(f"Сохранено: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

    def _cancel_operation(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()
            self._set_status("Отмена…")

    def _run_async(self, coro, on_done) -> None:
        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                self.after(0, lambda: on_done(result, None))
            except Exception as exc:
                self.after(0, lambda error=exc: on_done(None, error))
            finally:
                loop.close()

        threading.Thread(target=worker, daemon=True).start()

    def _start_embed(self) -> None:
        cover = self._box_text(self.cover_text)
        secret = self._box_text(self.secret_text, strip=True)
        if not secret:
            messagebox.showwarning("Внимание", "Укажите секрет")
            return
        method_key = LABEL_TO_KEY.get(self.embed_method_var.get(), "classic_synonyms")
        if method_key not in NEURAL_KINDS and not cover.strip():
            messagebox.showwarning("Внимание", "Для классического метода нужен контейнер")
            return
        if method_key in NEURAL_KINDS and not cover.strip():
            cover = "Продолжи связный текст на русском языке:\n"

        self.btn_embed.configure(state="disabled")
        self.btn_cancel_embed.configure(state="normal")
        self._cancel_event = asyncio.Event()
        self._set_status("Внедрение…")
        self._set_progress(0)
        self._set_metrics("")
        self._save_ui_state()

        password = self.embed_password.get().strip() or None
        provider = self._selected_provider()
        model = self._selected_model()
        use_marker = bool(self.use_marker_var.get())

        async def task():
            method = create_method(method_key, logits_provider=provider, model=model)

            async def progress(p, msg):
                self.after(0, lambda: (self._set_progress(p), self._set_status(msg)))

            try:
                return await method.embed(
                    secret=secret,
                    cover_text=cover,
                    prompt=cover,
                    password=password,
                    use_marker=use_marker,
                    progress=progress,
                    cancel_event=self._cancel_event,
                )
            finally:
                await method.close()

        def done(result, error):
            self.btn_embed.configure(state="normal")
            self.btn_cancel_embed.configure(state="disabled")
            self._cancel_event = None
            if error:
                messagebox.showerror("Ошибка", str(error))
                self._set_status("Ошибка")
                return
            if not result.success:
                messagebox.showerror("Ошибка", result.message)
                self._set_status(result.message)
                return
            self.stego_result.delete("1.0", tk.END)
            self.stego_result.insert("1.0", result.text)
            self.extract_prompt.delete(0, tk.END)
            self.extract_prompt.insert(0, cover.strip())
            self._set_status(result.message)
            self._set_metrics(
                f"бит={result.bits_embedded} · шагов={result.positions_used} · "
                f"время={result.elapsed_sec:.2f}с · {result.details}"
            )
            self._set_progress(1.0)

        self._run_async(task(), done)

    def _start_extract(self) -> None:
        stego = self._box_text(self.extract_input)
        if not stego.strip():
            messagebox.showwarning("Внимание", "Укажите стеготекст")
            return
        method_key = LABEL_TO_KEY.get(self.extract_method_var.get(), "classic_synonyms")
        self.btn_extract.configure(state="disabled")
        self.btn_cancel_extract.configure(state="normal")
        self._cancel_event = asyncio.Event()
        self._set_status("Извлечение…")
        self._set_progress(0)

        password = self.extract_password.get().strip() or None
        provider = self._selected_provider()
        model = self._selected_model()
        use_marker = bool(self.extract_marker_var.get())
        prompt = self.extract_prompt.get()

        async def task():
            method = create_method(method_key, logits_provider=provider, model=model)

            async def progress(p, msg):
                self.after(0, lambda: (self._set_progress(p), self._set_status(msg)))

            try:
                return await method.extract(
                    stego_text=stego,
                    prompt=prompt,
                    password=password,
                    use_marker=use_marker,
                    progress=progress,
                    cancel_event=self._cancel_event,
                )
            finally:
                await method.close()

        def done(result, error):
            self.btn_extract.configure(state="normal")
            self.btn_cancel_extract.configure(state="disabled")
            self._cancel_event = None
            if error:
                messagebox.showerror("Ошибка", str(error))
                return
            if not result.success:
                messagebox.showerror("Ошибка", result.message)
                self._set_status(result.message)
                return
            self.extract_result.delete("1.0", tk.END)
            self.extract_result.insert("1.0", result.secret)
            self._set_status(result.message)
            self._set_metrics(
                f"бит={result.bits_extracted} · время={result.elapsed_sec:.2f}с · {result.details}"
            )
            self._set_progress(1.0)

        self._run_async(task(), done)

    def _start_experiments(self) -> None:
        try:
            runs = max(1, int(self.exp_runs.get().strip() or "5"))
        except ValueError:
            runs = 5
        secret = self.exp_secret.get().strip() or "test"
        method_key = LABEL_TO_KEY.get(self.exp_method_var.get(), "classic_synonyms")
        provider = self._selected_provider()
        model = self._selected_model()
        use_marker = bool(self.exp_marker_var.get())
        cover = generate_cover_local(secret, method_key, seed=42)

        self.btn_exp.configure(state="disabled")
        self.exp_log.delete("1.0", tk.END)
        self._set_status("Эксперименты…")

        async def task():
            rows = [["run", "success", "bits", "steps", "seconds", "message"]]
            out_dir = settings.project_root / "experiments" / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(runs):
                method = create_method(method_key, logits_provider=provider, model=model)
                t0 = time.perf_counter()
                try:
                    emb = await method.embed(
                        secret=secret,
                        cover_text=cover,
                        prompt=cover,
                        use_marker=use_marker,
                    )
                    ok = emb.success
                    if ok:
                        ext = await method.extract(
                            stego_text=emb.text,
                            prompt=cover,
                            use_marker=use_marker,
                        )
                        ok = ext.success and ext.secret == secret
                    dt = time.perf_counter() - t0
                    msg = emb.message if emb else ""
                    rows.append([
                        i + 1,
                        ok,
                        getattr(emb, "bits_embedded", 0),
                        getattr(emb, "positions_used", 0),
                        f"{dt:.3f}",
                        msg,
                    ])
                    self.after(
                        0,
                        lambda i=i, ok=ok, dt=dt: self.exp_log.insert(
                            tk.END, f"#{i+1} ok={ok} t={dt:.2f}s\n"
                        ),
                    )
                except Exception as exc:
                    rows.append([i + 1, False, 0, 0, "0.000", str(exc)])
                finally:
                    await method.close()
            path = out_dir / f"exp_{method_key}_{int(time.time())}.csv"
            write_experiment_csv(path, rows)
            return str(path)

        def done(path, error):
            self.btn_exp.configure(state="normal")
            if error:
                messagebox.showerror("Ошибка", str(error))
                return
            self._set_status(f"Эксперименты сохранены: {path}")
            self.exp_log.insert(tk.END, f"\nCSV: {path}\n")

        self._run_async(task(), done)


def run_app() -> None:
    setup_logger()
    NeuralTextStegoApp().mainloop()
