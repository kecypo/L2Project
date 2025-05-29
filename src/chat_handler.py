import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab
import pytesseract
import threading
import time
import json
import os

from area_selector import AreaSelector  # Импорт твоего AreaSelector


class ChatMessageHandler:
    """Обрабатывает сообщения чата по настраиваемым ключевым фразам."""

    def __init__(self, phrases=None):
        self.phrases = phrases or {
            "spoiled": ["вы используете: spoil", 'умение "оценить" активировано'],
            "not_spoiled": ["спойл не удался", "не удалось"],
            "already_spoiled": ["цель уже оценена"],
        }
        self.is_spoiled = False
        self.can_sweep = False

    def process_message(self, msg: str):
        msg_lower = msg.lower()
        if any(
            phrase.lower() in msg_lower for phrase in self.phrases.get("spoiled", [])
        ):
            self.is_spoiled = True
            self.can_sweep = True
            print("[ChatMessageHandler] Цель спойлена")
            return "spoiled"
        elif any(
            phrase.lower() in msg_lower
            for phrase in self.phrases.get("not_spoiled", [])
        ):
            self.is_spoiled = False
            self.can_sweep = False
            print("[ChatMessageHandler] Спойл не удался")
            return "not_spoiled"
        elif any(
            phrase.lower() in msg_lower
            for phrase in self.phrases.get("already_spoiled", [])
        ):
            self.is_spoiled = True
            self.can_sweep = True
            print("[ChatMessageHandler] Цель уже была оценена")
            return "already_spoiled"
        return None

    def get_state(self):
        return self.is_spoiled, self.can_sweep

    def save_phrases_to_file(self, filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.phrases, f, ensure_ascii=False, indent=4)
            print(f"[ChatMessageHandler] Фразы сохранены в {filepath}")
        except Exception as e:
            print(f"[ChatMessageHandler] Ошибка сохранения фраз: {e}")

    def load_phrases_from_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"[ChatMessageHandler] Файл с фразами не найден: {filepath}")
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.phrases = json.load(f)
            print(f"[ChatMessageHandler] Фразы загружены из {filepath}")
        except Exception as e:
            print(f"[ChatMessageHandler] Ошибка загрузки фраз: {e}")


class ChatSettingsWindow(tk.Toplevel):
    """Окно для настройки ключевых фраз."""

    def __init__(self, master, message_handler, on_settings_changed):
        super().__init__(master)
        self.title("Настройка фраз чата")
        self.geometry("600x600")
        self.message_handler = message_handler
        self.on_settings_changed = on_settings_changed

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.phrase_vars = {}

        self.create_tab("spoiled", "Успешный спойл")
        self.create_tab("not_spoiled", "Неудачный спойл")
        self.create_tab("already_spoiled", "Уже оценена")

        save_btn = tk.Button(self, text="Сохранить", command=self.save_settings)
        save_btn.pack(pady=10)

    def create_tab(self, state_name, tab_title):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=tab_title)

        phrases = self.message_handler.phrases.get(state_name, [])
        self.phrase_vars[state_name] = []

        for i, phrase in enumerate(phrases):
            tk.Label(tab, text=f"Фраза {i + 1}:").grid(
                row=i, column=0, sticky="w", padx=5, pady=2
            )
            var = tk.StringVar(value=phrase)
            entry = tk.Entry(tab, textvariable=var, width=60)
            entry.grid(row=i, column=1, padx=5, pady=2, sticky="we")
            self.phrase_vars[state_name].append(var)

        add_btn = tk.Button(
            tab, text="Добавить фразу", command=lambda s=state_name: self.add_phrase(s)
        )
        add_btn.grid(row=len(phrases), column=0, columnspan=2, pady=5)

    def add_phrase(self, state_name):
        tab = self.notebook.nametowidget(self.notebook.select())
        i = len(self.phrase_vars[state_name])
        tk.Label(tab, text=f"Фраза {i + 1}:").grid(
            row=i, column=0, sticky="w", padx=5, pady=2
        )
        var = tk.StringVar()
        entry = tk.Entry(tab, textvariable=var, width=60)
        entry.grid(row=i, column=1, padx=5, pady=2, sticky="we")
        self.phrase_vars[state_name].append(var)

    def save_settings(self):
        new_phrases = {}
        for state_name, vars_list in self.phrase_vars.items():
            new_phrases[state_name] = [v.get() for v in vars_list if v.get().strip()]
        self.on_settings_changed(new_phrases)
        self.destroy()


class ChatOCR(threading.Thread):
    """Поток для захвата экрана и распознавания текста."""

    def __init__(self, bbox, message_handler, hp_action_controller=None, interval=0.2):
        super().__init__(daemon=True)
        self.bbox = bbox  # (x, y, w, h)
        self.message_handler = message_handler
        self.hp_action_controller = hp_action_controller
        self.interval = interval
        self.running = False

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            x, y, w, h = self.bbox
            try:
                img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                text = pytesseract.image_to_string(img, lang="rus+eng")
                print(f"[ChatOCR] Распознанный текст:\n{text}\n{'-'*40}")
                state = self.message_handler.process_message(text)
                if state:
                    print(f"[ChatOCR] Обнаружено состояние: {state}")
                    if self.hp_action_controller:
                        is_spoiled, can_sweep = self.message_handler.get_state()
                        self.hp_action_controller.set_spoil_state(is_spoiled, can_sweep)
                time.sleep(self.interval)
            except Exception as e:
                print(f"[ChatOCR] Ошибка распознавания: {e}")
                time.sleep(5)


class ChatHandlerWindow(tk.Toplevel):
    PHRASES_FILE = "chat_phrases.json"

    def __init__(self, master=None, hp_action_controller=None):
        super().__init__(master)
        self.title("Обработчик чата")
        self.geometry("600x350")

        self.message_handler = ChatMessageHandler()
        self.hp_action_controller = hp_action_controller
        self.ocr = None
        self.selected_area = None

        btn_select_area = tk.Button(
            self, text="Выбрать область чата", command=self.select_area
        )
        btn_select_area.pack(pady=5)

        btn_start = tk.Button(self, text="Старт", command=self.start_ocr)
        btn_start.pack(pady=5)

        btn_stop = tk.Button(self, text="Стоп", command=self.stop_ocr)
        btn_stop.pack(pady=5)

        btn_settings = tk.Button(
            self, text="Настройки ключевых фраз", command=self.open_settings
        )
        btn_settings.pack(pady=5)

        # Индикатор состояния — цветной кружок
        self.status_indicator = tk.Label(self, text=" ", bg="red", width=2, height=1)
        self.status_indicator.pack(pady=5)

        # Текстовый статус
        self.status_label = tk.Label(self, text="Статус: Ожидание")
        self.status_label.pack(pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Загрузка фраз при старте
        self.message_handler.load_phrases_from_file(self.PHRASES_FILE)

    def update_status(self, text, color):
        self.status_label.config(text=f"Статус: {text}")
        self.status_indicator.config(bg=color)

    def select_area(self):
        def on_area_selected(area):
            self.selected_area = area
            self.update_status(f"Область чата выбрана: {area}", "orange")

        selector = AreaSelector(self, on_area_selected)
        selector.grab_set()

    def start_ocr(self):
        if not self.selected_area:
            self.update_status("Область не выбрана", "red")
            return
        if self.ocr:
            self.ocr.stop()
        self.ocr = ChatOCR(
            self.selected_area,
            self.message_handler,
            hp_action_controller=self.hp_action_controller,
        )
        self.ocr.start()
        self.update_status("Распознавание запущено", "green")

    def stop_ocr(self):
        if self.ocr:
            self.ocr.stop()
            self.ocr = None
            self.update_status("Распознавание остановлено", "red")

    def open_settings(self):
        def on_settings_changed(new_phrases):
            self.message_handler.phrases = new_phrases
            self.message_handler.save_phrases_to_file(self.PHRASES_FILE)
            self.update_status("Фразы чата обновлены и сохранены", "blue")

        settings_win = ChatSettingsWindow(
            self, self.message_handler, on_settings_changed
        )
        settings_win.grab_set()

    def on_close(self):
        self.stop_ocr()
        self.destroy()
