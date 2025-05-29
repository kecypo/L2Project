import tkinter as tk
import json
import os

SETTINGS_FILE = "events_settings.json"


class EventsControllerWindow(tk.Toplevel):
    def __init__(self, master, hp_action_controller):
        super().__init__(master)
        self.title("Контроллер событий HP")
        self.geometry("600x600")  # чуть увеличил высоту под новые поля
        self.resizable(False, False)
        self.hp_action_controller = hp_action_controller

        # Команды и cooldown (как было)
        tk.Label(self, text="Команда при отсутствии цели:").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.no_target_var = tk.StringVar()
        tk.Entry(self, textvariable=self.no_target_var).pack(fill=tk.X, padx=10)

        tk.Label(self, text="Команда при мёртвой цели:").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.dead_target_var = tk.StringVar()
        tk.Entry(self, textvariable=self.dead_target_var).pack(fill=tk.X, padx=10)

        tk.Label(self, text="Команда при живой цели:").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.alive_target_var = tk.StringVar()
        tk.Entry(self, textvariable=self.alive_target_var).pack(fill=tk.X, padx=10)

        tk.Label(self, text="Cooldown (сек):").pack(anchor="w", padx=10, pady=(10, 0))
        self.cooldown_var = tk.DoubleVar()
        tk.Entry(self, textvariable=self.cooldown_var).pack(fill=tk.X, padx=10)

        # --- Добавляем спойл ---

        self.spoil_enabled_var = tk.BooleanVar()
        self.spoil_enabled_check = tk.Checkbutton(
            self,
            text="Включить спойл мобов",
            variable=self.spoil_enabled_var,
            command=self.on_spoil_enabled_changed,
        )
        self.spoil_enabled_check.pack(anchor="w", padx=10, pady=(15, 0))

        tk.Label(self, text="Клавиша Spoil:").pack(anchor="w", padx=10, pady=(10, 0))
        self.spoil_key_var = tk.StringVar()
        tk.Entry(self, textvariable=self.spoil_key_var).pack(fill=tk.X, padx=10)

        tk.Label(self, text="Клавиша Sweep:").pack(anchor="w", padx=10, pady=(10, 0))
        self.sweep_key_var = tk.StringVar()
        tk.Entry(self, textvariable=self.sweep_key_var).pack(fill=tk.X, padx=10)

        # --- Конец добавления спойла ---

        # Кнопка сохранения настроек
        self.save_btn = tk.Button(
            self, text="Сохранить настройки", command=self.save_settings
        )
        self.save_btn.pack(pady=10)

        # Кнопка старта/стопа обработки событий
        self.toggle_btn = tk.Button(
            self, text="Старт обработки", command=self.toggle_processing
        )
        self.toggle_btn.pack(pady=5)

        # Статус
        self.status_label = tk.Label(self, text="Обработка событий: ВЫКЛ")
        self.status_label.pack(pady=10)

        # Сохраняем состояние обработки
        self.processing_active = self.hp_action_controller.enabled
        self.update_toggle_button()

        # Загружаем настройки из файла или из контроллера
        self.load_settings()

    def load_settings(self):
        """Загрузить настройки из файла или из контроллера (если файла нет)."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.no_target_var.set(data.get("no_target_command", ""))
                self.dead_target_var.set(data.get("dead_target_command", ""))
                self.alive_target_var.set(data.get("alive_target_command", ""))
                self.cooldown_var.set(float(data.get("cooldown_sec", 1)))

                # --- Загрузка спойла ---
                self.spoil_enabled_var.set(data.get("spoil_enabled", False))
                self.spoil_key_var.set(data.get("spoil_key", ""))
                self.sweep_key_var.set(data.get("sweep_key", ""))
                # --- Конец загрузки спойла ---

                # Обновить контроллер событий
                self.hp_action_controller.set_no_target_command(
                    self.no_target_var.get()
                )
                self.hp_action_controller.set_dead_target_command(
                    self.dead_target_var.get()
                )
                if hasattr(self.hp_action_controller, "set_alive_target_command"):
                    self.hp_action_controller.set_alive_target_command(
                        self.alive_target_var.get()
                    )
                self.hp_action_controller.set_cooldown(self.cooldown_var.get())

                # --- Обновить спойл в контроллере ---
                self.hp_action_controller.set_spoil_enabled(
                    self.spoil_enabled_var.get()
                )
                self.hp_action_controller.set_spoil_key(self.spoil_key_var.get())
                self.hp_action_controller.set_sweep_key(self.sweep_key_var.get())
                # --- Конец обновления спойла ---

                self.status_label.config(text="Настройки загружены из файла")
            except Exception as e:
                self.status_label.config(text=f"Ошибка загрузки настроек: {e}")
                self.update_fields_from_controller()
        else:
            self.update_fields_from_controller()

    def update_fields_from_controller(self):
        """Обновить значения полей из контроллера событий."""
        self.no_target_var.set(self.hp_action_controller.no_target_command)
        self.dead_target_var.set(self.hp_action_controller.dead_target_command)
        self.alive_target_var.set(
            getattr(self.hp_action_controller, "alive_target_command", "")
        )
        self.cooldown_var.set(self.hp_action_controller.cooldown_sec)

        # --- Обновляем спойл поля ---
        self.spoil_enabled_var.set(
            getattr(self.hp_action_controller, "spoil_enabled", False)
        )
        self.spoil_key_var.set(getattr(self.hp_action_controller, "spoil_key", ""))
        self.sweep_key_var.set(getattr(self.hp_action_controller, "sweep_key", ""))
        # --- Конец обновления спойла ---

    def save_settings(self):
        """Сохранить настройки из полей в контроллер событий и в файл."""
        # В контроллер
        self.hp_action_controller.set_no_target_command(self.no_target_var.get())
        self.hp_action_controller.set_dead_target_command(self.dead_target_var.get())
        if hasattr(self.hp_action_controller, "set_alive_target_command"):
            self.hp_action_controller.set_alive_target_command(
                self.alive_target_var.get()
            )
        self.hp_action_controller.set_cooldown(self.cooldown_var.get())

        # --- Сохраняем спойл ---
        self.hp_action_controller.set_spoil_enabled(self.spoil_enabled_var.get())
        self.hp_action_controller.set_spoil_key(self.spoil_key_var.get())
        self.hp_action_controller.set_sweep_key(self.sweep_key_var.get())
        # --- Конец сохранения спойла ---

        # В файл
        data = {
            "no_target_command": self.no_target_var.get(),
            "dead_target_command": self.dead_target_var.get(),
            "alive_target_command": self.alive_target_var.get(),
            "cooldown_sec": self.cooldown_var.get(),
            "spoil_enabled": self.spoil_enabled_var.get(),
            "spoil_key": self.spoil_key_var.get(),
            "sweep_key": self.sweep_key_var.get(),
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.status_label.config(text="Настройки сохранены (и применены)")
        except Exception as e:
            self.status_label.config(text=f"Ошибка сохранения: {e}")

    def on_spoil_enabled_changed(self):
        enabled = self.spoil_enabled_var.get()
        self.hp_action_controller.set_spoil_enabled(enabled)

    def toggle_processing(self):
        """Запустить или остановить обработку событий."""
        if self.processing_active:
            self.hp_action_controller.stop()
            self.processing_active = False
        else:
            self.hp_action_controller.start()
            self.processing_active = True
        self.update_toggle_button()

    def update_toggle_button(self):
        """Обновить текст и статус кнопки и статус-лейбла."""
        if self.processing_active:
            self.toggle_btn.config(text="Стоп обработки")
            self.status_label.config(text="Обработка событий: ВКЛ")
        else:
            self.toggle_btn.config(text="Старт обработки")
            self.status_label.config(text="Обработка событий: ВЫКЛ")
