import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from arduino_controller import ArduinoController
from arduino_auto import auto_detect_all_ports
from hp_analyzer import HpAnalyzerThread
from area_selector import AreaSelector
from events import HpActionController
from events_controller_window import EventsControllerWindow
from chat_handler import ChatHandlerWindow


class HpDebugWindow(tk.Toplevel):
    """
    Окно для отладки HP анализа — отображает текущий кадр с выделением области.
    """

    def __init__(self, master):
        super().__init__(master)
        self.title("Отладка HP Анализатора")
        self.canvas = tk.Canvas(self, width=640, height=480, bg="black")
        self.canvas.pack(padx=10, pady=10)
        self.photo_image = None
        self.img_on_canvas = None
        # ... внутри __init__ после других self.*_btn.pack()

    def update_image(self, cv_img):
        # Конвертируем BGR OpenCV изображение в RGB и отображаем в Tkinter Canvas
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img_rgb)
        self.photo_image = ImageTk.PhotoImage(image=pil_img)
        if self.img_on_canvas is None:
            self.img_on_canvas = self.canvas.create_image(
                0, 0, anchor=tk.NW, image=self.photo_image
            )
        else:
            self.canvas.itemconfig(self.img_on_canvas, image=self.photo_image)
        self.update_idletasks()


class Interface(tk.Tk):
    """
    Основной класс GUI приложения.
    """

    def __init__(self, arduino_ports, key_names, on_select_area, on_arduino_found):
        super().__init__()
        self.title("projectx")
        self.geometry("800x700")
        self.minsize(400, 400)
        self.resizable(True, True)
        self.chat_handler_window = None  # ссылка на окно, чтобы не открывать повторно
        self.open_chat_handler_btn = tk.Button(
            self, text="Открыть обработчик чата", command=self.open_chat_handler_window
        )
        self.open_chat_handler_btn.pack(pady=10)

        # Колбэки
        self.on_select_area = on_select_area
        self.on_arduino_found = on_arduino_found

        # Списки и переменные
        self.arduino_ports = arduino_ports.copy()
        self.key_names = key_names
        self.selected_area = None
        self.hp_analyzer_thread = None
        self.hp_debug_window = None
        self.arduino = None  # Экземпляр ArduinoController

        # Метка статуса
        self.status_label = tk.Label(self, text="Статус: Инициализация...")
        self.status_label.pack(pady=5)

        # Кнопка открытия окна контроллера событий
        self.events_window = None
        self.open_events_btn = tk.Button(
            self,
            text="Открыть окно контроллера событий",
            command=self.open_events_controller_window,
        )
        self.open_events_btn.pack(pady=10)

        # Выбор порта Arduino
        tk.Label(self, text="Выберите порт Arduino:").pack()
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(
            self,
            values=self.arduino_ports,
            textvariable=self.port_var,
            state="readonly",
        )
        if self.arduino_ports:
            self.port_combo.current(0)
        self.port_combo.pack(pady=5)
        self.port_combo.bind("<<ComboboxSelected>>", self.on_port_selected)

        # Выбор клавиши для отправки
        tk.Label(self, text="Выберите клавишу:").pack()
        self.key_var = tk.StringVar()
        self.key_combo = ttk.Combobox(
            self, values=self.key_names, textvariable=self.key_var, state="readonly"
        )
        if self.key_names:
            self.key_combo.current(0)
        self.key_combo.pack(pady=5)

        # Кнопка отправки команды вручную
        self.send_btn = tk.Button(
            self, text="Отправить команду", command=self.send_command
        )
        self.send_btn.pack(pady=10)

        # Кнопка выбора области экрана
        self.area_btn = tk.Button(
            self, text="Выбрать область экрана", command=self.start_select_area
        )
        self.area_btn.pack(pady=10)

        # Кнопки запуска и остановки HP анализа
        self.hp_start_btn = tk.Button(
            self, text="Запустить анализ HP", command=self.start_hp_analysis
        )
        self.hp_start_btn.pack(pady=10)

        self.hp_stop_btn = tk.Button(
            self,
            text="Остановить анализ HP",
            command=self.stop_hp_analysis,
            state=tk.DISABLED,
        )
        self.hp_stop_btn.pack(pady=5)

        # Инициализация контроллера событий HpActionController
        self.hp_action_controller = HpActionController(
            send_command_callback=self.send_key_to_arduino,
        )

        # Запуск автоопределения Arduino портов
        self.start_auto_detect()

        # Обработка закрытия окна
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def open_events_controller_window(self):
        if self.events_window is None or not self.events_window.winfo_exists():
            self.events_window = EventsControllerWindow(self, self.hp_action_controller)
        else:
            self.events_window.lift()
            self.events_window.focus_set()

    def open_chat_handler_window(self):
        self.chat_handler_window = ChatHandlerWindow(
            self, hp_action_controller=self.hp_action_controller
        )

        if (
            self.chat_handler_window is None
            or not self.chat_handler_window.winfo_exists()
        ):
            self.chat_handler_window = ChatHandlerWindow(self)
        else:
            self.chat_handler_window.lift()
            self.chat_handler_window.focus_set()

    def start_auto_detect(self):
        """
        Запуск автоопределения Arduino портов в отдельном потоке.
        """

        def detect():
            self.after(
                0, lambda: self.status_label.config(text="Статус: Поиск Arduino...")
            )

            def on_port_found(port):
                self.after(0, lambda: self.handle_arduino_found(port))

            auto_detect_all_ports(on_port_found)
            self.after(
                0, lambda: self.status_label.config(text="Статус: Поиск завершён")
            )

        threading.Thread(target=detect, daemon=True).start()

    def handle_arduino_found(self, port):
        """
        Обработка найденного Arduino порта:
        добавление в список, установка выбранного порта и создание ArduinoController.
        """
        if port not in self.arduino_ports:
            self.arduino_ports.append(port)
            self.port_combo["values"] = self.arduino_ports
        self.port_var.set(port)
        if self.arduino:
            self.arduino.close()
        self.arduino = ArduinoController(port)
        self.status_label.config(text=f"Arduino найден на порту: {port}")
        if self.on_arduino_found:
            self.on_arduino_found(port)

    def on_port_selected(self, event=None):
        """
        Обработка выбора порта вручную пользователем.
        """
        port = self.port_var.get()
        if self.arduino:
            self.arduino.close()
        if port:
            self.arduino = ArduinoController(port)
            print(f"[GUI] ArduinoController создан для порта {port}")

    def send_key_to_arduino(self, key_name):
        """
        Отправка команды на Arduino через ArduinoController.
        """
        if self.arduino:
            self.arduino.send_key_by_name(key_name)
        else:
            print("Arduino не подключен!")

    def send_command(self):
        """
        Отправка выбранной вручную команды на Arduino.
        """
        key = self.key_var.get()
        if self.arduino and key:
            self.arduino.send_key_by_name(key)
        else:
            messagebox.showwarning("Ошибка", "Выберите порт и клавишу")

    def start_select_area(self):
        """
        Запуск выбора области экрана.
        """
        AreaSelector(self, self.handle_area_selected)

    def handle_area_selected(self, selected_area):
        """
        Обработка выбранной области экрана.
        """
        self.selected_area = selected_area
        self.status_label.config(text=f"Выбрана область: {self.selected_area}")
        if self.hp_analyzer_thread and self.hp_analyzer_thread.is_alive():
            self.stop_hp_analysis()
        if self.on_select_area:
            self.on_select_area(selected_area)

    def start_hp_analysis(self):
        """
        Запуск анализа HP.
        """
        if self.selected_area is None:
            messagebox.showwarning("Ошибка", "Сначала выберите область экрана")
            return
        if self.hp_analyzer_thread and self.hp_analyzer_thread.is_alive():
            messagebox.showinfo("Информация", "Анализ уже запущен")
            return
        if self.hp_debug_window is None or not self.hp_debug_window.winfo_exists():
            self.hp_debug_window = HpDebugWindow(self)
        self.hp_analyzer_thread = HpAnalyzerThread(
            self.selected_area,
            self.hp_analysis_callback,
            debug_window=self.hp_debug_window,
            interval=0.2,
        )
        self.hp_analyzer_thread.start()
        self.status_label.config(text="Анализ HP запущен")
        self.hp_start_btn.config(state=tk.DISABLED)
        self.hp_stop_btn.config(state=tk.NORMAL)

    def stop_hp_analysis(self):
        """
        Остановка анализа HP.
        """
        if self.hp_analyzer_thread:
            self.hp_analyzer_thread.stop()
            self.hp_analyzer_thread.join()
            self.hp_analyzer_thread = None
            self.status_label.config(text="Анализ HP остановлен")
            self.hp_start_btn.config(state=tk.NORMAL)
            self.hp_stop_btn.config(state=tk.DISABLED)

    def hp_analysis_callback(self, status, hp_percent):
        def update():
            self.status_label.config(text=f"{status} | HP: {hp_percent:.2f}%")

            previous_status = getattr(self, "_previous_status", None)
            new_target = False

            # Новая цель — когда статус меняется с "Цели нет" на "Цель жива"
            if previous_status == "Цели нет" and status == "Цель жива":
                new_target = True

            # Также можно считать новой целью появление живой цели после смерти предыдущей
            if previous_status == "Цель мертва" and status == "Цель жива":
                new_target = True

            self._previous_status = status

            # Преобразуем статус для HpActionController
            state_map = {
                "Цели нет": "no_target",
                "Цель мертва": "dead_target",
                "Цель жива": "alive_target",
            }
            state = state_map.get(status, "no_target")

            self.hp_action_controller.update(state, hp_percent)

            if new_target and getattr(self, "spoil_manager", None):
                self.spoil_manager.on_new_target()
                self.spoil_manager.try_spoil()

        self.after(0, update)

    def on_closing(self):
        """
        Обработка закрытия окна — корректное завершение потоков и закрытие порта.
        """
        self.stop_hp_analysis()
        if self.arduino:
            self.arduino.close()
        self.hp_action_controller.stop()
        self.destroy()


if __name__ == "__main__":
    arduino_ports = []
    key_names = [
        "1",
        "PP",
        "2",
        "3",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "F9",
        "F10",
        "F11",
        "F12",
    ]

    def dummy_select(*args):
        pass

    def dummy_found(*args):
        pass

    app = Interface(arduino_ports, key_names, dummy_select, dummy_found)
    app.mainloop()
