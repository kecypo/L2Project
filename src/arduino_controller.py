import serial
import time
import serial.tools.list_ports


class ArduinoController:
    # Карта команд — сопоставляет имена клавиш с командами для Arduino
    key_map = {
        # Команды из твоего скетча
        "F11_random": "11",
        "F11_once": "FF",
        "Ping": "PP",
        # Новые функциональные клавиши (пример)
        "F1": "F1",
        "F2": "F2",
        "F3": "F3",
        "F4": "F4",
        "F5": "F5",
        "F6": "F6",
        "F7": "F7",
        "F8": "F8",
        "F9": "F9",
        "F10": "F10",
        "F12": "F12",
        # Цифры
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "0": "0",
        # Буквы (заглавные)
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "E",
        "F": "F",
        "G": "G",
        "H": "H",
        "I": "I",
        "J": "J",
        "K": "K",
        "L": "L",
        "M": "M",
        "N": "N",
        "O": "O",
        "P": "P",
        "Q": "Q",
        "R": "R",
        "S": "S",
        "T": "T",
        "U": "U",
        "V": "V",
        "W": "W",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
        # Модификаторы и спецклавиши (примеры)
        "Alt": "Alt",
        "Ctrl": "Ctrl",
        "Shift": "Shift",
        "Space": "Space",
        "Enter": "Enter",
        "Esc": "Esc",
        "Tab": "Tab",
    }

    def __init__(self, port, baudrate=9600, timeout=0.5):
        self.ser = None
        self.port = port
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Подключено к порту {port} с baudrate {baudrate}")
            time.sleep(0.5)  # Ждём инициализацию Arduino
        except serial.SerialException as e:
            print(f"Ошибка при подключении к порту {port}: {e}")

    def send_command(self, command):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(command.encode("utf-8"))
                print(f"Отправлено: {repr(command)}")
            except Exception as e:
                print(f"Ошибка при отправке команды: {e}")
        else:
            print("Порт не открыт или не инициализирован")

    def send_key_by_name(self, key_name):
        cmd = self.key_map.get(key_name)
        if cmd:
            self.send_command(cmd + "\n")
        else:
            print(f"Клавиша '{key_name}' не найдена в карте команд.")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Порт закрыт")
