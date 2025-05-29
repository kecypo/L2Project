import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.gui import Interface


def on_select_area(selected_area):
    print("Выбрана область:", selected_area)


def on_arduino_found(port):
    print(f"Arduino найден на порту: {port}")


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

    app = Interface(
        arduino_ports=arduino_ports,
        key_names=key_names,
        on_select_area=on_select_area,
        on_arduino_found=on_arduino_found,
    )
    app.mainloop()
