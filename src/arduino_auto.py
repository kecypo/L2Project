import threading
import serial
import serial.tools.list_ports

def check_port(port, result_callback):
    """
    Проверяет порт на наличие Arduino.
    Отправляет команду 'PP' и ждёт ответ 'pong'.
    Если ответ получен, вызывает result_callback с портом.
    """
    try:
        ser = serial.Serial(port, 9600, timeout=0.5)
        ser.reset_input_buffer()
        ser.write(b'PP\n')  # Команда, которая есть в твоём скетче
        response = ser.readline().decode(errors='ignore').strip()
        ser.close()
        if response.lower() == 'pong':
            result_callback(port)
    except Exception:
        # Игнорируем ошибки, например, если порт занят или не отвечает
        pass

def auto_detect_all_ports(result_callback):
    """
    Параллельно запускает проверку всех доступных COM-портов.
    Для каждого порта создаёт отдельный поток.
    Не блокирует вызывающий поток.
    """
    ports = [p.device for p in serial.tools.list_ports.comports()]

    for port in ports:
        thread = threading.Thread(target=check_port, args=(port, result_callback), daemon=True)
        thread.start()
