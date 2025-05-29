import serial
import serial.tools.list_ports
import time
import threading

def check_port(port, baudrate, handshake_command, expected_response, timeout, result, lock):
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(1)  # Ждём Arduino после открытия порта
        ser.reset_input_buffer()
        ser.write((handshake_command + '\n').encode('utf-8'))
        response = ser.readline().decode('utf-8').strip()
        ser.close()
        if response == expected_response:
            with lock:
                result.append(port)
    except serial.SerialException as e:
        # Порт занят или недоступен
        pass
    except Exception as e:
        # Другие ошибки
        pass

def auto_detect_arduino_threaded(baudrate=9600, handshake_command='PP', expected_response='PONG', timeout=1, max_thread_time=3):
    ports = [port.device for port in serial.tools.list_ports.comports()]
    print(f"Доступные порты: {ports}")

    threads = []
    result = []
    lock = threading.Lock()

    for port in ports:
        t = threading.Thread(target=check_port, args=(port, baudrate, handshake_command, expected_response, timeout, result, lock))
        t.daemon = True
        threads.append(t)
        t.start()

    # Ждём максимум max_thread_time секунд на все потоки
    start_time = time.time()
    while time.time() - start_time < max_thread_time:
        if result:
            # Найден подходящий порт, можно прервать ожидание
            break
        time.sleep(0.1)

    # Завершаем все потоки (демон-потоки завершатся с выходом из программы)
    return result[0] if result else None

if __name__ == "__main__":
    port = auto_detect_arduino_threaded()
    if port:
        print(f"Arduino найдена на порту: {port}")
    else:
        print("Arduino не найдена.")
