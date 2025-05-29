import threading
import time
import mss
import numpy as np
import cv2
from ultralytics import YOLO


class HpAnalyzerThread(threading.Thread):
    def __init__(self, region, update_callback, debug_window=None, interval=0.2):
        """
        :param region: Кортеж (x, y, width, height) — абсолютные координаты экрана для захвата
        :param update_callback: Функция с сигнатурой (status: str, hp_percent: float), вызывается после анализа
        :param debug_window: Объект окна для отображения отладочного изображения (может быть None)
        :param interval: Интервал между кадрами в секундах
        """
        super().__init__()
        self.region = region
        self.update_callback = update_callback
        self.debug_window = debug_window
        self.interval = interval
        self.running = True

        # Загружаем обученную модель YOLOv8
        self.model = YOLO(r"E:\Projectx\src\PEREUCHKA3\weights\best.pt")

    def analyze_hp_in_window(self, img, hp_box):
        """
        Анализирует заполненность полосы HP по выделенной области hp_box в изображении img.
        Возвращает процент заполненности (0-100).
        """
        x1, y1, x2, y2 = map(int, hp_box)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        hp_roi = img[y1:y2, x1:x2]
        if hp_roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(hp_roi, cv2.COLOR_BGR2HSV)
        # Маска красного цвета (учитываем два диапазона красного в HSV)
        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 120])
        upper_red2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        hp_line = np.max(mask, axis=0)
        red_cols = np.sum(hp_line > 100)
        hp_percent = red_cols / mask.shape[1] * 100
        return min(hp_percent, 100)

    def detect_and_analyze(self, img):
        """
        Выполняет детекцию объектов на изображении и анализ полосы HP.
        Возвращает статус, процент HP и координаты окна цели.
        """
        results = self.model(img, conf=0.25)
        boxes = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, score, class]

        window_box = None
        hp_box = None

        for box in boxes:
            x1, y1, x2, y2, score, cls = box
            cls = int(cls)
            if cls == 0 and score > 0.25:
                window_box = (x1, y1, x2, y2)
            elif cls == 1 and score > 0.25:
                hp_box = (x1, y1, x2, y2)

        if window_box is None:
            return "Цели нет", 0.0, None

        hp_percent = self.analyze_hp_in_window(img, hp_box) if hp_box else 0.0
        status = "Цель мертва" if hp_percent < 1.5 else "Цель жива"
        return status, hp_percent, window_box

    def run(self):
        with mss.mss() as sct:
            while self.running:
                x, y, w, h = self.region
                monitor = sct.monitors[0]
                x = max(monitor["left"], x)
                y = max(monitor["top"], y)
                w = min(w, monitor["width"] - (x - monitor["left"]))
                h = min(h, monitor["height"] - (y - monitor["top"]))

                monitor_region = {"top": y, "left": x, "width": w, "height": h}
                img = np.array(sct.grab(monitor_region))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                status, hp_percent, window_box = self.detect_and_analyze(img_bgr)
                self.update_callback(status, hp_percent)

                if self.debug_window and self.debug_window.winfo_exists():
                    debug_img = img_bgr.copy()
                    if window_box is not None:
                        x1, y1, x2, y2 = map(int, window_box)
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.debug_window.after(
                        0, lambda m=debug_img: self.debug_window.update_image(m)
                    )

                time.sleep(self.interval)

    def stop(self):
        self.running = False
