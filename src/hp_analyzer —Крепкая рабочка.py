import threading
import time
import mss
import numpy as np
import cv2
from ultralytics import YOLO


class HpAnalyzerThread(threading.Thread):
    def __init__(self, region, update_callback, debug_window=None, interval=0.2):
        """
        :param region: (x, y, width, height) — координаты области экрана для анализа
        :param update_callback: функция callback(status: str, hp_percent: float)
        :param debug_window: окно для отладки (может быть None)
        :param interval: интервал анализа (сек)
        """
        super().__init__()
        self.region = region
        self.update_callback = update_callback
        self.debug_window = debug_window
        self.interval = interval
        self.running = True

        # Загружаем YOLO-модель
        self.model = YOLO(r"E:\Projectx\src\FinalDodep\weights\best.pt")

    def analyze_hp_in_box(self, img, hp_box):
        """
        Анализирует процент заполнения HP bar по цвету внутри найденного бокса.
        Работает для красного, зелёного, жёлтого HP bar.
        """
        x1, y1, x2, y2 = map(int, hp_box)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        hp_roi = img[y1:y2, x1:x2]
        if hp_roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(hp_roi, cv2.COLOR_BGR2HSV)

        # Маски для красного, жёлтого, зелёного (можно добавить другие цвета)
        mask_red1 = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv, (160, 120, 120), (179, 255, 255))
        mask_yellow = cv2.inRange(hsv, (15, 120, 120), (35, 255, 255))
        mask_green = cv2.inRange(hsv, (36, 80, 80), (85, 255, 255))

        mask = cv2.bitwise_or(mask_red1, mask_red2)
        mask = cv2.bitwise_or(mask, mask_yellow)
        mask = cv2.bitwise_or(mask, mask_green)

        # Проекция по горизонтали (или вертикали — зависит от ориентации HP bar)
        hp_line = np.max(mask, axis=0)
        filled_cols = np.sum(hp_line > 100)
        hp_percent = filled_cols / mask.shape[1] * 100
        return min(max(hp_percent, 0.0), 100.0)

    def detect_and_analyze(self, img):
        results = self.model(img, conf=0.25)
        boxes = results[0].boxes.data.cpu().numpy()
        window_box = None
        hp_box = None

        for box in boxes:
            x1, y1, x2, y2, score, cls = box
            cls = int(cls)
            if cls == 0 and score > 0.25:
                window_box = (x1, y1, x2, y2)
            elif cls == 1 and score > 0.25:
                hp_box = (x1, y1, x2, y2)

        if window_box is None or hp_box is None:
            return "Цели нет", 0.0, None, None

        hp_percent = self.analyze_hp_in_box(img, hp_box)
        status = "Цель мертва" if hp_percent < 1.5 else "Цель жива"
        return status, hp_percent, window_box, hp_box

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

                status, hp_percent, window_box, hp_box = self.detect_and_analyze(
                    img_bgr
                )
                self.update_callback(status, hp_percent)

                # Визуализация для debug_window
                if self.debug_window and self.debug_window.winfo_exists():
                    debug_img = img_bgr.copy()
                    if window_box is not None:
                        x1, y1, x2, y2 = map(int, window_box)
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if hp_box is not None:
                        x1, y1, x2, y2 = map(int, hp_box)
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    self.debug_window.after(
                        0, lambda m=debug_img: self.debug_window.update_image(m)
                    )

                time.sleep(self.interval)

    def stop(self):
        self.running = False
