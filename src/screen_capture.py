import mss
import numpy as np
import cv2

class ScreenCapture:
    def __init__(self, region):
        self.region = region
        self.sct = mss.mss()

    def capture(self):
        x, y, w, h = self.region
        monitor = {"top": y, "left": x, "width": w, "height": h}
        img = np.array(self.sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
