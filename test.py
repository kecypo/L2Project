from ultralytics import YOLO
import cv2

model = YOLO(r"E:\Projectx\src\my_yolo_model3\weights\best.pt")

img = cv2.imread(r"E:\Projectx\screenshot.jpg")

results = model([img], conf=0.1)  # Передаём список изображений

for result in results:
    result.plot()  # Отрисовать предсказания на изображении
    result.show()  # Показать изображение с предсказаниями

print(results)  # Выводит детали предсказаний
