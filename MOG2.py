import time
import cv2
import numpy as np

# Открытие видео
cap = cv2.VideoCapture('./Видосы/5.mp4')

# Создание фона с помощью Mixture of Gaussians - MOG2, чтобы отслеживать движущиеся объекты
# Объекты на видео, скорее всего станут статичными после первого кадра
mog = cv2.createBackgroundSubtractorMOG2()

fourcc = cv2.VideoWriter.fourcc(*'XVID')  # Кодек для записи, для создания видеофайла
out = cv2.VideoWriter('./Result/MOG2/5_MOG2.avi', fourcc, 20.0, (1280, 720))  # Создание выходного файла для видео

start = time.time()

# Цикл для обработки видео и отслеживания движущихся объектов
while True:
    # Чтение следующего кадра
    ret, frame = cap.read()
    # Если false -> остановка
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))

    # Применение метода для выделения движущихся объектов к кадру
    mogMask = mog.apply(frame)

    # Находим контуры объектов на маске и рисуем их на исходном кадре
    contours, hierarchy = cv2.findContours(mogMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Получаем координаты и размеры прямоугольника вокруг движущегося объекта
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50 and w / h > 1.3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Записываем текущий кадр в выходной файл
    out.write(frame)
    cv2.imshow('TrackMOG2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.time()

# Выводим статистику о времени обработки и частоте кадров
if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время обработки одного кадра MOG2: {end - start:.5f} секунд")
    print(f"Частота кадров: {cap.get(cv2.CAP_PROP_FPS):.0f} кадров в секунду")
    print(f"Частота обработки объектов: {1 / ((end - start) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров в секунду")
else:
    print("Видеопоток был пуст")

cap.release()
out.release()

cv2.destroyAllWindows()
