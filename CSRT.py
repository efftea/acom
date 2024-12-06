import time
import cv2
import numpy as np


cap = cv2.VideoCapture('./Видосы/1.mp4')
fourcc = cv2.VideoWriter.fourcc(*'XVID')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('./Result/CSRT/1_CSRT.avi', fourcc, 25.0, (1280, 720))

start = time.time()

ret, frame = cap.read()

frame = cv2.resize(frame, (1280, 720))

bbox = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)

tracker = cv2.legacy.TrackerCSRT_create()
tracker.init(frame, bbox)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))

    success, bbox = tracker.update(frame)

    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking Success", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking Failure", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow('TrackCSRT', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.time()

if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время обработки одного кадра CSRT: {end - start:.5f} секунд")
    print(f"Частота кадров: {cap.get(cv2.CAP_PROP_FPS):.0f} кадров в секунду")
    print(f"Частота обработки объектов: {1 / ((end - start) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров в секунду")
else:
    print("Видеопоток был пуст")

cap.release()
out.release()
cv2.destroyAllWindows()