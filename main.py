from fileinput import filename
from sys import flags
import numpy as np
import cv2


# Задание 1
web_cap = cv2.VideoCapture("http://192.168.1.68:8080/video")

while True:
    ret, frame = web_cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('Rec', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()


#Задание 2
web_cap = cv2.VideoCapture("http://192.168.1.68:8080/video")

while True:
    ret, frame = web_cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.inRange(frame, (0, 100, 100), (30, 255, 255))
    cv2.imshow('Rec', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()


#Задание 3,4,5
web_cap = cv2.VideoCapture(r'http://192.168.1.68:8080/video')

while True:
    ret, frame = web_cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_frame = cv2.inRange(hsv_frame, (0, 100, 100), (30, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(red_frame, kernel, iterations=1)
    dilate = cv2.dilate(erode, kernel, iterations=1)


    moment = cv2.moments(erode)
    if (moment["m00"] != 0):
        print(f"Площадь: {moment['m00']}")
        print(f"Моменты 1 порядка: {moment['m01']}, {moment['m10']}")
        xc = int(moment['m10'] / moment['m00'])
        yc = int(moment['m01'] / moment['m00'])
    (contours, _) = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for countour in contours:
        (x, y, w, h) = cv2.boundingRect(countour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

    cv2.imshow('Rec', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()