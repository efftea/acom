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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 100, 100), (30, 255, 255))
    red_only = cv2.bitwise_and(frame, frame, mask=red_mask)
    cv2.imshow('Rec', red_only)
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
    dilate = cv2.dilate(red_frame, kernel, iterations=1)
    erode = cv2.erode(dilate, kernel, iterations=1)
    cv2.imshow('Close', erode)

    erode2 = cv2.erode(red_frame, kernel, iterations=1)
    dilate2 = cv2.dilate(erode2, kernel, iterations=1)
    cv2.imshow('Open', dilate2)


    moment = cv2.moments(dilate2)

    if (moment["m00"] != 0):
        print(f"Площадь: {moment['m00']}")
        print(f"Моменты 1 порядка: {moment['m01']}, {moment['m10']}")
        x = int(moment['m10'] / moment['m00'])
        y = int(moment['m01'] / moment['m00'])
        w = int(np.sqrt(moment['m00']))
        h = int((moment['m00'] / w))
        cv2.rectangle(frame, (x - (w//2), y - (h//2)), (x + (w//2), y + (h//2)), (255, 0, 0), 4)

    cv2.imshow('Moments', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()