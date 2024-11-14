import numpy as np
import cv2

cap = cv2.VideoCapture("ЛР4_main_video.mov", cv2.CAP_ANY)
ret, frame = cap.read()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("output.mp4", fourcc, 25, (w, h))

ret,frame=cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blured_one = cv2.GaussianBlur(gray, (7, 7), 100)

while True:
    ret, frame = cap.read()
    if not (ret):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (7, 7), 100)

    frame_diff = cv2.absdiff(blured_one, blured)

    thrash = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)[1]

    contours,_=cv2.findContours(thrash,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    try:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area > 50:
            video_writer.write(frame)
            print("Распознано")
    except:
        print("Не распознано")
    blured_one = blured

cap.release()
cv2.destroyAllWindows()