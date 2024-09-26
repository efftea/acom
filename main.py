from fileinput import filename
from sys import flags
import numpy as np
# Задание 1
import cv2


# Задание 2
img1 = cv2.imread(r'tKX6xpswJBY.jpg')
img2 = cv2.imread(r'tKX6xpswJBY.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(r'tKX6xpswJBY.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_8)

cv2.namedWindow('img 1',cv2.WINDOW_NORMAL)
cv2.imshow('img 1',img1)

cv2.namedWindow('img 2',cv2.WINDOW_FULLSCREEN)
cv2.imshow('img 2',img2)

cv2.namedWindow('img 3',cv2.WINDOW_AUTOSIZE)
cv2.imshow('img 3',img3)

cv2.waitKey(0)
cv2.destroyAllWindows()


#Задание 3
cap = cv2.VideoCapture(r'cat-watching-rain-moewalls-com.mp4', cv2.CAP_ANY)

while True:
    ret, frame = cap.read()
    if not(ret):
        break

    frame = cv2.resize(frame, (320, 240))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

#Задание 4
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width , frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out.write(frame)

cv2.destroyAllWindows()

#Задание 5
cv2.namedWindow('img 1',cv2.WINDOW_NORMAL)
cv2.imshow('img 1',img1)

hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

cv2.namedWindow('Hsv img 1',cv2.WINDOW_NORMAL)
cv2.imshow('Hsv img 1',hsv_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Задание 6
web_cap = cv2.VideoCapture("0")


ret, frame = web_cap.read()
if not ret:
    print("Не удалось получить кадр")
height, width, _ = frame.shape

center_x, center_y = width // 2, height // 2

points = np.array([
    [[center_x + 15, center_y - 125], [center_x + 15, center_y - 15]],
    [[center_x - 15, center_y - 125], [center_x - 15, center_y - 15]],
    [[center_x + 15, center_y + 125], [center_x + 15, center_y + 15]],
    [[center_x - 15, center_y + 125], [center_x - 15, center_y + 15]],

    [[center_x - 125, center_y - 15], [center_x + 125, center_y - 15]],
    [[center_x - 125, center_y + 15], [center_x + 125, center_y + 15]],
    [[center_x - 15, center_y - 125], [center_x + 15, center_y - 125]],
    [[center_x - 15, center_y + 125], [center_x + 15, center_y + 125]],

    [[center_x + 125, center_y - 15], [center_x + 125, center_y + 15]],
    [[center_x - 125, center_y + 15], [center_x - 125, center_y - 15]]
], dtype=np.int32)

for line in points:
    cv2.polylines(frame, [line], isClosed=True, color=(0, 0, 255), thickness=2)

cv2.imshow('Cross', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()


#Задание 7
web_cap = cv2.VideoCapture("0")
output_file = 'output_webc.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(web_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(web_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = web_cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width , frame_height))

while True:
    ret, frame = web_cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break
    out.write(frame)
    cv2.imshow('Rec', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

#Задание 8
web_cap = cv2.VideoCapture("0")

ret, frame = web_cap.read()
if not ret:
    print("Не удалось получить кадр")

height, width, _ = frame.shape

center_x, center_y = width // 2, height // 2

points = np.array([
    [center_x - 125, center_y - 15],
    [center_x - 15, center_y - 15],
    [center_x - 15, center_y - 125],
    [center_x + 15, center_y - 125],
    [center_x + 15, center_y - 15],
    [center_x + 125, center_y - 15],
    [center_x + 125, center_y + 15],
    [center_x + 15, center_y + 15],
    [center_x + 15, center_y + 125],
    [center_x - 15, center_y + 125],
    [center_x - 15, center_y + 15],
    [center_x - 125, center_y + 15],
    [center_x - 125, center_y - 15]
], np.int32)

pixel_color = img1[center_y, center_x]

color_max = [0,0,0]
for i in range(3):
    if max(pixel_color) == pixel_color[i]:
        color_max[i] = 255

cv2.fillPoly(img1, [points], color_max)

cv2.imshow('Cross', frame)
cv2.waitKey(0)

cv2.destroyAllWindows()

#Задание 9
web_cap = cv2.VideoCapture("http://192.168.1.68:8080/video")


ret, frame = web_cap.read()
if not ret:
    print("Не удалось получить кадр")

cv2.imshow('IP cam', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()