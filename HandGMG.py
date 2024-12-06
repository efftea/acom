import time
import cv2
import numpy as np


def gauss(x,y,a,b,om):
    return (1/(2*np.pi*(om**2))) * np.exp(-((((x-a)**2)+((y-b)**2))/(2*(om**2))))

def blur(blurImg, size, om):
    kernelMatrix = np.empty((size, size))
    a = (size)//2
    b = (size)//2
    sum = 0
    for i in range(size):
        for j in range(size):
            kernelMatrix[i][j] = gauss(i,j,a,b,om)
            sum += kernelMatrix[i][j]


    kernelMatrix /= sum
    for i in range(size):
        for j in range(size):
            sum += kernelMatrix[i][j]

    n_img = blurImg.copy()

    for i in range(0, blurImg.shape[0]):
        for j in range(0,  blurImg.shape[1]):
            val=0
            for k in range(i if (blurImg.shape[0]-i >= size) else i-(size-((blurImg.shape[0]-i)%size)), (i+size) if (blurImg.shape[0]-i >= size) else i+((blurImg.shape[0]-i)%size)):
                for l in range(j if (blurImg.shape[1]-j >= size) else j-(size-((blurImg.shape[1]-j)%size)), (j+size) if (blurImg.shape[1]-j >= size) else j+((blurImg.shape[1]-j)%size)):
                    val += blurImg[k,l] * kernelMatrix[(k-i) if (blurImg.shape[0]-i >= size) else (k - (i-(size-((blurImg.shape[0]-i)%size)))), (l - j) if (blurImg.shape[1]-j >= size) else (l - (j-(size-((blurImg.shape[1]-j)%size))))]
                    # print(f"{i}, {j}, {k}, {l}")
            n_img[i, j] = val


    return n_img

def kanny(gauss):
    border = np.zeros(gauss.shape)
    lengthVec = np.zeros(gauss.shape)
    color = np.zeros(gauss.shape)
    maxLen = np.max(lengthVec)
    x = 0
    y = 0
    for i in range(1, len(gauss)-1):
        for j in range(1, len(gauss[0])-1):
            GrX = (gauss[i + 1][j + 1] - gauss[i - 1][j - 1] + gauss[i + 1][j - 1] - gauss[i - 1][j + 1] + 2 * (gauss[i + 1][j] - gauss[i - 1][j]))
            GrY = (gauss[i + 1][j + 1] - gauss[i - 1][j - 1] + gauss[i - 1][j + 1] - gauss[i + 1][j - 1] + 2 * (gauss[i][j + 1] - gauss[i][j - 1]))
            lengthVec[i][j] = np.sqrt(GrX ** 2 + GrY ** 2)
            tg = np.arctan(GrY / GrX)

            if (GrX > 0 and GrY < 0 and tg < -2.414) or (GrX < 0 and GrY < 0 and tg > 2.414):
                color[i][j] = 0
                x = 0
                y = -1
            elif (GrX > 0 and GrY < 0 and tg < -0.414):
                color[i][j] = 1
                x = 1
                y = -1
            elif (GrX > 0 and GrY < 0 and tg > -0.414) or (GrX > 0 and GrY > 0 and tg > 0.414):
                color[i][j] = 2
                x = 1
                y = 0
            elif (GrX > 0 and GrY > 0 and tg < 2.414):
                color[i][j] = 3
                x = 1
                y = 1
            elif (GrX > 0 and GrY > 0 and tg > 2.414) or (GrX < 0 and GrY > 0 and tg < -2.414):
                color[i][j] = 4
                x = 0
                y = 1
            elif (GrX < 0 and GrY > 0 and tg < -0.414):
                color[i][j] = 5
                x = -1
                y = 1
            elif (GrX < 0 and GrY > 0 and tg > -0.414) or (GrX < 0 and GrY < 0 and tg < 0.414):
                color[i][j] = 6
                x = -1
                y = 0
            elif (GrX < 0 and GrY < 0 and tg < 2.414):
                color[i][j] = 7
                x = -1
                y = -1

            if (lengthVec[i][j]>lengthVec[i+x][j+y] and lengthVec[i][j]>lengthVec[i-x][j-y]):
                border[i][j] = 255
            else:
                border[i][j] = 0

    low_level = maxLen // 25
    high_level = maxLen // 10
    for x in range(1, (len(gauss) - 1)):
        for y in range(1, len(gauss[0]) - 1):
            if (border[x][y] == 255):
                if (lengthVec[x][y] < low_level):
                    border[x][y] = 0

            if (border[x][y] == 255):
                if (lengthVec[x][y] <= high_level):
                    if (border[x - 1][y - 1] == 255 or border[x - 1][y] == 255 or border[x - 1][y + 1] == 255 or border[x][y + 1] == 255 or border[x + 1][y + 1] == 255 or border[x + 1][y] == 255 or border[x + 1][y - 1] == 255 or border[x][y - 1] == 255):
                        border[x][y] = 255
                    else:
                        border[x][y] = 0

    return border

cap = cv2.VideoCapture('./Видосы/5.mp4')
fourcc = cv2.VideoWriter.fourcc(*'XVID')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('./Result/HandGMG/5_HandGMG.avi', fourcc, 25.0, (320, 180))

start = time.time()
ret, frame = cap.read()
frame = cv2.resize(frame, (320, 180))

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = blur(gray, 3, 10)
cv2.imshow('gaussian_blur_frame', blurred)
prev_frame = blurred

thrash = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('trashhold_frame', thrash)

kanny_frame = kanny(blurred)
cv2.imshow('kanny_frame', kanny_frame)

kernel = np.ones((4, 4), np.uint8)
erode = cv2.erode(thrash, kernel, iterations=1)
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(erode, kernel, iterations=1)
cv2.imshow('morph_frame', dilate)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (320, 180))
    cv2.imshow('Frame', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = blur(gray, 5, 1)
    cv2.imshow('gaussian_blur_frame', blurred)

    frame_diff = cv2.absdiff(prev_frame, blurred)
    cv2.imshow('diff_frame', frame_diff)
    prev_frame = blurred

    thrash = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('trashhold_frame', thrash)

    kanny_frame = kanny(thrash)
    cv2.imshow('kanny_frame', kanny_frame)

    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(kanny_frame, kernel, iterations=1)
    kernel = np.ones((20, 20), np.uint8)
    dilate = cv2.dilate(erode, kernel, iterations=1)
    dilate = dilate.astype(np.uint8)
    cv2.imshow('morph_frame', dilate)

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [c for c in contours if cv2.contourArea(c) > 200]

    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Рисуем прямоугольник вокруг объекта
    out.write(frame)
    cv2.imshow('Detected Objects', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.time()

if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время обработки видео: {end - start:.5f} секунд")
    print(f"Частота кадров: {cap.get(cv2.CAP_PROP_FPS):.0f} кадров в секунду")
    print(f"Частота обработки объектов: {1 / ((end - start) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров в секунду")
else:
    print("Видеопоток был пуст")

cap.release()
out.release()
cv2.destroyAllWindows()