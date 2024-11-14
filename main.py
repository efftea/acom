import numpy as np
import cv2

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

    cv2.imshow("lengths", lengthVec)
    cv2.imshow("color", color)
    cv2.imshow("border", border)

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

    cv2.imshow("borders filtered", border)

img = cv2.imread("9349507864_4e82203bf8_b.jpg",cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (480, 600))

size = int(input("Gauss Size: "))
om = int(input("Gauss Sigma: "))

gauss = cv2.GaussianBlur(img, (size, size), om)

kanny(gauss)

cv2.imshow("gray", img)
cv2.imshow("blured", gauss)

cv2.waitKey(0)
cv2.destroyAllWindows()