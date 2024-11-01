import numpy as np
import cv2

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

    print(kernelMatrix)
    print(sum)
    kernelMatrix /= sum
    print(kernelMatrix)
    for i in range(size):
        for j in range(size):
            sum += kernelMatrix[i][j]
    print(sum)

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

img = cv2.imread("2375-202111021051505426.png",cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("2375-202111021051505426.png",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (480, 600))
img1 = cv2.resize(img1, (480, 600))
size = int(input())
om = int(input())
cv2.imshow("not_gray", img)
cv2.imshow("not_blured", img1)
cv2.imshow("blured", blur(img1, size, om))
cv2.imshow("blured2", cv2.GaussianBlur(img1,(size,size),om))

cv2.waitKey(0)
cv2.destroyAllWindows()