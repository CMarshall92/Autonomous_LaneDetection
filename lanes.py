import cv2
import numpy as np

def canny(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

raw_image = cv2.imread('Road_Test.jpg')
raw_copy = np.copy(raw_image)
canny_final = canny(raw_copy)
cropped_image = region_of_interest(canny_final)
cv2.imshow('result', cropped_image)
cv2.waitKey(0)