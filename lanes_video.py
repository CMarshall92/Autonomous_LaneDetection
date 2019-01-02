import cv2
import numpy as np
raw_image = cv2.imread('./Content/Road_Test_Image.jpg')

def canny(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def average_slope(image, lines):
    left_lines, right_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        paramaters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = paramaters[0]
        intercept = paramaters[1]
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    return np.array([
        make_coordinates(image, np.average(left_lines, axis = 0)),
        make_coordinates(image, np.average(right_lines, axis = 0))
    ])
        
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

raw_copy = np.copy(raw_image)
cropped_image = region_of_interest(canny(raw_copy))
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
average_lines = average_slope(raw_copy, lines)
combine_image = cv2.addWeighted(raw_copy, 0.8, display_lines(raw_copy, average_lines), 1, 1)

# Dsiplay on screen
cv2.imshow('result', combine_image)
cv2.waitKey(0)