import cv2
import numpy as np

# Loads in the images using the cv2 lib
image = cv2.imread('Road_Test.jpg')

# Creates a copy of the image to be later converted to grayscale
lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# Displays the image onscreen in a new window titled result
cv2.imshow('result', gray)
cv2.waitKey(0)


