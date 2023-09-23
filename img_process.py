import numpy as np
import cv2

img = cv2.imread("opencv\samples\data\\apple.jpg")
b, g, r = cv2.split(img)
b = np.zeros(r.shape, np.uint8)
img = cv2.merge((b, g, r))
cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()