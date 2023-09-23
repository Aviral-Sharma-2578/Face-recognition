import os
import cv2

output_dir = 'cropped_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
img = cv2.imread("group2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (255, 0, 0), 2)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()