import os
import cv2

output_dir = 'cropped_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
img = cv2.imread("group.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.002, minNeighbors=5, minSize=(30, 30))

for idx, (x, y, w, h) in enumerate(faces):
    face = img[y-5:y+h+5, x-5:x+w+5]
    output_path = os.path.join(output_dir, f'face_{idx}.jpg')
    cv2.imwrite(output_path, face)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()