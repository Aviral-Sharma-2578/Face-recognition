import os
import cv2

output_dir = 'cropped_face'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
img = cv2.imread("group2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
for idx, (x, y, w, h) in enumerate(faces):
    face = img[y:y+h, x:x+w]
    output_path = os.path.join(output_dir, f'face_{idx}.jpg')
    cv2.imwrite(output_path, face)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()