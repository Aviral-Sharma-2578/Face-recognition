import os
import cv2

output_dir = 'cropped_faces_refined'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
idx = 0

for images in os.listdir('cropped_faces'):
    img = cv2.imread(os.path.join('cropped_faces', images))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        output_path = os.path.join(output_dir, f'face_{idx}.jpg')
        idx += 1
        cv2.imwrite(output_path, face)




