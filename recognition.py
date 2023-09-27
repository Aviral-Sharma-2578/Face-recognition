import cv2
import numpy
import face_recognition

imgElon = face_recognition.load_image_file('elon_musk\elon_musk_1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgBill = face_recognition.load_image_file('bill_gates\\bill_gates_1.jpg')
imgBill = cv2.cvtColor(imgBill, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('elon_musk\elon_musk_2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

facLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (facLoc[3], facLoc[0]), (facLoc[1], facLoc[2]), (255, 0, 255), 2)

facLoc = face_recognition.face_locations(imgBill)[0]
encodeBill = face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill, (facLoc[3], facLoc[0]), (facLoc[1], facLoc[2]), (255, 0, 255), 2)

facLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facLoc[3], facLoc[0]), (facLoc[1], facLoc[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encodeBill], encodeTest)
print(result)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Test Image', imgTest)
cv2.waitKey(0)