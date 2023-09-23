import cv2

cap = cv2.VideoCapture(0)

output_path = "grayscale_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30  
frame_size = (int(cap.get(3)), int(cap.get(4)))  
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size, isColor=False)  

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(gray_frame)
    cv2.imshow("Grayscale Video", gray_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()
