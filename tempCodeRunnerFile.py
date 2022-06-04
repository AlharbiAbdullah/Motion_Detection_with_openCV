import cv2 , time

cap = cv2.VideoCapture(1)


check , frame = cap.read()


time.sleep(14)

cap.release()