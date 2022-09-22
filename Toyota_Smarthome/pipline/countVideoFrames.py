import cv2

cap = cv2.VideoCapture("C:/Users/leech/Toyota_Smarthome/pipline/P18T15C03.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

