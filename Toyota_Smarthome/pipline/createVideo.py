import os
print(os.getcwd())

# Python program to write
# text on video


import cv2

cap = cv2.VideoCapture('C:/Users/leech/Toyota_Smarthome/pipline/P18T15C03.mp4')

# Get video metadata
video_fps = cap.get(cv2.CAP_PROP_FPS),
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# we are using x264 codec for mp4
fourcc = cv2.VideoWriter_fourcc(*'X264')
writer = cv2.VideoWriter('OUTPUT_PATH.mp4', apiPreference=0, fourcc=fourcc,
                     fps=video_fps[0], frameSize=(int(width), int(height)))

i = 0
while (True):

    # Capture frames in the video
    ret, frame = cap.read()
    i = 1
    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # inserting text on video
    cv2.putText(frame,
                'TEXT ON VIDEO' + str(i),
                (50, 50),
                font, 1,
                (0, 0, 0),
                2,
                cv2.LINE_4)

    i += 1
    # Display the resulting frame
    cv2.imshow('video', frame)

    writer.write(frame)

    # creating 'q' as the quit
    # button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if not ret:
        break


writer.release()
# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()