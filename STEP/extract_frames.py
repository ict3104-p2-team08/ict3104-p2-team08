# Python3 program to extract and save video frame
# using OpenCV Python

# import computer vision module
import cv2
import os
from config import parse_config
import shutil
from tqdm import tqdm

def add(x, y) -> str:  # remove type hint for Python 2
    return str(int(x) + int(y)).zfill(len(x))

# remove directory /frames and content after /frames/*
shutil.rmtree('./datasets/demo/frames')

args = parse_config()
# define the video name
file = args.video_name
# print(file)

# capture the video
cap = cv2.VideoCapture('../data/input_files/' + file)

# get fps of video
# framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
i = '00000'  # frame index to save frames

# remove .mp4 from filename string
filename_womp4 = file[:-4]

if not os.path.isdir('./datasets/demo/frames/'):
    os.makedirs('./datasets/demo/frames/')

if not os.path.isdir('./datasets/demo/frames/' + filename_womp4):
    os.makedirs('./datasets/demo/frames/' + filename_womp4)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
loop = tqdm(total=total_frames, position=0, leave=True)
# extract and save the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('datasets/demo/frames/' + filename_womp4 + '/frame' + str(i) + '.jpg', frame)
        i = add('00001', i)

        # show progress bar
        loop.set_description("Extracting frames..")
        loop.update(1)
    else:
        break
cap.release()
cv2.destroyAllWindows()
