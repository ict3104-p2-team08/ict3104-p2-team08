from config import parse_config
import os
import cv2

args = parse_config()
frame_arr = os.listdir('./datasets/demo/results/' + args.video_name[:-4])
image = cv2.imread('./datasets/demo/results/' + args.video_name[:-4] + '/' + frame_arr[0])
height = list(image.shape)[0]
width = list(image.shape)[1]
print(height)

# create video from frames
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./video_output/' + args.video_name, 0, 1, (width, height))

for i in range(len(frame_arr)):
    img = cv2.imread('./datasets/demo/results/' + args.video_name[:-4] + "/" + frame_arr[i])
    print(img)
    video.write(img)

cv2.destroyAllWindows()
video.release()
print("Done")