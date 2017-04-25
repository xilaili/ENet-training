import cv2
import moviepy
from moviepy.editor import VideoFileClip

# camera input
'''
video_cap = cv2.VideoCapture(0)
fps = int(video_cap.get(cv2.CAP_PROP_FPS))
print fps

count = 0
while True:
    ret, frame = video_cap.read()
    if count == 0:
        frame = frame[0:424, :, :]
        frame = cv2.resize(frame, (1024, 512))
        print frame.shape
        cv2.imwrite('cfl.jpg', frame)
        count = 10
    else:
        count -= 1
'''
# video input
clip1 = VideoFileClip('../../../CFL/IMG_4817.MOV')
clip2 = VideoFileClip('../../../CFL/IMG_4818.MOV')
clip3 = VideoFileClip('../../../CFL/IMG_4819.MOV')
clip4 = VideoFileClip('../../../CFL/IMG_4820.MOV')

count = 0
for clip in [clip1, clip2, clip3, clip4]:
    for frame in clip.iter_frames():
        if count == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame[:960, :, :]
            frame = cv2.resize(frame, (1024, 512))
            print frame.shape
            cv2.imwrite('cfl.jpg', frame)
            count = 50
        else:
            count -= 1


