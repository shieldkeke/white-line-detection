import cv2
import imageio

video_list = ['hough','canny','binary_line','binary','mobile']

for video_name in video_list:
    try:
        cap = cv2.VideoCapture(video_name+'.mp4')
        print(video_name+'.mp4')
        image_lst = []
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_lst.append(frame_rgb)
        cap.release()
        imageio.mimsave(video_name+'.gif', image_lst, fps=10)
    except Exception as e:
        print(e)
        continue
