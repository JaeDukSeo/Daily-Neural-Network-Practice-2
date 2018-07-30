import cv2
import os
import natsort
image_folder = 'mall_frame'
video_name = 'mall_frame.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = natsort.natsorted(images)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter(video_name,fourcc, 8.0, (width,height))

for image in images:
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()