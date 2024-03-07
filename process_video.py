import cv2
import numpy as np
import detectors
import time


# load IMG_1187.MOV
cap = cv2.VideoCapture('videos/adit.MOV')
# cap = cv2.VideoCapture()
# Initial ROI dimensions
roi_x, roi_y, roi_width, roi_height = 0, 0, 1080, 10805

intensity = 20

# timer variables
capture = False
start_time = time.time()
elapsed_seconds = 0

# initiate reocording interface

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Continue the loop to restart the video
        continue

    # Adjust ROI dimensions based on arrow key input
    key = cv2.waitKey(30)
    if key == ord('q'):
        break
    elif key == 0:  # Up arrow key
        intensity += 1
    elif key == 32:  # Down arrow key
        capture = not capture


    current_time = time.time()
    if current_time - start_time >= 1:
        elapsed_seconds += 1
        # print(f"Elapsed seconds: {elapsed_seconds}")
        start_time = current_time


cv2.destroyAllWindows()


    