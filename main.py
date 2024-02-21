import cv2
import numpy as np
import detectors
import time


# cap = cv2.VideoCapture("eye_recording.flv")
cap = cv2.VideoCapture(0)
# Initial ROI dimensions
roi_x, roi_y, roi_width, roi_height = 0, 0, 1080, 1080

intensity = 20

# timer variables
capture = False
start_time = time.time()
elapsed_seconds = 0

# initiate reocording interface

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Adjust ROI dimensions based on arrow key input
    key = cv2.waitKey(30)
    if key == ord('q'):
        break
    elif key == 0:  # Up arrow key
        intensity += 1
    elif key == 32:  # Down arrow key
        capture = not capture



    # Ensure ROI dimensions are within frame boundaries
    roi_height = max(1, min(roi_height, frame.shape[0] - roi_y))
    roi_width = max(1, min(roi_width, frame.shape[1] - roi_x))

    # Extract ROI
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    rows, cols, _ = roi.shape

    # Create reticle in the center of the ROI
    x = int(cols/2) + 400
    y = int(rows/2)
    width = 100
    # cv2.rectangle(roi, (x,y), (x+100, y+100) (0, 0, 255), 2)
    cv2.rectangle(roi, (x-width, y-width), (x+width, y+width), (0, 0, 255), 2)
    reticle_roi = roi[y-width:y+width, x-width:x+width]

    # Detect eyes in the ROI
    # eye_roi = detectors.detect_features(roi, reticle_roi)

    # Detect pupils in the eye ROI
    detectors.get_keypoints(reticle_roi, capture=capture)

    # Display the ROI
    cv2.imshow("ROI", roi)

    
    
    current_time = time.time()
    if current_time - start_time >= 1:
        elapsed_seconds += 1
        # print(f"Elapsed seconds: {elapsed_seconds}")
        start_time = current_time


cv2.destroyAllWindows()


    