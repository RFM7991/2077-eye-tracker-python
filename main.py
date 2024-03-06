import cv2
import numpy as np
import detectors
import time


# load IMG_1187.MOV
cap = cv2.VideoCapture('nate.MOV')
# cap = cv2.VideoCapture()
# Initial ROI dimensions
roi_x, roi_y, roi_width, roi_height = 0, 0, 1080, 10805

intensity = 20

# timer variables
capture = False
start_time = time.time()
elapsed_seconds = 0

# initiate reocording interface

# The image is already grayscale, so we can skip the color conversion step
test_img = cv2.imread('adit1.jpg')
detectors.print_hue(test_img)


# #fill in bright spots with black pixels 
# test_img[test_img > 140] = 8
# # mean blur
# test_img = cv2.blur(test_img, (9, 9))

# # Threshold for change in intensity to detect the pupil boundary
# intensity_threshold = 25  # This value may need to be adjusted for real images

# center_x, center_y =  test_img.shape[1]//2, test_img.shape[0]//2 
# cv2.line(test_img, (center_x, 0), (center_x, test_img.shape[0]), (0, 255, 0), 1)
# cv2.line(test_img, (0, center_y), (test_img.shape[1], center_y), (0, 255, 0), 1)


# # Calculate the pupil radius
# pupil_radius = detectors.find_pupil_radius(test_img, (center_x, center_y), initial_radius=5, threshold=intensity_threshold)

# # If a radius was found, draw the circle on the original image
# if pupil_radius:
#     cv2.circle(test_img, (center_x, center_y), pupil_radius, (255, 0, 0), 1)


# cv2.imshow("centered_img", detectors.resize_img(test_img))

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

    # Ensure ROI dimensions are within frame boundaries
    roi_height = max(1, min(roi_height, frame.shape[0] - roi_y))
    roi_width = max(1, min(roi_width, frame.shape[1] - roi_x))

    # Extract ROI
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    rows, cols, _ = roi.shape

    # Create reticle in the center of the ROI
    x = int(cols/2) 
    y = int(rows/2)
    width = 300
    # cv2.rectangle(roi, (x,y), (x+100, y+100) (0, 0, 255), 2)
    cv2.rectangle(roi, (x-width, y-width), (x+width, y+width), (0, 0, 255), 2)
    reticle_roi = roi[y-width:y+width, x-width:x+width]

    # Detect eyes in the ROI
    # eye_roi = detectors.detect_features(roi, reticle_roi)

    # Display the ROI
    # cv2.imshow("ROI", reticle_roi)

    # detectors.get_keypoints(roi, capture=capture)

    # detectors.blob_detect(roi)
    # detectors.preprocessed_detect(roi)

    # Detect pupils in the eye ROI
    # detectors.get_keypoints(roi, capture=capture)

    detectors.blob_detect(reticle_roi)


    current_time = time.time()
    if current_time - start_time >= 1:
        elapsed_seconds += 1
        # print(f"Elapsed seconds: {elapsed_seconds}")
        start_time = current_time


cv2.destroyAllWindows()


    