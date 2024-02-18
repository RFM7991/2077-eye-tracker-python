import cv2
import numpy as np
import detectors


# cap = cv2.VideoCapture("eye_recording.flv")
cap = cv2.VideoCapture(0)
# Initial ROI dimensions
roi_x, roi_y, roi_width, roi_height = 0, 0, 1080, 1080

intensity = 20

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
    elif key == 1:  # Down arrow key
        intensity -= 1

    print(key, intensity)


    # Ensure ROI dimensions are within frame boundaries
    roi_height = max(1, min(roi_height, frame.shape[0] - roi_y))
    roi_width = max(1, min(roi_width, frame.shape[1] - roi_x))

    # Extract ROI
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)

    # Detect eyes in the ROI
    eye_roi = detectors.detect_features(gray_roi)
    # _, threshold = cv2.threshold(eye_roi, intensity, 255, cv2.THRESH_BINARY_INV)
    # contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    detectors.detect_pupils(eye_roi)

    # for cnt in contours:
    #     (x, y, w, h) = cv2.boundingRect(cnt)

    #     cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
    #     cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     # cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
    #     # cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
    #     break

    # if threshold is not None:
    #     cv2.imshow("Threshold", threshold)
    #     cv2.imshow("Threshold Zoom", detectors.resize_img(threshold))

    cv2.imshow("gray roi", gray_roi)
    if eye_roi is not None:
        cv2.imshow("Eye Zoom", detectors.resize_img(eye_roi))
    
    # cv2.imshow("Roi", roi)


cv2.destroyAllWindows()


    