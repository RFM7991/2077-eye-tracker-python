import cv2
import numpy as np
import os
import math 

# Load the precomputed Haar cascade classifiers
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

curr_face = [0,0,0,0]
curr_eyes = [0,0,0,0]

# Set up the SimpleBlobDetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
# params.minThreshold = 10
# params.maxThreshold = 200
params.filterByArea = True
params.minArea = 100  # min size for pupil area
params.maxArea = 1000  # max size for pupil area
# params.filterByCircularity = True
# params.minCircularity = 0.2
# params.filterByConvexity = True
# params.minConvexity = 0.87
# params.filterByInertia = True
# params.minInertiaRatio = 0.2
detector = cv2.SimpleBlobDetector_create(params)

def detect_face(frame):
    # Detect faces in the frame
    faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    return faces

def detect_eyes(face):
    return eye_cascade.detectMultiScale(face, scaleFactor=1.3, minNeighbors=5)

def detect_features(frame, roi):
    global curr_eyes
    eye_delta_threshold = 200

    eyes = detect_eyes(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    if len(eyes) == 0:
        return

    for eye in eyes:
        x, y, w, h = eye
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)    

    # sort eyes by largest first
    eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = eyes[0]

    if w > 0 and h > 0:
        if abs(curr_eyes[0] - x) > eye_delta_threshold or abs(curr_eyes[1] - y) > eye_delta_threshold or abs(curr_eyes[2] - w) > eye_delta_threshold or abs(curr_eyes[3] - h) > eye_delta_threshold:
            curr_eyes = [x, y, w, h]
        
        cx, cy, cw, ch = curr_eyes 
        # debug rectangle around eyes
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)
        return frame[cy:cy+ch, cx:cx+cw]

def resize_img(img, scale_factor = 1):
    if img is None:
        return
    width = int(480)
    height = int(480)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)   

def get_keypoints(img, threshold=60, prev_area=0):
    if img is None or img.size == 0:
        return

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert image to binary
    _, thresholded_image = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
  
    keypoints = detector.detect(thresholded_image)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    bin_im_with_keypoints = cv2.drawKeypoints(thresholded_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for i, keypoint in enumerate(keypoints):
        # Extract the position of the keypoint
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        # Label the keypoint
        diameter = keypoint.size
        radius = diameter / 2.0
        area = math.pi * (radius ** 2)
        
        label = f"A: {area:.0f}"
        # draw line for diameter of pupil
        cv2.line(im_with_keypoints, (x, y), (x + int(radius), y), (0, 255, 0), 2)
        
        cv2.putText(bin_im_with_keypoints, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show keypoints
    cv2.imshow("Binary", resize_img(bin_im_with_keypoints))
    # cv2.imshow("Keypoints", resize_img(im_with_keypoints))

    # Assuming the pupil is the largest blob detected
    # Calculate dimensions from the size attribute
    if keypoints:
        diameter = keypoints[0].size
        radius = diameter / 2.0
        area = 3.1415 * (radius ** 2)

        # Output the dimensions of the pupil
        print(f"Pupil diameter: {diameter} pixels")
        print(f"Pupil radius: {radius} pixels")
        print(f"Pupil area: {area} square pixels")
