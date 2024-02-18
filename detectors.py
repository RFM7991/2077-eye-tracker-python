import cv2
import numpy as np
import os

# Load the precomputed Haar cascade classifiers
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

curr_face = [0,0,0,0]
curr_eyes = [0,0,0,0]

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

def detect_face(frame):
    # Detect faces in the frame
    faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    return faces

def detect_eyes(face):
    eyes = []

    # Detect eyes in the frame
    all_eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.3, minNeighbors=5)

    fh = face.shape[0]
    fw = face.shape[1]

    # debug draw line for face cut off
    # cv2.line(face, (0, int(fh/4)), (fw, int(fh/4)), (0, 255, 0), 2)
     # debug draw line for face cut off
    # cv2.line(face, (0, int(fh/2)), (fw, int(fh/2)), (0, 255, 0), 2)

    for eye in all_eyes:
        x, y, w, h = eye
        # Ensure the eye is in 2nd 4th of the face
        if y > fh/4 and y < fh/2:
            eyes.append(eye)    

    return eyes

def detect_features(frame):
    global curr_face
    global curr_eyes
    face_delta_threshold = 50
    eye_delta_threshold = 50
    faces = detect_face(frame)

    if len(faces) < 1:
        return

    # Draw rectangles around the faces
    x, y, w, h = faces[0]
    if abs(curr_face[0] - x) > face_delta_threshold or abs(curr_face[1] - y) > face_delta_threshold or abs(curr_face[2] - w) > face_delta_threshold or abs(curr_face[3] - h) > face_delta_threshold:
        curr_face = [x, y, w, h]

    cx, cy, cw, ch = curr_face
    # print(curr_face)

    # draw rectangle around face
    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)

    # Extract the region of interest (ROI) containing the face
    face_roi = frame[cy:cy + ch, cx:cx + cw]
    fw = face_roi.shape[1]
    eyes = detect_eyes(face_roi)

    # Draw rectangles around the eyes
    left_eye = [0,0,0,0]
    right_eye = [0,0,0,0]
    for eye in eyes:
        ex, ey, ew, eh = eye

        if ex < fw/2:
            right_eye = eye
        else:
            left_eye = eye
    

    lx, ly, lw, lh = left_eye
    rx, ry, rw, rh = right_eye
    scale_factor = 4
    if lw > 0 and lh > 0:
        # print(left_eye - curr_eyes)
        if abs(curr_eyes[0] - lx) > eye_delta_threshold or abs(curr_eyes[1] - ly) > eye_delta_threshold or abs(curr_eyes[2] - lw) > eye_delta_threshold or abs(curr_eyes[3] - lh) > eye_delta_threshold:
            curr_eyes = [lx, ly, lw, lh]
        
        cx, cy, cw, ch = curr_eyes   
        
        # debug draw rectangle around left eye
        cv2.rectangle(face_roi, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

        return face_roi[cy:cy+ch, cx:cx+cw]

def resize_img(img, scale_factor = 1):
    if img is None:
        return
    width = int(480)
    height = int(480)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)   

def detect_pupils(frame):
    if frame is None:
        return
    keypoints = get_keypoints(frame)
    if keypoints:
        x, y = keypoints[0].pt
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        return frame, (x, y)


def get_keypoints(img, threshold=20, prev_area=0):
    global blob_detector
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)

    keypoints = blob_detector.detect(img)
    if keypoints and len(keypoints) > 1:
        tmp = 1000
        for keypoint in keypoints:  # filter out odd blobs
            if abs(keypoint.size - prev_area) < tmp:
                ans = keypoint
                tmp = abs(keypoint.size - prev_area)

        keypoints = (ans,)
    return keypoints