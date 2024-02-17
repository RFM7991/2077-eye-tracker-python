import cv2
import numpy as np
import os

# Load the precomputed Haar cascade classifiers
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

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
    cv2.line(face, (0, int(fh/4)), (fw, int(fh/4)), (0, 255, 0), 2)
     # debug draw line for face cut off
    cv2.line(face, (0, int(fh/2)), (fw, int(fh/2)), (0, 255, 0), 2)

    for eye in all_eyes:
        x, y, w, h = eye
        # Ensure the eye is in 2nd 4th of the face
        if y > fh/4 and y < fh/2:
            eyes.append(eye)    

    return eyes

def detect_features(frame):
    faces = detect_face(frame)

    if len(faces) < 1:
        return

    # Draw rectangles around the faces
    x, y, w, h = faces[0]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Extract the region of interest (ROI) containing the face
    face_roi = frame[y:y + h, x:x + w]
    fw = face_roi.shape[1]
    eyes = detect_eyes(face_roi)

    # Draw rectangles around the eyes
    left_eye = [0,0,0,0]
    right_eye = [0,0,0,0]
    for eye in eyes:
        ex, ey, ew, eh = eye
        cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        if ex < fw/2:
            right_eye = eye
        else:
            left_eye = eye
    
 
    lx, ly, lw, lh = left_eye
    rx, ry, rw, rh = right_eye
    scale_factor = 4
    if lw > 0 and lh > 0:
        cv2.imshow("Left Eye", resize_img(face_roi[ly:ly + lh, lx:lx + lw], scale_factor))

    # if rw > 0 and rh > 0:
    #     cv2.imshow("Right Eye", resize_img(face_roi[ry:ry + rh, rx:rx + rw], scale_factor))

def resize_img(img, scale_factor):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)   