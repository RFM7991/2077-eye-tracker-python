import cv2
import numpy as np
import os
import math 
import graph

# Load the precomputed Haar cascade classifiers
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

curr_face = [0,0,0,0]
curr_eyes = [0,0,0,0]

# Set up the SimpleBlobDetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
# params.filterByArea = True
# params.minArea = 100  # min size for pupil area
# params.maxArea = 1000  # max size for pupil area
# params.filterByCircularity = True
# params.minCircularity = 0.2
# params.filterByConvexity = True
# params.minConvexity = 0.87
# params.filterByInertia = True
# params.minInertiaRatio = 0.2
detector = cv2.SimpleBlobDetector_create(params)

# frame capture
caps = []


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

def get_keypoints(img, threshold=60, prev_area=0, capture=False):
    if img is None or img.size == 0:
        return

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (15,15), 0)
    equalized_img = cv2.equalizeHist(blurred_img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clahe_img = clahe.apply(equalized_img)

    cv2.imshow("Equalized", resize_img(gray_img))

    # Convert image to binary
    # _, thresholded_image = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    thresholded_image = cv2.adaptiveThreshold(equalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

    # morphing
    # kernel = np.ones((5,5),np.uint8)
    # thresholded_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)
    
    # blob_detect(thresholded_image)
    contour_detect(thresholded_image)

    if capture:         
        if keypoints:
            diameter = keypoints[0].size
            radius = diameter / 2.0
            area = 3.1415 * (radius ** 2)
            caps.append([area, radius, diameter])
            # print(f"Pupil diameter: {diameter} pixels")
            print(f"Pupil area: {area} pixels")
    else:
        if len(caps) > 0:
            # print average area of captured pupils
            print(f"Average area: {sum([cap[0] for cap in caps]) / len(caps)}")
            print(f"Average radius: {sum([cap[1] for cap in caps]) / len(caps)}")
            print(f"Average diameter: {sum([cap[2] for cap in caps]) / len(caps)}")
            graph.graph_pupils(caps, diameter=True, area=True)

            caps.clear()


def blob_detect(thresholded_image):
    keypoints = detector.detect(thresholded_image)

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    bin_im_with_keypoints = cv2.drawKeypoints(thresholded_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # for i, keypoint in enumerate(keypoints):
    #     # Extract the position of the keypoint
    #     x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    #     # Label the keypoint
    #     diameter = keypoint.size
    #     radius = diameter / 2.0
    #     area = math.pi * (radius ** 2)
        
    #     label = f"D: {diameter:.0f}"
    #     # label = f" A: {area:.0f}"
    #     # draw line for diameter of pupil
    #     # cv2.line(bin_im_with_keypoints, (x, y), (x + int(radius), y), (0, 255, 0), 2)

    cv2.imshow("Binary", resize_img(bin_im_with_keypoints))
 
def contour_detect(thresholded_image):
    # Find contours
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Number of contours: {len(contours)}")
    # Filter contours
    # pupil_contours = []
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     # Assuming that the pupil will have a significant area,
    #     # you can adjust the threshold according to your image
    #     if area > 50:
    #         perimeter = cv2.arcLength(contour, True)
    #         # Check if contour is circular
    #         if perimeter == 0: continue
    #         circularity = 4 * np.pi * (area / (perimeter * perimeter))
    #         # This threshold can be adjusted based on how circular you expect the pupil to be
    #         if circularity > 0.7:
    #             pupil_contours.append(contour)

    # # Assuming the largest circular contour is the pupil
    # pupil_contours.sort(key=cv2.contourArea, reverse=True)
    # if len(pupil_contours) == 0:
    #     return
    # pupil = pupil_contours[0]

    # Draw the contour on the original image
    # cv2.drawContours(thresholded_image, [pupil], -1, (255, 0, 0), 2)

    cv2.imshow("Binary", resize_img(thresholded_image))
