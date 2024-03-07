import cv2
import numpy as np
import os
import math 
import graph
import matplotlib.pyplot as plt
from pupil_detectors import Detector2D

# Load the precomputed Haar cascade classifiers
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

curr_face = [0,0,0,0]
curr_eyes = [0,0,0,0]


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
    return cv2.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor), interpolation=cv2.INTER_AREA)   

def bin_threshold_detect(img, threshold=60):
    if img is None or img.size == 0:
        return

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img[gray_img > 140] = 60
    blurred_img = cv2.GaussianBlur(gray_img, (5,5), 0)
    equalized_img = cv2.equalizeHist(blurred_img)

    # Apply adaptive thresholding
    thresholded_image = cv2.adaptiveThreshold(equalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow("Threshold", resize_img(thresholded_image))


def threshold_detect(img, threshold=60, prev_area=0, capture=False):
    if img is None or img.size == 0:
        return

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img[gray_img > 120] = 60
    cv2.imshow("Threshold", resize_img(gray_img))
    blurred_img = cv2.GaussianBlur(gray_img, (11,11), 0)
    equalized_img = cv2.equalizeHist(blurred_img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clahe_img = clahe.apply(blurred_img)

    # cv2.imshow("Equalized", resize_img(gray_img))

    # Convert image to binary
    # _, thresholded_image = cv2.threshold(blurred_img, 50, 200, cv2.THRESH_BINARY)

   # Start with a mid-range block size and adjust as necessary
    block_size = 53  # Consider trying other odd numbers: 9, 15, 17, etc.

    # Start with a small C value and adjust based on the segmentation results
    constant_C = 2  # If segmentation is too aggressive, try values like 0, -1, -2, etc.

    # Apply adaptive thresholding
    thresholded_image = cv2.adaptiveThreshold(
        blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, constant_C
)
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


def blob_detect_old(thresholded_image):
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

    cv2.imshow("Binary", bin_im_with_keypoints)
 
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

def color_detect(image):
    # Convert to an appropriate color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for dark colors that could represent the pupil
    dark_color_lower = np.array([0, 0, 0])
    dark_color_upper = np.array([180, 255, 100])

    # Threshold the HSV image to get only dark colors
    mask = cv2.inRange(hsv_image, dark_color_lower, dark_color_upper)

    # Perform morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours or blobs in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Color Detection", result)

    # Filter contours and find the pupil
    # for contour in contours:
    #     # Your filtering conditions
    #     pass

    # Assume you have (x, y) as the center of the pupil and radius for its size
    # Draw the contour or a circle around the pupil
    # cv2.circle(image, (x, y), radius, (255, 0, 0), 2)

    # Display the result
    # cv2.imshow('Pupil Detected', image)



def circle_detect(label="", image=None):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = unsharp_mask(gray_image, sigma=2.0, strength=10)

    # Gaussian blur parameters
    gaussian_blur_kernel_size = (5, 5)
    gaussian_blur_sigma_x = 1

    # Canny edge detection parameters
    canny_threshold1 = 50
    canny_threshold2 = 200  # Try values like 100, 150, 200, etc.

    # Hough Circle Transform parameters
    hough_dp = 1  # Increase if too many false circles are detected
    hough_minDist = 20  # Decrease if too many circles are close together
    hough_param1 = 50  # Increase if too many edges are detected
    hough_param2 = 1  # Increase to reduce false positives
    hough_minRadius = 20  # Adjust based on the minimum expected size of the pupil
    hough_maxRadius = 1000  # Adjust based on the maximum expected size of the pupil

    
    # Apply Gaussian blur to reduce noise
    blurred_gray = cv2.GaussianBlur(gray_image, gaussian_blur_kernel_size, gaussian_blur_sigma_x)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred_gray, canny_threshold1, canny_threshold2)

    print(f"Number of edges: {len(edges)}")

    # Perform Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=hough_dp, minDist=hough_minDist,
                            param1=hough_param1, param2=hough_param2,
                            minRadius=hough_minRadius, maxRadius=hough_maxRadius)

    # Draw the detected circles
    if circles is not None:
        print(f"Number of circles: {len(circles)}")

        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)

    # Display the result
    cv2.imshow(label+'_Detected Pupil', image)

def rgb_diff_detect(image):

    # Convert to RGB since OpenCV loads in BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the channels
    R, G, B = cv2.split(image_rgb)

    # Calculate the absolute difference between channels
    diff_rg = cv2.absdiff(R, G)
    diff_rb = cv2.absdiff(R, B)
    diff_gb = cv2.absdiff(G, B)

    # Threshold the differences: this threshold can be adjusted
    threshold = 20
    _, mask_rg = cv2.threshold(diff_rg, threshold, 255, cv2.THRESH_BINARY_INV)
    _, mask_rb = cv2.threshold(diff_rb, threshold, 255, cv2.THRESH_BINARY_INV)
    _, mask_gb = cv2.threshold(diff_gb, threshold, 255, cv2.THRESH_BINARY_INV)

    # Combine the masks to isolate colors that are uniformly spread
    uniform_colors = cv2.bitwise_and(mask_rg, mask_rb)
    uniform_colors = cv2.bitwise_and(uniform_colors, mask_gb)

    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(image_rgb, image_rgb, mask=uniform_colors)

    # Convert back to BGR for OpenCV
    filtered_image_bgr = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR)

    # Display the result
    cv2.imshow('Filtered Image', filtered_image_bgr)

def rgb_sum_detect(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate the sum of the RGB channels
    sum_rgb = cv2.add(cv2.add(image_rgb[:, :, 0], image_rgb[:, :, 1]), image_rgb[:, :, 2])
    
    # Define a threshold for the sum. Pixels with a sum below this value are likely to be darker.
    sum_threshold = 160  # Adjust this threshold to your needs
    
    # Threshold the sum of RGB channels
    _, mask = cv2.threshold(sum_rgb, sum_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Apply the mask to the original image to isolate dark regions
    filtered_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    # Convert back to BGR for OpenCV
    filtered_image_bgr = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR)
    
    # Display the result
    cv2.imshow('Filtered Image', filtered_image_bgr)





def pupil_detect(img):
    detector = Detector2D()

    # read image as numpy array from somewhere, e.g. here from a file
    img = cv2.imread("conor1.jpeg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = detector.detect(gray)
    ellipse = result["ellipse"]

    # draw the ellipse outline onto the input image
    # note that cv2.ellipse() cannot deal with float values
    # also it expects the axes to be semi-axes (half the size)
    cv2.ellipse(
        gray,
        tuple(int(v) for v in ellipse["center"]),
        tuple(int(v / 2) for v in ellipse["axes"]),
        ellipse["angle"],
        0, 360, # start/end angle for drawing
        (0, 0, 255) # color (BGR): red
    )
    cv2.imshow("Image", gray)
    cv2.waitKey(0)

def preprocessed_detect(image):
    # Read the image
    # image = cv2.imread('arnav.jpeg')

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # Apply median blur
    median_blurred = cv2.medianBlur(equalized_image, 5)

    # Apply Gaussian blur
    # gaussian_blurred = cv2.GaussianBlur(median_blurred, (9, 9), 0)

    # Morphological closing to close small holes and dark spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_closed = cv2.morphologyEx(median_blurred, cv2.MORPH_CLOSE, kernel)

    # Remove large reflections if necessary
    # This might involve more complex image processing techniques and is not covered by a simple function call.

    # Apply adaptive thresholding
    max_output_value = 255
    neighborhood_size = 3
    subtract_from_mean = 0

    # thresholded_image = cv2.adaptiveThreshold(
    #     morph_closed,
    #     max_output_value,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV,  # Invert so that pupil becomes white and the rest becomes black
    #     neighborhood_size,
    #     subtract_from_mean
    # )

    _, thresholded_image = cv2.threshold(morph_closed, 0, 60, cv2.THRESH_BINARY)


    # Display the image
    cv2.imshow('Preprocessed Image', gray_image)



def get_circle_values_old(image, center, radius):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    return image[mask == 255]

def find_pupil_radius(image, center, initial_radius, threshold):
    # convert to grayscale 
    gray_image = image #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_image = gray_image.copy()
    mean_val_prev = np.mean(get_circle_get_circle_values_oldvalues(gray_image, center, initial_radius))
    for r in range(initial_radius, gray_image.shape[0]//2, 10):
        # Create a copy of the image for debug drawing
        # Draw the debug circle on the copy
        cv2.circle(debug_image, center, r, (0, 0, 255), 1) 
        # Display the debug image
        # cv2.waitKey(0)

        mean_val_current = np.mean(get_circle_values_old(gray_image, center, r))
        print(mean_val_current - mean_val_prev)
        if abs(mean_val_current - mean_val_prev) > threshold:
            return r
        # mean_val_prev = mean_val_current
        
    return None
 
def print_hue(image):
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the channels to the range [0, 1] for Matplotlib
    normalized_hue = hsv_image[:, :, 0] / 179.0  # Hue values range from 0 to 179
    normalized_saturation = hsv_image[:, :, 1] / 255.0  # Saturation values range from 0 to 255
    normalized_value = hsv_image[:, :, 2] / 255.0  # Value values range from 0 to 255

    # Create the color maps
    hue_colormap = plt.cm.hsv(normalized_hue)
    saturation_colormap = plt.cm.jet(normalized_saturation)  # Jet is a common colormap for intensity
    value_colormap = plt.cm.jet(normalized_value)

    # Convert from RGBA to BGR format for OpenCV
    hue_colormap_bgr = cv2.cvtColor((hue_colormap[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    saturation_colormap_bgr = cv2.cvtColor((saturation_colormap[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    value_colormap_bgr = cv2.cvtColor((value_colormap[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save the color maps as images
    cv2.imwrite('output/hue_colormap.png', hue_colormap_bgr)
    cv2.imwrite('output/saturation_colormap.png', saturation_colormap_bgr)
    cv2.imwrite('output/value_colormap.png', value_colormap_bgr)
    cv2.imwrite('output/gray_image.png', gray_image)

    # Provide the paths for download
    hue_colormap_path = 'hue_colormap.png'
    saturation_colormap_path = 'saturation_colormap.png'
    value_colormap_path = 'value_colormap.png'

    hue_colormap_path, saturation_colormap_path, value_colormap_path




def get_circle_values(image, center, radius, num_sections):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    
    # Convert the center and radius to integer values
    center = tuple(map(int, center))
    radius = int(radius)
    
    # Create an array to store the mean values of each section
    means = np.zeros(num_sections)
    
    # Angle step for each section
    angle_step = 360 / num_sections
    
    for i in range(num_sections):
        # Create a mask for the current section
        section_mask = np.zeros(image.shape[:2], dtype="uint8")
        
        # Define the start and end angles for the current section
        start_angle = int(angle_step * i)
        end_angle = int(angle_step * (i + 1))
        
        # Draw the section on the mask
        cv2.ellipse(section_mask, center, (radius, radius), 0, start_angle, end_angle, 255, -1)
        
        # Combine the section mask with the original circle mask
        combined_mask = cv2.bitwise_and(mask, section_mask)
        
        # Extract the pixel values from the image using the combined mask
        section_pixels = image[combined_mask == 255]
        
        # Calculate the mean value of the pixels in the section
        if section_pixels.size > 0:
            means[i] = np.mean(section_pixels)
        else:
            means[i] = 0  # If there are no pixels in the section, set the mean to 0
    
    return means

def find_pupil_radius_out(image, center, threshold = 2, precision=10, step = 5, delta_acceptance=0.7, max_brightness=140):
    # convert to grayscale 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image[gray_image > max_brightness] = 60
    # filter
    gray_image = unsharp_mask(gray_image, sigma=2.0, strength=10)
    # gray_image = cv2.GaussianBlur(gray_image, (0,0), 5)

    initial_radius = image.shape[0]//2 
    debug_image = gray_image.copy()
    prev_segments = get_circle_values(gray_image, center, initial_radius, precision)
    mean_val_init = np.mean(prev_segments)
    print(f"mean_val_init: {mean_val_init}")
    diff_segment_goal = int(precision * delta_acceptance)

    attempt = 1
    for r in range(initial_radius-step, 0, -step):
        print(f"Attempt {attempt}:")
        # Draw the debug circle on the copy
        cv2.circle(debug_image, center, r, (0, 0, 255), 1) 
        cv2.imshow("Debug", debug_image)
        cv2.moveWindow("Debug", 0, 850)
        # cv2.waitKey(0)

        current_segments = get_circle_values(gray_image, center, r, precision)
        # print(f"prev_segments: {prev_segments}")
        # print(f"Current segments: {current_segments}")
        # print(f"DIFF: {prev_segments - current_segments}")
        # print(f"RATIO: {prev_segments / current_segments}")
        
        

        diff_sections = 0
        for i in range(len(prev_segments)):
            if (prev_segments[i] - current_segments[i]) > threshold: 
            # if (prev_segments[i] / current_segments[i]) > 1.05: 
                diff_sections += 1

        print(f"Acceptance: {diff_sections / precision}")

        # print(f"diff_sections: {diff_sections}")
        if (diff_sections >= diff_segment_goal):
            print(f"Found pupil radius: {r}")
            return r

        prev_segments = current_segments
        attempt += 1
        
    return None


def unsharp_mask(image, sigma=1.0, strength=1.5):
    # Blurring the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Calculating the unsharp mask
    unsharp_mask = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    return unsharp_mask

def binary_threshold_detect(label, image, max_brightness=180):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center_val = np.mean(gray_image[gray_image.shape[0]//2-3:gray_image.shape[0]//2+3, gray_image.shape[1]//2-5:gray_image.shape[1]//2+5])
    # get darkest value in image
    max_brightness = np.max(gray_image)
    min_brightness = np.min(gray_image)

    for i in range(10):
        bright_pixels_mask = gray_image >= max_brightness - 10
        gray_image[bright_pixels_mask] = min_brightness+10
        max_brightness = np.max(gray_image)

    # gray_image = apply_reverse_vignette(gray_image)

    # morph close image
    kernel = np.ones((5,5),np.uint8)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

   # blur image
    # gray_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    cv2.imshow(label+"_Gray", gray_image)
    cv2.moveWindow(label+"_Gray", 0, 600)

    # thresholded_im = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 0)
    # get mean brightness of center 10X10 pixels 
    center_val = np.mean(gray_image[gray_image.shape[0]//2-10:gray_image.shape[0]//2+10, gray_image.shape[1]//2-5:gray_image.shape[1]//2+5])
    print(f"Center brightness: {center_val}")
    print(f"Min brightness: {min_brightness}")
    thresholded_im = cv2.threshold(gray_image, min_brightness+10, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5,5),np.uint8)
    # thresholded_im = cv2.morphologyEx(thresholded_im, cv2.MORPH_CLOSE, kernel)

    cv2.imshow(label+"_Thresholded", thresholded_im)
    cv2.moveWindow(label+"_Thresholded", 0, 800)

    # detect blobs
    # blob_detect(label, image, thresholded_im)
    contour_detection(label, image, thresholded_im)


def blob_detect(label, image, processed_img):

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.minArea = 100  # min size for pupil area
    params.maxArea = 1000  # max size for pupil area
    params.filterByCircularity = False
    params.minCircularity = 0.2
    params.filterByConvexity = False
    params.minConvexity = 0.1
    params.filterByInertia = False
    params.minInertiaRatio = 0.2
    params.blobColor = 0

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(processed_img)

    # circle keypoints 
    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        s = int(keypoint.size)
        cv2.circle(processed_img, (x, y), s, (0, 0, 255), 1)

    print(f"Label: {label}")
    print(f"Number of keypoints: {len(keypoints)}")

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    if len(keypoints) > 0:
        blank = np.zeros((1, 1))
        cv2.drawKeypoints(processed_img, keypoints, blank, (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow(label+"_Keypoints", processed_img)


def apply_reverse_vignette(image, darken_factor=1.2):
    # Create a coordinate grid
    rows, cols = image.shape[:2]
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    x, y = np.meshgrid(x, y)

    # Compute the radial distance from the center normalized to [0, 1]
    r = np.sqrt(x**2 + y**2)
    r = r / np.max(r)

    # Create the reverse vignette mask
    reverse_vignette_mask = r**darken_factor  # Darken towards the center

    # Apply the reverse vignette mask to each channel of the image
    if len(image.shape) == 3:  # For colored images
        reverse_vignette_mask = np.stack([reverse_vignette_mask] * 3, axis=-1)
    
    return np.uint8(image * reverse_vignette_mask)


def contour_detection(label, image, processed_img):
    # Assuming processed_img is a binary image with a black blob on a white background
    # You may need to threshold or invert the image accordingly
    ret, thresh = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return

    # Find the largest contour which will presumably be the pupil
    largest_contour = max(contours, key=cv2.contourArea)

    # Assuming we're looking for a circular blob, calculate its center and radius
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    print(f" Radius: {radius}")

    # Draw the circle on the original image
    # output_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    cv2.circle(image, center, radius, (0, 255, 0), 2)

    cv2.imshow(label+"_Detected Pupil", image)