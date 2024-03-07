import cv2
import numpy as np
import detectors
import time
from vidstab import VidStab

# best so far 
# threshold=3
# precision=25
# step=7 
# delta_acceptance=0.4

threshold=2.5
precision=25
step=7 
delta_acceptance=0.4



def run_detect(label, image, x_offset=0, y_offset=0):
    # draw circle in center of image 
    rows, cols, _ = image.shape
    x = int(cols/2) + x_offset
    y = int(rows/2) + y_offset
    width = 60
    height = 60
    # cv2.rectangle(image, (x-width, y-height), (x+width, y+height), (0, 0, 255), 2)
    reticle_roi = image[y-height:y+height, x-width:x+width]

    # show image with reticle
    # cv2.imshow(label+"_Eye", image)
    y_offset = image.shape[0] + 50
    cv2.imshow("Reticle", detectors.resize_img(reticle_roi, 2))
    # cv2.moveWindow(label+"Reticle", 0, y_offset)


    # center = (reticle_roi.shape[1]//2, reticle_roi.shape[0]//2)
    # radius = detectors.find_pupil_radius_out(reticle_roi, center, threshold, precision, step, delta_acceptance)

    # # draw pupil radius on image     
    # cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
    # cv2.imshow(label+"_Eye", image)
    # cv2.moveWindow(label+"_Eye", 400, 0)

    detectors.binary_threshold_detect(label, reticle_roi)



# run_detect("1", cv2.imread('images/adit1.jpg'), x_offset=0, y_offset=-10)
# run_detect("2", cv2.imread('images/adit2.jpg'), x_offset=-20, y_offset=0)
# run_detect("3", cv2.imread('images/ar_cen.jpg'), x_offset=0, y_offset=-10)
# run_detect("4", cv2.imread('images/nate_cen.jpg'), x_offset=0, y_offset=0)
# run_detect("5", cv2. imread('images/con_cen.jpg'), x_offset=5, y_offset=0)
# run_detect("6", cv2.imread('images/blair.jpg'), x_offset=-35, y_offset=0)
# run_detect("7", cv2.imread('images/vic.jpg'), x_offset=0, y_offset=0)
# run_detect("8", cv2.imread('images/zoe.jpg'), x_offset=0, y_offset=0)
# cv2.waitKey(0)

# Load the video
stabilizer = VidStab()
# stabilizer.stabilize(input_path='videos/arnav.MOV', output_path='videos/arnav_stable.mov')
width = 900
height = 1200
cap = cv2.VideoCapture('videos/arnav_stable.mov')

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for MP4 format
out = cv2.VideoWriter('output/output.mov', fourcc, 20.0, (width, height))

# Loop the video
while True:
    ret, frame = cap.read() 
    
    # If the frame was not read successfully, restart the video
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    frame = cv2.resize(frame, (width, height))
    
    # Display the frame
    run_detect("1", frame, x_offset=100, y_offset=-10)
    out.write(frame)
    cv2.imshow("Video", frame)
    # cv2.waitKey(0)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()



cv2.waitKey(0)