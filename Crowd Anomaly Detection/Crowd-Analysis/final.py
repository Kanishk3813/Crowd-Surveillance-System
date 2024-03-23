from ultralytics import YOLO
import cv2
import math
import numpy as np
import imutils
from config import YOLO_CONFIG, VIDEO_CONFIG, SHOW_PROCESSING_OUTPUT, DATA_RECORD_RATE, FRAME_SIZE, TRACK_MAX_AGE
from video_process import video_process
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

# Initialize YOLO models
fire_model = YOLO('C:/Users/kanis/neDrive/Desktop/crowd 4/fire detection/fire.pt')
crowd_fight_model = cv2.dnn.readNetFromDarknet(YOLO_CONFIG["CONFIG_PATH"], YOLO_CONFIG["WEIGHTS_PATH"])

# Other initialization steps

# Read video stream
cap = cv2.VideoCapture('drone.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fire detection
    fire_result = fire_model(frame, stream=True)
    for info in fire_result:
        # Process fire detection results

        # Crowd fight detection
        crowd_result = crowd_fight_model(frame)
        # Process crowd fight detection results

    # Display processed frame
    cv2.imshow('Frame', frame)
    
    # User interaction
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
