import cv2
from ultralytics import YOLO
import numpy as np

def main():
    # Need to have two cameras connected to camera at videoX and videoX. Just adjust cam inputs
    cam(0,2)

def detect(model, frame):
    results = model(frame, device=0)
    return results

def cam(camID1, camID2):
    cap1 = cv2.VideoCapture(camID1)
    cap2 = cv2.VideoCapture(camID2)
 
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Loop for fetching frames
    while True:
        # Get Frame
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Detect objects
        results1 = detect(model, frame1)
        results2 = detect(model, frame2)

        h_cat = np.concatenate((results1[0].plot(),results2[0].plot()), axis=1)

        cv2.imshow('Cam 1 & 2', h_cat)

        if (cv2.waitKey(30) == 27): # ESC key to break loop
            break

if __name__ == '__main__':
    main()