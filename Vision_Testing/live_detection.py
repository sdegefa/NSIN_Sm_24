import cv2
from ultralytics import YOLO
from threading import Thread

def main():
    cam(0)

def detect(model, frame):
    results = model(frame, device=0)
    return results


def cam(camID):
    cap = cv2.VideoCapture(camID)

    # Load YOLOv8 model
    model = YOLO('thermal_yolov8n_6_4_24.pt')

    # Loop for fetching frames
    while True:
        # Get Frame
        ret, frame = cap.read()

        # Detect objects
        results = detect(model, frame)

        cv2.imshow('Cam' + str(camID), results[0].plot())

        if (cv2.waitKey(30) == 27):  # ESC key to break loop
            break


if __name__ == '__main__':
    main()