import cv2
from ultralytics import YOLO
from threading import Thread

# Apparently, imshow() is not thread safe. Just use live_d_2cams.py for 2 camera detection.

def main():
    # Create two threads for running cam function
    thread1 = Thread(target=cam, args=(0,))
    thread2 = Thread(target=cam, args=(2,))

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for the threads to finish
    thread1.join()
    thread2.join()

def detect(model, frame):
    results = model(frame, device=0)
    return results

def cam(camID):
    cap = cv2.VideoCapture(camID)
 
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Loop for fetching frames
    while True:
        # Get Frame
        ret, frame = cap.read()

        # Detect objects
        results = detect(model, frame)

        cv2.imshow('Cam' + str(camID), results[0].plot())

        if (cv2.waitKey(30) == 27): # ESC key to break loop
            break

if __name__ == '__main__':
    main()