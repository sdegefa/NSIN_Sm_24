import cv2
from ultralytics import YOLO

def main():
    # Setup video capture
    cap = cv2.VideoCapture(0)
 
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Loop for fetching frames
    while True:
        # Get Frame
        ret, frame = cap.read()

        # Detect objects
        results = detect(model, frame)

        cv2.imshow('yolov8', results[0].plot())

        if (cv2.waitKey(30) == 27): # ESC key to break loop
            break

def detect(model, frame):
    results = model(frame, device=0)
    return results

if __name__ == '__main__':
    main()