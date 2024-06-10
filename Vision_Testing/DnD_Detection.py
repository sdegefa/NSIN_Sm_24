import cv2
import tensorflow as tf
from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img
from ultralytics import YOLO
import numpy as np
from torchvision import transforms
import torch

# DnD stands for Direction and Distance

# Assume focal length of 2 mm
focal_length = 2
# Assume baseline of 30 mm
baseline = 30
# No compass module for now so assume always facing 0 degrees North
facing_angle = 0
# Assuming a camera FOV of 84 degrees; found online (might be 90 degrees)
fov = 74.9
# load disparity model
model_path = "models/eth3d.pb"
model_type = ModelType.eth3d
hitnet_depth = HitNet(model_path, model_type)
# load YOLO model
model = YOLO('thermal_yolov8n_6_4_24.pt')


def main():
    cam_loop(0)

def detect(frame):
    results = model.predict(frame, device=0, classes=[0])
    return results

def cam_loop(camID):
    cap = cv2.VideoCapture(camID)

    # Loop for fetching frames
    while True:
        # Get Frame
        ret, frame = cap.read()

        # Detect objects
        results = detect(frame)

        # Get object disparity:
        disparity_map, disparity_img = disparity_calculation(results)

        # Convert disparity to depth
        depths = depth_from_disparity(disparity_map, results)

        # Get object directions:
        object_angles = angle_calculation(facing_angle, fov, results)

        # Get compass directions:
        compass_directions = compass_direction_32_point(object_angles)

        # Get top left bounding box coordinates
        top_lefts = top_left_points(results)

        # Add compass directions and object angles to the plot
        img_ann = dir_ann(results[0].plot(), compass_directions, object_angles, top_lefts, depths)

        # Concat annotated image with disparity image
        img_ann = np.concatenate((img_ann, disparity_img), axis=1)

        cv2.imshow('Cam' + str(camID), img_ann)

        if (cv2.waitKey(30) == 27):  # ESC key to break loop
            break

def angle_calculation(facing_angle, fov, results):
    angles = []

    for i in range(len(results[0].boxes.xyxy)):
        # Calculate the center of the bounding box
        center_x = (results[0].boxes.xyxy[i][0] + results[0].boxes.xyxy[i][2]) / 2

        # Calculate the angle of the object from the center of the camera
        angle = ((center_x) * fov / 640) - fov / 2
        if angle < 0:
            angle = 360 + angle

        # Calculate the direction of the object from the facing angle of the camera
        angle = facing_angle + angle
        if angle > 360:
            angle = angle - 360
        
        angle = round(angle.item(), 2)
        angles.append(angle)

        # Print the direction of the object
        # print(f"Object {i} has a center at {center_x}")
        # print(f"Object {i} is at {angle} degrees")

    return angles

def compass_direction_32_point(object_angles):
    directions = []

    # 32 point compass
    compass = {0: 'N', 11.25: 'NbE', 22.5: 'NNE', 33.75: 'NEbN', 45: 'NE', 56.25: 'NEbe', 67.5: 'ENE', 78.75: 'EbN', 90: 'E',
               101.25: 'EbS', 112.5: 'ESE', 123.75: 'SEbE', 135: 'SE', 146.25: 'SEbS', 157.5: 'SSE', 168.75: 'SbE', 180: 'S',
               191.25: 'SbW', 202.5: 'SSW', 213.75: 'SWbS', 225: 'SW', 236.25: 'SWbW', 247.5: 'WSW', 258.75: 'WbS', 270: 'W',
               281.25: 'WbN', 292.5: 'WNW', 303.75: 'NWbW', 315: 'NW', 326.25: 'NWbN', 337.5: 'NNW', 348.75: 'NbW', 360: 'N'}
    
    # Find the closest compass direction
    for angle in object_angles:
        # round angle to nearest 11.25 degrees
        angle = round(angle / 11.25) * 11.25
        directions.append(compass[angle])

    return directions


def top_left_points(results):
    top_lefts = []

    for i in range(len(results[0].boxes.xyxy)):
        top_left = results[0].boxes.xyxy[i][0].item(), results[0].boxes.xyxy[i][1].item()
        top_lefts.append(top_left)

    return top_lefts

def dir_ann(img, directions, angles, top_lefts, depths):
    for i in range(len(directions)):
        
        org = tuple(map(round, top_lefts[i]))
        img = cv2.rectangle(img, (org[0], org[1]-35), (org[0]+250, org[1]-20), (0, 0, 0), -1)
        org = org[0], org[1]-22
        org2 = org[0]+15*len(directions[i]), org[1]
        org3 = org2[0] + 70, org[1]
        img = cv2.putText(img, directions[i], org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, str(angles[i]) + ",", org2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1 , cv2.LINE_AA)
        img = cv2.putText(img, "Depth: " + str(depths[i]) + " mm", org3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1 , cv2.LINE_AA)

    return img

def disparity_calculation(results):
    # For this script, using the same camera input with the image offset to emulate a stereo camera
    im1 = results[0].orig_img
    im2 = results[0].orig_img
    im2 = torch.tensor(im2, dtype=torch.float32).permute(2, 0, 1)
    im2 = transforms.functional.affine(im2, angle=0, translate=(30, 0), scale=1, shear=0)
    im2 = im2.permute(1, 2, 0).numpy().astype(np.uint8)

    # show images
    # cv2.imshow('Stereo Images', im2)
    # cv2.waitKey(0)

    # Calculate disparity
    disparity = hitnet_depth(im1, im2)

    # Draw disparity
    disparity_img = draw_disparity(disparity)
    # cv2.imshow('Disparity', disparity_img)

    return disparity, disparity_img

def depth_from_disparity(disparity_map, results):
    depths = []
    
    for i in range(len(results[0].boxes.xyxy)):
        # Calculate the center of the bounding box
        center_x = (results[0].boxes.xyxy[i][0] + results[0].boxes.xyxy[i][2]) / 2
        center_y = (results[0].boxes.xyxy[i][1] + results[0].boxes.xyxy[i][3]) / 2
        coord = (round(center_x.item()), round(center_y.item()))
        disparity = disparity_map[coord[1]][coord[0]]

        # Calculate depth (in mm)
        depth = focal_length * baseline / disparity
        depth = round(depth.item(), 2)
        print(f"Object {i} has a depth of {depth} mm")
        depths.append(depth)

    return depths

if __name__ == '__main__':
    main()
