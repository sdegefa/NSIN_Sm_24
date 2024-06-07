from ultralytics import YOLO
import cv2

def main():
    img = cv2.imread('test_thermal.jpg')
    model = YOLO('thermal_yolov8n_6_4_24.pt')
    results = model(img, device=0)
    angles=angle_calculation(0, 84, results)
    directions=compass_direction_32_point(angles)

    # Get top left bounding box coordinates
    top_lefts = top_left_points(results)

    # Add compass directions and object angles to the plot
    img_ann = dir_ann(results[0].plot(), directions, angles,top_lefts)

    # print(results[0].boxes.xyxy.tolist()[0][1])
    cv2.imshow('Test', img_ann)
    cv2.waitKey(0)


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
        print(f"Object {i} has a center at {center_x}")
        print(f"Object {i} is at {angle} degrees")

    return angles

def compass_direction_32_point(object_angles):
    directions = []

    # 32 point compass
    compass = {0: 'N', 11.25: 'NbE', 22.5: 'NNE', 33.75: 'NEbN', 45: 'NE', 56.25: 'NEbe', 67.5: 'ENE', 78.75: 'EbN', 90: 'E',
               101.25: 'EbS', 112.5: 'ESE', 123.75: 'SEbE', 135: 'SE', 146.25: 'SEbS', 157.5: 'SSE', 168.75: 'SbE', 180: 'S',
               191.25: 'SbW', 202.5: 'SSW', 213.75: 'SWbS', 225: 'SW', 236.25: 'SWbW', 247.5: 'WSW', 258.75: 'WbS', 270: 'W',
               281.25: 'WbN', 292.5: 'WNW', 303.75: 'NWbW', 315: 'NW', 326.25: 'NWbN', 337.5: 'NNW', 348.75: 'NbW'}
    
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

def dir_ann(img, directions, angles, top_lefts):
    for i in range(len(directions)):
        
        org = tuple(map(round, top_lefts[i]))
        img = cv2.rectangle(img, (org[0], org[1]-35), (org[0]+120, org[1]-20), (0, 0, 0), -1)
        org = org[0], org[1]-22
        org2 = org[0]+15*len(directions[i]), org[1]
        img = cv2.putText(img, directions[i], org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, str(angles[i]), org2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1 , cv2.LINE_AA)

    return img

if __name__ == '__main__':
    main()