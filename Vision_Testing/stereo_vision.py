import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    # import test images
    img1 = cv2.imread('im0.png', 0)
    img2 = cv2.imread('im1.png', 0)

    # calculate disparity
    disparity_map = disparity(img1, img2)

    # Normalize the disparity map
    min = disparity_map.min()
    max = disparity_map.max()
    disparity_map = np.uint8(255 * (disparity_map - min) / (max - min))

    # Display the disparity map
    disparity_map = cv2.applyColorMap(disparity_map, cv2.COLORMAP_WINTER)
    cv2.imshow('Disparity Map', disparity_map)
    cv2.waitKey(0)


def disparity(img1,img2):
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=25)
    disparity = stereo.compute(img1, img2)
    return disparity


if __name__ == '__main__':
    main()