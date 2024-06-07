import cv2
from torchvision import transforms
import torch
import numpy as np

im = cv2.imread('test_thermal.jpg')
im = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)

print(im.shape)
im = transforms.functional.affine(im, angle=0, translate=(10, 0), scale=1, shear=0)
im = im.permute(1, 2, 0).numpy().astype(np.uint8)
cv2.imshow('Test', im)
cv2.waitKey(0)