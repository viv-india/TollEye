import cv2
import numpy as np

img = cv2.imread('img4.jpg', cv2.IMREAD_GRAYSCALE)
print(img)

n_white_pix = np.sum(img == 255)
print('Number of white pixels:', n_white_pix)