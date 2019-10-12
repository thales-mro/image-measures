import cv2
import math
import numpy as np

def transform_colors(img):
    """
    It transform colored objects in white background image in grayscale ones

    Keyword arguments:
    img -- the image itself (numpy array)
    """
    b = g = r = img.copy()
    b[b[:, :, 0] < 255] = 0
    g[g[:, :, 1] < 255] = 0
    r[r[:, :, 2] < 255] = 0

    return b*g*r

def get_contours(img):
    """
    It returns image with the contours of the objects in the input image

    Keyword arguments:
    img -- the image itself (numpy array)
    """
    contours_img = np.full_like(img, 255)

    _, thresh = cv2.threshold(img[:, :, 0], 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contours_img, contours[1:], -1, (0, 0, 255), 1)
    return contours_img, contours
