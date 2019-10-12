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
