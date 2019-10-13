import cv2
import math
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

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
    It returns image with the contours of the objects in the input image,
    and the contours themselves

    Keyword arguments:
    img -- the image itself (numpy array)
    """
    contours_img = np.full_like(img, 255)

    _, thresh = cv2.threshold(img[:, :, 0], 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contours_img, contours[1:], -1, (0, 0, 255), 1)
    return contours_img, contours

def get_measurements(img, contours):
    """
    It returns measurements related to the image (centroid, perimeter and
    area), with numbers identifying the respective objects in the output
    image

    Keyword arguments:
    img -- the image itself (numpy array)
    contours -- the contour array of the objects within the image
    """
    #output_img = np.zeros_like(img, dtype=np.uint8)
    hsvImg = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2HSV)
    hsvImg[...,2] = np.multiply(hsvImg[...,2], 1.1).astype(np.uint8)

    output_img = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR) + 100#np.multiply(img.copy(), 1.2).astype(np.uint8)#np.add(0.8*img.copy(), 10)
    print(output_img[193][147])

    idx = 1
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        #print("Perimeter:", perimeter)
        area = cv2.contourArea(contour)
        print("Area:", area)
        moments = cv2.moments(contour)
        c_x = int(moments['m10']/moments['m00'])
        c_y = int(moments['m01']/moments['m00'])
        output_img[c_y, c_x] = [0, 0, 0]
        offset_x = 10 if idx > 9 else 5
        cv2.putText(output_img, str(idx), (c_x - offset_x, c_y + 5), font, 0.5, (0, 0, 0), 2)
        idx += 1

    return output_img