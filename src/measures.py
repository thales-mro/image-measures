import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def get_measures(img, contours):
    """
    It returns measures related to the image (centroid, perimeter and
    area), with numbers identifying the respective objects in the output
    image, and also the area of the objects

    Keyword arguments:
    img -- the image itself (numpy array)
    contours -- the contour array of the objects within the image
    """
    contours = np.flip(contours)
    areas = []

    # modify image colors to make the labels more readable
    hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    hsv_img[..., 2] = np.multiply(hsv_img[..., 2], 1.1).astype(np.uint8)
    output_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR) + 100

    idx = 0
    for contour in contours:
        # calculate required measures
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        c_x = int(moments['m10']/moments['m00'])
        c_y = int(moments['m01']/moments['m00'])
        areas.append(area)

        # label image
        offset_x = 10 if idx > 9 else 5
        cv2.putText(output_img, str(idx), (c_x - offset_x, c_y + 5), font, 0.5, (0, 0, 0), 2)

        print("Region %2d: area: %6.1f perimeter: %9.5f " % (idx, area, perimeter))
        idx += 1

    return output_img, areas

def areas_histogram(areas, img_name):
    """
    It returns a histogram of the object's area

    Keyword arguments:
    areas -- the areas of all objects (array)
    """

    counts, _, _ = plt.hist(areas, [0, 1500, 3000, 4500], color='#0504aa',
                            alpha=0.7)
    plt.xlabel("Area")
    plt.ylabel("Number of objects")
    plt.title("Number of objects per area")
    plt.grid(axis='y', alpha=0.75)

    print("Classification of objects based on their respective areas:")
    print("Number of small regions: %d" % counts[0])
    print("Number of medium regions: %d" % counts[1])
    print("Number of big regions: %d" % counts[2])

    plt.savefig('output/' + img_name + "-histogram" + '.png')
    plt.show()
