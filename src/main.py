import cv2
import matplotlib.pyplot as plt
import numpy as np
from measurements import transform_colors

def open_image(name):
    """
    it makes calls for openCV functions for reading an image based on a name

    Keyword arguments:
    name -- the name of the image to be opened
    grayscale -- whether image is opened in grayscale or not
        False (default): image opened normally (with all 3 color channels)
        True: image opened in grayscale form
    """
    img_name = 'input/' + name  + '.png'
    return cv2.imread(img_name, cv2.IMREAD_UNCHANGED)

def display_image(img, label):
    """
    it opens graphic user interface displaying the provided input image

    Keyword arguments:
    img -- the image itself (numpy array)
    label -- text to appear in the window, describing the image
    """
    cv2.imshow(label,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(name, image):
    """
    it makes calls for openCV function for saving an image based on a name (path)
    and the image itself

    Keyword arguments:
    name -- the name (path) of the image to be saved
    image -- the image itself (numpy array)
    """
    image_name = 'output/' + name + '.png'
    cv2.imwrite(image_name, image)

def save_histogram(hist, name):
    """
    it makes calls for matplotlib function for painting the histogram on canvas, then saving it

    Keyword arguments:
    name -- the name (path) of the histogram to be saved
    hist -- the histogram itself
    """
    plt.clf()
    plt.plot(hist, color='k')
    plt.savefig('output/' + name + '.png')


def main():
    """
    Entrypoint for the code of project 02 MO443/2s2019

    For every input image, it creates thresholded images with global and
    local methods, for different global thresholds and different window
    sizes for local methods
    """

    # for inserting other images, add tem to /input folder and list them here
    images = (
        'i-0',
        'i-1',
        'i-2'
    )

    for image_name in images:
        print(image_name, "image:")

        image = open_image(image_name)
        display_image(image, "Original input " + image_name)

        grayscale_v = transform_colors(image)
        display_image(grayscale_v, "Grayscale " + image_name)
        save_image(image_name + "-grayscale", grayscale_v)

main()
