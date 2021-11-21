import os

import cv2 as cv
import numpy as np
from PIL import Image


def convert_images():
    folder_path = os.path.join(os.getcwd(), "myapp")
    c = 1
    for filename in os.listdir(folder_path):
        convert_image(os.path.join(folder_path, filename))
        if c%500 == 0:
            print("first" ,c," images done")
        c+=1


def convert_image(image_path):

    img = cv.imread(image_path)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    lower_yellow = np.array([0, 49, 90], dtype="uint8")
    upper_yellow = np.array([100, 255, 255], dtype="uint8")

    lower_gray = np.array([0, 0, 95], np.uint8)
    upper_gray = np.array([179, 50, 255], np.uint8)

    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    grey_mask = cv.inRange(hsv, lower_gray, upper_gray)

    res_grey = cv.bitwise_and(img, img, mask=grey_mask)
    res_yellow = cv.bitwise_and(img, img, mask=yellow_mask)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 250, 400, apertureSize=3)

    result = cv.add(cv.add(np.stack((edges,) * 3, axis=-1), res_yellow), res_grey)

    im_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

    img = Image.fromarray(im_rgb, 'RGB')
    w, h = img.size
    crop_height = 120
    img.crop((0, crop_height, w, h)).save(image_path)

convert_images()
