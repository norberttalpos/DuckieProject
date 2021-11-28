import os

import cv2 as cv
import numpy as np
from PIL import Image



def change_colors(cv2img): #TODO elvárt, de lehet hogy rosszabb tanításra
    hsv = cv.cvtColor(cv2img, cv.COLOR_BGR2HSV)

    lower_yellow = np.array([0, 49, 90], dtype="uint8")
    upper_yellow = np.array([100, 255, 255], dtype="uint8")

    lower_gray = np.array([0, 0, 95], np.uint8)
    upper_gray = np.array([179, 50, 255], np.uint8)

    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    grey_mask = cv.inRange(hsv, lower_gray, upper_gray)

    res_grey = cv.bitwise_and(cv2img, cv2img, mask=grey_mask)
    res_yellow = cv.bitwise_and(cv2img, cv2img, mask=yellow_mask)

    gray = cv.cvtColor(cv2img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 250, 400, apertureSize=3)

    result = cv.add(cv.add(np.stack((edges,) * 3, axis=-1), res_yellow), res_grey)

    return result


def preprocess_image(cv2img):
    im_rgb = change_colors(cv2img)

    img = Image.fromarray(im_rgb, 'RGB')
    w, h = img.size  # original images w,h :  640 x 480px
    crop_height = 120  # cropping it to 640x360
    img = img.crop((0, crop_height, w, h))

    # resizing the image for training to 85x48 (48 height and the according width to keep cropped image size ratio
    img = img.resize((85,48))

    return img
