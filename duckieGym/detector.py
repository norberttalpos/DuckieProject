import os

import cv2 as cv
import numpy as np
from PIL import Image


sarga = cv.imread(os.path.join(os.getcwd(), "szinAlapok", "sarga.png"))
sarga = cv.cvtColor(sarga, cv.COLOR_RGB2BGR)
szurke = cv.imread(os.path.join(os.getcwd(), "szinAlapok", "szurke.png"))
szurke = cv.cvtColor(szurke, cv.COLOR_RGB2BGR)
barna = cv.imread(os.path.join(os.getcwd(), "szinAlapok", "tablabarna.png"))
barna = cv.cvtColor(barna, cv.COLOR_RGB2BGR)
fekete = cv.imread(os.path.join(os.getcwd(), "szinAlapok", "fekete.png"))
narancs = cv.imread(os.path.join(os.getcwd(), "szinAlapok", "narancs.png"))
narancs = cv.cvtColor(narancs, cv.COLOR_RGB2BGR)

def change_colors(cv2img): #TODO elvárt, de lehet hogy rosszabb tanításra
    hsv = cv.cvtColor(cv2img, cv.COLOR_BGR2HSV)
     

    lower_yellow = np.array([0, 49, 90], dtype="uint8")
    upper_yellow = np.array([100, 255, 255], dtype="uint8")
    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    
    lower_org = np.array([87, 17, 0], dtype="uint8")
    upper_org = np.array([255, 185, 5], dtype="uint8")
    org_mask = cv.inRange(cv2img, lower_org, upper_org)
    res_org = cv.bitwise_and(cv2img,cv2img, mask=org_mask)
    res_org = cv.bitwise_not(res_org,res_org, mask=yellow_mask)
    res_org = cv.bitwise_and(res_org,cv2img, mask=org_mask)
    res_org = np.where(res_org!=0,narancs,fekete)

    lower_gray = np.array([0, 0, 95], np.uint8)
    upper_gray = np.array([179, 50, 255], np.uint8)



    grey_mask = cv.inRange(hsv, lower_gray, upper_gray)

    res_grey = cv.bitwise_and(cv2img, cv2img, mask=grey_mask)
    res_yellow = cv.bitwise_and(cv2img, cv2img, mask=yellow_mask)
    
    res_grey = np.where(res_grey!=0,szurke,fekete)
    
    lower_yellow2 = np.array([120, 120, 0], dtype="uint8")
    upper_yellow2 = np.array([255, 235, 200], dtype="uint8")
    
    lower_brown = np.array([10, 10, 10], dtype="uint8")
    upper_brown = np.array([63, 84, 98], dtype="uint8")
    
    #lower_brown = np.array([10, 43, 66], dtype="uint8")
    #upper_brown = np.array([150, 150, 100], dtype="uint8")
    
    #brown_mask = cv.inRange(cv2img, lower_brown, upper_brown)
    #res_brown = cv.bitwise_and(cv2img, cv2img, mask=brown_mask)
    #res_brown = np.where(res_brown!=0,barna,fekete)

    yellow_mask = cv.inRange(res_yellow, lower_yellow2, upper_yellow2)
    res_yellow = cv.bitwise_and(res_yellow, res_yellow, mask=yellow_mask)
    res_yellow = np.where(res_yellow!=0,sarga,fekete)

    gray = cv.cvtColor(cv2img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 250, 400, apertureSize=3)

    #result = cv.add(cv.add(np.stack((edges,) * 3, axis=-1), res_yellow), res_grey)
    result = cv.add(res_yellow, res_grey)
    result = cv.add(result, res_org)

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
