# -*- coding: utf-8 -*-
'''
Identify license plate number
'''

import sys
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import plate_detect.HyperLPRLite as pr
import cv2
import numpy as np
import time
from importlib import reload

fontC = ImageFont.truetype("Font/platech.ttf", 14, 0)


# return the [result, confidence, position] of license plate
def recognize_plate(image, smallest_confidence = 0.7):
    # # grr = cv2.imread(image_path)

    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    model.SimpleRecognizePlateByE2E(image)
    return_all_plate = []
    for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(image):
        if confidence>smallest_confidence:
            return_all_plate.append([pstr,confidence,rect])
    return return_all_plate


# draw rectangle and number on image
def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText.encode('utf-8').decode("utf-8"), (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex


# test result and show image
def visual_draw_position(grr):
    model = pr.LPR("model/cascade.xml","model/model12.h5","model/ocr_plate_all_gru.h5")
    for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(grr):
        if confidence>0.7:
            grr = drawRectBox(grr, rect, pstr+" "+str(round(confidence,3)))
            print("车牌号:")
            print(pstr)
            print("置信度")
            print(confidence)
    cv2.imshow("image",grr)
    cv2.waitKey(0)


# SpeedTest("Images/test3.jpg")
'''
if __name__ == '__main__':
    test_image = cv2.imread("car/1.jpg")
    print(recognize_plate(test_image))
    visual_draw_position(test_image)
'''