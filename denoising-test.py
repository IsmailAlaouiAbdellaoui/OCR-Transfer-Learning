# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 01:18:57 2018

@author: smail
"""

import cv2
image = cv2.imread("FIZEBGPMKUPNVFBR.png",0)
#    image = imutils.resize(image, width=800)
(thresh, image) = cv2.threshold(image, 105, 255, cv2.THRESH_BINARY)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()