# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 19:42:49 2018

@author: smail
"""

import cv2
import imutils
import numpy as np
#image = cv2.imread("CIN3.JPG")
#image = imutils.resize(image, width=800)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##(thresh, im_bw) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl1 = clahe.apply(gray)
##equ = cv2.equalizeHist(gray)
##res = np.hstack((image,equ))
#
#cv2.imshow('image',cl1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def show_color(name):
    image = cv2.imread(name)
#    image = imutils.resize(image, width=800)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show_gray(name):
    image = cv2.imread(name)
#    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show_CLAHE(name):
    image = cv2.imread(name)
#    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    cv2.imshow('image',cl1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show_bw(name):
    image = cv2.imread(name,0)
#    image = imutils.resize(image, width=800)
    (thresh, image) = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_edges(name,low_threshold=80, high_threshold=150):
    image = cv2.imread(name)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    cv2.imshow('image',edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

import transform

test = transform.DocScanner()
trans = test.scan("raw1.jpg")
cv2.imshow('image',trans)
cv2.waitKey(0)
cv2.destroyAllWindows()




#scan("raw1.jpg")




    
    