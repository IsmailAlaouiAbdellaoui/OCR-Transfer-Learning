# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:34:03 2018

@author: smail
"""
import random
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import string
     

def generate_image_train():
    fnt = ImageFont.truetype('arial.ttf', 20)
    img = Image.new('L', (270, 20), color = random.randint(185,185))
    d = ImageDraw.Draw(img)
    name = ''.join(random.choices(string.ascii_uppercase, k=16))
    d.text((5,0),name ,font=fnt,fill=(90))
    list1 = list(img.getdata())
    length = len(list1)    
    list2 = np.random.normal(0, 25, length)
    list3 = list1+list2
    img.putdata(list3)
    img.save('Names- Training/'+name+'.png')

def generate_image_test():
    fnt = ImageFont.truetype('arial.ttf', 20)
    img = Image.new('L', (270, 20), color = random.randint(185,185))
    d = ImageDraw.Draw(img)
    name = ''.join(random.choices(string.ascii_uppercase, k=16))
    d.text((5,0),name ,font=fnt,fill=(90))
    list1 = list(img.getdata())
    length = len(list1)    
    list2 = np.random.normal(0, 25, length)
    list3 = list1+list2
    img.putdata(list3)
    img.save('Names- Testing/'+name+'.png')
    
def generate_image_validate():
    fnt = ImageFont.truetype('arial.ttf', 20)
    img = Image.new('L', (270, 20), color = random.randint(185,185))
    d = ImageDraw.Draw(img)
    name = ''.join(random.choices(string.ascii_uppercase, k=16))
    d.text((5,0),name ,font=fnt,fill=(90))
    list1 = list(img.getdata())
    length = len(list1)    
    list2 = np.random.normal(0, 25, length)
    list3 = list1+list2
    img.putdata(list3)
    img.save('Names- Validation/'+name+'.png')


    
def generate_all_train_images():
    i=0
    while(i<10000):
        generate_image_train()
        i = i+1
        
def generate_all_test_images():
    i=0
    while(i<2250):
        generate_image_test()
        i = i+1
        
def generate_all_validate_images():
    i=0
    while(i<250):
        generate_image_validate()
        i = i+1
        
        























