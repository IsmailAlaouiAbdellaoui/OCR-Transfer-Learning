
from PIL import Image
import numpy as np
import cv2
import time
import os


def letter_to_digit(letters):
    digits = []
    for item in letters:
        number = ord(item) - 65
        digits.append(number)
    return digits

    
def generate_train_x():
    print("generating training set X ...")
    start_time = time.time()
    train_X = []
    for root, dirs, files in os.walk("Names- Training/"): 
        for filename in files:                        
            im=Image.open("D://OCR//Names- Training//"+filename)
            im = np.asarray(im)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            train_X.append(im)
    train_X = np.asarray(train_X)
    print("done generating training set X !")
    print("Elapsed time : "+str(time.time()-start_time))
    return train_X

def generate_train_y():
    print("generating training set Y ...")
    start_time = time.time()
    train_Y = []
    for root, dirs, files in os.walk("Names- Training/"): 
        for filename in files:                        
            temp2 = filename.split(".")
            name = temp2[0]        
            train_Y.append(letter_to_digit(list(name)))
    print("done generating training set Y !")
    print("Elapsed time : "+str(time.time()-start_time))
    return train_Y

    
def generate_test_x():
    print("generating testing set X ...")
    start_time = time.time()
    test_X = []
    for root, dirs, files in os.walk("Names- Testing/"): 
        for filename in files:
            im=Image.open("D://OCR//Names- Testing//"+filename)
            im = np.asarray(im)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            test_X.append(im)
    test_X = np.asarray(test_X)
    print("done generating testing set Y !")
    print("Elapsed time : "+str(time.time()-start_time))    
    return test_X

def generate_test_y():
    print("generating testing set Y ...")
    start_time = time.time()
    test_Y = []
    for root, dirs, files in os.walk("Names- Testing/"): 
        for filename in files:
            temp2 = filename.split(".")
            name = temp2[0]
            test_Y.append(letter_to_digit(list(name)))
    print("done generating testing set Y !")
    print("Elapsed time : "+str(time.time()-start_time))
    
    return test_Y

def generate_validate_x():
    print("generating validation set X ...")
    start_time = time.time()
    test_X = []
    for root, dirs, files in os.walk("Names- Validation/"): 
        for filename in files:
            im=Image.open("D://OCR//Names- Validation//"+filename)
            im = np.asarray(im)
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            test_X.append(im)
    test_X = np.asarray(test_X)
    print("done generating testing set Y !")
    print("Elapsed time : "+str(time.time()-start_time))    
    return test_X

def generate_validate_y():
    print("generating validation set Y ...")
    start_time = time.time()
    test_Y = []
    for root, dirs, files in os.walk("Names- Validation/"): 
        for filename in files:
            temp2 = filename.split(".")
            name = temp2[0]
            test_Y.append(letter_to_digit(list(name)))
    print("done generating testing set Y !")
    print("Elapsed time : "+str(time.time()-start_time))
    
    return test_Y
















#test_X = []
#
#test_Y=[]
#train_X=[]
#train_Y=[]
##model = []
#valid_x = generate_validate_x()
#valid_y= generate_validate_y()
#predictions = model.predict(valid_x)
#i=0
while(i<len(valid_x)):
    print("Real value : "+str(valid_y[i])+" ,  Prediction : "+ str(predictions[i]))
    i = i+1