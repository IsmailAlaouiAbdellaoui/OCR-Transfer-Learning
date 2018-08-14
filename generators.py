# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 20:19:15 2018

@author: smail
"""

import random
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import string
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
#import generate_sets as gen
from keras import backend as K

def letter_to_digit(letters):
    digits = []
    for item in letters:
        number = ord(item) - 65
        digits.append(number)
    return digits

def generator(batch_size):
    i = 0
    train_x = []
    train_y = []
    while True:
        
        while(i<batch_size):        
            fnt = ImageFont.truetype('arial.ttf', 20)
            img = Image.new('L', (270, 20), color = random.randint(185,185))
            d = ImageDraw.Draw(img)
            name = ''.join(random.choices(string.ascii_uppercase, k=16))
            d.text((5,0),name ,font=fnt,fill=(90))
            list1 = list(img.getdata())
            length = len(list1)    
            list2 = np.random.normal(0, 25, length)
            list3 = list1+list2
            train_x.append(list3)
            train_y.append(letter_to_digit(list(name)))
            i = i+1
            
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        train_x = train_x.reshape(-1, 20,270, 1)
        train_x = train_x.astype('float')
        train_x /= 255.
        yield train_x,train_y
        

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(50,270,3),padding='same'))
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D((2, 2),padding='same'))
#model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
#model.add(LeakyReLU(alpha=0.1))                  
#model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#model.add(Flatten())
#model.add(Dense(128, activation='linear'))
#model.add(LeakyReLU(alpha=0.1))                  
#model.add(Dense(16, activation='softmax'))
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#model.summary()
#train_x = gen.generate_train_x()
#train_y = gen.generate_train_y()
#test_x = gen.generate_test_x()
#test_y = gen.generate_test_y()
##model.fit_generator(generator = generator(50), steps_per_epoch=200,epochs=20, verbose=1,validation_data=generator(50),validation_steps=40)
#
#model.fit(x=train_x, y=train_y, batch_size=50, epochs=20, verbose=1, validation_data=(test_x,test_y))

# import random
# from PIL import Image
# from PIL import ImageDraw, ImageFont
# import string

def paint_text(text,w,h,rotate=False, ud=False, multi_fonts=False):
   fnt = ImageFont.truetype('arial.ttf', 20)
   img = Image.new('L', (w, h), color = random.randint(185,185))
   d = ImageDraw.Draw(img)
 #   name = ''.join(random.choices(string.ascii_uppercase, k=16))
   name = ''.join(text)
   d.text((5,0),name ,font=fnt,fill=(90))
   list1 = list(img.getdata())
   length = len(list1)    
   list2 = np.random.normal(0, 25, length)
   list3 = list1+list2
   train_x = np.asarray(list3)
   return train_x

#a = paint_text('A GAME',h = 60, w = 270)
#a = a.reshape(1,60,270)
#print(a.shape)
   
def insert_space_random(s):
    r = random.randint(1, len(s)-1)
    return s[:r] + ' ' + s[r:]

def generate_data_file():
    print("generating the file ...")
    file = open('data.txt','w') 
    i = 0
    while(i<85000):
        length = random.randint(4,16)
        s = ''.join(random.choices(string.ascii_uppercase, k=length))
        s = insert_space_random(s)
        file.write(s+'\n')
        i= i+1
    print("done !")
    file.close()
#generate_data_file()
import re
import codecs
tmp_string_list = []
max_string_len = 17
data_file='data.txt'
num_words = 16000
regex = r'^[A-Z ]+$'
def is_valid_str(in_str):
    search = re.compile(regex, re.UNICODE).search
    return bool(search(in_str))
with codecs.open(data_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(tmp_string_list) == num_words:
                    break
                columns = line.split()
                word = columns[0] + ' ' + columns[1]
#                print(word)
                if is_valid_str(word):# and (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(word)
#        if len(tmp_string_list) != num_words:
#            print("length of tmp_string_list is : "+str(len(tmp_string_list)))
#            print("num_words is : "+str(num_words))
#            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
#        # interlace to mix up the easy and hard words
#        string_list[::2] = tmp_string_list[:self.num_words // 2]
#        string_list[1::2] = tmp_string_list[self.num_words // 2:]
                    

print("length of tmp_string_list is : "+str(len(tmp_string_list)))        




