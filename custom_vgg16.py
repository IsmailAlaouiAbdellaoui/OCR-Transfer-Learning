# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:53:21 2018

@author: smail
"""

from keras.models import Sequential
from keras.optimizers import  Adam
#from keras.layers.merge import add
from keras.layers.core import Flatten, Dense
import keras
import generate_sets as gen


#MODEL STARTS HERE ! !
def create_custom_vgg(): 
    model = Sequential()
    tlearning = keras.applications.vgg16.VGG16(include_top=False, input_shape=(50,270,3))
#    tlearning.summary()#uncomment to see the difference between the two ones
    for layer in tlearning.layers:
        model.add(layer)
    
    model.layers.pop()
    
#    for layer in model.layers:
#        layer.trainable = False
        
    model.add(Flatten())
    model.add(Dense(16,activation='softmax',name='dense_last2'))
    
    model.summary()
    model.compile(Adam(lr=.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.load_weights('vgg16_weights_tl.hd5')
    return model
    #model.fit(x=train_X, y=train_Y, batch_size=None, epochs=30, verbose=1, validation_data=(test_X,test_Y))
    #model.save_weights('vgg16_weights_tl.hd5')
    
train_X= gen.generate_train_x()
train_Y = gen.generate_train_y()
#
test_X= gen.generate_test_x()
test_Y = gen.generate_test_y()
#
train_X = train_X.reshape(-1, 50,270, 3)
test_X = test_X.reshape(-1, 50,270, 3)