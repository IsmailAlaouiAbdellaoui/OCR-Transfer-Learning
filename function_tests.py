# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:43:15 2018

@author: smail
"""

import numpy as np
import codecs
import re
import keras
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import generate_sets as gen
import generators

absolute_max_string_len = 16
minibatch_size = 50
val_split = 0.2


regex = r'^[A-Z ]+$'
#alphabet = u'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
alphabet = u'abcdefghijklmnopqrstuvwxyz '


def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret


def is_valid_str(in_str):
    search = re.compile(regex, re.UNICODE).search
    return bool(search(in_str))

def build_word_list(num_words, max_string_len=None, mono_fraction=0.5):
        assert max_string_len <= absolute_max_string_len
        assert num_words % minibatch_size == 0
        assert (val_split * num_words) % minibatch_size == 0
        num_words = num_words
        string_list = [''] * num_words
        tmp_string_list = []
        max_string_len = max_string_len
        Y_data = np.ones([num_words, absolute_max_string_len]) * -1
        X_text = []
        Y_len = [0] * num_words

        # monogram file is sorted by frequency in english speech
        with codecs.open('wordlist_mono_clean.txt', mode='r', encoding='utf-8') as f:
            for line in f:
                if len(tmp_string_list) == int(num_words * mono_fraction):
                    break
                word = line.rstrip()
                if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
                    tmp_string_list.append(word)

        # bigram file contains common word pairings in english speech
        with codecs.open('wordlist_bi_clean.txt', mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(tmp_string_list) == num_words:
                    break
                columns = line.lower().split()
                word = columns[0] + ' ' + columns[1]
                if is_valid_str(word) and \
                        (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(word)
        if len(tmp_string_list) != num_words:
            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
        # interlace to mix up the easy and hard words
        string_list[::2] = tmp_string_list[:num_words // 2]
        string_list[1::2] = tmp_string_list[num_words // 2:]

        for i, word in enumerate(string_list):
            Y_len[i] = len(word)
            Y_data[i, 0:len(word)] = text_to_labels(word)
            X_text.append(word)
        Y_len = np.expand_dims(np.array(Y_len), 1)

        cur_val_index = val_split
        cur_train_index = 0
#        print("X_text content : "+str(X_text))
        print("Y_data content : "+str(Y_data))
#        print("Y_len content : "+str(Y_len))
        print(type(Y_data))
        
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



#build_word_list(16000, 4, 1)
input_shape = (20, 270, 1)
img_h = 64
words_per_epoch = 16000
val_split = 0.2
val_words = int(words_per_epoch * (val_split))

# Network parameters
conv_filters = 16
kernel_size = (3, 3)
pool_size = 2
time_dense_size = 32
rnn_size = 512
act = 'relu'
input_data = Input(name='the_input', shape=input_shape, dtype='float32')
inner = Conv2D(conv_filters, kernel_size, padding='same',
               activation=act, kernel_initializer='he_normal',
               name='conv1')(input_data)
inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
inner = Conv2D(conv_filters, kernel_size, padding='same',
               activation=act, kernel_initializer='he_normal',
               name='conv2')(inner)
inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

conv_to_rnn_dims = (270 // (pool_size ** 2), (20 // (pool_size ** 2)) * conv_filters)
inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

# cuts down input size going into RNN:
inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

# Two layers of bidirectional GRUs
# GRU seems to work as well, if not better than LSTM:
gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
gru1_merged = add([gru_1, gru_1b])
gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

#inner = Flatten()(concatenate([gru_2, gru_2b]))
# transforms RNN output to character activations:
inner = Dense(28, kernel_initializer='he_normal',
              name='dense2')(concatenate([gru_2, gru_2b]))
y_pred = Activation('softmax', name='softmax')(inner)        
#model = Model(inputs=input_data, outputs=y_pred)
#model.summary()

labels = Input(name='the_labels', shape=[16], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
#model.summary()

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

#train_x = gen.generate_train_x()
#train_x = train_x.reshape(-1, 50,270, 1)
#train_x = train_x.astype('float')
#train_x /= 255.
#train_y = gen.generate_train_y()
#
#
#
#
#test_x = gen.generate_test_x()
#test_x = train_x.reshape(-1, 50,270, 1)
#test_x = train_x.astype('float')
#test_x /= 255.
#test_y = gen.generate_test_y()
#print(len(test_x))
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#model.fit(x=train_x, y=train_y, batch_size=50, epochs=20, verbose=1, validation_data=(test_x,test_y))
model.fit_generator(generators.generator(50), steps_per_epoch=200,epochs=20, verbose=1,validation_data=generators.generator(50),validation_steps=40)
#print(loss_out[0])

























