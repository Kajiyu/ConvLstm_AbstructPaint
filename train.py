#coding: utf-8

import numpy as np
import pandas as pd
import os, sys
import glob


from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def create_model(input_height, input_width):
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),input_shape=(None, input_height, input_width, 1),padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),activation='sigmoid',padding='same', data_format='channels_last'))
    seq.compile(loss='binary_crossentropy', optimizer='adadelta')
    return seq


if __name__ == '__main__':
    image_datagen = ImageDataGenerator(rescale=1./255.)