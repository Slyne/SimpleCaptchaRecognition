#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, GRU, TimeDistributed
from keras.layers import Dense, Dropout, Activation, Flatten, RepeatVector,Input
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential,Model
from keras import backend as K
K.set_image_dim_ordering("th")
vocab_size = 36  # 26 + 10
C, H, W = 3, 60, 250
max_caption_len = 5


def load_data():
    with open("../Data/pic", "rb") as f:
        images = np.load(f)
    with open("../Data/labels", "rb") as f:
        labels = np.load(f)
    labels_categorical = np.asarray([to_categorical(label, vocab_size) for label in labels])
    print "images shape", images.shape
    # print images[0]
    print "input labels shape", labels_categorical.shape
    return images,labels_categorical


def create_simpleCnnRnn():
    image_model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(C, H, W)))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(32, 3, 3))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))
    # image_model.add(Dropout(0.25))
    image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(64, 3, 3))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))
    # image_model.add(Dropout(0.25))
    image_model.add(Flatten())
    # Note: Keras does automatic shape inference.
    image_model.add(Dense(128))
    image_model.add(RepeatVector(max_caption_len))
    image_model.add(GRU(output_dim=128, return_sequences=True))
    image_model.add(GRU(output_dim=128, return_sequences=True))
    image_model.add(TimeDistributed(Dense(vocab_size)))
    image_model.add(Activation('softmax'))
    return image_model


def create_multiOutputCnn():
    image_model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(C, H, W)))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(32, 3, 3))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))
    # image_model.add(Dropout(0.25))
    image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(64, 3, 3))
    image_model.add(BatchNormalization())
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))
    # image_model.add(Dropout(0.25))
    image_model.add(Flatten())
    # Note: Keras does automatic shape inference.
    image_input = Input(shape=(C, H, W))
    encoded_image = image_model(image_input)
    out1 = Dense(128, activation="relu")(encoded_image)
    out2 = Dense(128, activation="relu")(encoded_image)
    out3 = Dense(128, activation="relu")(encoded_image)
    out4 = Dense(128, activation="relu")(encoded_image)
    out5 = Dense(128, activation="relu")(encoded_image)
    output1 = Dense(vocab_size, activation="softmax")(out1)
    output2 = Dense(vocab_size, activation="softmax")(out2)
    output3 = Dense(vocab_size, activation="softmax")(out3)
    output4 = Dense(vocab_size, activation="softmax")(out4)
    output5 = Dense(vocab_size, activation="softmax")(out5)
    model = Model([image_input], [output1, output2, output3, output4, output5])
    return model
