#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy import misc
import numpy as np
from utils.utils import create_simpleCnnRnn, create_multiOutputCnn

image_model = create_simpleCnnRnn()
# image_model = create_multiOutputCnn()
image_model.load_weights("../model/CnnRnn.hdf5")

predir = "../Data/type2_train/"
pics = []
base = 16001
for i in range(4000):
    index = str(base + i)
    pic = misc.imread(predir + "type2_train_"+index+".jpg")
    pic = np.rollaxis(pic, 2, 0)
    pics.append(pic)
pics = np.asarray(pics)

result_all = image_model.predict(pics)

with open("../Data/labels", "rb") as f:
    labels = np.load(f)

correct = 0
val_labels = labels[16000:]
for (true, predict) in zip(val_labels,result_all):
    predict_value = np.argmax(predict, axis=1)
    if list(true) == list(predict_value):
        correct += 1
    else:
        print true,
        print predict_value
print correct
