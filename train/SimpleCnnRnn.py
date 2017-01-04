#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import SGD
from utils.utils import create_simpleCnnRnn, load_data
import numpy as np

image_model = create_simpleCnnRnn()
sgd = SGD(lr=0.0002, decay=1e-6, momentum=0.9, nesterov=True)
image_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
images, labels = load_data()  # categorical labels


val_testx = images[-4000:]
with open("../Data/labels", "rb") as f:
    index_labels = np.load(f)
val_testy = index_labels[-4000:]


class ValidateAcc(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print '\n———————————--------'
        image_model.load_weights('../model/weights.%02d.hdf5' % epoch)
        r = image_model.predict(val_testx, verbose=0)
        y_predict = np.asarray([np.argmax(i, axis=1) for i in r])
        length = len(y_predict) * 1.0
        correct = 0
        for (true,predict) in zip(val_testy,y_predict):
            if list(true) == list(predict):
                correct += 1
        print "Validation set acc is: ", correct/length
        print '\n———————————--------'


val_acc_check_pointer = ValidateAcc()
check_pointer = ModelCheckpoint(filepath="../model/weights.{epoch:02d}.hdf5")

image_model.fit(images, labels,
                validation_split=0.2,  # split data into 4:1  the last 4000(0.2*20000) is used as val data set
                shuffle=True,batch_size=16, nb_epoch=20,callbacks=[check_pointer, val_acc_check_pointer])
image_model.save("../model/model2.hdf5")

