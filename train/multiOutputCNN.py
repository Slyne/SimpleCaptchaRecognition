#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import SGD
from utils.utils import load_data, create_multiOutputCnn

images, labels = load_data()
print labels.shape
digit1 = labels[:,0,:]
digit2 = labels[:,1,:]
digit3 = labels[:,2,:]
digit4 = labels[:,3,:]
digit5 = labels[:,4,:]

model = create_multiOutputCnn()
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
checkpointer= ModelCheckpoint(filepath="../model/weights.{epoch:02d}.hdf5")

model.fit(images, [digit1, digit2, digit3, digit4, digit5], validation_split=0.2, shuffle=True,batch_size=16, nb_epoch=20,callbacks=[checkpointer])
model.save("../model/model2.hdf5")