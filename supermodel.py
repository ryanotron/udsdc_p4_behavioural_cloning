#!/opt/conda/bin/python

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os, csv, math
import matplotlib.image as mpim
import numpy as np

import model as m

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
print(model.output_shape)

fname = os.path.join('.', 'record', 'driving_log.csv')
samples = m.get_samples_list(fname)
train_samples, valid_samples = train_test_split(samples, test_size=0.25)
    
batch_size = 32

train_gen = m.datagen(train_samples, batch_size=batch_size)
valid_gen = m.datagen(valid_samples, batch_size=batch_size)
    
history = model.fit_generator(train_gen, steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                              validation_data=valid_gen, validation_steps=math.ceil(len(valid_samples)/batch_size),
                              epochs=5, verbose=1)
model.save('model.h5')