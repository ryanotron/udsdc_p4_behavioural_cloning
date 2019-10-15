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

def net2():
    model = Sequential()
    
#     model.add(Lambda(lambda x: (x/255)-0.5, input_shape=(160, 320, 3)))
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def network():
    model = Sequential()
    
    # input processing
    # crop (50 px from top, 20 px from bottom)
    model.add(Cropping2D(((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    
    # normalise
    model.add(Lambda(lambda x: (x/255)-0.5))
    
    # conv1
    model.add(Conv2D(16, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    
    # conv2
    model.add(Conv2D(32, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    
    # conv3
    model.add(Conv2D(64, kernel_size=(5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    
    # flatten
    model.add(Flatten())
    
    # dense1
    model.add(Dense(1000))
    model.add(Activation('relu'))
    
    # dense2
    model.add(Dense(500))
    model.add(Activation('relu'))
    
    # dense3
    model.add(Dense(200))
    model.add(Activation('relu'))
    
    # output
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    return model


def get_samples_list(fname):
    samples = []
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

                    
def datagen(samples, batch_size=32):
    n_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                fname = os.path.join('.', 'record', 'IMG', batch_sample[0].split('\\')[-1])
                center_im = mpim.imread(fname)
                center_th = float(batch_sample[3])
                images.append(center_im)
                angles.append(center_th)
                
            images = np.array(images)
            angles = np.array(angles)
            yield shuffle(images, angles)

            
def train(model, batch_size=32, epochs=5):
    fname = os.path.join('.', 'record', 'driving_log.csv')
    samples = get_samples_list(fname)
    train_samples, valid_samples = train_test_split(samples, test_size=0.25)
    
    train_gen = datagen(train_samples, batch_size=batch_size)
    valid_gen = datagen(valid_samples, batch_size=batch_size)
    
    model = network()
    history = model.fit_generator(train_gen, steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                                  validation_data=valid_gen, validation_steps=math.ceil(len(valid_samples)/batch_size),
                                  epochs=epochs, verbose=1)
    model.save('model.h5')
    return history

if __name__ == "__main__":
    print("Hello")
    mdl = network()
    print("Start training!")
    hst = train(mdl)