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
import cv2

angle_offset = 0.2 # what angle to add/subtract for side cameras

def net_nvidia():
    model = Sequential()
    
    # input processing
    # crop (50 px from top, 20 px from bottom)
    model.add(Cropping2D(((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    
    # normalise
    model.add(Lambda(lambda x: (x/255)-0.5))
    
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2)))
    model.add(Activation('relu')) # the paper doesn't mention activation function, but isn't that needed?
    
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2)))
    model.add(Activation('relu')) 
    
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2)))
    model.add(Activation('relu')) 
    
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))
    model.add(Activation('relu')) 
    
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))
    model.add(Dropout(0.30))
    model.add(Activation('relu')) 
    
    model.add(Flatten())
    
    model.add(Dense(100))
    model.add(Activation('relu'))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('relu'))
    
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


def get_samples_list(fname, mirror=True):
    global angle_offset
    samples = []
    logpath = fname.split(os.sep)[:-1]
    logpath = os.path.join(*logpath)
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            theta = float(line[3])
            
            # add centre image and mirror
            sep = '\\'
            if 'linux' in line[0]:
                sep = '/'
                
            impath = os.path.join(logpath, 'IMG', line[0].split(sep)[-1])
            sample = [impath, theta, False]
            samples.append(sample)
            if mirror:
                sample = [impath, -1*theta, True]
                samples.append(sample)
            
            # add left image and mirror
            impath = os.path.join(logpath, 'IMG', line[1].split(sep)[-1])
            sample = [impath, theta + angle_offset, False]
            samples.append(sample)
            if mirror:
                sample = [impath, -1*(theta + angle_offset), True]
                samples.append(sample)
            
            # add right image and mirror
            impath = os.path.join(logpath, 'IMG', line[2].split(sep)[-1])
            sample = [impath, theta - angle_offset, False]
            samples.append(sample)
            if mirror:
                sample = [impath, -1*(theta - angle_offset), True]
                samples.append(sample)
            
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
                fname = batch_sample[0]
                center_im = mpim.imread(fname)
                if batch_sample[2]:
                    center_im = cv2.flip(center_im, 1)
                center_th = batch_sample[1]
                images.append(center_im)
                angles.append(center_th)
                
            images = np.array(images)
            angles = np.array(angles)
            yield shuffle(images, angles)

            
def train(model, train_fname=None, batch_size=32, epochs=5, mirror=True):
    if not train_fname:
        fname = os.path.join('.', 'record', 'driving_log.csv')
    else:
        fname = train_fname
    samples = get_samples_list(fname, mirror=mirror)
        
    train_samples, valid_samples = train_test_split(samples, test_size=0.25)
    
    train_gen = datagen(train_samples, batch_size=batch_size)
    valid_gen = datagen(valid_samples, batch_size=batch_size)
    
    history = model.fit_generator(train_gen, steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                                  validation_data=valid_gen, validation_steps=math.ceil(len(valid_samples)/batch_size),
                                  epochs=epochs, verbose=1)
    model.save('model.h5')
    return history

if __name__ == "__main__":
    print("Hello, driver")
    mdl = net_nvidia()
    
    print("Begin training, now!")
    hst = train(mdl)
