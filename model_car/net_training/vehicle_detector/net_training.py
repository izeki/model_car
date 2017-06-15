'''
Created on May 29, 2017

@author: Sascha Hornauer
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from os.path import expanduser
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

from utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_box

if __name__ == '__main__':
    
    # TODO ... add tensorflow gpu support
    
    keras.backend.set_image_dim_ordering('th')
    
    ###
    
    model = Sequential()
    model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(64,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(128,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(256,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(512,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    
    ###
    
    model.summary()
    
    load_weights(model,'/home/picard/ICSI/yoloUdacity/yolo-tiny.weights')
    
    imagePath = '/home/picard/ICSI/VideoWorkspace/followingBlue/behind3.png'
    
    image = plt.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image_crop = image[300:650,500:,:]
    resized = cv2.resize(image_crop,(448,448))
    
    batch = np.transpose(resized,(2,0,1))
    batch = 2*(batch/255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    
    boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.0005)
    
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    ax1.imshow(image)
    ax2.imshow(draw_box(boxes,plt.imread(imagePath),[[500,1280],[300,650]]))
    plt.show()
    