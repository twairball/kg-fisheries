from __future__ import division,print_function

import os, json, sys
import os.path
from glob import glob
from shutil import copyfile
import numpy as np
import pandas as pd

#import modules
from utils import *
from vgg16 import Vgg16
# from vgg16bn import Vgg16BN as Vgg16
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.utils.np_utils import to_categorical

from kaggle import submit, push_to_kaggle

from base_model import lazy_property


##
## Batches
##
def create_gen(use_da=False):
    if use_da:
        gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.05,
            zoom_range=0.05, channel_shift_range=10, height_shift_range=0.05, 
            shear_range=0.05, horizontal_flip=True)
    else:
        gen = image.ImageDataGenerator()
    return gen 

def create_batches(path, shuffle=True, use_da=False, target_size=(224, 224), batch_size=64):
    gen = create_gen(use_da)
    return gen.flow_from_directory(path, 
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = shuffle
    )

class PrecalcFeats():
    """
    precalc VGG conv features
    """
    def __init__(self, input_shape=(3,224,224)):
        self.model = self.create_model(input_shape=input_shape)
    
    def create_model(self, input_shape=(3,224,224)):
        vgg = Vgg16() # TODO: pass input_shape
        model=vgg.model
        last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D][-1]
        conv_layers = model.layers[:last_conv_idx+1]
        return Sequential(conv_layers)

    def model_output_shape(self):
        return self.model.layers[-1].output_shape[1:]
    
    def calc_feats_on_batch(self, batches):
        return self.model.predict_generator(batches, batches.nb_sample)

