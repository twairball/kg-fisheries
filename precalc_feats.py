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

def create_batches(path, shuffle=True, use_da=False):
    batch_size = 64
    target_size = (224, 224)
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
    def __init__(self):
        self.model = self.create_model()
    
    def create_model(self):
        vgg = Vgg16()
        model=vgg.model
        last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D][-1]
        conv_layers = model.layers[:last_conv_idx+1]
        return Sequential(conv_layers)

    def model_output_shape(self):
        return self.model.layers[-1].output_shape[1:]
    
    def calc_feats_on_batch(self, batches):
        return self.model.predict_generator(batches, batches.nb_sample)

## main scripts
def calc_train_da_feats():
    nb_augm = 5
    print("===== (TRAIN) Precalc data-augmented conv features =====")

    pcf = PrecalcFeats()
    for aug in range(nb_augm):
        print("===== data-aug: %d =====" % aug)
        batches = create_batches('data/train/', shuffle=True, use_da=True)
        print("    (precalc) calculating features...")
        feats = pcf.calc_feats_on_batch(batches)
        labels = to_categorical(batches.classes)

        # save
        labels_file = "data/results/da%d_conv_labels.h5" % aug
        feats_file = "data/results/da%d_conv_feats.h5" % aug
        save_array(labels_file, labels)
        save_array(feats_file, feats)
        print("    (precalc) feats: %s" % (feats.shape,))
        print("    (precalc) saved feats to: %s" % feats_file)
        print("    (precalc) saved labels to: %s" % labels_file)

def calc_val_feats():
    print("===== (VALID) Precalc validation conv features =====")
    pcf = PrecalcFeats()
    val_batches = create_batches('data/valid/', shuffle=True, use_da=False)
    print("    (precalc) calculating features...")
    feats = pcf.calc_feats_on_batch(batches)
    labels = to_categorical(batches.classes)

    # save
    labels_file = "data/results/conv_val_labels.h5" 
    feats_file = "data/results/conv_val_feats.h5"
    save_array(labels_file, labels)
    save_array(feats_file, feats)
    print("    (precalc) feats: %s" % (feats.shape,))
    print("    (precalc) saved feats to: %s" % feats_file)
    print("    (precalc) saved labels to: %s" % labels_file)


def calc_test_da_feats():
    nb_augm = 5
    print("===== (TEST) Precalc data-augmented conv features =====")

    pcf = PrecalcFeats()
    for aug in range(nb_augm):
        print("===== data-aug: %d =====" % aug)
        batches = create_batches('data/test/', shuffle=True, use_da=True)
        print("    (precalc) calculating features...")
        feats = pcf.calc_feats_on_batch(batches)
        labels = to_categorical(batches.classes)

        # save
        labels_file = "data/results/da%d_conv_test_labels.h5" % aug
        feats_file = "data/results/da%d_conv_test_feats.h5" % aug
        save_array(labels_file, labels)
        save_array(feats_file, feats)
        print("    (precalc) feats: %s" % (feats.shape,))
        print("    (precalc) saved feats to: %s" % feats_file)
        print("    (precalc) saved labels to: %s" % labels_file)


if __name__ == "__main__":
    calc_train_da_feats()
    calc_val_feats()
    calc_test_da_feats()
    