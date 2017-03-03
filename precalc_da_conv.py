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
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

from kaggle import submit, push_to_kaggle
from precalc_conv import DenseModel, PrecalcConvTestModel, PrecalcConvModel

class PrecalcDAConvModel(PrecalcConvModel):
    """
        Precalculate convolution features with data augmentation
    """
    def __init__(self, path, data_augment_size=3):
        self.path = path
        self.model = self.conv_model()
        self.data_augment_size = data_augment_size
        self.conv_feat_path = path+'results/conv_da_feat.dat'
        self.conv_val_feat_path = path+'results/conv_da_val_feat.dat'

    def create_da_batches(self, path):
        # image augmentation generator
        gen_t = image.ImageDataGenerator(rotation_range=45, height_shift_range=0.2, 
            shear_range=0.3, channel_shift_range=40, width_shift_range=0.3)
        batch_size = 64
        target_size = (224, 224)
        return gen_t.flow_from_directory(path, 
            target_size = target_size,
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = False
        )
    
    def create_train_batches(self):
        return self.create_da_batches(self.path + 'train')
    
    def calc_train_conv_feats(self):
        print("(train) calculating convolution features")
        train_batches = self.create_train_batches()
        conv_feat = self.model.predict_generator(train_batches, train_batches.nb_sample * (self.data_augment_size+1))

        print("(train) saving feats to file....")
        print("(train) conv feats: %s" % (conv_feat.shape,))
        print("(train) path: %s" % self.conv_feat_path)
        
        save_array(self.conv_feat_path, conv_feat)
        return conv_feat


class DenseDAModel(DenseModel):
    def __init__(self, path, p=0.8, input_shape=(512, 14, 14), data_augment_size=3):
        dense_layers = self.dense_layers(p, input_shape)
        self.path = path
        self.model = self.dense_model(dense_layers)
        self.model_path = path + 'models/conv_da_weights.h5'
        self.preds_path = path + 'results/preds_da.h5'
        self.nb_epoch = 15
        self.data_augment_size = data_augment_size

    def get_train_labels(self):
        trn_labels = self.get_labels(self.path + 'train')
        da_trn_labels = np.concatenate([trn_labels]*(self.data_augment_size+1))
        return  da_trn_labels

def precalc():
    print("===== conv features =====")
    pcm = PrecalcDAConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.calc_conv_feats()

def train_da_model():
    print("===== loading data-augmented conv features =====")
    pcm = PrecalcDAConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()

    print("===== train dense model =====")
    dm = DenseDAModel('data/')
    dm.train(conv_feat, conv_val_feat)

def run_test():
    print("====== load test conv feats ======")
    tm = PrecalcConvTestModel('data/')
    conv_test_feat = tm.get_conv_feats()

    # run test
    print("====== load dense model ======")
    dm = DenseDAModel('data/')
    dm.load_model()
    print("====== run test ======")
    preds = dm.test(conv_test_feat)

def run_submit():
    print("======= making submission ========")
    preds = load_array('data/results/preds_da.h5/')
    test_batch = get_batches('data/test/')
    submit(preds, test_batch, 'submits/da_subm.gz')

    print("======= pushing to kaggle ========")
    push_to_kaggle('submits/da_subm.gz')

if __name__ == "__main__":
    # precalc()
    train_da_model()
    run_test()
    run_submit()