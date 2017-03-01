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
from precalc_conv import *

class PrecalcPseudoModel(PrecalcConvModel):
    """
        Precalculate convolution features + pseudo labelling
    """
    def __init__(self, path):
        self.path = path
        self.model = self.conv_model()
        self.conv_feat_path = path+'results/conv_pseudo_feat.dat'
        self.conv_val_feat_path = path+'results/conv_pseudo_val_feat.dat'

    def create_pseudo_batches(self, path):
        # predict validation features and combine with training
        # val_pseudo = bn_model.predict(conv_val_feat, batch_size=batch_size)
        # comb_pseudo = np.concatenate([da_trn_labels, val_pseudo])
        # comb_feat = np.concatenate([da_conv_feat, conv_val_feat])

        # fine tune with model
        # bn_model.load_weights(path+'models/da_conv8_1.h5')
        # bn_model.fit(comb_feat, comb_pseudo, batch_size=batch_size, nb_epoch=1, 
            #  validation_data=(conv_val_feat, val_labels))
        pass
    
    def create_train_batches(self):
        return self.create_da_batches(self.path + 'train')
    
        def calc_train_conv_feats(self):
        print("(train) calculating convolution features")
        train_batches = self.create_train_batches()
        conv_feat = self.model.predict_generator(train_batches, train_batches.nb_sample * self.data_augment_size)

        print("(train) saving feats to file....")
        print("(train) conv feats: %s" % (conv_feat.shape,))
        print("(train) path: %s" % self.conv_feat_path)
        
        save_array(self.conv_feat_path, conv_feat)
        return conv_feat


class DensePseudoMode(DenseModel):
    def __init__(self, path, p=0.8, input_shape=(512, 14, 14), data_augment_size=3):
        dense_layers = self.dense_layers(p, input_shape)
        self.path = path
        self.model = self.dense_model(dense_layers)
        self.model_path = path + 'models/conv_da_weights.h5'
        self.preds_path = path + 'results/preds_da.h5'
        self.data_augment_size = data_augment_size

    def get_train_labels(self):
        return self.get_labels(self.path + 'train') * self.data_augment_size

if __name__ == "__main__":
    print("======= pseudo labeling =======")