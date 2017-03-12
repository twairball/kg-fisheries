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

class DensePseudoModel(DenseModel):
    def __init__(self, path, p=0.8, input_shape=(512, 14, 14)):
        dense_layers = self.dense_layers(p, input_shape)
        self.path = path
        self.model = self.dense_model(dense_layers)
        self.model_path = path + 'models/conv_psuedo_weights.h5'
        self.preds_path = path + 'results/preds_pseudo.h5'
        self.nb_epoch = 15

    def pseudo_train(self, conv_feat, conv_val_feat, pseudo_feat):
        batch_size = 32
        nb_epoch = 5
        trn_labels = self.get_train_labels()
        val_labels = self.get_val_labels()

        # predict pseudo features and combine with training
        pseudo_labels = self.model.predict(pseudo_feat, batch_size=batch_size)
        comb_pseudo = np.concatenate([trn_labels, pseudo_labels])
        comb_feat = np.concatenate([conv_feat, pseudo_feat])
        print("(pseudo) labels: %d" % comb_pseudo.shape[0])
        print("(pseudo) combined features: %s" %(comb_feat.shape,))

        # fine tune with model
        self.model.fit(comb_feat, comb_pseudo, batch_size=batch_size, nb_epoch=nb_epoch,
            validation_data=(conv_val_feat, val_labels))
        self.model.save_weights(self.model_path)

def train_model():
    print("===== loading conv features =====")
    pcm = PrecalcConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()

    print("===== train dense model =====")
    dm = DensePseudoModel('data/')
    dm.train(conv_feat, conv_val_feat)

    print("===== Fine-tuning with pseudo labels on validation set =====")
    dm.pseudo_train(conv_feat, conv_val_feat, conv_val_feat)

    print("===== Fine-tuning with pseudo labels on test set =====")
    tm = PrecalcConvTestModel('data/')
    conv_test_feat = tm.get_conv_feats()
    dm.pseudo_train(conv_feat, conv_val_feat, conv_test_feat)


def run_test():
    print("====== load test conv feats ======")
    tm = PrecalcConvTestModel('data/')
    conv_test_feat = tm.get_conv_feats()

    # run test
    print("====== load dense model ======")
    dm = DensePseudoModel('data/')
    dm.load_model()
    print("====== run test ======")
    preds = dm.test(conv_test_feat)

def run_submit():
    print("======= making submission ========")
    preds = load_array('data/results/preds_pseudo.h5/')
    test_batch = get_batches('data/test/')
    submit(preds, test_batch, 'submits/pseudo_subm.gz')

    print("======= pushing to kaggle ========")
    push_to_kaggle('submits/pseudo_subm.gz')

if __name__ == "__main__":
    train_model()
    run_test()
    run_submit()