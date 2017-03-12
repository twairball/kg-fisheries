from __future__ import division,print_function

import numpy as np
import pandas as pd

#import modules
from utils import *
from vgg16 import Vgg16
# from vgg16bn import Vgg16BN as Vgg16

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.utils.np_utils import to_categorical

from base_model import lazy_property

class DenseModel():
    """
        Dense layer with batch norm. 
        Feed convolution features as input
    """
    def __init__(self, path, input_shape=(512, 14, 14), lr=0.0001, dropout_p=0.5, dense_nodes=512):
        self.path = path
        self.model = self.create_model(input_shape=input_shape, lr=lr, dropout_p=dropout_p, dense_nodes=dense_nodes)
        self.model_name = "precalc_lr%s_p%s_dn%s" % (lr, dropout_p, dense_nodes)

    @lazy_property
    def model_path(self):
        return self.path + 'models/' + self.model_name + '.h5'

    @lazy_property
    def preds_path(self):
        return self.path + 'results/' + self.model_name + '.h5'
    
    @lazy_property
    def log_path(self):
        return 'logs/' + self.model_name  + '_log.csv'

    def dense_layers(self, input_shape=(512, 14, 14), dropout_p=0.5, dense_nodes=512):
        return [
            MaxPooling2D(input_shape=input_shape), 
            Flatten(),
            Dropout(dropout_p),
            Dense(dense_nodes, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_p),
            Dense(dense_nodes, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_p),
            Dense(8, activation='softmax')
            ]
    
    def create_model(self, lr=1e-4, input_shape=(512, 14, 14), dropout_p=0.5, dense_nodes=512):
        layers = self.dense_layers(input_shape=input_shape, dense_nodes=dense_nodes, dropout_p=dropout_p)
        model = Sequential(layers)
        optimizer = Adam(lr=lr)
        # optimizer = RMSprop(lr=0.00001, rho=0.7)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_model(self):
        self.model.load_weights(self.model_path)

    def train(self, conv_feat, trn_labels, conv_val_feat, val_labels, nb_epoch=15):
        batch_size = 32
        
        # csv logger
        csv_logger = CSVLogger(self.log_path, separator=',', append=False)

        # save model 
        checkpointer = ModelCheckpoint(filepath=self.model_path, 
            verbose=1, save_best_only=True)

        self.model.fit(conv_feat, trn_labels, 
            batch_size=batch_size, 
            nb_epoch=nb_epoch,
            validation_data=(conv_val_feat, val_labels), 
            callbacks=[csv_logger, checkpointer])
        self.model.save_weights(self.model_path)
        
    def test(self, conv_test_feat):
        batch_size = 32
        preds = self.model.predict(conv_test_feat, batch_size=batch_size)

        print("(test) saving predictions to file....")
        print("(test) preds: %s" % (preds.shape,))
        print("(test) path: %s" % self.preds_path)

        save_array(self.preds_path, preds)
        return preds
