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

from kaggle import submit, push_to_kaggle

class lazy_property(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    '''

    def __init__(self,fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self,obj,cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj,self.func_name,value)
        return value


class BaseModel():
    """
        VGG16 base model with fine-tuned dense layers
    """
    def __init__(self, path, lr=0.0001, dropout_p=0.5, dense_nodes=4096):
        self.path = path
        self.model = self.create_model(lr=lr, dropout_p=dropout_p, dense_nodes=dense_nodes)
        self.model_name = "model_lr%s_p%s_dn%s" % (lr, dropout_p, dense_nodes)

    @lazy_property
    def model_path(self):
        return self.path + 'models/' + self.model_name + '.h5'

    @lazy_property
    def preds_path(self):
        return self.path + 'results/' + self.model_name + '.h5'
    
    @lazy_property
    def log_path(self):
        return 'logs/' + self.model_name  + '_log.csv'

    def _vgg_pretrained(self):
        # VGG pretrained convolution layers
        model = Vgg16BN(include_top=False).model
        for layer in model.layers: layer.trainable=False
        return model 

    def _add_FCBlock(self, model, dropout_p, dense_nodes):
        for layer in self.dense_layers(dropout_p=dropout_p, dense_nodes=dense_nodes):
            model.add(layer)
        return model 

    def dense_layers(self, dropout_p, dense_nodes):
        return [
            Dense(dense_nodes, activation='relu'), 
            BatchNormalization(), 
            Dropout(dropout_p)
        ]

    def create_model(self, lr, dropout_p, dense_nodes):
        model = self._vgg_pretrained()
        model.add(Flatten())
        model.add(Dropout(0.5))

        # add 2 sets of dense layers. 
        model = self._add_FCBlock(model, 
            dropout_p=dropout_p, dense_nodes=dense_nodes)
        model = self._add_FCBlock(model, 
            dropout_p=dropout_p, dense_nodes=dense_nodes)
        
        # classification layer -- 8 classes
        model.add(Dense(8, activation='softmax'))

        # compile with learning rate
        model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])

        print("(model) lr=%s, dense_nodes=%s, dropout=%s" % (lr, dense_nodes, dropout_p))
        return model

    ##
    ## Batches
    ##
    def create_train_batches(self, use_da=False):
        return self.create_batches(self.path + 'train', use_da=use_da)
    
    def create_val_batches(self):
        # validation never uses data-augmentation
        return self.create_batches(self.path + 'valid')
    
    def create_gen(self, use_da=False):
        if use_da:
            gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.05,
                zoom_range=0.05, channel_shift_range=10, height_shift_range=0.05, 
                shear_range=0.05, horizontal_flip=True)
        else:
            gen = image.ImageDataGenerator()
        return gen 

    def create_batches(self, path, shuffle=True, use_da=False):
        batch_size = 64
        target_size = (224, 224)
        gen = self.create_gen(use_da)

        return gen.flow_from_directory(path, 
            target_size = target_size,
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = shuffle
        )
    
    ##
    ## Training
    ##
    def train(self, nb_epoch = 15, use_da=False):
        batch_size = 32
        train_batches = self.create_train_batches(use_da=use_da)
        val_batches = self.create_val_batches()

        # csv logger
        csv_logger = CSVLogger(self.log_path, separator=',', append=False)

        # save model 
        checkpointer = ModelCheckpoint(filepath=self.model_path, 
            verbose=1, save_best_only=True)

        self.model.fit_generator(train_batches, 
            samples_per_epoch = train_batches.nb_sample, 
            nb_epoch = nb_epoch, 
            validation_data = val_batches, 
            nb_val_samples = val_batches.nb_sample, 
            callbacks=[checkpointer, csv_logger])

        return self.model
    
    def load_model(self):
        self.model.load_weights(self.model_path)
    
    ##
    ## Test
    ##
    def create_test_batches(self, use_da=False):
        return self.create_batches(self.path + 'test', shuffle=False, use_da=use_da)

    def test(self, use_da=False):
        test_batches = self.create_test_batches(use_da=use_da)
        preds = self.model.predict_generator(test_batches, test_batches.nb_sample)

        print("(test) saving predictions to file....")
        print("(test) preds: %s" % (preds.shape,))
        print("(test) path: %s" % self.preds_path)

        save_array(self.preds_path, preds)
        return (preds, test_batches)
    
    def test_on_batch(self, test_batches):
        return self.model.predict_generator(test_batches, test_batches.nb_sample)


if __name__ == "__main__":
    print("====== training model ======")
    m = BaseModel('data/')
    m.train(nb_epoch = 20, use_da=True)

    print("====== running test ======")
    preds, test_batches = m.test(use_da=True)
