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

from kaggle import submit, push_to_kaggle


class BaseModel():
    """
        VGG16 base model with fine-tuned dense layers
    """
    def __init__(self, path, p=0.8):
        self.path = path
        self.model = self.create_model()
        self.model_path = path + 'models/model_weights.h5'
        self.preds_path = path + 'results/model_preds.h5'
    
    def _vgg_pretrained(self):
        # VGG pretrained convolution layers
        model = Vgg16BN(include_top=False).model
        for layer in model.layers: layer.trainable=False
        return model 

    def _add_FCBlock(self, model, dropout_p=0.5):
        for layer in self.dense_layers(dropout_p):
            model.add(layer)
        return model 

    def dense_layers(self, dropout_p=0.5):
        return [
            Dense(4096, activation='relu'), 
            BatchNormalization(), 
            Dropout(dropout_p)
        ]

    def create_model(self, lr=0.00001):
        model = self._vgg_pretrained()
        model.add(Flatten())

        # add 2 sets of dense layers. 
        model = self._add_FCBlock(model)
        model = self._add_FCBlock(model)
        
        # classification layer -- 8 classes
        model.add(Dense(8, activation='softmax'))

        # compile with learning rate
        model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    ##
    ## Batches
    ##
    def create_train_batches(self):
        return self.create_batches(self.path + 'train')
    
    def create_val_batches(self):
        return self.create_batches(self.path + 'valid')

    def create_batches(self, path, shuffle=True):
        batch_size = 64
        target_size = (224, 224)
        return image.ImageDataGenerator().flow_from_directory(path, 
            target_size = target_size,
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = shuffle
        )
    
    ##
    ## Training
    ##
    def train(self, nb_epoch = 10):
        batch_size = 32
        train_batches = self.create_train_batches()
        val_batches = self.create_val_batches()

        self.model.fit_generator(train_batches, 
            samples_per_epoch = train_batches.nb_sample, 
            nb_epoch = nb_epoch, 
            validation_data = val_batches, 
            nb_val_samples = val_batches.nb_sample)

        self.model.save_weights(self.model_path)
        return self.model
    
    def load_model(self):
        self.model.load_weights(self.model_path)
    
    ##
    ## Test
    ##
    def create_test_batches(self):
        return self.create_batches(self.path + 'test')

    def test(self):
        test_batches = self.create_test_batches()
        preds = self.model.predict_generator(test_batches, test_batches.nb_sample)

        print("(test) saving predictions to file....")
        print("(test) preds: %s" % (preds.shape,))
        print("(test) path: %s" % self.preds_path)

        save_array(self.preds_path, preds)
        return (preds, test_batches)

##
## main scripts
##
if __name__ == "__main__":
    print("====== training model ======")
    m = BaseModel('data/')
    m.train()

    print("====== running test ======")
    preds, test_batches = m.test()

    print("======= making submission ========")
    submits_path = 'submits/base_model_subm.gz'
    submit(preds, test_batches, submits_path)

    print("======= pushing to kaggle ========")
    push_to_kaggle(submits_path)