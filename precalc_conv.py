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


class PrecalcConvModel():
    """
        Precalculate convolution features using pretrained VGG
    """
    def __init__(self, path):
        self.path = path
        self.model = self.conv_model()
        self.conv_feat_path = path+'results/conv_feat.dat'
        self.conv_val_feat_path = path+'results/conv_val_feat.dat'

    ##
    ## Build convolution layers with VGG
    ## We can use pre-trained VGG conv layers and predict output, and use as input to dense layer. 
    def conv_model(self):
        vgg = Vgg16()
        model=vgg.model
        last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D][-1]
        conv_layers = model.layers[:last_conv_idx+1]
        return Sequential(conv_layers)

    def model_output_shape(self):
        return self.model.layers[-1].output_shape[1:]

    def create_batches(self, path):
        batch_size = 64
        target_size = (224, 224)
        return image.ImageDataGenerator().flow_from_directory(path, 
            target_size = target_size,
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = False
        )
    def create_train_batches(self):
        return self.create_batches(self.path + 'train')
    
    def create_val_batches(self):
        return self.create_batches(self.path + 'valid')

    def get_conv_feats(self):
        if os.path.isdir(self.conv_feat_path) & os.path.isdir(self.conv_val_feat_path):
            return self.load_conv_feats()
        else:
            return self.calc_conv_feats()

    def calc_conv_feats(self):
        print("(train) calculating convolution features")
        train_batches = self.create_train_batches()
        conv_feat = self.model.predict_generator(train_batches, train_batches.nb_sample)
        print("(train) saving feats to file....")
        save_array(self.conv_feat_path, conv_feat)
        
        print("(valid) calculating convolution features")
        val_batches = self.create_val_batches()
        conv_val_feat = self.model.predict_generator(val_batches, val_batches.nb_sample)
        print("(valid) saving feats to file....")
        save_array(self.conv_val_feat_path, conv_val_feat)

        return (conv_feat, conv_val_feat)
    
    def load_conv_feats(self):
        print("loading convolution features from file...")
        conv_feat = load_array(self.conv_feat_path)
        conv_val_feat = load_array(self.conv_val_feat_path)
        return (conv_feat, conv_val_feat)


class PrecalcDAConvModel(PrecalcConvModel):
    """
        Precalculate convolution features with data augmentation
    """
    def __init__(self, path):
        self.path = path
        self.model = self.conv_model()
        self.conv_feat_path = path+'results/conv_da_feat.dat'
        self.conv_val_feat_path = path+'results/conv_da_val_feat.dat'

    def create_da_batches(self, path):
        # image augmentation generator
        gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05, 
            shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
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

class PrecalcConvTestModel(PrecalcConvModel):
    """
        Precalculate test convolution features 
    """

    def get_conv_feats(self):
        conv_test_feat_path = path+'results/conv_test_feat.dat'
        if os.path.isdir(conv_test_feat_path):
            return self.load_conv_feats()
        else:
            return self.calc_conv_feats()

    def calc_conv_feats(self):
        print("(test) calculating convolution features")
        test_batches = self.create_batches(self.path+'test')
        conv_test_feat = self.model.predict_generator(test_batches, test_batches.nb_sample)

        print("(test) saving feats to file....")
        conv_test_feat_path = path+'results/conv_test_feat.dat'
        save_array(conv_test_feat_path, conv_test_feat)
        return conv_test_feat

    def load_conv_feats(self):
        print("(test) loading convolution features from file...")
        conv_test_feat_path = path+'results/conv_test_feat.dat'
        conv_test_feat = load_array(conv_test_feat_path)
        return conv_test_feat

class DenseModel():
    """
        Dense layer with batch norm. 
        Feed convolution features as input
    """
    def __init__(self, path, p=0.8, input_shape=(512, 14, 14)):
        dense_layers = self.dense_layers(p, input_shape)
        self.model = self.dense_model(dense_layers)
        self.model_path = path + 'models/conv_weights.h5'

    def dense_layers(self, p=0.8, input_shape=(512, 14, 14)):
        return [
            # MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
            MaxPooling2D(input_shape=input_shape), 
            Flatten(),
            Dropout(p/2),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(p/2),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(p),
            Dense(8, activation='softmax')
            ]
    
    def dense_model(self, layers):
        model = Sequential(layers)
        optimizer = Adam(lr=0.0001)
        # optimizer = RMSprop(lr=0.00001, rho=0.7)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def get_labels(self, dir):
        # get labels for dir
        gen = image.ImageDataGenerator()
        batch = gen.flow_from_directory('data/train', target_size=(224, 224), batch_size=1, class_mode='categorical', shuffle=False)
        labels = to_categorical(batch.classes)
        return labels
        
    def train(self, conv_feat, conv_val_feat):
        batch_size = 32
        nb_epoch = 10
        trn_labels = self.get_labels(self.path + 'train')
        val_labels = self.get_labels(self.path + 'valid')

        self.model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=nb_epoch,
            validation_data=(conv_val_feat, val_labels))
        self.model.save_weights(self.model_path)
        
        
if __name__ == "__main__":
    print("===== conv features =====")
    pcm = PrecalcConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.calc_conv_feats()

    del pcm

    print("===== data-augmented conv features ======")
    pdm = PrecalcDAConvModel('data/')
    pdm.calc_conv_feats()

    del pdm

    print("===== test conv features =====")
    tm = PrecalcConvTestModel('data/')
    tm.calc_conv_feats()

    del tm 

    print("===== train dense model =====")
    dm = DenseModel('data/')
    dm.train(conv_feat, conv_val_feat)

    
