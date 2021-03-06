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

from base_model import lazy_property

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

    def calc_train_conv_feats(self):
        print("(train) calculating convolution features")
        train_batches = self.create_train_batches()
        conv_feat = self.model.predict_generator(train_batches, train_batches.nb_sample)

        print("(train) saving feats to file....")
        print("(train) conv feats: %s" % (conv_feat.shape,))
        print("(train) path: %s" % self.conv_feat_path)
        
        save_array(self.conv_feat_path, conv_feat)
        return conv_feat

    def calc_val_conv_feats(self):
        print("(valid) calculating convolution features")
        val_batches = self.create_val_batches()
        conv_val_feat = self.model.predict_generator(val_batches, val_batches.nb_sample)

        print("(valid) saving feats to file....")
        print("(valid) conv feats: %s" % (conv_val_feat.shape,))
        print("(valid) path: %s" % self.conv_val_feat_path)

        save_array(self.conv_val_feat_path, conv_val_feat)
        return conv_val_feat

    def calc_conv_feats(self):
        conv_feat = self.calc_train_conv_feats()
        conv_val_feat = self.calc_val_conv_feats()
        return (conv_feat, conv_val_feat)
    
    def load_conv_feats(self):
        print("loading convolution features from file...")
        conv_feat = load_array(self.conv_feat_path)
        conv_val_feat = load_array(self.conv_val_feat_path)
        return (conv_feat, conv_val_feat)

class PrecalcConvTestModel(PrecalcConvModel):
    """
        Precalculate test convolution features 
    """
    def __init__(self, path):
        self.path = path
        self.model = self.conv_model()
        self.conv_test_feat_path = self.path+'results/conv_test_feat.dat'

    def get_conv_feats(self):
        if os.path.isdir(self.conv_test_feat_path):
            return self.load_conv_feats()
        else:
            return self.calc_conv_feats()

    def calc_conv_feats(self):
        print("(test) calculating convolution features")
        test_batches = self.create_batches(self.path+'test')
        conv_test_feat = self.model.predict_generator(test_batches, test_batches.nb_sample)

        print("(test) saving feats to file....")
        print("(test) conv feats: %s" % (conv_test_feat.shape,))
        print("(test) path: %s" % self.conv_test_feat_path)

        save_array(self.conv_test_feat_path, conv_test_feat)
        return conv_test_feat

    def load_conv_feats(self):
        print("(test) loading convolution features from file...")
        conv_test_feat = load_array(self.conv_test_feat_path)
        return conv_test_feat

class DenseModel():
    """
        Dense layer with batch norm. 
        Feed convolution features as input
    """
    def __init__(self, path, input_shape=(512, 14, 14), lr=0.0001, dropout_p=0.5, dense_nodes=512):
        dense_layers = self.dense_layers(input_shape=input_shape, dropout_p=dropout_p, dense_nodes=dense_nodes)
        self.path = path
        self.model = self.dense_model(dense_layers)
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
    
    def dense_model(self, layers, lr=0.0001, dropout_p=0.5, dense_nodes=512):
        model = Sequential(layers)
        optimizer = Adam(lr=lr)
        # optimizer = RMSprop(lr=0.00001, rho=0.7)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def get_labels(self, dir_path):
        # get labels for dir
        gen = image.ImageDataGenerator()
        batch = gen.flow_from_directory(dir_path, target_size=(224, 224), batch_size=1, class_mode='categorical', shuffle=False)
        labels = to_categorical(batch.classes)
        return labels
    
    def get_train_labels(self):
        return self.get_labels(self.path + 'train')
    
    def get_val_labels(self):
        return self.get_labels(self.path + 'valid')

    def train(self, conv_feat, conv_val_feat, nb_epoch=15):
        batch_size = 32
        trn_labels = self.get_train_labels()
        val_labels = self.get_val_labels()
        
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
    
    def load_model(self):
        self.model.load_weights(self.model_path)
    
    def test(self, conv_test_feat):
        batch_size = 32
        preds = self.model.predict(conv_test_feat, batch_size=batch_size)

        print("(test) saving predictions to file....")
        print("(test) preds: %s" % (preds.shape,))
        print("(test) path: %s" % self.preds_path)

        save_array(self.preds_path, preds)
        return preds

# main scripts
def precalc():
    print("===== conv features =====")
    pcm = PrecalcConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.calc_conv_feats()

    print("===== test conv features =====")
    tm = PrecalcConvTestModel('data/')
    tm.calc_conv_feats()

def train_model(m):
    print("===== loading conv features =====")
    pcm = PrecalcConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()

    print("===== train dense model =====")
    m.train(conv_feat, conv_val_feat, nb_epoch=14)

def run_test(m):
    print("====== load test conv feats ======")
    tm = PrecalcConvTestModel('data/')
    conv_test_feat = tm.get_conv_feats()

    print("====== run test ======")
    preds = m.test(conv_test_feat)

def run_submit(m):
    print("======= making submission ========")
    preds = load_array(m.preds_path)
    test_batch = get_batches('data/test/')
    submit_path = 'submits/' + m.model_name + '_subm.gz'
    submit(preds, test_batch, submit_path)

    print("======= pushing to kaggle ========")
    push_to_kaggle(submit_path)

if __name__ == "__main__":
    m = DenseModel('data/', lr=1e-5, dense_nodes=512)
    m.load_model()
    # train_model(m)
    run_test(m)
    run_submit(m)
