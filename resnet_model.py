from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

from kaggle import submit, push_to_kaggle
from base_model import BaseModel
from precalc_feats import *
from dense_model import DenseModel

class PrecalcResnet(PrecalcFeats):
    """
    Precalculate conv features using Resnet50
    """
    def create_model(self, input_shape=(3,224,224)):
        return ResNet50(include_top=False, input_shape=input_shape)

class ResnetDenseModel(DenseModel):
    """
    Dense layer with batch norm. 
    Feed convolution features as input
    """
    def __init__(self, path, input_shape=(2048,1,1), lr=0.001):
        self.path = path
        self.model = self.dense_model(lr=lr, input_shape=input_shape)
        self.model_name = "rn_dense_lr%s" % lr
        self.model_path = path + 'models/' + self.model_name + '.h5'
        self.preds_path = path + 'results/' + self.model_name + '.h5'
        self.log_path = 'logs/' + self.model_name  + '_log.csv'
    
    def dense_model(self, lr=0.001, input_shape=(2048,1,1), dropout_p=0.5):
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dropout(dropout_p)(x)
        x = Dense(8, activation='softmax', name='fc')(x)

        model = Model(input, x)
        optimizer = RMSprop(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        return model 

class ResnetModel(BaseModel):
    """
    ResNet50 with fine-tuned dense layers
    """
    def __init__(self, path, lr=0.001, dropout_p=0.5):
        self.path = path
        self.model_name = "resnet50_lr%s" % lr
        self.model = self.create_model(lr=lr)

    def create_model(self, lr, input_shape=(3, 224, 244), dropout_p=0.5):
        # pretrained ResNet50
        resnet_model = ResNet50(include_top=False, input_shape=input_shape)
        for layer in resnet_model.layers: layer.trainable=False

        # fully connected layers
        output = resnet_model.get_layer(index=-2).output # output before average pooling
        x = AveragePooling2D((7, 7), name='avg_pool')(output)
        x = Flatten()(x)
        x = Dropout(dropout_p)(x)
        x = Dense(8, activation='softmax', name='fc8')(x)

        model = Model(resnet_model.input, x)
        optimizer = RMSprop(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        return model 

    def fine_tune(self):
        for layer in self.model.layers: layer.trainable=False
        # find index to last indentity block
        l = self.model.get_layer(name='res5a_branch2a')
        idx = self.model.layers.index(l)
        for layer in self.model.layers[idx:]: layer.trainable=True
    
