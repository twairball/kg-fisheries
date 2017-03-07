from keras.applications.resnet50 import ResNet50
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
from precalc_conv import PrecalcConvModel, PrecalcConvTestModel, DenseModel

class PrecalcResnet(PrecalcConvModel):
    """
    Precalculate conv features using Resnet50
    """
    def __init__(self, path):
        self.path = path
        self.model = self.conv_model()
        self.conv_feat_path = path+'results/resnet_conv_feat.dat'
        self.conv_val_feat_path = path+'results/resnet_conv_val_feat.dat'

    def conv_model(self):
        return ResNet50(include_top=False)

class PrecalcResnetTest(PrecalcConvTestModel):
    """
    Precalculate test convolution features 
    """
    def __init__(self, path):
        self.path = path
        self.model = self.conv_model()
        self.conv_test_feat_path = self.path+'results/resnet_conv_test_feat.dat'
    
    def conv_model(self):
        return ResNet50(include_top=False)

class ResnetDenseModel(DenseModel):
    """
    Dense layer with batch norm. 
    Feed convolution features as input
    """
    def __init__(self, path, input_shape=(2048,1,1), lr=1e-4, momentum=0.9, decay=0.0):
        self.path = path
        self.model = self.dense_model(lr=lr, 
            momentum=momentum, 
            decay=decay, 
            input_shape=input_shape)
        self.model_name = "rn_dense_lr%s_mm%s_dc%s" % (lr, momentum, decay)
        self.model_path = path + 'models/' + self.model_name + '.h5'
        self.preds_path = path + 'results/' + self.model_name + '.h5'
        self.log_path = 'logs/' + self.model_name  + '_log.csv'
    
    def dense_model(self, lr, momentum, decay, input_shape=(2048,1,1)):
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(8, activation='softmax', name='fc')(x)
        model = Model(input, x)
        return model 

class ResnetModel(BaseModel):
    """
    ResNet50 with fine-tuned dense layers
    """
    def __init__(self, path, lr=1e-4, momentum=0.9, decay=0.0):
        self.model_name = "resnet50_lr%s_mm%s_dc%s" % (lr, momentum, decay)
        self.model_path = path + 'models/' + self.model_name + '.h5'
        self.preds_path = path + 'results/' + self.model_name + '.h5'
        self.log_path = 'logs/' + self.model_name  + '_log.csv'
        self.model = self.create_model(lr=lr, momentum=momentum, decay=decay)

    
    def create_model(self, lr, momentum, decay):
        # pretrained ResNet50
        resnet_model = ResNet50(include_top=False)
        for layer in resnet_model.layers: layer.trainable=False

        # fully connected layers
        x = resnet_model.get_layer(index=-1).output
        x = Flatten()(x)
        x = Dense(8, activation='softmax', name='fc8')(x)

        model = Model(resnet_model.input, x)
        optimizer = SGD(lr=lr, momentum = momentum, decay = decay, nesterov = True)
        model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        return model 

if __name__ == "__main__":
    for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
        model = ResnetModel('data/', lr=lr)
        print("====== training model ======")
        print("model: %s" % model.model_name)
        model.train(nb_epoch=20)
        print("")
        print("")