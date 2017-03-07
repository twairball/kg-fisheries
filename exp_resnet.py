from resnet_model import ResnetModel, PrecalcResnet, PrecalcResnetTest, ResnetDenseModel
from kaggle import *
from setup import *

##
## main scripts
##
def precalc():
    print("===== conv features =====")
    pcm = PrecalcResnet('data/')
    print("model output shape: %s" % (pcm.model_output_shape(),))
    (conv_feat, conv_val_feat) = pcm.calc_conv_feats()

    print("===== test conv features =====")
    tm = PrecalcResnetTest('data/')
    tm.calc_conv_feats()

def pre_sample():
    print("===== conv sample features =====")
    pcm = PrecalcResnet('data/sample')
    print("model output shape: %s" % (pcm.model_output_shape(),))

def train_lr():
    for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
        model = ResnetDenseModel('data/', lr=lr)
        print("====== training model ======")
        print("model: %s" % model.model_name)
        model.train(nb_epoch=20)
        print("")
        print("")

if __name__ == "__main__":
    precalc()