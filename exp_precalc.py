from precalc_conv import *
from kaggle import *
from setup import *

##
## main scripts
##

def train_dense_nodes():
    print("===== loading conv features =====")
    pcm = PrecalcConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()

    for dn in [4096, 2048]:
        model = DenseModel('data/', dense_nodes=dn, lr=0.00001)
        print("====== training model ======")
        print("model: %s" % model.model_name)
        model.train(conv_feat, conv_val_feat, nb_epoch=15)
        print("")
        print("")

def train_lr():
    print("===== loading conv features =====")
    pcm = PrecalcConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()

    for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
        model = DenseModel('data/', dense_nodes=512, lr=lr)
        print("====== training model ======")
        print("model: %s" % model.model_name)
        model.train(conv_feat, conv_val_feat, nb_epoch=20)
        print("")
        print("")

if __name__ == "__main__":
    train_lr()