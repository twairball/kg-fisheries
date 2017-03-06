from precalc_conv import *
from precalc_da_conv import *
from kaggle import *
from setup import *

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

##
## model experiments
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

def train_da_lr():
    print("===== loading conv features =====")
    pcm = PrecalcDAConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()

    for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
        model = DenseDAModel('data/', dense_nodes=512, lr=lr)
        print("====== training model ======")
        print("model: %s" % model.model_name)
        model.train(conv_feat, conv_val_feat, nb_epoch=20)
        print("")
        print("")

##
## test runs
## 
def train_precalc():
    print("===== loading conv features =====")
    pcm = PrecalcConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()

    print("===== train dense model =====")
    m = DenseModel('data/', lr=1e-5, dense_nodes=512)
    m.train(conv_feat, conv_val_feat, nb_epoch=14)

    run_test(m)
    run_submit(m)

def train_precalc_da():
    print("===== loading conv da features =====")
    pcm = PrecalcDAConvModel('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()

    print("===== train dense model =====")
    m = DenseDAModel('data/', lr=1e-5, dense_nodes=512)
    m.train(conv_feat, conv_val_feat, nb_epoch=14)


    run_test(m)
    run_submit(m)


if __name__ == "__main__":
    train_da_lr()