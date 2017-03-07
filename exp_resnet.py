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
    print("===== loading conv features =====")
    pcm = PrecalcResnet('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()

    for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
        model = ResnetDenseModel('data/', lr=lr)
        print("====== training model ======")
        print("model: %s" % model.model_name)
        model.train(conv_feat, conv_val_feat, nb_epoch=20)
        print("")
        print("")

def train_model():
    print("===== loading conv features =====")
    pcm = PrecalcResnet('data/')
    (conv_feat, conv_val_feat) = pcm.get_conv_feats()
    
    model = ResnetDenseModel('data/', lr=1e-3)
    print("====== training model ======")
    print("model: %s" % model.model_name)
    model.train(conv_feat, conv_val_feat, nb_epoch=50)
    return model

def run_test(m):
    print("====== load test conv feats ======")
    tm = PrecalcResnetTest('data/')
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
    m = train_model()
    run_test(m)
    run_submit(m)