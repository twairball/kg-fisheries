from dense_model import DenseModel
from precalc_feats import *

from kaggle import *
from setup import *
from utils import save_array

## main scripts
def calc_train_da_feats():
    nb_augm = 5
    print("===== (TRAIN) Precalc data-augmented conv features =====")

    pcf = PrecalcFeats()
    for aug in range(nb_augm):
        print("===== data-aug: %d =====" % aug)
        batches = create_batches('data/train/', shuffle=False, use_da=True)
        print("    (precalc) calculating features...")
        feats = pcf.calc_feats_on_batch(batches)
        labels = to_categorical(batches.classes)

        # save
        labels_file = "data/results/da%d_conv_labels.h5" % aug
        feats_file = "data/results/da%d_conv_feats.h5" % aug
        save_array(labels_file, labels)
        save_array(feats_file, feats)
        print("    (precalc) feats: %s" % (feats.shape,))
        print("    (precalc) saved feats to: %s" % feats_file)
        print("    (precalc) saved labels to: %s" % labels_file)

def calc_val_feats():
    print("===== (VALID) Precalc validation conv features =====")
    pcf = PrecalcFeats()
    batches = create_batches('data/valid/', shuffle=False, use_da=False)
    print("    (precalc) calculating features...")
    feats = pcf.calc_feats_on_batch(batches)
    labels = to_categorical(batches.classes)

    # save
    labels_file = "data/results/conv_val_labels.h5" 
    feats_file = "data/results/conv_val_feats.h5"
    save_array(labels_file, labels)
    save_array(feats_file, feats)
    print("    (precalc) feats: %s" % (feats.shape,))
    print("    (precalc) saved feats to: %s" % feats_file)
    print("    (precalc) saved labels to: %s" % labels_file)


def calc_test_da_feats():
    nb_augm = 5
    print("===== (TEST) Precalc data-augmented conv features =====")

    pcf = PrecalcFeats()
    for aug in range(nb_augm):
        print("===== data-aug: %d =====" % aug)
        batches = create_batches('data/test/', shuffle=False, use_da=True)
        print("    (precalc) calculating features...")
        feats = pcf.calc_feats_on_batch(batches)
        labels = to_categorical(batches.classes)

        # save
        labels_file = "data/results/da%d_conv_test_labels.h5" % aug
        feats_file = "data/results/da%d_conv_test_feats.h5" % aug
        save_array(labels_file, labels)
        save_array(feats_file, feats)
        print("    (precalc) feats: %s" % (feats.shape,))
        print("    (precalc) saved feats to: %s" % feats_file)
        print("    (precalc) saved labels to: %s" % labels_file)

def train_ensemble():
    nb_models = 5 # train 5 ensemble models
    models = []
    model_paths = []

    # load validation 
    val_labels = load_array("data/results/conv_val_labels.h5")
    conv_val_feat = load_array("data/results/conv_val_feats.h5")

    for aug in range(nb_models):
        print("===== data-aug: %d =====" % aug)

        # load
        labels_file = "data/results/da%d_conv_labels.h5" % aug
        feats_file = "data/results/da%d_conv_feats.h5" % aug

        trn_labels = load_array(labels_file)
        conv_feat = load_array(feats_file)

        print("====== Ensemble model: %d ======" % aug)
        m = DenseModel('data/', dense_nodes=4096)
        model_prefix = "da_dense_r%d_" % aug 
        m.model_name = model_prefix + m.model_name

        print("====== training model ======")
        m.train(conv_feat, trn_labels, conv_val_feat, val_labels, nb_epoch=15)

        # append model 
        models = models + [m]
        model_paths = model_paths + [m.model_path]

    return models, model_paths

def test_ensemble(models):
    nb_test_samples = 1000
    nb_classes = 8
    nb_augmentations = 5
    preds = np.zeros((nb_test_samples, nb_classes))

    for test_run in range(nb_augmentations):
        # make test batch randomly with data aug
        print("====== data-aug test batch: %d ======" % test_run)
        preds_aug = np.zeros((nb_test_samples, nb_classes))
        conv_test_feat = load_array("data/results/da%d_conv_test_feats.h5" % test_run)

        for ind, m in enumerate(models): 
            print("====== running test model: %d ======" % ind)
            _preds = m.test(conv_test_feat)
            preds_aug = preds_aug + _preds
        
        preds_aug /= len(models) 
        preds = preds + preds_aug
        

    preds /= nb_augmentations
    save_array('data/results/ensemble_dense_preds.h5', preds)
    return preds 


def submit_ensemble(preds):
    # get test batch from any model 
    test_batches = create_batches('data/test/', shuffle=False, use_da=False)

    print("======= making submission ========")
    submits_path = 'submits/ens_dense_preds.gz'
    submit(preds, test_batches, submits_path)

    print("======= pushing to kaggle ========")
    push_to_kaggle(submits_path)

def load_models():
    models = []
    nb_models = 5
    for run in range(nb_models):
        print("====== Loading ensemble model: %d ======" % run)
        m = DenseModel('data/', dense_nodes=4096)
        model_prefix = "da_dense_r%d_" % aug 
        m.model_name = model_prefix + m.model_name
        print("model path: %s" % m.model_path)
        m.load_model()

        models = models + [m]
    return models

if __name__ == "__main__":
    # calc_train_da_feats()
    # calc_val_feats()
    # calc_test_da_feats()

    # models, model_paths = train_ensemble()

    models = load_models()
    preds = test_ensemble(models)
    submit_ensemble(preds)
