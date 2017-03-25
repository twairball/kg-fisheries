from precalc_feats import *
from resnet_model import *

from kaggle import *
from setup import *
from utils import save_array

def train_ensemble():
    nb_models = 5 # train 5 ensemble models
    models = []
    model_paths = []

    for run in range(nb_models):
        print("====== Ensemble model: %d ======" % run)
        m = ResnetModel('data/')
        model_prefix = "resnet_ft_da%d" % run 
        m.model_name = model_prefix + m.model_name

        # finetune for last identity block
        m.fine_tune()

        print("====== training model ======")
        train_batches = create_batches('data/train/', shuffle=True, use_da=True)
        val_batches = create_batches('data/valid/', shuffle=True, use_da=True)
        # m.train(nb_epoch=20, use_da=True)
        m.train_on_batches(train_batches, val_batches, nb_epoch=20)

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
        test_batches = create_batches('data/test/', shuffle=False, use_da=True)
        preds_aug = np.zeros((nb_test_samples, nb_classes))

        for ind, m in enumerate(models): 
            print("====== running test model: %d ======" % ind)
            _preds = m.test_on_batch(test_batches)
            preds_aug = preds_aug + _preds
        
        preds_aug /= len(models) 
        preds = preds + preds_aug
        

    preds /= nb_augmentations
    save_array('submits/resnet_ft_ens_preds.gz', preds)
    return preds 

def submit_ensemble(preds):
    # get test batch from any model 
    test_batches = create_batches('data/test/', shuffle=False, use_da=False)

    print("======= making submission ========")
    submits_path = 'submits/resnet_ft_ens_preds.gz'
    submit(preds, test_batches, submits_path)

    print("======= pushing to kaggle ========")
    push_to_kaggle(submits_path)


def load_models():
    models = []
    nb_models = 5
    for run in range(nb_models):
        print("====== Loading ensemble model: %d ======" % run)
        m = ResnetModel('data/')
        model_prefix = "resnet_ft_da%d" % run 
        m.model_name = model_prefix + m.model_name
        print("model path: %s" % m.model_path)
        m.load_model()

        models = models + [m]
    return models

if __name__ == "__main__":
    models, model_paths = train_ensemble()
    preds = test_ensemble(models)
    submit_ensemble(preds)
