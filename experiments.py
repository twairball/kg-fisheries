from base_model import BaseModel
from kaggle import *
from setup import *
from utils import save_array

##
## main scripts
##
def test_and_submit():
    print("====== loading model ======")
    m = BaseModel('data/')
    m.load_model()

    print("====== running test ======")
    preds, test_batches = m.test()

    print("======= making submission ========")
    submits_path = 'submits/base_model_subm.gz'
    submit(preds, test_batches, submits_path)

    print("======= pushing to kaggle ========")
    push_to_kaggle(submits_path)

def train_one(model):
    print("====== training model ======")
    print("model: %s" % model.model_name)
    model.train()
    print("====== running test ======")
    model.test()
    print("")
    print("")

def train_dense_nodes():
    for dn in [4096, 2048, 1024, 512, 256, 128]:
        m = BaseModel('data/', dense_nodes=dn)
        train_one(m)

def train_ensemble():
    nb_models = 5 # train 5 ensemble models
    models = []
    model_paths = []

    for run in range(nb_models):
        print("====== Ensemble model: %d ======" % run)
        m = BaseModel('data/', dense_nodes=4096)
        model_prefix = "da_r%d_" % run 
        m.model_name = model_prefix + m.model_name

        print("====== training model ======")
        m.train(nb_epoch = 20, use_da=True)

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
        test_batches = models[0].create_test_batches(use_da=True)
        preds_aug = np.zeros((nb_test_samples, nb_classes))

        for ind, m in enumerate(models): 
            print("====== running test model: %d ======" % ind)
            _preds = m.test_on_batch(test_batches)
            preds_aug = preds_aug + _preds
        
        preds_aug /= len(models) 
        preds = preds + preds_aug
        

    preds /= nb_augmentations
    save_array('data/results/ensemble_dn512_ep20_da_test_preds.h5', preds)
    return preds 


def submit_ensemble(preds):
    # get test batch from any model 
    test_batches = BaseModel('data/').create_test_batches()    

    print("======= making submission ========")
    submits_path = 'submits/ens_dn4096_ep20_da_subm.gz'
    submit(preds, test_batches, submits_path)

    print("======= pushing to kaggle ========")
    push_to_kaggle(submits_path)

def load_models():
    models = []
    nb_models = 5
    for run in range(nb_models):
        print("====== Loading ensemble model: %d ======" % run)
        m = BaseModel('data/', dense_nodes=4096)
        model_prefix = "da_r%d_" % run 
        m.model_name = model_prefix + m.model_name
        print("model path: %s" % m.model_path)
        m.load_model()

        models = models + [m]

    return models

if __name__ == "__main__":
    # models, model_paths = train_ensemble()
    models = load_models()
    preds = test_ensemble(models)
    submit_ensemble(preds)

    