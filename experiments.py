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
    nb_test_samples = 1000
    nb_classes = 8
    preds = np.zeros((nb_test_samples, nb_classes))

    for run in range(nb_models):
        print("====== Ensemble model: %d ======" % run)
        m = BaseModel('data/', dense_nodes=512)
        model_prefix = "da_r%d_" % run 
        m.model_name = model_prefix + m.model_name

        print("====== training model ======")
        m = BaseModel('data/', dense_nodes=512)
        m.train(nb_epoch = 20, use_da=True)

        print("====== running test ======")
        _preds, _test_batches = m.test(use_da=True)

        # append predictions
        preds = preds + _preds

        # append model 
        models = models + [m]
    
    # average
    preds /= nb_models
    save_array('data/results/ensemble_dn512_ep20_da_preds.h5', preds)

    return preds, models
    
if __name__ == "__main__":
    preds, models = train_ensemble()

    del models 

    # get test batch from any model 
    test_batches = BaseModel('data/').create_test_batches()    

    print("======= making submission ========")
    submits_path = 'submits/ens_dn512_ep20_da_subm.gz'
    submit(preds, test_batches, submits_path)

    print("======= pushing to kaggle ========")
    push_to_kaggle(submits_path)
    