from base_model import BaseModel
from kaggle import *
from setup import *

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
    for run in range(nb_models):
        m = BaseModel('data/', dense_nodes=512)

        print("====== training model ======")
        m = BaseModel('data/', dense_nodes=512)
        m.train(nb_epoch = 20, use_da=True)

        print("====== running test ======")
        preds, test_batches = m.test(use_da=True)
        
        models = models + [m]
    return models
    
if __name__ == "__main__":
    train_ensemble()