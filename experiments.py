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

def train_lots():
    for dn in [4096, 2048, 1024, 512, 256, 128]:
        m = BaseModel('data/', dense_nodes=dn)
        train_one(m)

if __name__ == "__main__":
    train_lots()