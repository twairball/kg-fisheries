from __future__ import division,print_function

import pandas as pd
import numpy as np
import os, json, sys
import os.path
from glob import glob
from utils import *


"""
    Util methods for submitting to kaggle
"""
def classnames():
    return ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def submit(preds, test_batches, filepath):
    def do_clip(arr, mx): 
        return np.clip(arr, (1-mx)/9, mx)

    def img_names(filenames):
        df = pd.DataFrame(filenames, columns=['image'])
        df.loc[:, 'image'] = df['image'].str.replace('unknown/', '')
        return df

    def preprocess(subm):
        # make classes
        classes = classnames()
        submission = pd.DataFrame(subm, columns=classes)
        return submission
    
    # make submission dataframe
    df_img_names = img_names(test_batches.filenames)
    subm = do_clip(preds,0.98)
    submission = preprocess(subm)
    submission = pd.concat([df_img_names, submission], axis=1)

    print(submission.head())
    print("saving to csv: " + filepath)
    submission.to_csv(filepath, index=False, compression='gzip')


def push_to_kaggle(filepath):
    command = "kg submit -c the-nature-conservancy-fisheries-monitoring " + filepath
    os.system(command)

if __name__ == "__main__":
    print("======= making submission ========")
    preds = load_array('data/results/preds.h5/')
    test_batch = get_batches('data/test/')
    submit(preds, test_batch, 'subm.gz')

    print("======= pushing to kaggle ========")
    push_to_kaggle('subm.gz')