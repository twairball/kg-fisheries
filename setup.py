import pandas as pd
import numpy as np
from glob import glob
import os, json, sys
from shutil import copyfile

import errno
def mkdir_p(path):
    """ 'mkdir -p' in Python """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def setup_folders(root_path):
    for folder in ['valid', 'results', 'models', 
                   'sample/train', 'sample/test', 
                   'sample/valid', 'sample/results', 'sample/models']:
        folder_path = root_path + '/' + folder
        mkdir_p(folder_path)

    # make subfolders for each class
    for c in classnames():
        mkdir_p(root_path + '/sample/train/' + c)
        mkdir_p(root_path + '/sample/valid/' + c)
        mkdir_p(root_path + '/valid/' + c)

def df_dataset(dir_path):
    # get training data stats
    df = pd.DataFrame(glob(dir_path + '/*'), columns=['dir'])
    class_names = df['dir'].str.extract(dir_path + '/(?P<classname>\w+)')
    df = df.join(class_names)
    count = df['dir'].apply(lambda x: len(glob(x+'/*'))).rename('count')
    df = df.join(count)
    return df

def train_data_stats():
    # get training data stats
    return df_dataset('data/train')

def classnames():
    return ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
def df_draw_random(from_dir, sample_size):
    # draw random sample from dir
    files = glob(from_dir+'/*')
    sample = np.random.permutation(files)[0:sample_size-1]
    return pd.DataFrame(sample, columns=['filepath'])

"""
    Validation dataset
"""
def df_validation(row):
    # draw random sample for validation data
    validation_size = int(row['count'] * 0.15)
    validation_dir = 'data/valid/'
    
    df = df_draw_random(row['dir'], validation_size)
    # filename
    filename = df['filepath'].str.extract('(?P<filename>img.*jpg)$')
    df = df.join(filename)
    # destination filepath
    df['dest_filepath'] = validation_dir + row['classname'] + '/' + filename    
    return df

def move_valid(row):
    df_v = df_validation(row)
    df_v.apply(lambda x: os.rename(x['filepath'], x['dest_filepath']), axis=1)
    
def make_validation(df):    
    df.apply(lambda row: move_valid(row), axis=1)
    

"""
    Sample dataset
"""

def df_sample(row, sample_dir, sample_pct):
    # draw random sample for sample data
    sample_size = max(int(row['count'] * sample_pct),1)
    print("sample_size: %d" % sample_size)
    df = df_draw_random(row['dir'], sample_size)
    # filename
    filename = df['filepath'].str.extract('(?P<filename>img.*jpg)$')
    df = df.join(filename)
    # destination filepath
    df['dest_filepath'] = sample_dir + row['classname'] + '/' + filename    
    return df

def copy_sample(row, sample_dir, sample_pct):
    print("row: %s" % row['dir'])
    df_s = df_sample(row, sample_dir, sample_pct)
    df_s.apply(lambda x: copyfile(x['filepath'], x['dest_filepath']), axis=1)

def make_sample(df, sample_dir = 'data/sample/train/', sample_pct = 0.3):
    df.apply(lambda row: copy_sample(row, sample_dir, sample_pct), axis=1)

if __name__ == "__main__":
    # make validation 
    print("======= data/valid ===========")
    df = df_dataset('data/train')
    make_validation(df)

    # sample training data
    print("======= data/sample/train ===========")
    df = df_dataset('data/train')
    make_sample(df)

    # sample valid data
    print("======= data/sample/valid ===========")
    df_v = df_dataset('data/valid')
    make_sample(df_v, sample_dir = 'data/sample/valid/', sample_pct = 1.)