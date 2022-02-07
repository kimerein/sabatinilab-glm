import os
import sys
dir_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.append(f'{dir_path}/sabatinilab-glm/backend')
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/backend')
sys.path.append(f'{dir_path}/../backend')
# sys.path.append('./backend')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
import time
import random

import sglm
import sglm_cv
import sglm_pp
import sglm_ez
import sglm_plt as splt
import lynne_pp as lpp

import cProfile


def preprocess_lynne(df):
    '''
    Preprocess dataframe for GLM
    Args:
        df: dataframe with entry, exit, lick, reward, and dFF columns
    Returns:
        dataframe with entry, exit, lick, reward, and
    '''
        df = df[[_ for _ in df.columns if 'Unnamed' not in _]]
        print(df.columns)
        df = lpp.rename_columns(df)


        tmp = sglm_pp.detrend_data(df, y_col, [], 200)
        df[y_col] = tmp
        df = df.dropna()

        df = lpp.define_trial_starts_ends(df)

        print('Percent of Data in ITI:', (df['nTrial'] == df['nEndTrial']).mean())

        df = lpp.set_reward_flags(df)
        df = lpp.set_port_entry_exit_rewarded_unrewarded_indicators(df)
        df = lpp.define_side_agnostic_events(df)

        if 'index' in df.columns:
            df = df.drop('index', axis=1)
    return



start = time.time()

files_list = [
                'Ach_rDAh_WT63_11082021.csv',
                'Ach_rDAh_WT63_11162021.csv',
                'Ach_rDAh_WT63_11182021.csv'
                ]

res = {}

for y_col in ['zsrdFF', 'zsgdFF']:

    # Loop through files to be processed
    for filename in files_list:
        df = pd.read_csv(f'{dir_path}/../{filename}')
        ___ = preprocess_lynne(df)
        
        df = drop_unnamed_columns(df)
        
        [[_ for _ in df.columns if 'Unnamed' not in _]]