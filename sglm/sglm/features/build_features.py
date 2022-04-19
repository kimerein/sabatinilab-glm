# import os
# import sys
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
# import time
# import random

# import glob
# from sglm.features import sglm_pp
# import freely_moving_helpers as lpp
from tqdm import tqdm, trange

def get_rename_columns_by_file(files_list, channel_definitions, verbose=0):
    """
    Rename channels in a file.
    Args:
        files_list : list(str)
            List of filenames with which to associate different names for channels
        channel_definitions : dict(dict)
            Dictionary of Keys: tuple of filename identifiers (all must be matched) to Vals: dictionary mapping initial channel names to final channel names
        verbose : int
            Verbosity level
    Returns:
        channel_assignments : dict
            Dictionary of Keys: filenames with required renamings to Vals: 

    Example:
        files_list = glob.glob(f'{dir_path}/../GLM_SIGNALS_WT61_*') + \
                    glob.glob(f'{dir_path}/../GLM_SIGNALS_WT63_*') + \
                    glob.glob(f'{dir_path}/../GLM_SIGNALS_WT64_*')
        channel_definitions = {
            ('WT61',): {'Ch1': 'gACH', 'Ch2': 'rDA'},
            ('WT64',): {'Ch1': 'gACH', 'Ch2': 'empty'},
            ('WT63',): {'Ch1': 'gDA', 'Ch2': 'empty'},
        }
        channel_assignments = rename_channels(files_list, channel_definitions)

        (channel_assignments will map each individual filename to a renaming dictionary of columns)
    """
    channel_assignments = {}
    for file_lookup in channel_definitions:
        print(file_lookup)
        channel_renamings = channel_definitions[file_lookup]
        relevant_files = [f for f in files_list if all(x in f for x in file_lookup)]
        for relevant_file in relevant_files:
            relevant_file = relevant_file.split('/')[-1]
            print('>', relevant_file)
            channel_assignments[relevant_file] = channel_renamings
    
    return channel_assignments

def rename_consistent_columns(df, rename_columns={'Ch1':'Ch1',
                                                  'Ch2':'Ch2',
                                                  'Ch5':'Ch5',
                                                  'Ch6':'Ch6',

                                                  'centerOcc':'cpo',
                                                  'centerIn':'cpn',
                                                  'centerOut':'cpx',
                                                  'rightOcc':'rpo',
                                                  'rightIn':'rpn',
                                                  'rightOut':'rpx',
                                                  'rightLick':'rl',
                                                  'leftOcc':'lpo',
                                                  'leftIn':'lpn',
                                                  'leftOut':'lpx',
                                                  'leftLick':'ll',
                                                  'reward':'r',
                                                  'noreward':'nr'}):
    '''
    Simplify variable names to match the GLM

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe in which to rename columns
    rename_columns : dict
        Dictionary of old column names to rename to new column names

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with renamed columns
    '''
    # Simplify variable names
    df = df.rename(rename_columns, axis=1)
    return df



def set_port_entry_exit_rewarded_unrewarded_indicators(df):
    '''
    Set port entry, exit, and intersecting reward / non-reward indicators
    Args:
        df: dataframe with right / left port entry / exit columns and reward/no_reward indicators
    Returns:
        dataframe with right / left, rewarded / unrewarded intersection indicators
    '''
    # Identify combined reward vs. non-rewarded / left vs. right / entries vs. exits
    df = df.assign(**{
        'rpxr':df['r']*df['rpx'],
        'rpxnr':df['nr']*df['rpx'],
        'lpxr':df['r']*df['lpx'],
        'lpxnr':df['nr']*df['lpx'],

        'rpnr':df['r']*df['rpn'],
        'rpnnr':df['nr']*df['rpn'],
        'lpnr':df['r']*df['lpn'],
        'lpnnr':df['nr']*df['lpn'],

    })
    return df


def define_side_agnostic_events(df):
    '''
    Define side agnostic events
    Args:
        df: dataframe with left / right entry / exit and rewarded / unrewarded indicators
    Returns:
        dataframe with added port entry/exit, and reward indicators
    '''
    df = df.assign(**{
        'spn':df['rpn']+df['lpn'],
        'spx':df['rpx']+df['lpx'],

        'spnr':df['rpnr']+df['lpnr'],
        'spnnr':df['rpnnr']+df['lpnnr'],
        'spxr':df['rpxr']+df['lpxr'],
        'spxnr':df['rpxnr']+df['lpxnr'],

        'sl':df['rl']+df['ll'],
    })

    return df







def generate_toy_data():
    pass