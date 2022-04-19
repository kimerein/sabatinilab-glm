import os
import sys
dir_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.append(f'{dir_path}/sabatinilab-glm/backend')
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/backend')
sys.path.append(f'{dir_path}/../backend')

import time
import numpy as np
import pandas as pd

from sglm.models import sglm
from sglm.models import sglm_cv
from sglm.features import sglm_pp
from sglm.visualization import sglm_plt as splt
from sglm.data import save_results as ssave


def define_trial_starts_ends(df, trial_shift_bounds=7):
    '''
    Define trial starts and ends.
    Args:
        df: dataframe on which to define trial starts and ends
        trial_shift_bounds: define how many timesteps before / after first / last event to include as non-ITI
    Returns:
        dataframe with added nTrial and nEndTrial columns to identify the number of the trial counts for start & end
    '''
    df['event_col_a'] = df['cpn'].replace(0, np.nan) * 1.0
    df['event_col_b'] = df['lpx'].replace(0, np.nan) * 2.0
    df['event_col_c'] = df['rpx'].replace(0, np.nan) * 2.0

    df['event_col'] = df['event_col_a'].combine_first(df['event_col_b']).combine_first(df['event_col_c'])
    df['event_col'] = df['event_col'].bfill()
    df['trial_start_flag'] = ((df['event_col'] == 1.0)&(df['event_col'].shift(-1) != 1.0)).shift(-trial_shift_bounds) * 1.0

    df['nTrial'] = df['trial_start_flag'].cumsum()

    df['event_col_end'] = df['event_col_b'].combine_first(df['event_col_c']).combine_first(df['trial_start_flag'].replace(0.0, np.nan))
    df['event_col_end'] = df['event_col_end'].ffill()
    df['trial_end_flag'] = ((df['event_col_end'] == 2.0)&(df['event_col_end'].shift(1) != 2.0)&(df['nTrial'] > 0)).shift(trial_shift_bounds) * 1.0
    df['nEndTrial'] = df['trial_end_flag'].cumsum()

    return df.drop(['event_col_a', 'event_col_b', 'event_col_c'], axis=1)


def rename_columns(df):
    '''
    Simplify variable names to match the GLM
    Args:
        df: dataframe with entry, exit, lick, reward, and dFF columns
    Returns:
        dataframe with renamed columns
    '''
    # # Simplify variable names
    # df = df.rename({'center port occupancy': 'cpo',
    #                 'center port entry': 'cpn',
    #                 'center port exit': 'cpx',

    #                 'left port occupancy': 'lpo',
    #                 'left port entry': 'lpn',
    #                 'left port exit': 'lpx',
    #                 'left licks': 'll',

    #                 'right port occupancy': 'rpo',
    #                 'right port entry': 'rpn',
    #                 'right port exit': 'rpx',
    #                 'right licks': 'rl',

    #                 'no reward': 'nr',
    #                 'reward': 'r',


    #                 'dF/F green (Ach3.0)': 'gdFF',
    #                 'zscored green (Ach3.0)': 'zsgdFF',

    #                 'dF/F green (dLight1.1)': 'gdFF',
    #                 'zscored green (dLight1.1)': 'zsgdFF',

    #                 'dF/F green (dlight1.1)': 'gdFF',
    #                 'zscored green (dlight1.1)': 'zsgdFF',

    #                 'dF/F (dlight1.1)': 'gdFF',
    #                 'zscore dF/F (dlight)': 'zsgdFF',

    #                 'zscore dF/F (Ach)': 'zsgdFF',
    #                 'zscore dF/F (Ach3.0)': 'zsgdFF',

    #                 'zscore dF/F (rGRAB-DA)' : 'zsrdFF',
    #                 }, axis=1)
    # Simplify variable names
    df = df.rename({'Ch1':'Ch1',
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
                    'noreward':'nr'}, axis=1)
    return df


def set_reward_flags(df):
    '''
    Set reward flags
    Args:
        df: dataframe with nTrial, r, and nr columns
    Returns:
        dataframe with added rewarded trial and not rewarded trial columns
    '''
    # Identify rewarded vs. unrewarded trials
    df['r_trial'] = (df.groupby('nTrial')['r'].transform(np.sum) > 0) * 1.0
    # df['nr_trial'] = (df.groupby('nTrial')['nr'].transform(np.sum) > 0) * 1.0
    df['nr_trial'] = (df.groupby('nTrial')['r'].transform(np.sum) <= 0) * 1.0
    return df

def get_first_time_events(dfrel):
    '''
    Returns a list of first time events
    Args:
        dfrel: dataframe with entry, exit, reward, non-reward columns
    Returns:
        first_time_events: list of first time events
    '''
    
    dfrel['nn'] = dfrel[['lpn', 'rpn']].sum(axis=1)
    dfrel['xx'] = dfrel[['lpx', 'rpx']].sum(axis=1)

    first_trans = dfrel.groupby('nTrial')[['nn', 'xx', 'lpn', 'rpn', 'spn', 'lpx', 'rpx', 'spx', 'cpn']].cumsum()
    first_trans = ((first_trans == 1)*1).diff()
    first_trans *= first_trans >= 0
    first_trans['lpn'] = dfrel['nn']*dfrel['lpn']
    first_trans['rpn'] = dfrel['nn']*dfrel['rpn']
    first_trans['spn'] = dfrel['nn']*dfrel['spn']
    first_trans['lpx'] = dfrel['xx']*dfrel['lpx']
    first_trans['rpx'] = dfrel['xx']*dfrel['rpx']
    first_trans['spx'] = dfrel['xx']*dfrel['spx']

    first_trans = first_trans.rename({_k:f'ft_{_k}' for _k in first_trans.columns}, axis=1)
    dfrel[first_trans.columns] = first_trans

    dfrel['ft_r_rpn'] = dfrel['ft_rpn'] * dfrel['r']
    dfrel['ft_r_lpn'] = dfrel['ft_lpn'] * dfrel['r']
    dfrel['ft_r_spn'] = dfrel['ft_spn'] * dfrel['r']
    dfrel['ft_nr_rpn'] = dfrel['ft_rpn'] * dfrel['nr']
    dfrel['ft_nr_lpn'] = dfrel['ft_lpn'] * dfrel['nr']
    dfrel['ft_nr_spn'] = dfrel['ft_spn'] * dfrel['nr']


    return dfrel

def preprocess_lynne(df, trial_shift_bounds=7):
    '''
    Preprocess Lynne's dataframe for GLM
    Args:
        df: dataframe with entry, exit, lick, reward, and dFF columns
    Returns:
        dataframe with entry, exit, lick, reward, and
    '''
    df = df[[_ for _ in df.columns if 'Unnamed' not in _]]
    # print(df.columns)
    df = rename_columns(df)
    # print(df.columns)
    df = define_trial_starts_ends(df, trial_shift_bounds=trial_shift_bounds)

    print('Percent of Data in ITI:', (df['nTrial'] == df['nEndTrial']).mean())

    # print(df)

    df = set_reward_flags(df)
    df = set_port_entry_exit_rewarded_unrewarded_indicators(df)
    df = define_side_agnostic_events(df)

    if 'index' in df.columns:
        df = df.drop('index', axis=1)
    
    dfrel = df.copy()
    dfrel = dfrel.replace('False', 0).astype(float)
    dfrel = dfrel*1
    # dfrel = overwrite_response_with_toy(dfrel)

    dfrel = dfrel[[_ for _ in dfrel.columns if 'Unnamed' not in _]]
    dfrel = get_first_time_events(dfrel)
    return dfrel

def detrend(df, y_col):
    tmp = sglm_pp.detrend_data(df, y_col, [], 200)
    df[y_col] = tmp
    df = df.dropna()
    return df

def timeshift_vals(dfrel, X_cols, neg_order=-7, pos_order=20):
    '''
    Timeshift values
    Args:
        dfrel: full dataframe
        X_cols: list of columns to shift
        neg_order: negative order of the timeshift
        pos_order: positive order of the timeshift
    Returns:
        dfrel: dataframe with additional timeshifted columns
        X_cols_sftd: list of shifted columns
    '''
    dfrel = sglm_pp.timeshift_cols(dfrel, X_cols[1:], neg_order=neg_order, pos_order=pos_order)
    X_cols_sftd = sglm_pp.add_timeshifts_to_col_list(X_cols, X_cols[1:], neg_order=neg_order, pos_order=pos_order)
    return dfrel, X_cols_sftd


def get_first_entry_time(tmp):
    '''
    Get first entry time
    Args:
        tmp: dataframe with ITI removed, and first_time (ft_rpn / ft_lpn / ft_cpn) columns defined
    Returns:
        dataframe with added time_adjusted columns releatvive to first entry
    '''
    # Get first entry time
    tmp['1'] = 1
    tmp['tim'] = tmp.groupby('nTrial')['1'].cumsum()

    entry_timing_r = tmp.groupby('nTrial')['ft_rpn'].transform(lambda x: x.argmax()).astype(int)
    entry_timing_l = tmp.groupby('nTrial')['ft_lpn'].transform(lambda x: x.argmax()).astype(int)
    entry_timing = (entry_timing_r > entry_timing_l)*entry_timing_r + (entry_timing_r < entry_timing_l)*entry_timing_l

    adjusted_time = (tmp['tim'] - entry_timing)
    tmp['adjusted_time'] = adjusted_time
    adjusted_time.index = tmp.index

    entry_timing_c = tmp.groupby('nTrial')['ft_cpn'].transform(lambda x: x.argmax()).astype(int)
    adjusted_time_c = (tmp['tim'] - entry_timing_c)
    adjusted_time_c.index = tmp.index
    tmp['cpn_adjusted_time'] = adjusted_time_c
    return tmp

if __name__ == '__main__':
    df = pd.read_csv('/Users/josh/Documents/Harvard/GLM/GLM_SIGNALS_WT68_12152021.txt')
    df = df[[_ for _ in df.columns if 'Unnamed' not in _]]
    print(df.columns)
    df = rename_columns(df)
    print(df.columns)
    df = define_trial_starts_ends(df, trial_shift_bounds=1)
    print(df)