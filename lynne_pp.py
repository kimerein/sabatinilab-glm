import numpy as np


def define_trial_starts_ends(df, trial_shift_bounds=7):
    '''
    Define trial starts and ends.
    Args:
        df: dataframe on which to define trial starts and ends
        trial_shift_bounds: define how many timesteps before / after first / last event to include as non-ITI
    Returns:
        dataframe with added nTrial and nEndTrial columns to identify the number of the trial counts for start & end
    '''
    df['event_col_a'] = ((df['cpo'].diff() > 0)*1).replace(0, np.nan) * 1.0
    df['event_col_b'] = df['nr'].replace(0, np.nan) * 2.0
    df['event_col_c'] = df['r'].replace(0, np.nan) * 3.0
    df['event_col'] = df['event_col_a'].combine_first(df['event_col_b']).combine_first(df['event_col_c'])
    df['event_col'] = df['event_col'].bfill()
    df['trial_start_flag'] = ((df['event_col'] == 1.0)&(df['event_col'].shift(-1) != df['event_col']) * 1.0).shift(-trial_shift_bounds) * 1.0
    df['nTrial'] = df['trial_start_flag'].cumsum()
    df['event_col_d'] = ((df['lpx'] > 0)*1.0).replace(0, np.nan) * 1.0
    df['event_col_e'] = ((df['rpx'] > 0)*1.0).replace(0, np.nan) * 1.0
    df['event_col_end'] = df['event_col_d'].combine_first(df['event_col_e']).combine_first(df['trial_start_flag'].replace(0.0, np.nan)*2.0)
    df['event_col_end'] = df['event_col_end'].ffill()
    df['trial_end_flag'] = ((df['event_col_end'] == 1.0)&(df['event_col_end'].shift(1) == 2.0)&(df['event_col_end'].shift(1) != df['event_col_end'])&(df['nTrial'] > 0) * 1.0).shift(trial_shift_bounds) * 1.0
    df['nEndTrial'] = df['trial_end_flag'].cumsum()
    return df.drop(['event_col_a', 'event_col_b', 'event_col_c', 'event_col_d', 'event_col_e'], axis=1)


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
    df = df.rename({'Ch1':'resp1',
                    'Ch2':'resp2',
                    'Ch5':'resp3',
                    'Ch6':'resp4',
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
    df['nr_trial'] = (df.groupby('nTrial')['nr'].transform(np.sum) > 0) * 1.0
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
        'rpxr':df['r_trial']*df['rpx'],
        'rpxnr':df['nr_trial']*df['rpx'],
        'lpxr':df['r_trial']*df['lpx'],
        'lpxnr':df['nr_trial']*df['lpx'],

        'rpnr':df['r_trial']*df['rpn'],
        'rpnnr':df['nr_trial']*df['rpn'],
        'lpnr':df['r_trial']*df['lpn'],
        'lpnnr':df['nr_trial']*df['lpn'],

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
