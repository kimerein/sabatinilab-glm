import glob
import numpy as np
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm, trange
from sglm.models import sglm
from sglm.features import table_file as tbf
from sglm.features import sglm_pp


def timeshift_vals(dfrel, X_cols, neg_order=-20, pos_order=20, exclude_columns=None):
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

    if type(exclude_columns) != type(None):
        X_cols_reduced = [_ for _ in X_cols if _ not in exclude_columns]
    else:
        X_cols_reduced = X_cols

    dfrel = sglm_pp.timeshift_cols(dfrel, X_cols_reduced, neg_order=neg_order, pos_order=pos_order)
    X_cols_sftd = sglm_pp.add_timeshifts_to_col_list(X_cols, X_cols_reduced, neg_order=neg_order, pos_order=pos_order)

    return dfrel, X_cols_sftd

def multi_file_analysis_prep(signal_files, neg_order=-20, pos_order=20, exclude_columns=None):
    '''
    
    Args:
        
    Returns:
        
    '''
    if type(exclude_columns) != type(None):
        exclude_columns = [ 'nTrial',
                            'nEndTrial',
                            'wi_trial_keep',
                            'r_trial',
                            'nr_trial',
                            ]

    signal_df_lst = []

    for file_num in trange(len(signal_files)):
        signal_fn = signal_files[file_num]
        tmp_signal_df = pd.read_csv(signal_fn, index_col='index').copy()
        tmp_signal_df['file_num'] = file_num


        tmp_signal_df, X_cols_sftd = timeshift_vals(tmp_signal_df,
                                                list(tmp_signal_df.columns),
                                                neg_order=neg_order,
                                                pos_order=pos_order,
                                                exclude_columns=exclude_columns
                                                )

        signal_df_lst.append(tmp_signal_df)
    signal_df = pd.concat(signal_df_lst, axis=0)

    signal_df['nTrial'] = signal_df['nTrial'].astype(int)
    signal_df['nEndTrial'] = signal_df['nEndTrial'].astype(int)

    max_num_trial = len(str(signal_df['nTrial'].max()))
    signal_df['nTrial_filenum'] = signal_df['nTrial'] + signal_df['file_num'] * 10**(max_num_trial)
    signal_df['nEndTrial_filenum'] = signal_df['nEndTrial'] + signal_df['file_num'] * 10**(max_num_trial)

    signal_filenames = '-'.join([_.split('/')[-1].split('.')[0] for _ in signal_files])
    
    return [signal_df], X_cols_sftd, None

def single_file_analysis_prep(signal_files, neg_order=-20, pos_order=20, exclude_columns=None):
    '''
    
    Args:
        
    Returns:
        
    '''
    if type(exclude_columns) != type(None):
        exclude_columns = [ 'nTrial',
                            'nEndTrial',
                            'wi_trial_keep',
                            'r_trial',
                            'nr_trial',
                            ]
    
    signal_df_lst = []
    signal_filenames = []

    for file_num in trange(len(signal_files)):
        signal_fn = signal_files[file_num]
        tmp_signal_df = pd.read_csv(signal_fn, index_col='index').copy()
        tmp_signal_df['file_num'] = file_num

        tmp_signal_df['nTrial'] = tmp_signal_df['nTrial'].astype(int)
        tmp_signal_df['nEndTrial'] = tmp_signal_df['nEndTrial'].astype(int)

        max_num_trial = len(str(tmp_signal_df['nTrial'].max()))
        tmp_signal_df['nTrial_filenum'] = tmp_signal_df['nTrial'] + tmp_signal_df['file_num'] * 10**(max_num_trial)
        tmp_signal_df['nEndTrial_filenum'] = tmp_signal_df['nEndTrial'] + tmp_signal_df['file_num'] * 10**(max_num_trial)


        tmp_signal_df, X_cols_sftd = timeshift_vals(tmp_signal_df,
                                                list(tmp_signal_df.columns),
                                                neg_order=neg_order,
                                                pos_order=pos_order,
                                                exclude_columns=exclude_columns
                                                )

        
        signal_df_lst.append(tmp_signal_df)
        signal_filenames.append(signal_files[file_num].split('/')[-1].split('.')[0].replace('GLM_SIGNALS_', '').replace('INTERIM_', ''))

    return signal_df_lst, X_cols_sftd, signal_filenames
