import threading
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









def timeshift_vals_by_dict(df, X_cols_dict, keep_nans=False):
    '''
    Timeshift values
    Args:
        dfrel: full dataframe
    '''

    # for X_col in X_cols_dict:
    #     X_cols_lst = list(X_cols_dict.keys())
    #     col_nums = [df.columns.get_loc(_) for _ in X_cols_lst]
    #     neg_order, pos_order = X_cols_dict[X_col]
    #     dfrel = sglm_pp.timeshift_multiple(df, shift_inx=col_nums, shift_amt_list=[0]+list(range(neg_order, 0))+list(range(1, pos_order + 1)))
    #     dfrel, X_cols_sftd = timeshift_vals(df, X_cols_lst, exclude_columns=X_col)

    # # col_nums = [df.columns.get_loc(_) for _ in column_names]
    # # if len([_ for _ in col_nums if type(_) == np.ndarray]):
    # #     raise ValueError('Duplicate column found in X column names.')
    # # dfrel = timeshift_multiple(dfrel, shift_inx=col_nums, shift_amt_list=[0]+list(range(neg_order, 0))+list(range(1, pos_order + 1)))
    # # X_cols_sftd = sglm_pp.add_timeshifts_to_col_list(X_cols, X_cols_reduced, neg_order=neg_order, pos_order=pos_order)

    # # return dfrel, X_cols_sftd

    X_cols_sftd = []

    df = df.copy()
    for X_col in X_cols_dict:
        # X_cols_lst = list(X_cols_dict.keys())
        # col_nums = [df.columns.get_loc(_) for _ in X_cols_lst]
        neg_order, pos_order = X_cols_dict[X_col]
        # dfrel = sglm_pp.timeshift_multiple(df, shift_inx=col_nums, shift_amt_list=[0]+list(range(neg_order, 0))+list(range(1, pos_order + 1)))
        # dfrel, X_cols_sftd = timeshift_vals(df, X_cols_lst, exclude_columns=X_col)
        
        shift_amt_list = list(range(neg_order, pos_order + 1))
        for shift_amt in shift_amt_list:
            df[X_col + '_' + str(shift_amt)] = df[X_col].shift(shift_amt)
            X_cols_sftd += [X_col + '_' + str(shift_amt)]

    if not keep_nans:
        na_drop_cols = [X_col + '_' + str(neg_order) for X_col in X_cols_dict] + [X_col + '_' + str(pos_order) for X_col in X_cols_dict]
        
        # print(df.isna().sum(axis=0))
        
        df = df.dropna(subset=na_drop_cols)
    
    return df, X_cols_sftd

def X_cols_dict_to_default(X_cols_dict, neg_order=-20, pos_order=20):
    X_cols_dict = X_cols_dict.copy()
    for X_col in X_cols_dict:
        # print(X_cols_dict[X_col])
        if X_cols_dict[X_col] == (0,0) or X_cols_dict[X_col] is None:
            X_cols_dict[X_col] = (neg_order, pos_order)
            
    return X_cols_dict


def xy_pairs_to_widest_orders(X_y_pairings):
    """
    """
    widest_shifts = {}
    for xy_pair in X_y_pairings:
        X_dict = xy_pair['X_cols']
        for X_col in X_dict:
            neg_order, pos_order = (X_dict[X_col][0], X_dict[X_col][1])

            if X_col not in widest_shifts:
                widest_shifts[X_col] = (neg_order, pos_order)
                continue
            
            most_neg_order, most_pos_order = widest_shifts[X_col]

            if most_neg_order > neg_order:
                most_neg_order = neg_order
            if most_pos_order < pos_order:
                most_pos_order = pos_order
            
            widest_shifts[X_col] = (most_neg_order, most_pos_order)
            
    return widest_shifts




def multi_file_analysis_prep(signal_files, X_cols_dict):
    '''
    
    Args:
        
    Returns:
        
    '''
    # if type(exclude_columns) != type(None):
    #     exclude_columns = [ 'nTrial',
    #                         'nEndTrial',
    #                         'wi_trial_keep',
    #                         'r_trial',
    #                         'nr_trial',
    #                         ]

    signal_df_lst = []
    X_cols_sftd_lst = []
    for file_num in trange(len(signal_files)):
        signal_fn = signal_files[file_num]
        tmp_signal_df = pd.read_csv(signal_fn, index_col='index').copy()
        tmp_signal_df['file_num'] = file_num


        # tmp_signal_df, X_cols_sftd = timeshift_vals(tmp_signal_df,
        #                                         list(tmp_signal_df.columns),
        #                                         neg_order=neg_order,
        #                                         pos_order=pos_order,
        #                                         exclude_columns=exclude_columns
        #                                         )
        tmp_signal_df, X_cols_sftd = timeshift_vals_by_dict(tmp_signal_df, X_cols_dict)

        signal_df_lst.append(tmp_signal_df)
        X_cols_sftd_lst += [_ for _ in X_cols_sftd if _ not in X_cols_sftd_lst]
    
    signal_df = pd.concat(signal_df_lst, axis=0)


    signal_df['nTrial'] = signal_df['nTrial'].astype(int)
    signal_df['nEndTrial'] = signal_df['nEndTrial'].astype(int)

    max_num_trial = len(str(signal_df['nTrial'].max()))
    signal_df['nTrial_filenum'] = signal_df['nTrial'] + signal_df['file_num'] * 10**(max_num_trial)
    signal_df['nEndTrial_filenum'] = signal_df['nEndTrial'] + signal_df['file_num'] * 10**(max_num_trial)

    signal_filenames = '-'.join([_.split('/')[-1].split('.')[0] for _ in signal_files])
    
    return [signal_df], X_cols_sftd_lst, None

def single_file_analysis_prep(signal_files, X_cols_dict):
    '''
    
    Args:
        
    Returns:
        
    '''
    # if type(exclude_columns) != type(None):
    #     exclude_columns = [ 'nTrial',
    #                         'nEndTrial',
    #                         'wi_trial_keep',
    #                         'r_trial',
    #                         'nr_trial',
    #                         ]
    
    X_cols_sftd_lst = []
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


        # tmp_signal_df, X_cols_sftd = timeshift_vals(tmp_signal_df,
        #                                         list(tmp_signal_df.columns),
        #                                         neg_order=neg_order,
        #                                         pos_order=pos_order,
        #                                         exclude_columns=exclude_columns
        #                                         )
        tmp_signal_df, X_cols_sftd = timeshift_vals_by_dict(tmp_signal_df, X_cols_dict)

        X_cols_sftd_lst += [_ for _ in X_cols_sftd if _ not in X_cols_sftd_lst]
        signal_df_lst.append(tmp_signal_df)
        signal_filenames.append(signal_files[file_num].split('/')[-1].split('.')[0].replace('GLM_SIGNALS_', '').replace('INTERIM_', ''))

    return signal_df_lst, X_cols_sftd_lst, signal_filenames
