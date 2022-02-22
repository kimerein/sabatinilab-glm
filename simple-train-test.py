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
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
from sklearn.model_selection import GroupShuffleSplit
import time
import random

import sglm
import sglm_cv
import sglm_pp
import sglm_ez
import sglm_plt as splt
import sglm_save as ssave
import lynne_pp as lpp
from tqdm import tqdm, trange

import cProfile
import pickle


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

    first_trans = dfrel.groupby('nTrial')[['nn', 'xx', 'lpn', 'rpn', 'lpx', 'rpx', 'cpn']].cumsum()
    first_trans = ((first_trans == 1)*1).diff()
    first_trans *= first_trans >= 0
    first_trans['lpn'] = dfrel['nn']*dfrel['lpn']
    first_trans['rpn'] = dfrel['nn']*dfrel['rpn']
    first_trans['lpx'] = dfrel['xx']*dfrel['lpx']
    first_trans['rpx'] = dfrel['xx']*dfrel['rpx']

    first_trans = first_trans.rename({_k:f'ft_{_k}' for _k in first_trans.columns}, axis=1)
    dfrel[first_trans.columns] = first_trans

    dfrel['ft_r_rpn'] = dfrel['ft_rpn'] * dfrel['r']
    dfrel['ft_r_lpn'] = dfrel['ft_lpn'] * dfrel['r']
    dfrel['ft_nr_rpn'] = dfrel['ft_rpn'] * dfrel['nr']
    dfrel['ft_nr_lpn'] = dfrel['ft_lpn'] * dfrel['nr']

    return dfrel

def preprocess_lynne(df):
    '''
    Preprocess Lynne's dataframe for GLM
    Args:
        df: dataframe with entry, exit, lick, reward, and dFF columns
    Returns:
        dataframe with entry, exit, lick, reward, and
    '''
    df = df[[_ for _ in df.columns if 'Unnamed' not in _]]
    print(df.columns)
    df = lpp.rename_columns(df)
    df = lpp.define_trial_starts_ends(df)

    print('Percent of Data in ITI:', (df['nTrial'] == df['nEndTrial']).mean())

    df = lpp.set_reward_flags(df)
    df = lpp.set_port_entry_exit_rewarded_unrewarded_indicators(df)
    df = lpp.define_side_agnostic_events(df)

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

def get_is_not_iti(df):
    '''
    Returns a boolean array of whether the trial is not ITI
    Args:
        df: dataframe with entry, exit, lick, reward, and dFF columns
    Returns:
        boolean array of whether the trial is not ITI
    '''
    return df['nTrial'] != df['nEndTrial']

def get_x(df, x_cols, keep_rows=None):
    '''
    Get x values for fitting/scoring
    Args:
        df: dataframe that includes x_cols
        x_cols: list of column names to include in prediction
        keep_rows: boolean array of which rows to keep
    Returns:
        df[x_cols]: dataframe only including prediction columns and keep_rows
    '''
    if type(keep_rows) != type(None):
        df = df[keep_rows]
    return df[x_cols]

def get_y(df, y_col, keep_rows=None):
    '''
    Get y values for fitting/scoring
    Args:
        df: dataframe that includes x_cols
        y_cols: column name to use for response
        keep_rows: boolean array of which rows to keep
    Returns:
        df[y_col]: dataframe only including response column and keep_rows
    '''
    if type(keep_rows) != type(None):
        df = df[keep_rows]
    return df[y_col]


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
    dfrel = sglm_ez.timeshift_cols(dfrel, X_cols[1:], neg_order=neg_order, pos_order=pos_order)
    X_cols_sftd = sglm_ez.add_timeshifts_to_col_list(X_cols, X_cols[1:], neg_order=neg_order, pos_order=pos_order)
    return dfrel, X_cols_sftd


def holdout_splits(dfrel_setup, id_cols=['nTrial'], perc_holdout=0.2):
    '''
    Create holdout splits
    Args:
        dfrel_setup: full setup dataframe
        id_cols: list of columns to use as trial identifiers
        perc_holdout: percentage of data to holdout
    Returns:
        dfrel_setup: full setup dataframe
        dfrel_holdout: full holdout dataframe
    '''
    # Create holdout splits
    holdout = sglm_ez.holdout_split_by_trial_id(dfrel_setup, id_cols=id_cols, perc_holdout=perc_holdout)
    dfrel_holdout = dfrel_setup.loc[holdout]
    dfrel_setup = dfrel_setup.loc[~holdout]
    return dfrel_setup, dfrel_holdout

def print_best_model_info(X_setup, best_score, best_params, best_model, start):
    """
    Print best model info
    Args:
        X_setup: setup prediction dataframe
        best_score: best score
        best_params: best parameters
        best_model: best model
        start: start time
    """

    print()
    print('---')
    print()

    # Print out all non-zero coefficients
    print('Non-Zero Coeffs:')
    epsilon = 1e-10
    for ic, coef in enumerate(best_model.coef_):
        if np.abs(coef) > epsilon:
            print(f'> {coef}: {X_setup.columns[ic]}')

    # Print out information related to the best model
    print(f'Best Score: {best_score}')
    print(f'Best Params: {best_params}')
    print(f'Best Model: {best_model}')
    print(f'Best Model — Intercept: {best_model.intercept_}')

    # Print out runtime information
    print(f'Overall RunTime: {time.time() - start}')

    print()
    return

def training_fit_holdout_score(X_setup, y_setup, X_holdout, y_holdout, best_params):
    '''
    Fit GLM on training data, and score on holdout data
    Args:
        X_setup: X training data on which to fit model
        y_setup: response column for training data
        X_holdout: X holdout data on which to score model
        y_holdout: response column for holdout data
        best_params: dictionary of best parameters
    Returns:
        glm: Fitted model
        holdout_score: Score on holdout data
        holdout_neg_mse_score: Negative mean squared error on holdout data
    '''
    # Refit the best model on the full setup (training) data
    glm = sglm_ez.fit_GLM(X_setup, y_setup, **best_params)

    # Get the R^2 and MSE scores for the best model on the holdout (test) data
    holdout_score = glm.r2_score(X_holdout, y_holdout)
    holdout_neg_mse_score = glm.neg_mse_score(X_holdout, y_holdout)

    return glm, holdout_score, holdout_neg_mse_score


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


def to_profile():
    start = time.time()


    ssave_folder = 'model_outputs/ssave'
    all_models_folder = 'model_outputs/all_models'
    all_data_folder = 'model_outputs/all_data'
    all_reconstruct_folder = 'model_outputs/all_reconstructions'
    all_coeffs_folder = 'model_outputs/all_coeffs'
    best_reconstruct_folder = 'model_outputs/best_reconstructions'
    best_coeffs_folder = 'model_outputs/best_coeffs'

    prefix = 'no_cv_shuffle_kwargs'
    avg_reconstruct_basename = 'arr'
    all_betas_basename = 'betas'
    model_c_basename = 'coeffs'
    model_i_basename = 'intercept'
    tmp_data_basename = 'tmp_data'

    files_list = [
                    'Ach_rDAh_WT63_11082021.csv',
                    # 'Ach_rDAh_WT63_11162021.csv',
                    # 'Ach_rDAh_WT63_11182021.csv'
                    ]

    # Select column names to use for GLM predictors
    # 'spn', 'spx',
    X_cols_all = [
        'nTrial',
        'cpn', 'cpx',
        'spnr', 'spxr',
        'spnnr', 'spxnr',
        'sl',
    ]


    score_method = 'r2'        

    # Select hyper parameters for GLM to use for model selection
    # Step 1: Create a dictionary of lists for these relevant keywords...
    kwargs_iterations = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 0.9, 1.0],
        'l1_ratio': [0.0, 0.001, 0.1],

        # 'alpha': [0.001, 0.01,],
        # 'l1_ratio': [0.0, 0.001,],

        # 'alpha': [0.001,],
        # 'l1_ratio': [0.0,],
    }

    # Step 2: Create a dictionary for the fixed keyword arguments that do not require iteration...
    kwargs_fixed = {
        'max_iter': 10000,
        'fit_intercept': True
    }

    neg_order, pos_order = -14, 14
    folds = 50
    pholdout = 0.2
    # pgss = 0.2
    # pgss = 0.05
    pgss = 0.01

    # Step 3: Generate iterable list of keyword sets for possible combinations
    glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)

    res = {}

    # leave_one_out_list = [[]]
    leave_one_out_list = [[]] + [[_] for _ in X_cols_all if _ != 'nTrial'] # Excluding column for groupby, 'nTrial'


    # Loop through files to be processed
    for filename in tqdm(files_list, 'filename'):
        fn = filename.split('.')[0].split('/')[-1]

        glmsave = ssave.GLM_data(ssave_folder, f'{prefix}_{fn}.pkl')
        glmsave.set_uid(prefix)
        glmsave.set_filename(filename)
        glmsave.set_timeshifts(neg_order, pos_order)
        glmsave.set_X_cols(X_cols_all)
        glmsave.set_gss_info(folds, pholdout, pgss, gssid=None)


        # Load file
        df = pd.read_csv(f'{dir_path}/../{filename}')
        df = preprocess_lynne(df)
        # df = df[df['r_trial'] > 0]
        df['wi_trial_keep'] = get_is_not_iti(df)

        glmsave.set_basedata(df)

        for y_col in tqdm(['zsrdFF', 'zsgdFF'], 'ycol'):

            # df = detrend(df, y_col)

            for left_out in tqdm(leave_one_out_list, 'left_out'):
                
                X_cols = [_ for _ in X_cols_all if _ not in left_out]

                if len(leave_one_out_list) > 1:
                    run_id = f'{prefix}_{fn}_{y_col}_drop={"_".join(left_out)}'
                else:
                    run_id = f'{prefix}_{fn}_{y_col}'

                print("Run ID:", run_id)

                res_lst = []

                for glm_kwargs in glm_kwarg_lst:
                    res_dct = {}
                    print(glm_kwargs)

                    
                    dfrel = df.copy()
                    
                    # Timeshift X_cols forward by pos_order times and backward by neg_order times
                    dfrel, X_cols_sftd = timeshift_vals(dfrel, X_cols, neg_order=neg_order, pos_order=pos_order)
                    dfrel = dfrel.dropna()
                    dfrel_setup, dfrel_holdout = holdout_splits(dfrel,
                                                                id_cols=['nTrial'],
                                                                perc_holdout=pholdout)

                    prediction_X_cols = [_ for _ in X_cols if _ not in ['nTrial']]
                    prediction_X_cols_sftd = [_ for _ in X_cols_sftd if _ not in ['nTrial']]

                    X_setup = get_x(dfrel_setup, prediction_X_cols_sftd, keep_rows=None)
                    y_setup = get_y(dfrel_setup, y_col, keep_rows=None)
                    X_setup_noiti = get_x(dfrel_setup, prediction_X_cols_sftd, keep_rows=dfrel_setup['wi_trial_keep'])
                    y_setup_noiti = get_y(dfrel_setup, y_col, keep_rows=dfrel_setup['wi_trial_keep'])
                    
                    X_holdout_witi = get_x(dfrel_holdout, prediction_X_cols_sftd, keep_rows=None)
                    y_holdout_witi = get_y(dfrel_holdout, y_col, keep_rows=None)
                    X_holdout_noiti = get_x(dfrel_holdout, prediction_X_cols_sftd, keep_rows=dfrel_holdout['wi_trial_keep'])
                    y_holdout_noiti = get_y(dfrel_holdout, y_col, keep_rows=dfrel_holdout['wi_trial_keep'])
                    # glm, holdout_score, holdout_neg_mse_score = training_fit_holdout_score(X_setup, y_setup, X_holdout_noiti, y_holdout_noiti, best_params)
                    

                    if glm_kwargs['l1_ratio'] == 0:
                        no_l1_glmkwarg = glm_kwargs.copy()
                        del no_l1_glmkwarg['l1_ratio']
                        glm = Ridge(**no_l1_glmkwarg)
                    elif glm_kwargs['l1_ratio'] == 1:
                        no_l1_glmkwarg = glm_kwargs.copy()
                        del no_l1_glmkwarg['l1_ratio']
                        glm = Lasso(**no_l1_glmkwarg)
                    else:
                        glm = ElasticNet(**glm_kwargs)
                    
                    glm.fit(X_setup, y_setup)
                    res_dct['model'] = glm
                    res_dct['glm_kwargs'] = glm_kwargs
                    res_dct['tr_witi'] = glm.score(X_setup, y_setup)
                    res_dct['tr_noiti'] = glm.score(X_setup_noiti, y_setup_noiti)
                    res_dct['holdout_witi'] = glm.score(X_holdout_witi, y_holdout_witi)
                    res_dct['holdout_noiti'] = glm.score(X_holdout_noiti, y_holdout_noiti)
                    res_lst.append(res_dct)

                    print(res_dct)


                
                for fitted_model_dict in res_lst:
                    fitted_model = fitted_model_dict['model']
                    kwarg_info = "_".join([f"{_k}_{fitted_model_dict['glm_kwargs'][_k]}" for _k in fitted_model_dict["glm_kwargs"]])
                    glmsave.append_fit_results(y_col, fitted_model_dict["glm_kwargs"], glm_model=fitted_model, dropped_cols=left_out,
                                            scores={
                                                'tr_witi':fitted_model_dict['tr_witi'],
                                                'tr_noiti':fitted_model_dict['tr_noiti'],
                                                'gss_witi':None,
                                                'gss_noiti':None,
                                                'holdout_witi':fitted_model_dict['holdout_witi'],
                                                'holdout_noiti':fitted_model_dict['holdout_noiti']
                                            },
                                            gssids=None)
                
                print(glmsave.data['fit_results'])
                    
        glmsave.save(overwrite=True)

    # For every file iterated, for every result value, for every model fitted, print the reslts
    print(f'Final Results:')
    for k in res:
        print(f'> {k}')
        for k_ in res[k]:
            if type(res[k][k_]) != list:
                print(f'>> {k_}: {res[k][k_]}')
            else:
                lst_str_setup = f'>> {k_}: ['
                lss_spc = ' '*(len(lst_str_setup)-1)
                print(lst_str_setup)
                for v_ in res[k][k_]:
                    print((f'{lss_spc} R^2: {np.round(v_[0], 5)} — MSE: {np.round(v_[1], 5)} —'+
                        f' L1: {np.round(v_[2], 5)} — L2: {np.round(v_[3], 5)} — '+
                        f'Params: {v_[4]}'))
                print(lss_spc + ']')


    # print('X_cols_plot:', X_cols_plot)
    # print('X_cols_sftd_plot:', X_cols_sftd_plot)

    end = time.time()
    print('Runtime:',end-start)

if __name__ == '__main__':
    profile = cProfile.run('to_profile()', filename='./profile_val.prof', sort='cumtime')