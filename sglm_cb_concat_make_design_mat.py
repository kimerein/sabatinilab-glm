#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:47:35 2022

@author: celiaberon
"""

import os
import sys
dir_path = '/Users/celiaberon/GitHub/'
sys.path.append(f'{dir_path}/sabatinilab-glm/backend')
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/backend')
sys.path.append(f'{dir_path}/../backend')
import numpy as np
import pandas as pd
import time
import random
import pickle

# import sglm_cv
import sglm_ez
from copy import deepcopy

import load_sessions

import importlib
importlib.reload(sglm_ez)
importlib.reload(load_sessions)


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
    dfrel = sglm_ez.timeshift_cols(dfrel, X_cols, neg_order=neg_order, pos_order=pos_order)
    X_cols_sftd = sglm_ez.add_timeshifts_to_col_list(X_cols, X_cols, neg_order=neg_order, pos_order=pos_order)
    return dfrel, X_cols_sftd


def CB_timeshifts(X, cols, neg_order, pos_order, step_size=1):

    '''CB version of timeshifts'''

    X_ = X.copy()

    X_cols_sftd = [] #[f'{col}_0' for col in cols]
    X_cols_sftd.extend(cols)

    if (step_size>1) & ((neg_order % step_size)!=0):
        neg_order += step_size - (neg_order % step_size)  # make sure ends at symmetric gap from zero as pos_order

    for shift_size in np.arange(-neg_order, 0, step=step_size):

        X_[[f'{col}_{shift_size}' for col in cols]] = X_[cols].shift(shift_size)
        X_cols_sftd.extend([f'{col}_{shift_size}' for col in cols])

    for shift_size in np.arange(0+step_size, pos_order, step=step_size):

        X_[[f'{col}_{shift_size}' for col in cols]] = X_[cols].shift(shift_size)
        X_cols_sftd.extend([f'{col}_{shift_size}' for col in cols])

    # X_ = X_.rename(columns={col: f'{col}_0' for col in cols})
    
    return X_, X_cols_sftd, [neg_order, pos_order]


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
    
    return dfrel_setup, dfrel_holdout, holdout


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
    #print('Non-Zero Coeffs:')
    #epsilon = 1e-10
    #for ic, coef in enumerate(best_model.coef_):
    #    if np.abs(coef) > epsilon:
    #        print(f'> {coef}: {X_setup.columns[ic]}')

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


def backtrack_single_column(df, current_set):
    
    '''
    remove interaction term from pair of features by backtracking to nearest node
    INPUTS:
        -df (pandas DataFrame): design matrix
        -current_set (list): list containing root (node) feature, and bifurcated pair
    OUTPUTS:
        -df_reduced (pandas DataFrame): design matrix with compressed representation across paired feature set
    '''
    
    col_root, *matched_cols = current_set
    
    df_reduced=pd.DataFrame()

    for col in df.columns:

        if col in matched_cols:
            continue

        df_reduced[col] = df[col].copy() # copy features not in specified pair

    if len(matched_cols)==1:
        df_reduced[col_root] = df[matched_cols[0]].copy() # if no pair, solo feature stays as is

    else:
        df_reduced[col_root] = df[matched_cols].copy().sum(axis=1) # if paired set, sum across columns to remove interaction effect

    return df_reduced


def trim_to_active_periods(design_matrix, features):

    reduced_design_matrix = design_matrix.copy()
    within_trial_labels = [col for col in reduced_design_matrix.columns if col.startswith(tuple(features))]

    active_rows = reduced_design_matrix[within_trial_labels].values.any(axis=1)
    reduced_design_matrix['active_rows'] = active_rows > 0
    reduced_design_matrix = reduced_design_matrix.loc[reduced_design_matrix.active_rows>0]
    # reduced_design_matrix = reduced_design_matrix.drop(columns=['active_rows'])

    print(len(reduced_design_matrix) / len(design_matrix))
    
    return reduced_design_matrix


def cb_to_profile_concat(n_timeshifts, dm_kwargs, mouse='C38', trim_to_events=False,  **kwargs):
    
    model_tag = kwargs.get('model_tag', '')
    
    folds = 3  # k folds for cross validation
    pholdout = 0.2 
    pgss = 0.2
    
    score_method = 'r2'        

    glm_hyperparams = [{
        'alpha': 0.0, # 0 is OLS
        'l1_ratio': 0.0,
        'max_iter': 1000,
        'fit_intercept': False
    }]
    start = time.time()
    
    # Read in all sessions, making design matrices with specified features.
    dfs = load_sessions.read_in_multi_sessions(mouse, **dm_kwargs); 
    df = dfs['analog']

    # Define columns to exclude from timeshifting but include as features.
    not_sftd_features = ['time_from_enl_onset', 'time_from_enlp_onset']
    # Define columns to store for heatmaps but exclude from GLM.
    hm_cols = [col for col in df.columns if str(col).startswith('hm')]
    # Define photometry columns to include for analysis.
    photo_cols = [col for col in df.columns if 'grn' in col]
    y_cols_passed = [col for col in photo_cols if len(df.dropna(subset=col))>0]
    # Define columns that contain trial-wide constants
    trial_constants = ['iBlock', 'nTrial']
    # Special cases that don't get used as group...
    special_cols = ['nTrial_orig', 'session', 'flag']
    # from concatenating where conditions may have not appeared in single
    # sessions, use 0s, but don't fill any photometry NaNs.
    df = df.fillna(value={col:0 for col in df.columns if 'grn' not in col }) 
    # Specify list of columns to shift tht excludes above cases.
    X_cols = [col for col in df.columns if col not in trial_constants
                                                      + not_sftd_features
                                                      + hm_cols
                                                      + photo_cols
                                                      + special_cols] # CB all X cols to shift

    df = df.convert_dtypes()

    # Iterate over photometry channels, clean data based on where channel contains signal,
    # add timeshifts, fit glm, save model and metadata.
    for y_ in y_cols_passed:

        root = f'/Volumes/Neurobio/MICROSCOPE/Celia/data/lickTask/model_outputs/new_save/{mouse}/{model_tag}/{y_}'
        if not os.path.isdir(root):
            os.makedirs(root)
            print(f'starting analysis on {mouse} {y_}')
    
        df_y = df.dropna(subset=y_) # drop NaNs from used photometry column
        drop_col = set(photo_cols).difference({y_}) # get column names for photometry channels not analyzed in this run
        df_y = df_y.drop(columns=drop_col) # remove other photometry channel that's not being analyzed
        df_y = df_y.reset_index(drop=True) # df_y contains only one photometry channel

        dates_included = df_y.session.unique() # list of sessions in analyzed dataset

        # list of trials nested in list of sessions for reconstructions
        trials_included = [df_y.loc[df_y.session==sess].nTrial.unique() 
                          for sess in df_y.session.unique()]

        df_y = pd.get_dummies(df_y, columns=['session']) # get dummy variables for each session (fixed effects)
        session_constants = [col for col in df_y.columns if 'session' in col]
        res = {} # results dictionary

        run_id = f'z_{mouse}_{y_}_{model_tag}'
        print("Run ID:", run_id)

        
        # Timeshift X_cols forward by pos_order times and backward by 
        # neg_order times. CB version of function allows for skipped steps 
        # at specified interval.
        dfrel, X_cols_sftd, n_timeshifts = CB_timeshifts(
                                                df_y, X_cols, 
                                                neg_order=n_timeshifts[0], 
                                                pos_order=n_timeshifts[1]
                                                ) 

        # Set flagged trial events to zero (e.g. timeouts) -- these
        # will be dropped after timeshifts, but still need their 
        # photometry data for any shifts crossing trial boundaries.
        dfrel = dfrel.loc[df_y['flag']==0]
        print('timeouts? ', len(dfrel)/len(df_y))
        df_y = df_y.loc[df_y['flag']==0] # to match for adding columns
        print(f'{n_timeshifts=}')

        # Add columns without timeshifts
        dfrel[trial_constants] = df_y[trial_constants] 
        
        # Check that indices still match so we can use shortcut for 
        # using dfrel masking on df_y.
        assert(np.all(dfrel.index==df_y.index))

        # Drop NAs for non-existant timeshifts (e.g., session boundaries)
        dfrel = dfrel.dropna()
        df_y = df_y.loc[df_y.index.isin(dfrel.index.values)]

        # if flagged, trim each trial to only contain period spanned by 
        # non-zero event values
        if trim_to_events:
            dfrel = trim_to_active_periods(dfrel, X_cols_sftd + not_sftd_features)
            df_y = df_y.loc[df_y.index.isin(dfrel.index.values)]  # to match when add cols later
            dfrel = dfrel.drop(columns=['active_rows'])
        
        # CB add any single occurrence variables here, i.e. iBlock, reward_seq
        X_cols_sftd.extend(['nTrial'] + not_sftd_features + session_constants) 
        print(X_cols_sftd)

        # Split data into setup (training) and holdout (test) sets
        np.random.seed(30186)
        random.seed(30186)

        dfrel_setup, dfrel_holdout, holdout_mask = holdout_splits(dfrel,
                                                    id_cols=['nTrial'],
                                                    perc_holdout=pholdout)

        print(holdout_mask.index[0])
        metadata = pd.DataFrame(
                    {'Mouse':[mouse],
                    'sessions':[dates_included], 
                    'trials':[trials_included],
                    'cv_folds': folds, 'cv_pholdout':pholdout, 'cv_pgss':pgss,
                    'timeshifts':[n_timeshifts],
                    'channel':y_,
                    'trials_holdout':[dfrel_holdout.nTrial.values],
                    'trials_train':[dfrel_setup.nTrial.values]
                    })
        metadata.to_csv(os.path.join(root, 'metadata.csv'))

        # Generate cross-validation (technically, group / shuffle split) sets
        # for training / model selection
        X_setup, X_holdout = dfrel_setup[X_cols_sftd].copy(), dfrel_holdout[X_cols_sftd].copy()
        y_setup, y_holdout = dfrel_setup[y_].copy(),  dfrel_holdout[y_].copy()
        
        # Josh's code -- indices for each fold of cross validation
        kfold_cv_idx = sglm_ez.cv_idx_by_trial_id(X_setup,
                                                  y=y_setup, 
                                                  trial_id_columns=['nTrial'],
                                                  num_folds=folds, 
                                                  test_size=pgss)
        
        dfrel['holdout_mask'] = holdout_mask  # to reproduce splits after
        dfrel[hm_cols] = df_y[hm_cols] # add columns for viz, not used in glm
        dfrel.to_parquet(os.path.join(root, 'data_ref_full.parquet.gzip'), 
            compression='gzip')
        
        # Drop nTrial column from X_setup. (It is only used for group 
        # identification in group/shuffle/split)
        X_setup = X_setup.drop(columns=['nTrial']) 
        X_holdout = X_holdout.drop(columns=['nTrial']) 

        best_score, _, best_params, best_model, _ = sglm_ez.simple_cv_fit(X_setup, 
                                                                y_setup, 
                                                                kfold_cv_idx, 
                                                                glm_hyperparams, 
                                                                model_type='Normal', 
                                                                verbose=0, 
                                                                score_method=score_method
                                                                )


        print_best_model_info(X_setup, best_score, best_params, best_model, start)

        glm, holdout_score, holdout_neg_mse_score = training_fit_holdout_score(X_setup, y_setup, X_holdout, y_holdout, best_params)

        # Collect
        res[f'{run_id}'] = {'holdout_score':holdout_score,
                            'holdout_neg_mse_score':holdout_neg_mse_score,
                            'best_score':best_score,
                            'best_params':best_params,
                            }

        model_metadata = pd.DataFrame({'score_train':glm.r2_score(X_setup, y_setup), 
                            'score_gss':best_score, 
                            'score_holdout':glm.r2_score(X_holdout, y_holdout),
                            'hyperparams': [best_params],
                            'gssids': [kfold_cv_idx],
                            'features':[[_ for _ in X_cols if _ in X_setup.columns]],
                            'full_features':[[_ for _ in X_cols_sftd if _ in X_setup.columns]],
                            'features_sftd': [X_cols],
                            'session_constants':[session_constants],
                            'trial_constants': [trial_constants],
                            'not_sftd_features': [not_sftd_features]})

        model_metadata.to_csv(os.path.join(root, 'model_metadata.csv'))

        with open(os.path.join(root, 'glm.pkl'), "wb") as file_save:
            pickle.dump(glm, file_save)

    # For every file iterated, for every result value, for every model fitted, print the reslts
    print(f'Final Results:')
    for k in res:
        print(f'> {k}') # print key, filename
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


    end = time.time()
    print('Runtime:',end-start)
                
    
    return glm, X_holdout, y_holdout, X_setup, y_setup, holdout_mask





