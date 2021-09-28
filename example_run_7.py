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

import sglm
import sglm_cv
import sglm_pp
import sglm_ez

#  from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
#                                      StratifiedKFold, GroupShuffleSplit,
#                                      GroupKFold, StratifiedShuffleSplit)
# group_len = 60*2 * Fs # seconds * Fs
# n_splits = 10
# test_size = 0.3
# groups = np.arange(X.shape[0])//group_len
# n_groups = np.max(groups)
# cv = GroupShuffleSplit(n_splits, test_size=test_size)
# cv_idx = cross_validation.make_cv_indices(cv,
#                                         groups,
#                                         lw=5,
#                                         plot_pref=True)


def to_profile():

    start = time.time()

    # Load Data
    df = pd.read_csv(f'{dir_path}/../C39_2020_11_03_designMat.csv').drop('Unnamed: 0', axis=1)#.drop('index', axis=1)
    if 'index' in df.columns:
        df = df.drop('index', axis=1)

    y_setup_col = 'grnL' # photometry response
    df = sglm_ez.diff_cols(df, ['grnL'])

    X_cols = [
        'nTrial',
        'iBlock',
        'ENLPenalty',
        'responseTime',
        'Cue',
        'decision',
        'switch',
        'selReward',
        'post',
        'lickReward',
        'lickNoReward',
        'lickLeft',
        'lickRight',
        'lickSwitch',
    ]

    

    # y_col = 'grnL_diff'
    y_col = 'grnL'
    # y_col = 'grnR'

    dfrel = df[X_cols + [y_col]].copy()
    dfrel = dfrel.replace('False', 0).astype(float)
    dfrel = dfrel*1

    print(dfrel)
    print(X_cols)

    neg_order = -40
    pos_order = 40


    dfrel = sglm_ez.timeshift_cols(dfrel, X_cols[2:], neg_order=neg_order, pos_order=pos_order)
    X_cols_sftd = sglm_ez.add_timeshifts_to_col_list(X_cols, X_cols[2:], neg_order=neg_order, pos_order=pos_order)

    print(dfrel.shape)

    print(X_cols_sftd)
    
    dfrel = dfrel.dropna()

    X_setup = dfrel[X_cols_sftd].copy()
    y_setup = dfrel[y_col].copy()

    X_setup.head()


    holdout = sglm_ez.holdout_split_by_trial_id(X_setup, y_setup, id_cols=['nTrial', 'iBlock'], perc_holdout=0.2)
    X_holdout = X_setup.loc[holdout]
    y_holdout = y_setup.loc[holdout]
    X_setup = X_setup.loc[~holdout]
    y_setup = y_setup.loc[~holdout]


    #######################
    #######################
    #######################

    kfold_cv_idx = sglm_ez.cv_idx_by_trial_id(X_setup, y=y_setup, trial_id_columns=['nTrial', 'iBlock'], num_folds=5, test_size=0.2)

    X_setup = X_setup[[_ for _ in X_setup.columns if _ not in ['nTrial', 'iBlock']]]

    # Step 1: Create a dictionary of lists for these relevant keywords...
    kwargs_iterations = {
        'alpha': reversed([0.1, 1.0, 10.0]),
        'l1_ratio': [0.1, 0.5, 0.9],
        'fit_intercept': [True]
    }

    # Step 2: Create a dictionary for the fixed keyword arguments that do not require iteration...
    kwargs_fixed = {
        'max_iter': 1000
    }

    score_method = 'r2'

    # Step 3: Generate iterable list of keyword sets for possible combinations
    glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)
    best_score, best_score_std, best_params, best_model = sglm_ez.simple_cv_fit(X_setup, y_setup, kfold_cv_idx, glm_kwarg_lst, model_type='Normal', verbose=2, score_method=score_method)

    print()
    print('---')
    print()

    print('Non-Zero Coeffs:')
    epsilon = 1e-10
    for ic, coef in enumerate(best_model.coef_):
        if np.abs(coef) > epsilon:
            print(f'> {coef}: {X_setup.columns[ic]}')


    print(f'Best Score: {best_score}')
    print(f'Best Params: {best_params}')
    print(f'Best Model: {best_model}')
    print(f'Best Model — Intercept: {best_model.intercept_}')

    print(f'Overall RunTime: {time.time() - start}')

    print()

    glm = sglm_ez.fit_GLM(X_setup, y_setup, **best_params)
    holdout_score = glm.r2_score(X_holdout[X_setup.columns], y_holdout)
    print(f'Holdout Score: {holdout_score}')

to_profile()
