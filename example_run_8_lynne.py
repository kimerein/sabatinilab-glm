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

def to_profile():

    start = time.time()
    print(dir_path)
    df = pd.read_csv(f'{dir_path}/../dlight_only_WT35_12212020.csv')
    # df = pd.read_csv(f'{dir_path}/../dlight_only_WT36_12212020.csv')
    df = df[[_ for _ in df.columns if 'Unnamed' not in _]]

    print(df.columns)


    df = df.rename({'center port occupancy': 'cpo',
                    'center port entry': 'cpn',
                    'center port exit': 'cpx',

                    'left port occupancy': 'lpo',
                    'left port entry': 'lpn',
                    'left port exit': 'lpx',
                    'left licks': 'll',

                    'right port occupancy': 'rpo',
                    'right port entry': 'rpn',
                    'right port exit': 'rpx',
                    'right licks': 'rl',

                    'no reward': 'nr',
                    'reward': 'r',

                    'dF/F green (dLight1.1)': 'gdFF',
                    'zscored green (dLight1.1)': 'zsgdFF'}, axis=1)
    
    df['event_col_a'] = ((df['cpo'].diff() > 0)*1).replace(0, np.nan) * 1.0
    df['event_col_b'] = df['nr'].replace(0, np.nan) * 2.0
    df['event_col_c'] = df['r'].replace(0, np.nan) * 3.0

    df['event_col'] = df['event_col_a'].combine_first(df['event_col_b']).combine_first(df['event_col_c'])

    df = df.drop(['event_col_a', 'event_col_b', 'event_col_c'], axis=1)

    df['event_col'] = df['event_col'].bfill()
    
    df['trial_start_flag'] = ((df['event_col'] == 1.0)&(df['event_col'].shift(-1) != df['event_col']) * 1.0).shift(-10) * 1.0
    df['trial_end_flag'] = ((df['event_col'] != 1.0)&(df['event_col'].shift(-1) == 1.0)&(df['event_col'].shift(-1) != df['event_col']) * 1.0).shift(10) * 1.0
    df['nTrial'] = df['trial_start_flag'].cumsum()
    df['nEndTrial'] = df['trial_end_flag'].cumsum()

    wi_trial_keep = (df['nTrial'] != df['nEndTrial'])

    if 'index' in df.columns:
        df = df.drop('index', axis=1)
    
    # y_setup_col = 'gdFF' # photometry response
    # df = sglm_ez.diff_cols(df, ['gdFF'])

    X_cols = [
       'nTrial',
    #    'index',

       #'cpo',
       'cpn', 'cpx',
       #'lpo',
       'lpn', 'lpx',
       'll',
       #'rpo',
       'rpn', 'rpx',
       'rl',
       'nr', 'r'
    ]

    y_col = 'zsgdFF'

    dfrel = df[X_cols + [y_col]].copy()
    dfrel = dfrel.replace('False', 0).astype(float)
    dfrel = dfrel*1
    
    neg_order = -40
    pos_order = 40

    dfrel = sglm_ez.timeshift_cols(dfrel, X_cols[1:], neg_order=neg_order, pos_order=pos_order)
    X_cols_sftd = sglm_ez.add_timeshifts_to_col_list(X_cols, X_cols[1:], neg_order=neg_order, pos_order=pos_order)

    dfrel = dfrel.dropna()

    X_setup = dfrel[X_cols_sftd].copy()
    y_setup = dfrel[y_col].copy()





    X_setup = X_setup[wi_trial_keep]
    y_setup = y_setup[wi_trial_keep]





    X_setup.head()

    

    holdout = sglm_ez.holdout_split_by_trial_id(X_setup, y_setup, id_cols=['nTrial'], perc_holdout=0.2)

    X_holdout = X_setup.loc[holdout]
    y_holdout = y_setup.loc[holdout]
    X_setup = X_setup.loc[~holdout]
    y_setup = y_setup.loc[~holdout]


    #######################
    #######################
    #######################

    kfold_cv_idx = sglm_ez.cv_idx_by_trial_id(X_setup, y=y_setup, trial_id_columns=['nTrial'], num_folds=5, test_size=0.2)

    X_setup = X_setup[[_ for _ in X_setup.columns if _ not in ['nTrial']]]
    # X_setup = X_setup[[_ for _ in X_setup.columns]]

    # Step 1: Create a dictionary of lists for these relevant keywords...
    kwargs_iterations = {
        'alpha': reversed([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]),
        'l1_ratio': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99],
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
