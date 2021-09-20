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

    df = pd.read_csv(f'{dir_path}/../C39_2020_11_03_designMat.csv').drop('Unnamed: 0', axis=1)#.drop('index', axis=1)

    if 'index' in df.columns:
        df = df.drop('index', axis=1)


    print(df.shape)

    y_setup_col = 'grnL' # photometry response
    df = sglm_ez.diff_cols(df, ['grnL'])

    # # Demonstrative first 5 timesteps of photometry signal vs. differential
    # print(df[['grnL', 'grnL_diff']].head())

    # # Plotting original photometry output (excluding first timestep)
    # plt.figure()
    # df['grnL'].iloc[1:].plot(color='c')
    # plt.title('Original Photometry Signal vs. Time')
    # plt.ylabel('Original Photometry Output')
    # plt.xlabel('Timestep Index')

    # # Plotting photometry differential output (excluding first timestep)
    # plt.figure()
    # df['grnL_diff'].iloc[1:].plot(color='g')
    # plt.title('Differential Photometry Signal vs. Time')
    # plt.ylabel('Differential Photometry Output')
    # plt.xlabel('Timestep Index')

    X_cols = [
        # 'nTrial', # trial ID
        # 'iBlock', # block number within session
        # 'CuePenalty', # lick during cue period (no directionality yet, so binary 0,1)
        # 'ENLPenalty', # lick during ENL period (no directionality yet, 0,1)
        # 'Select', # binary selection lick
        # # 'Consumption', # consumption period (from task perspective)
        # # # 'TO', # timeout trial
        # # 'responseTime', # task state cue to selection window
        # # 'ENL', # task state ENL window
        # 'Cue', # task state Cue window
        # 'decision', # choice lick direction (aligned to select but with directionality -1,1)
        # 'switch', # switch from previous choice on selection (-1,1)
        # 'selR', # select reward (-1,1) aligned to selection
        # 'selHigh', # select higher probability port (-1,1)
        # 'Reward', # reward vs no reward during consumption period (-1,1)
        # 'post', # log-odds probability

        'nTrial',
        'iBlock',
        'ENLPenalty',
        # 'TO',
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

    # dfrel = sglm_ez.diff_cols(dfrel, X_cols)
    # X_cols = X_cols[:2] + [_+'_diff' for _ in X_cols][2:]
    print(dfrel)
    print(X_cols)

    neg_order = -5
    pos_order = 5


    dfrel = sglm_ez.timeshift_cols(dfrel, X_cols[2:], neg_order=neg_order, pos_order=pos_order)
    X_cols_sftd = sglm_ez.add_timeshifts_to_col_list(X_cols, X_cols[2:], neg_order=neg_order, pos_order=pos_order)

    # dfrel = sglm_ez.timeshift_cols(dfrel, [_ for _ in X_cols[2:] if _ not in ['Cue', 'Reward', 'post']], neg_order=neg_order, pos_order=pos_order)
    # X_cols_sftd = sglm_ez.add_timeshifts_to_col_list(X_cols, [_ for _ in X_cols[2:] if _ not in ['Cue', 'Reward', 'post']], neg_order=neg_order, pos_order=pos_order)

    # dfrel, sft_orders = sglm_ez.timeshift_cols_by_signal_length(dfrel, [_ for _ in X_cols[2:] if _ in ['Cue', 'Reward', 'post']], neg_order=-10, pos_order=10, shift_amt_ratio=2)
    # X_cols_sftd = sglm_ez.add_timeshifts_by_sl_to_col_list(X_cols_sftd, [_ for _ in X_cols[2:] if _ in ['Cue', 'Reward', 'post']], sft_orders)

    print(dfrel.shape)

    # print(list(dfrel.columns))
    print(X_cols_sftd)

    # X_setup = sglm_ez.diff_cols(X_setup, ['A', 'B'])
    # X_setup = sglm_ez.setup_autoregression(X_setup, ['B'], 4)


    dfrel = dfrel.dropna()

    # X_setup = dfrel[X_cols].copy()
    X_setup = dfrel[X_cols_sftd].copy()
    y_setup = dfrel[y_col].copy()

    X_setup.head()

    # # glm = sglm_ez.fit_GLM(X_setup, y_setup, reg_lambda=0.1)
    # glm = sglm_ez.fit_GLM(X_setup, y_setup, alpha=0.1)
    # pred = glm.predict(X_setup)
    # mse = np.mean((y_setup - pred)**2)

    perc_holdout = 0.2
    id_cols = ['nTrial', 'iBlock']
    for i, idc in enumerate(id_cols):
        if i == 0:
            bucket_ids = X_setup[idc].astype(str).str.len().astype(str) + ':' + X_setup[idc].astype(str)
        else:
            bucket_ids = bucket_ids + '_' + X_setup[idc].astype(str)
    bucket_ids = bucket_ids.astype("category").cat.codes

    num_bucket_ids = int(bucket_ids.max() + 1)
    num_buckets_for_test = int(num_bucket_ids * perc_holdout)

    test_ids = np.random.choice(num_bucket_ids, size=num_buckets_for_test)
    holdout = bucket_ids.isin(test_ids)

    X_holdout = X_setup.loc[holdout]
    y_holdout = y_setup.loc[holdout]
    X_setup = X_setup.loc[~holdout]
    y_setup = y_setup.loc[~holdout]


    #######################
    #######################
    #######################

    kfold_cv_idx = sglm_ez.cv_idx_by_trial_id(X_setup, y=y_setup, trial_id_columns=['nTrial', 'iBlock'], num_folds=5, test_size=0.2)


    # X_setup = X_setup[[_ for _ in X_setup.columns if _ not in ['nTrial', 'iBlock', 'TO', 'Select', 'Consumption', 'selHigh']]]
    X_setup = X_setup[[_ for _ in X_setup.columns if _ not in ['nTrial', 'iBlock']]]

    # Step 1: Create a dictionary of lists for these relevant keywords...
    kwargs_iterations = {
        # 'reg_lambda': [0.0001],
        # 'reg_lambda': [0.001, 0.01, 0.1, 1.0, 10.0],
        # 'alpha': [0.01, 0.1, 1.0, 10.0],
        'alpha': reversed([0.1, 1.0, 10.0]),
        'l1_ratio': [0.1, 0.5, 0.9],
        # 'l1_ratio': [0.001, 0.1, 0.5, 0.9, 0.999],
        # 'fit_intercept': [True, False]
        'fit_intercept': [True, False]
    }

    # Step 2: Create a dictionary for the fixed keyword arguments that do not require iteration...
    kwargs_fixed = {
        'max_iter': 1000
    }

    # hold_out_idx = kfold_cv_idx[0:1]
    # kfold_cv_idx = kfold_cv_idx[1:]

    # Step 3: Generate iterable list of keyword sets for possible combinations
    glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)
    best_score, best_params, best_model = sglm_ez.simple_cv_fit(X_setup, y_setup, kfold_cv_idx, glm_kwarg_lst, model_type='Normal', verbose=2)

    print()
    print('---')
    print()

    # print('Coeffs:')
    # for ic, coef in enumerate(best_model.coef_):
    #     print(f'> {coef}: {X_setup.columns[ic]}')

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
    # pred = glm.predict(X_holdout)
    # mse = np.mean((y_holdout - pred)**2)

    holdout_score = glm.score(X_holdout[X_setup.columns], y_holdout)

    print(f'Holdout Score: {holdout_score}')

# import cProfile
# cProfile.run('to_profile()')

to_profile()
