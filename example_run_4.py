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

    df = pd.read_csv(f'{dir_path}/../C39v2_sampleDesignMat.csv').drop('Unnamed: 0', axis=1).drop('index', axis=1)

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
        'nTrial', # trial ID
        'iBlock', # block number within session
        'CuePenalty', # lick during cue period (no directionality yet, so binary 0,1)
        'ENLPenalty', # lick during ENL period (no directionality yet, 0,1)
        'Select', # binary selection lick
        # 'Consumption', # consumption period (from task perspective)
        # # 'TO', # timeout trial
        # 'responseTime', # task state cue to selection window
        # 'ENL', # task state ENL window
        'Cue', # task state Cue window
        'decision', # choice lick direction (aligned to select but with directionality -1,1)
        'switch', # switch from previous choice on selection (-1,1)
        'selR', # select reward (-1,1) aligned to selection
        'selHigh', # select higher probability port (-1,1)
        'Reward', # reward vs no reward during consumption period (-1,1)
        'post', # log-odds probability
    ]

    # y_col = 'grnL_diff'
    y_col = 'grnL'

    dfrel = df[X_cols + [y_col]].copy()
    dfrel = dfrel.replace('False', 0).astype(float)
    dfrel = dfrel*1

    # dfrel = sglm_ez.diff_cols(dfrel, X_cols)
    # X_cols = X_cols[:2] + [_+'_diff' for _ in X_cols][2:]
    print(dfrel)
    print(X_cols)

    neg_order = -10
    pos_order = 10

    dfrel = sglm_ez.timeshift_cols(dfrel, X_cols[2:], neg_order=neg_order, pos_order=pos_order)
    X_cols_sftd = sglm_ez.add_timeshifts_to_col_list(X_cols, X_cols[2:], neg_order=neg_order, pos_order=pos_order)
    # X_setup = sglm_ez.diff_cols(X_setup, ['A', 'B'])
    # X_setup = sglm_ez.setup_autoregression(X_setup, ['B'], 4)


    dfrel = dfrel.dropna()

    # X_setup = dfrel[X_cols].copy()
    X_setup = dfrel[X_cols_sftd].copy()
    y_setup = dfrel[y_col].copy()

    X_setup.head()

    # glm = sglm_ez.fit_GLM(X_setup, y_setup, reg_lambda=0.1)
    glm = sglm_ez.fit_GLM(X_setup, y_setup, alpha=0.1)
    pred = glm.predict(X_setup)
    mse = np.mean((y_setup - pred)**2)












    kfold_cv_idx = sglm_ez.cv_idx_by_trial_id(X_setup, y=y_setup, trial_id_columns=['nTrial', 'iBlock'])


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

    # Step 3: Generate iterable list of keyword sets for possible combinations
    glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)
    best_score, best_params, best_model = sglm_ez.simple_cv_fit(X_setup, y_setup, kfold_cv_idx, glm_kwarg_lst, model_type='Normal')

    print(f'Best Score: {best_score}')
    print(f'Best Params: {best_params}')
    print(f'Best Model: {best_model}')
    print(f'Best Model — Coeffs: {best_model.coef_}, Intercept: {best_model.intercept_}')

    print(f'Overall RunTime: {time.time() - start}')

# import cProfile
# cProfile.run('to_profile()')

to_profile()
