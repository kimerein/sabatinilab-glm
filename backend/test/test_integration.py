import sys

# sys.path.append('..')
# sys.path.append('../backend')
# sys.path.append('/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/backend')
import os 
dir_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/backend')
sys.path.append(f'{dir_path}/../backend')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

import sglm
import sglm_cv
import sglm_pp


if __name__ == '__main__':
    df = pd.read_csv('/Users/josh/Documents/Harvard/GLM/C39v2_sampleDesignMat.csv').drop('Unnamed: 0', axis=1).drop('index', axis=1)
    print('Data Loaded...')
    y_setup_col = 'grnL' # photometry response
    df['grnL_diff'] = sglm_pp.diff(df['grnL'])
    print('NP Diff Completed...')

    X_cols = [
        'nTrial', # trial ID
        'iBlock', # block number within session
        'CuePenalty', # lick during cue period (no directionality yet, so binary 0,1)
        'ENLPenalty', # lick during ENL period (no directionality yet, 0,1)
        'Select', # binary selection lick
        'Consumption', # consumption period (from task perspective)
        'TO', # timeout trial
        'responseTime', # task state cue to selection window
        'ENL', # task state ENL window
        'Cue', # task state Cue window
        'decision', # choice lick direction (aligned to select but with directionality -1,1)
        'switch', # switch from previous choice on selection (-1,1)
        'selR', # select reward (-1,1) aligned to selection
        'selHigh', # select higher probability port (-1,1)
        'Reward', # reward vs no reward during consumption period (-1,1)
        'post', # log-odds probability
    ]

    y_col = 'grnL_diff'

    dfrel = df[X_cols + [y_col]].copy()
    dfrel = dfrel.replace('False', 0).astype(float)
    dfrel = dfrel*1

    X_setup = dfrel[X_cols]
    y_setup = dfrel[y_col]

    ts = 2

    shift_amt_list = [0]
    shift_amt_list += list(range(-ts, 0))
    shift_amt_list += list(range(1, ts+1))

    dfrel = sglm_pp.timeshift_multiple(X_setup, shift_amt_list=shift_amt_list)

    print('Timeshift Multiple Done...')
    
    full_dataset = dfrel.copy()
    full_dataset['grnL_diff'] = y_setup
    full_dataset['grnL_sft'] = y_setup.shift(1)
    full_dataset['grnL_sft2'] = y_setup.shift(2)
    full_dataset = full_dataset.iloc[5:]
    full_dataset = full_dataset.dropna().copy()

    X = full_dataset.drop(y_col, axis=1)
    y = full_dataset[y_col]
    
    print('SKGLM Starting...')

    use_cols = list(range(16, 20))
    
    # print(list(X.columns))
    column_names = X.columns[use_cols]
    print(column_names)
    # glm3 = sglm.SKGLM('Normal', max_iter=10000, alpha=0)
    # glm3.fit(X.values[1:, use_cols], y.values[1:])

    glm4 = LinearRegression()
    glm4.fit(np.float64(X.values[1:, use_cols]), np.float64(y.values[1:]))
    glm3 = glm4

    # pred = glm3.model.predict(X.values)
    # true = y.values

    print('SKGLM Done...')
    sk_int = glm3.intercept_
    sk_coef = glm3.coef_

    print('Starting PY GLM Net Done...')

    glm = sglm.GLM('Normal', max_iter=10000, reg_lambda=0, alpha=0)
    glm.fit(np.float64(X.values[1:, use_cols]), np.float64(y.values[1:]))

    print('PY GLM Net Done...')

    print(glm.intercept_, glm.coef_)



    sk_int = glm3.intercept_
    pgn_int = glm.intercept_
    
    sk_coef = glm3.coef_
    pgn_coef = glm.coef_

    int_vals = [(sk_int, pgn_int)]
    coef_vals = [(column_names[_], sk_coef[_], pgn_coef[_]) for _ in range(len(use_cols))]

    print(int_vals + coef_vals)


    # Problem Columns...
    # > 'nTrial', 'iBlock'
    [
        'CuePenalty',
        'Penalty',
        'Select',
        'TO'
    ]
    [
        'switch',
        'selHigh'
    ]