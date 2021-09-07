import pytest

import sys
import os 
dir_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/backend')
sys.path.append(f'{dir_path}/../backend')

import sglm
import sglm_cv
import sglm_pp
import sklearn.linear_model
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.model_selection
from sklearn.linear_model import PoissonRegressor



# Normal (OLS)
def test_normal_ols(epsilon=0.01):

    print('Testing Normal GLM...')

    true_x = np.linspace(-3, 3, 21)
    true_y = true_x * 0.5 + 0.5
    obs_y = (true_y + (-1)**np.arange(len(true_x)) * 0.1)

    x = true_x[:,None]
    y = obs_y

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, alpha = 0.25)

    from sklearn.linear_model import LinearRegression
    sklr = LinearRegression()
    sklr.fit(x, y)

    glm = sglm.GLM('Normal', alpha=0, reg_lambda=0)
    glm.fit(x, y)
    coef, intercept = glm.coef_, glm.intercept_

    print(f"> SKL: Intercept — {sklr.intercept_}, Coefs — {sklr.coef_}")
    print(f"> GLM: Intercept — {glm.intercept_}, Coefs — {glm.coef_}")

    assert(np.abs(sklr.intercept_ - glm.intercept_) < epsilon)
    assert(np.all(np.abs(sklr.coef_ - glm.coef_) < epsilon))
    
    return 

# Poisson GLM
def test_poisson_glm(epsilon=0.01):

    print('Testing Poisson GLM...')

    true_x = np.linspace(-2, 2, 51)
    true_y = np.exp(true_x)
    obs_y = np.array([np.floor(np.exp(_)) for _ in true_x]).reshape(-1)

    x = true_x[:,None]
    y = obs_y

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, alpha = 0.25)

    sklr = PoissonRegressor(alpha=0)
    sklr.fit(x, y)

    plt.plot(true_x, sklr.predict(x), label='SKLearn')

    glm = sglm.GLM('Poisson', reg_lambda=0)
    glm.fit(x, y)
    coef, intercept = glm.coef_, glm.intercept_

    plt.plot(true_x, glm.model.predict(x), label='Sabatini GLM')
    plt.legend()

    view_x = np.linspace(x.min(), x.max(), num=100)
    view_y = np.exp(view_x*coef + intercept)

    print(f"> SKL: Intercept — {sklr.intercept_}, Coefs — {sklr.coef_}")
    print(f"> GLM: Intercept — {glm.intercept_}, Coefs — {glm.coef_}")

    assert(np.abs(sklr.intercept_ - glm.intercept_) < epsilon)
    assert(np.all(np.abs(sklr.coef_ - glm.coef_) < epsilon))

    return

# Logistic GLM
def test_logistic_glm():

    print('Testing Logistic GLM...')

    epsilon = 0.01

    z = np.linspace(-10, 10, 201)
    sigmoid_z = 1/(1+np.exp(-z))
    np.random.seed(117)
    sample = np.random.binomial(1,sigmoid_z)

    true_x = z #np.linspace(-10, 10, 101)
    true_y = sigmoid_z #(true_x > 0)*1.0
    obs_y = sample

    x = true_x[:,None]
    y = obs_y

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, alpha = 0.25)

    from sklearn.linear_model import LogisticRegression
    sklr = LogisticRegression(C=np.inf)
    sklr.fit(x, y)

    plt.plot(true_x, sklr.predict(x), label='SKLearn')

    glm = sglm.GLM('Logistic', reg_lambda=0.0, alpha=0, max_iter=10000)
    glm.fit(x, y)
    coef, intercept = glm.coef_, glm.intercept_

    plt.plot(true_x, glm.model.predict(x), label='Sabatini GLM')
    plt.legend()

    print(f"> SKL: Intercept — {sklr.intercept_}, Coefs — {sklr.coef_}")
    print(f"> GLM: Intercept — {glm.intercept_}, Coefs — {glm.coef_}")

    assert(np.abs(sklr.intercept_ - glm.intercept_) < epsilon)
    assert(np.all(np.abs(sklr.coef_ - glm.coef_) < epsilon))
    return

# def pull_and_format_test_data():
#     df = pd.read_csv('../../C39v2_sampleDesignMat.csv').drop('Unnamed: 0', axis=1).drop('index', axis=1)
#     y_setup_col = 'grnL' # photometry response
#     df['grnL_diff'] = sglm_pp.diff(df['grnL'])

#     # Demonstrative first 5 timesteps of photometry signal vs. differential
#     print(df[['grnL', 'grnL_diff']].head())

#     # Plotting original photometry output (excluding first timestep)
#     plt.figure()
#     df['grnL'].iloc[1:].plot(color='c')
#     plt.title('Original Photometry Signal vs. Time')
#     plt.ylabel('Original Photometry Output')
#     plt.xlabel('Timestep Index')

#     # Plotting photometry differential output (excluding first timestep)
#     plt.figure()
#     df['grnL_diff'].iloc[1:].plot(color='g')
#     plt.title('Differential Photometry Signal vs. Time')
#     plt.ylabel('Differential Photometry Output')
#     plt.xlabel('Timestep Index')

#     X_cols = [
#     'nTrial', # trial ID
#     'iBlock', # block number within session
#     'CuePenalty', # lick during cue period (no directionality yet, so binary 0,1)
#     'ENLPenalty', # lick during ENL period (no directionality yet, 0,1)
#     'Select', # binary selection lick
#     'Consumption', # consumption period (from task perspective)
#     'TO', # timeout trial
#     'responseTime', # task state cue to selection window
#     'ENL', # task state ENL window
#     'Cue', # task state Cue window
#     'decision', # choice lick direction (aligned to select but with directionality -1,1)
#     'switch', # switch from previous choice on selection (-1,1)
#     'selR', # select reward (-1,1) aligned to selection
#     'selHigh', # select higher probability port (-1,1)
#     'Reward', # reward vs no reward during consumption period (-1,1)
#     'post', # log-odds probability
#     ]

#     y_col = 'grnL_diff'

#     dfrel = df[X_cols + [y_col]].copy()
#     dfrel = dfrel.replace('False', 0).astype(float)
#     dfrel = dfrel*1

#     X_setup = dfrel[X_cols]
#     y_setup = dfrel[y_col]

#     # ts = 10
#     ts = 2

#     shift_amt_list = [0]
#     shift_amt_list += list(range(-ts, 0))
#     shift_amt_list += list(range(1, ts+1))

#     dfrel = sglm_pp.timeshift_multiple(X_setup, shift_amt_list=shift_amt_list)

#     with pd.option_context('max_columns',None):
#         print('Example First 5 Rows of Timeshifted Columns:')
#         print(dfrel[['Cue', 'Cue_1', 'Cue_2', 'Cue_-1', 'Cue_-2']].head())
#         print('Example Last 5 Rows of Timeshifted Columns:')
#         print(dfrel[['Cue', 'Cue_1', 'Cue_2', 'Cue_-1', 'Cue_-2']].tail())

#     full_dataset = dfrel.copy()
#     full_dataset['grnL_diff'] = y_setup
#     full_dataset['grnL_sft'] = y_setup.shift(1)
#     full_dataset['grnL_sft2'] = y_setup.shift(2)
#     full_dataset = full_dataset.iloc[5:]
#     full_dataset = full_dataset.dropna().copy()

#     return full_dataset



# if __name__ == '__main__':
#     test_normal_ols()
#     test_poisson_glm()
#     test_logistic_glm()
#     test_normal_ols_cv()

#     full_dataset = pull_and_format_test_data()
    