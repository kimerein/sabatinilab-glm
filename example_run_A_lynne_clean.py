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
import sglm_plt as splt

import cProfile

def define_trial_starts_ends(df, trial_shift_bounds=7):
    '''
    Define trial starts and ends.
    Args:
        df: dataframe with trial_start and trial_end columns
        trial_shift_bounds: define how many timesteps before / after first & last event to include as non-ITI
    Returns:
        dataframe with added nTrial and nEndTrial columns to identify the number of the trial counts for start & end
    '''
    df['event_col_a'] = ((df['cpo'].diff() > 0)*1).replace(0, np.nan) * 1.0
    df['event_col_b'] = df['nr'].replace(0, np.nan) * 2.0
    df['event_col_c'] = df['r'].replace(0, np.nan) * 3.0
    df['event_col'] = df['event_col_a'].combine_first(df['event_col_b']).combine_first(df['event_col_c'])
    df['event_col'] = df['event_col'].bfill()
    df['trial_start_flag'] = ((df['event_col'] == 1.0)&(df['event_col'].shift(-1) != df['event_col']) * 1.0).shift(-trial_shift_bounds) * 1.0
    df['nTrial'] = df['trial_start_flag'].cumsum()
    df['event_col_d'] = ((df['lpx'] > 0)*1.0).replace(0, np.nan) * 1.0
    df['event_col_e'] = ((df['rpx'] > 0)*1.0).replace(0, np.nan) * 1.0
    df['event_col_end'] = df['event_col_d'].combine_first(df['event_col_e']).combine_first(df['trial_start_flag'].replace(0.0, np.nan)*2.0)
    df['event_col_end'] = df['event_col_end'].ffill()
    df['trial_end_flag'] = ((df['event_col_end'] == 1.0)&(df['event_col_end'].shift(1) == 2.0)&(df['event_col_end'].shift(1) != df['event_col_end'])&(df['nTrial'] > 0) * 1.0).shift(trial_shift_bounds) * 1.0
    df['nEndTrial'] = df['trial_end_flag'].cumsum()
    return df.drop(['event_col_a', 'event_col_b', 'event_col_c', 'event_col_d', 'event_col_e'], axis=1)

def to_profile():

    start = time.time()

    # List of files in directory "dir_path" to be processed
    files_list = [
        'dlight_only_WT36L_12172020.csv',
        'dlight_only_WT36L_12212020.csv',
        'dlight_only_WT36L_12242020.csv',
        'dlight_only_WT36L_12292020.csv',

        'Ach_only_WT53L_08262021xlsx.csv',
        'Ach_only_WT53L_09012021xlsx.csv',
        'Ach_only_WT53L_09032021xlsx.csv',
        'Ach_only_WT53L_09062021xlsx.csv',
    ]
    res = {}

    # Loop through files to be processed
    for filename in files_list:
        
        # Load file
        df = pd.read_csv(f'{dir_path}/../{filename}')
        df = df[[_ for _ in df.columns if 'Unnamed' not in _]]

        # Simplify variable names
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


                        'dF/F green (Ach3.0)': 'gdFF',
                        'zscored green (Ach3.0)': 'zsgdFF',

                        'dF/F green (dLight1.1)': 'gdFF',
                        'zscored green (dLight1.1)': 'zsgdFF',

                        'dF/F green (dlight1.1)': 'gdFF',
                        'zscored green (dlight1.1)': 'zsgdFF'
                        }, axis=1)
        

        df = define_trial_starts_ends(df)

        wi_trial_keep = (df['nTrial'] != df['nEndTrial'])

        if 'index' in df.columns:
            df = df.drop('index', axis=1)
        
        # Identify rewarded vs. unrewarded trials
        df['r_trial'] = df.groupby('nTrial')['r'].transform(np.sum)
        df['nr_trial'] = df.groupby('nTrial')['nr'].transform(np.sum)

        # Identify combined reward vs. non-rewarded / left vs. right / entries vs. exits
        df = df.assign(**{
            'rpxr':df['r_trial']*df['rpx'],
            'rpxnr':df['nr_trial']*df['rpx'],
            'lpxr':df['r_trial']*df['lpx'],
            'lpxnr':df['nr_trial']*df['lpx'],

            'rpnr':df['r_trial']*df['rpn'],
            'rpnnr':df['nr_trial']*df['rpn'],
            'lpnr':df['r_trial']*df['lpn'],
            'lpnnr':df['nr_trial']*df['lpn'],

        })


        # Select column names to use for GLM predictors
        X_cols = [
        'nTrial',
        #    'index',
        
        'cpn', 'cpx',
        'lpn', 'rpn',
        # 'lpnr', 'rpnr',
        # 'lpnnr', 'rpnnr',
        'lpx', 'rpx',
        # 'lpxr', 'rpxr',
        # 'lpxnr', 'rpxnr',
        'll', 'rl',
        'nr', 'r',
        #'cpo',
        #'lpo',
        #'rpo',
        ]

        # Select column name to use for outcome variable
        y_col = 'zsgdFF'

        # Simplify dataframe fro training
        dfrel = df[X_cols + [y_col]].copy()
        dfrel = dfrel.replace('False', 0).astype(float)
        dfrel = dfrel*1
        
        # Timeshift X_cols forward by pos_order times and backward by neg_order times
        neg_order = -7
        pos_order = 7
        dfrel = sglm_ez.timeshift_cols(dfrel, X_cols[1:], neg_order=neg_order, pos_order=pos_order)
        X_cols_sftd = sglm_ez.add_timeshifts_to_col_list(X_cols, X_cols[1:], neg_order=neg_order, pos_order=pos_order)

        # Drop NAs for non-existant timeshifts
        dfrel = dfrel.dropna()
        X_setup = dfrel[X_cols_sftd].copy()
        y_setup = dfrel[y_col].copy()

        # Fit GLM only on the non-ITI data
        X_setup = X_setup[wi_trial_keep]
        y_setup = y_setup[wi_trial_keep]

        # Split data into setup (training) and holdout (test) sets
        holdout = sglm_ez.holdout_split_by_trial_id(X_setup, y_setup, id_cols=['nTrial'], perc_holdout=0.2)
        X_holdout = X_setup.loc[holdout]
        y_holdout = y_setup.loc[holdout]
        X_setup = X_setup.loc[~holdout]
        y_setup = y_setup.loc[~holdout]


        #######################
        #######################
        #######################

        # Generate cross-validation (technically, group / shuffle split) sets for training / model selection
        kfold_cv_idx = sglm_ez.cv_idx_by_trial_id(X_setup, y=y_setup, trial_id_columns=['nTrial'], num_folds=50, test_size=0.2)

        # Drop nTrial column from X_setup. (It is only used for group identification in group/shuffle/split)
        X_setup = X_setup[[_ for _ in X_setup.columns if _ not in ['nTrial']]]

        

        score_method = 'r2'        

        # Select hyper parameters for GLM to use for model selection
        # Step 1: Create a dictionary of lists for these relevant keywords...
        kwargs_iterations = {
            # # 'alpha': [0.0, 0.01, 0.1, 0.5, 0.9, 1.0],
            # 'alpha': [0.0, 0.01, 0.1, 0.5, 0.9, 1.0],
            # 'l1_ratio': [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0]

            'alpha': [0.0, 1.0],
            'l1_ratio': [0.0, 1.0]
        }

        # Step 2: Create a dictionary for the fixed keyword arguments that do not require iteration...
        kwargs_fixed = {
            'max_iter': 1000,
            'fit_intercept': True
        }

        # Step 3: Generate iterable list of keyword sets for possible combinations
        glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)
        best_score, best_score_std, best_params, best_model, cv_results = sglm_ez.simple_cv_fit(X_setup, y_setup, kfold_cv_idx, glm_kwarg_lst, model_type='Normal', verbose=0, score_method=score_method)

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

        # Refit the best model on the full setup (training) data
        glm = sglm_ez.fit_GLM(X_setup, y_setup, **best_params)

        # Get the R^2 and MSE scores for the best model on the holdout (test) data
        holdout_score = glm.r2_score(X_holdout[X_setup.columns], y_holdout)
        holdout_neg_mse_score = glm.neg_mse_score(X_holdout[X_setup.columns], y_holdout)

        # Collect
        res[filename] = {'holdout_score':holdout_score,
                        'holdout_neg_mse_score':holdout_neg_mse_score,
                        'best_score':best_score,
                        'best_params':best_params,
                        'all_models':sorted([(_['cv_R2_score'],
                                              _['cv_mse_score'],
                                              sglm_ez.calc_l1(_['cv_coefs']),
                                              sglm_ez.calc_l2(_['cv_coefs']),
                                              _['glm_kwargs']) for _ in cv_results['full_cv_results']], key=lambda x: -x[0])
                        }
        print(f'Holdout Score: {holdout_score}')

        # Generate and save plots of the beta coefficients
        X_cols_plot = [_ for _ in X_cols if _ in X_setup.columns]
        X_cols_sftd_plot = [_ for _ in X_cols_sftd if _ in X_setup.columns]
        fn = filename.split(".")[0]
        splt.plot_all_beta_coefs(glm, X_cols_plot,
                                        X_cols_sftd_plot,
                                        plot_width=2,
                                        y_lims=(-2.0, 2.0),
                                        # filename=f'{fn}_coeffs.png',
                                        binsize=50,
                                        filename=f'{fn}_coeffs_R2_{np.round(holdout_score, 4)}.png',
                                        plot_name=f'{fn} — {best_params}'
                                        )
        






        tmp = X_holdout.set_index('nTrial').copy()
        tmp_y = y_holdout.copy()
        tmp_y.index = tmp.index
        tmp[y_holdout.name] = tmp_y

        tmp['1'] = 1
        tmp['tim'] = tmp.groupby('nTrial')['1'].cumsum()
        tmp['pred'] = glm.predict(tmp[X_setup.columns])

        # print(tmp)

        entry_timing_r = tmp.groupby('nTrial')['rpn'].agg(lambda x: x.argmax()).astype(int)
        entry_timing_l = tmp.groupby('nTrial')['lpn'].agg(lambda x: x.argmax()).astype(int)
        entry_timing = (entry_timing_r > entry_timing_l)*entry_timing_r + (entry_timing_r < entry_timing_l)*entry_timing_l

        adjusted_time = (tmp['tim'] - entry_timing)
        # print(adjusted_time)
        tmp['adjusted_time'] = adjusted_time
        adjusted_time.index = tmp.index

        entry_timing_c = tmp.groupby('nTrial')['cpn'].agg(lambda x: x.argmax()).astype(int)
        adjusted_time_c = (tmp['tim'] - entry_timing_c)
        adjusted_time_c.index = tmp.index
        tmp['cpn_adjusted_time'] = adjusted_time_c

        splt.plot_avg_reconstructions(tmp, binsize = 50, min_time = -20, max_time = 30, min_signal = -3.0, max_signal = 3.0, file_name=f'figure_outputs/average_response_reconstruction_{filename[:-4]}.png')



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

to_profile()

# cProfile.run('to_profile()', sort='tottime')
