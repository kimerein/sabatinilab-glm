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
import random

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


def rename_columns(df):
    '''
    Simplify variable names to match the GLM
    Args:
        df: dataframe with entry, exit, lick, reward, and dFF columns
    Returns:
        dataframe with renamed columns
    '''
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
    return df

def set_reward_flags(df):
    '''
    Set reward flags
    Args:
        df: dataframe with trial_start and trial_end columns
    Returns:
        dataframe with added rewarded trial and not rewarded trial columns
    '''
    # Identify rewarded vs. unrewarded trials
    df['r_trial'] = df.groupby('nTrial')['r'].transform(np.sum)
    df['nr_trial'] = df.groupby('nTrial')['nr'].transform(np.sum)
    return df

def set_port_entry_exit_rewarded_unrewarded_indicators(df):
    '''
    Set port entry, exit, and reward indicators
    Args:
        df: dataframe with trial_start and trial_end columns
    Returns:
        dataframe with added port entry/exit, and reward indicators
    '''
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
    return df

def define_side_agnostic_events(df):
    '''
    Define side agnostic events
    Args:
        df: dataframe with trial_start and trial_end columns
    Returns:
        dataframe with added port entry/exit, and reward indicators
    '''
    df = df.assign(**{
        'spn':df['rpn']+df['lpn'],
        'spx':df['rpx']+df['lpx'],

        'spnr':df['rpnr']+df['lpnr'],
        'spnnr':df['rpnnr']+df['lpnnr'],
        'spxr':df['rpxr']+df['lpxr'],
        'spxnr':df['rpxnr']+df['lpxnr'],

        'sl':df['rl']+df['ll'],
    })

    return df

def overwrite_response_with_toy(dfrel):
    '''
    Overwrite response column with dummy column
    Args:
        df: dataframe with trial_start and trial_end columns
    Returns:
        dataframe with added port entry/exit, and reward indicators
    '''
    exp_decay_delta_rt = np.convolve(dfrel['rpn'], np.exp(-np.arange(4)), mode='full')[:len(dfrel)]
    exp_decay_delta_lft = np.convolve(dfrel['lpn'], np.exp(-np.arange(10)/3), mode='full')[:len(dfrel)]
    exp_decay_delta_ll = np.convolve(dfrel['ll'], 0.3*np.exp(-np.arange(3)), mode='full')[:len(dfrel)]
    exp_decay_delta_rl = np.convolve(dfrel['rl'], 0.5*np.exp(-np.arange(5)/3), mode='full')[:len(dfrel)]
    noise = np.random.normal(0, 0.1, len(dfrel))
    dfrel['zsgdFF'] = exp_decay_delta_rt + exp_decay_delta_lft + exp_decay_delta_ll + exp_decay_delta_rl + noise
    return dfrel


def timeshift_vals(dfrel, X_cols, neg_order=-7, pos_order=20):
    '''
    Timeshift values
    Args:
        df: dataframe with trial_start and trial_end columns
    Returns:
        dataframe with added port entry/exit, and reward indicators
    '''
    dfrel = sglm_ez.timeshift_cols(dfrel, X_cols[1:], neg_order=neg_order, pos_order=pos_order)
    X_cols_sftd = sglm_ez.add_timeshifts_to_col_list(X_cols, X_cols[1:], neg_order=neg_order, pos_order=pos_order)
    return dfrel, X_cols_sftd

def create_setup_dfs(df):
    '''
    Create setup dataframes
    Args:
        df: dataframe with trial_start and trial_end columns
    Returns:
        dataframe with added port entry/exit, and reward indicators
    '''
    # Create setup dataframes
    setup_dfs = {}
    setup_dfs['df_setup'] = df.loc[df['trial_start'].values, ['nTrial', 'r_trial', 'nr_trial', 'rpn', 'rpx', 'lpn', 'lpx', 'll', 'rl']]
    setup_dfs['df_setup'].columns = ['nTrial', 'r_trial', 'nr_trial', 'rpn', 'rpx', 'lpn', 'lpx', 'll', 'rl']
    setup_dfs['df_setup_sftd'] = df.loc[df['trial_start'].values, ['nTrial', 'r_trial', 'nr_trial', 'rpn', 'rpx', 'lpn', 'lpx', 'll', 'rl']]
    setup_dfs['df_setup_sftd'].columns = ['nTrial', 'r_trial', 'nr_trial', 'rpn', 'rpx', 'lpn', 'lpx', 'll', 'rl']

    return setup_dfs


def create_setup_dfs(dfrel, X_cols_sftd, y_col):
    '''
    Create setup dataframes
    Args:
        df: dataframe with trial_start and trial_end columns
    Returns:
        dataframe with added port entry/exit, and reward indicators
    '''
    wi_trial_keep = (dfrel['nTrial'] != dfrel['nEndTrial'])

    X_setup = dfrel[X_cols_sftd].copy()
    y_setup = dfrel[y_col].copy()

    # Fit GLM only on the non-ITI data
    X_setup = X_setup[wi_trial_keep]
    y_setup = y_setup[wi_trial_keep]
    dfrel_setup = dfrel[wi_trial_keep].copy()
    return X_setup, y_setup, dfrel_setup

def holdout_splits(X_setup, y_setup, dfrel_setup, id_cols=['nTrial'], perc_holdout=0.2):
    '''
    Create holdout splits
    Args:
        df: dataframe with trial_start and trial_end columns
    Returns:
        dataframe with added port entry/exit, and reward indicators
    '''
    # Create holdout splits
    holdout = sglm_ez.holdout_split_by_trial_id(X_setup, y_setup, id_cols=id_cols, perc_holdout=perc_holdout)
    # holdout = sglm_ez.holdout_split_by_trial_id(X_setup, y_setup, id_cols=['nTrial'], strat_col='strat_id', strat_mode='stratify', perc_holdout=0.2)
    X_holdout = X_setup.loc[holdout]
    y_holdout = y_setup.loc[holdout]
    X_setup = X_setup.loc[~holdout]
    y_setup = y_setup.loc[~holdout]

    dfrel_holdout = dfrel_setup.loc[holdout]
    dfrel_setup = dfrel_setup.loc[~holdout]
    return X_setup, y_setup, X_holdout, y_holdout, dfrel_setup, dfrel_holdout

def print_best_model_info(X_setup, best_score, best_params, best_model, start):
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

    # # Select column names to use for GLM predictors
    # X_cols = [
    # 'nTrial',
    # # #    'index',
    
    # # # 'cpn', 'cpx',
    
    # # # 'lpn', 'rpn',
    # # 'lpnr', 'rpnr',
    # # 'lpnnr', 'rpnnr',

    # # 'lpx', 'rpx',
    # # # 'lpxr', 'rpxr',
    # # # 'lpxnr', 'rpxnr',
    # # 'll', 'rl',
    # # # 'nr', 'r',
    # # #'cpo',
    # # #'lpo',
    # # #'rpo',

    # 'cpn', 'cpx',
    # # 'spn', 'spx',
    # 'spnr', 'spxr',
    # 'spnnr', 'spxnr',
    # 'sl',
    # ]

def training_fit_holdout_score(X_setup, y_setup, X_holdout, y_holdout, best_params):
    '''
    Fit GLM on training data, and score on holdout data
    Args:
        X_setup: dataframe with trial_start and trial_end columns
        y_setup: dataframe with trial_start and trial_end columns
        X_holdout: dataframe with trial_start and trial_end columns
        y_holdout: dataframe with trial_start and trial_end columns
        best_params: dictionary of best parameters
    Returns:
        best_score: best score
    '''
    # Refit the best model on the full setup (training) data
    glm = sglm_ez.fit_GLM(X_setup, y_setup, **best_params)

    # Get the R^2 and MSE scores for the best model on the holdout (test) data
    holdout_score = glm.r2_score(X_holdout[X_setup.columns], y_holdout)
    holdout_neg_mse_score = glm.neg_mse_score(X_holdout[X_setup.columns], y_holdout)

    return glm, holdout_score, holdout_neg_mse_score

def get_first_entry_time(tmp, X_setup):
    '''
    Get first entry time
    Args:
        df: dataframe with trial_start and trial_end columns
    Returns:
        first_entry_time: first entry time
    '''
    # Get first entry time

    tmp['1'] = 1
    tmp['tim'] = tmp.groupby('nTrial')['1'].cumsum()
    # tmp['pred'] = glm.predict(tmp[X_setup.columns])

    # print(tmp)

    entry_timing_r = tmp.groupby('nTrial')['ft_rpn'].transform(lambda x: x.argmax()).astype(int)
    entry_timing_l = tmp.groupby('nTrial')['ft_lpn'].transform(lambda x: x.argmax()).astype(int)
    entry_timing = (entry_timing_r > entry_timing_l)*entry_timing_r + (entry_timing_r < entry_timing_l)*entry_timing_l

    adjusted_time = (tmp['tim'] - entry_timing)
    # print(adjusted_time)
    tmp['adjusted_time'] = adjusted_time
    adjusted_time.index = tmp.index

    entry_timing_c = tmp.groupby('nTrial')['ft_cpn'].transform(lambda x: x.argmax()).astype(int)
    adjusted_time_c = (tmp['tim'] - entry_timing_c)
    adjusted_time_c.index = tmp.index
    tmp['cpn_adjusted_time'] = adjusted_time_c
    return tmp


def to_profile():

    start = time.time()

    # List of files in directory "dir_path" to be processed
    files_list = [
        'dlight_only_WT36L_12172020.csv',
        'dlight_only_WT36L_12212020.csv',
        'dlight_only_WT36L_12242020.csv',
        'dlight_only_WT36L_12292020.csv',

        # 'Ach_only_WT53L_08262021xlsx.csv',
        # 'Ach_only_WT53L_09012021xlsx.csv',
        # 'Ach_only_WT53L_09032021xlsx.csv',
        # 'Ach_only_WT53L_09062021xlsx.csv',
    ]
    res = {}

    # Loop through files to be processed
    for filename in files_list:
        
        # Load file
        df = pd.read_csv(f'{dir_path}/../{filename}')
        df = df[[_ for _ in df.columns if 'Unnamed' not in _]]
        df = rename_columns(df)
        df = define_trial_starts_ends(df)
        df = set_reward_flags(df)
        df = set_port_entry_exit_rewarded_unrewarded_indicators(df)
        df = define_side_agnostic_events(df)

        if 'index' in df.columns:
            df = df.drop('index', axis=1)

        

        # Select column names to use for GLM predictors
        X_cols = [
        'nTrial',
        'cpn', 'cpx',
        # 'spn', 'spx',
        'spnr', 'spxr',
        'spnnr', 'spxnr',
        'sl',
        ]

        # Select column name to use for outcome variable
        y_col = 'zsgdFF'

        # Simplify dataframe for training
        dfrel = df.copy()
        dfrel = dfrel.replace('False', 0).astype(float)
        dfrel = dfrel*1
        # dfrel = overwrite_response_with_toy(dfrel)


        # Timeshift X_cols forward by pos_order times and backward by neg_order times
        dfrel, X_cols_sftd = timeshift_vals(dfrel, X_cols, neg_order=-7, pos_order=20)

        # Drop NAs for non-existant timeshifts
        dfrel = dfrel.dropna()

        # Create setup dfs by dropping the inter-trial intervals
        X_setup, y_setup, dfrel_setup = create_setup_dfs(dfrel, X_cols_sftd, y_col)

        # Split data into setup (training) and holdout (test) sets
        np.random.seed(30186)
        random.seed(30186)
        X_setup, y_setup, X_holdout, y_holdout, dfrel_setup, dfrel_holdout = holdout_splits(X_setup, y_setup, dfrel_setup, id_cols=['nTrial'], perc_holdout=0.2)

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
            'alpha': [0.0, 1.0],
            'l1_ratio': [0.0, 0.001],
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

        print_best_model_info(X_setup, best_score, best_params, best_model, start)


        glm, holdout_score, holdout_neg_mse_score = training_fit_holdout_score(X_setup, y_setup, X_holdout, y_holdout, best_params)

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
        # print('Keys:',)

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
        

        # dfrel_holdout
        # dfrel_setup


        # print('dfrel', list(X_setup.columns))

        
        dfrel_holdout['nn'] = dfrel_holdout[['lpn', 'rpn']].sum(axis=1)
        dfrel_holdout['xx'] = dfrel_holdout[['lpx', 'rpx']].sum(axis=1)

        first_trans = dfrel_holdout.groupby('nTrial')[['nn', 'xx', 'lpn', 'rpn', 'lpx', 'rpx', 'cpn']].cumsum()
        first_trans = ((first_trans == 1)*1).diff()
        first_trans *= first_trans >= 0
        first_trans['lpn'] = dfrel_holdout['nn']*dfrel_holdout['lpn']
        first_trans['rpn'] = dfrel_holdout['nn']*dfrel_holdout['rpn']
        first_trans['lpx'] = dfrel_holdout['xx']*dfrel_holdout['lpx']
        first_trans['rpx'] = dfrel_holdout['xx']*dfrel_holdout['rpx']

        first_trans = first_trans.rename({_k:f'ft_{_k}' for _k in first_trans.columns}, axis=1)
        dfrel_holdout[first_trans.columns] = first_trans



        dfrel_holdout['ft_r_rpn'] = dfrel_holdout['ft_rpn'] * dfrel_holdout['r']
        dfrel_holdout['ft_r_lpn'] = dfrel_holdout['ft_lpn'] * dfrel_holdout['r']
        dfrel_holdout['ft_nr_rpn'] = dfrel_holdout['ft_rpn'] * dfrel_holdout['nr']
        dfrel_holdout['ft_nr_lpn'] = dfrel_holdout['ft_lpn'] * dfrel_holdout['nr']






        for fitted_model_dict in cv_results['full_cv_results']:
            fitted_model = fitted_model_dict['model']
            
            # tmp_holdout_score = fitted_model.r2_score(X_holdout[X_setup.columns], y_holdout)
            tmp_holdout_score = fitted_model.r2_score(dfrel_holdout[X_setup.columns], y_holdout)

            # print("fitted_model_dict['glm_kwargs']",fitted_model_dict['glm_kwargs'])

            tmp = dfrel_holdout.set_index('nTrial').copy()
            tmp['pred'] = fitted_model.predict(tmp[X_setup.columns])
            tmp = get_first_entry_time(tmp, X_setup)
            
            # tmp = X_holdout.set_index('nTrial').copy()
            tmp_y = y_holdout.copy()
            tmp_y.index = tmp.index
            tmp[y_holdout.name] = tmp_y


            kwarg_info = "_".join([f"{_k}_{fitted_model_dict['glm_kwargs'][_k]}" for _k in fitted_model_dict["glm_kwargs"]])

            splt.plot_avg_reconstructions(tmp, binsize = 50, min_time = -20, max_time = 30, min_signal = -3.0, max_signal = 3.0, file_name=f'figure_outputs/avg_resp_reconstruction_{filename[:-4]}_{kwarg_info}_R2_{np.round(tmp_holdout_score, 4)}.png')
            splt.plot_all_beta_coefs(fitted_model, X_cols_plot,
                                            X_cols_sftd_plot,
                                            plot_width=2,
                                            y_lims=(-2.0, 2.0),
                                            # filename=f'{fn}_coeffs.png',
                                            binsize=50,
                                            filename=f'figure_outputs/betas_{filename[:-4]}_{kwarg_info}_coeffs_R2_{np.round(tmp_holdout_score, 4)}.png',
                                            plot_name=f'{fn} — {kwarg_info}'
                                            )

            
            # print('entry_timing_r',entry_timing_r)
            # print('entry_timing_l',entry_timing_l)
            # print('entry_timing_c',entry_timing_c)
            
            



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
