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

prefix = '99'

def to_profile():
    global prefix
    start = time.time()


    ssave_folder = 'model_outputs/ssave'
    all_models_folder = 'model_outputs/all_models'
    all_data_folder = 'model_outputs/all_data'
    all_reconstruct_folder = 'model_outputs/all_reconstructions'
    all_coeffs_folder = 'model_outputs/all_coeffs'
    best_reconstruct_folder = 'model_outputs/best_reconstructions'
    best_coeffs_folder = 'model_outputs/best_coeffs'

    # prefix = '99'
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
        'max_iter': 1000,
        'fit_intercept': True
    }

    neg_order, pos_order = -14, 14
    folds = 50
    pholdout = 0.2
    pgss = 0.2

    # Step 3: Generate iterable list of keyword sets for possible combinations
    glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)

    res = {}

    leave_one_out_list = [[]]
    # leave_one_out_list = [[]] + [[_] for _ in X_cols_all]

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
        df['wi_trial_keep'] = get_is_not_iti(df)

        glmsave.set_basedata(df)

        for y_col in ['zsrdFF']:
        # for y_col in tqdm(['zsrdFF', 'zsgdFF'], 'ycol'):


            for _ in range(20):
                prefix = str(10000+_)    


                df = detrend(df, y_col)

                for left_out in tqdm(leave_one_out_list, 'left_out'):
                    
                    X_cols = [_ for _ in X_cols_all if _ not in left_out]

                    if len(leave_one_out_list) > 1:
                        run_id = f'{prefix}_{fn}_{y_col}_drop={"_".join(left_out)}'
                    else:
                        run_id = f'{prefix}_{fn}_{y_col}'

                    print("Run ID:", run_id)
                    dfrel = df.copy()
                    
                    # Timeshift X_cols forward by pos_order times and backward by neg_order times
                    dfrel, X_cols_sftd = timeshift_vals(dfrel, X_cols, neg_order=neg_order, pos_order=pos_order)
                    dfrel = dfrel.dropna()
                    dfrel_setup, dfrel_holdout = holdout_splits(dfrel,
                                                                id_cols=['nTrial'],
                                                                perc_holdout=pholdout)

                    # Generate cross-validation (technically, group / shuffle split) sets for training / model selection
                    kfold_cv_idx = sglm_ez.cv_idx_by_trial_id(dfrel_setup,
                                                            trial_id_columns=['nTrial'],
                                                            num_folds=folds,
                                                            test_size=pgss)

                    prediction_X_cols = [_ for _ in X_cols if _ not in ['nTrial']]
                    prediction_X_cols_sftd = [_ for _ in X_cols_sftd if _ not in ['nTrial']]

                    X_setup = get_x(dfrel_setup, prediction_X_cols_sftd, keep_rows=None)
                    y_setup = get_y(dfrel_setup, y_col, keep_rows=None)

                    X_setup = X_setup.sample(frac=1.0, replace=True)
                    y_setup = y_setup.loc[X_setup.index]

                    best_score, best_score_std, best_params, best_model, cv_results = sglm_ez.simple_cv_fit(X_setup, y_setup, kfold_cv_idx, glm_kwarg_lst, model_type='Normal', verbose=0, score_method=score_method)
                    
                    print_best_model_info(X_setup, best_score, best_params, best_model, start)
                    
                    X_holdout_witi = get_x(dfrel_holdout, prediction_X_cols_sftd, keep_rows=None)
                    y_holdout_witi = get_y(dfrel_holdout, y_col, keep_rows=None)
                    X_holdout_noiti = get_x(dfrel_holdout, prediction_X_cols_sftd, keep_rows=dfrel_holdout['wi_trial_keep'])
                    y_holdout_noiti = get_y(dfrel_holdout, y_col, keep_rows=dfrel_holdout['wi_trial_keep'])
                    glm, holdout_score, holdout_neg_mse_score = training_fit_holdout_score(X_setup, y_setup, X_holdout_noiti, y_holdout_noiti, best_params)

                    # Collect
                    res[f'{run_id}'] = {'holdout_score':holdout_score,
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
                    X_cols_plot = prediction_X_cols
                    X_cols_sftd_plot = prediction_X_cols_sftd


                    holdout_score_rnd = np.round(holdout_score, 4)
                    best_beta_fn = f'{best_coeffs_folder}/{run_id}_best_{all_betas_basename}_R2_{holdout_score_rnd}.png'
                    splt.plot_all_beta_coefs(glm.coef_, X_cols_plot,
                                                    X_cols_sftd_plot,
                                                    plot_width=2,
                                                    y_lims=(-2.0, 2.0),
                                                    # filename=f'{fn}_coeffs.png',
                                                    binsize=50,
                                                    filename=best_beta_fn,
                                                    plot_name=f'Best Coeffs - {run_id} — {best_params}'
                                                    )

                    tmp = dfrel_holdout.set_index('nTrial').copy()
                    tmp['pred'] = glm.predict(get_x(dfrel_holdout, prediction_X_cols_sftd))
                    tmp = get_first_entry_time(tmp)


                    best_beta_fn = f'{best_reconstruct_folder}/{run_id}_best_{avg_reconstruct_basename}_R2_{holdout_score_rnd}.png'
                    splt.plot_avg_reconstructions(tmp,
                                                y_col=y_col,
                                                binsize = 50,
                                                min_time = -20,
                                                max_time = 30,
                                                min_signal = -3.0,
                                                max_signal = 3.0,
                                                file_name=best_beta_fn,
                                                title=f'Best Average Reconstruction - {run_id} — {best_params}'
                                                )
                    
                    for fitted_model_dict in tqdm(cv_results['full_cv_results'], 'cv_results'):
                        fitted_model = fitted_model_dict['model']
                        kwarg_info = "_".join([f"{_k}_{fitted_model_dict['glm_kwargs'][_k]}" for _k in fitted_model_dict["glm_kwargs"]])

                        model_coef = fitted_model.coef_
                        model_intercept = fitted_model.intercept_

                        std_name = f'{run_id}_{kwarg_info}'
                        np.save(f'{all_models_folder}/coeffs/{std_name}_{model_c_basename}.npy', model_coef)
                        np.save(f'{all_models_folder}/intercepts/{std_name}_{model_i_basename}.npy', model_intercept)
                        
                        tmp_holdout_score = fitted_model.r2_score(X_holdout_noiti, y_holdout_noiti)

                        glmsave.append_fit_results(y_col, fitted_model_dict["glm_kwargs"], glm_model=fitted_model, dropped_cols=left_out,
                                                scores={
                                                    'tr_witi':None,
                                                    'tr_noiti':None,
                                                    'gss_witi':fitted_model_dict['cv_R2_score'],
                                                    'gss_noiti':None,
                                                    'holdout_witi':fitted_model.r2_score(X_holdout_witi, y_holdout_witi),
                                                    'holdout_noiti':fitted_model.r2_score(X_holdout_noiti, y_holdout_noiti)
                                                },
                                                gssids=kfold_cv_idx)

                        tmp = dfrel_holdout.set_index('nTrial').copy()
                        tmp['pred'] = fitted_model.predict(get_x(dfrel_holdout, prediction_X_cols_sftd, keep_rows=None))
                        tmp = get_first_entry_time(tmp)
                        tmp_y = get_y(dfrel_holdout, y_col, keep_rows=None).copy()
                        tmp_y.index = tmp.index
                        tmp[y_holdout_noiti.name] = tmp_y

                        tmp.to_csv(f'{all_data_folder}/{std_name}_{tmp_data_basename}.csv')

                        holdout_score_rnd = np.round(tmp_holdout_score, 4)
                        splt.plot_avg_reconstructions(tmp,
                                                    y_col=y_col,
                                                    binsize = 50,
                                                    min_time = -20,
                                                    max_time = 30,
                                                    min_signal = -3.0,
                                                    max_signal = 3.0,
                                                    file_name=f'{all_reconstruct_folder}/{std_name}_{avg_reconstruct_basename}_R2_{holdout_score_rnd}.png',
                                                    title=f'Average Reconstruction - {run_id} — {kwarg_info}'
                                                    )
                        splt.plot_all_beta_coefs(fitted_model.coef_, X_cols_plot,
                                                        X_cols_sftd_plot,
                                                        plot_width=2,
                                                        y_lims=(-2.0, 2.0),
                                                        # filename=f'{fn}_coeffs.png',
                                                        binsize=50,
                                                        filename=f'{all_coeffs_folder}/{std_name}_{all_betas_basename}_R2_{holdout_score_rnd}.png',
                                                        plot_name=f'Coeffs by Timeshift - {run_id} — {kwarg_info}'
                                                        # plot_name=f'{fn} — {y_col} — {kwarg_info}'
                                                        )
                        
                        plt.close('all')
                    plt.close('all')
                    

            glmsave.save()

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


        print('X_cols_plot:', X_cols_plot)
        print('X_cols_sftd_plot:', X_cols_sftd_plot)

        end = time.time()
        print('Runtime:',end-start)

if __name__ == '__main__':
    # main()
    profile = cProfile.run('to_profile()', filename=f'./{prefix}_profile_val.prof', sort='cumtime')
    # pickle.dump(profile, open('./profile.pkl', 'wb'))

        # def overwrite_response_with_toy(dfrel, y_col='zsgdFF'):
        #     '''
        #     Overwrite response column with dummy column
        #     Args:
        #         dfrel: dataframe with rpn, lpn, ll, and rl columns
        #     Returns:
        #         dataframe with added port entry/exit, and reward indicators
        #     '''
        #     exp_decay_delta_rt = np.convolve(dfrel['rpn'], np.exp(-np.arange(4)), mode='full')[:len(dfrel)]
        #     exp_decay_delta_lft = np.convolve(dfrel['lpn'], np.exp(-np.arange(10)/3), mode='full')[:len(dfrel)]
        #     exp_decay_delta_ll = np.convolve(dfrel['ll'], 0.3*np.exp(-np.arange(3)), mode='full')[:len(dfrel)]
        #     exp_decay_delta_rl = np.convolve(dfrel['rl'], 0.5*np.exp(-np.arange(5)/3), mode='full')[:len(dfrel)]
        #     noise = np.random.normal(0, 0.1, len(dfrel))
        #     dfrel[y_col] = exp_decay_delta_rt + exp_decay_delta_lft + exp_decay_delta_ll + exp_decay_delta_rl + noise
        #     return dfrel



        # # def create_setup_dfs(df):
        # #     '''
        # #     Create setup dataframes
        # #     Args:
        # #         df: dataframe with trial_start and trial_end columns
        # #     Returns:
        # #         dataframe with added port entry/exit, and reward indicators
        # #     '''
        # #     # Create setup dataframes
        # #     setup_dfs = {}
        # #     setup_dfs['df_setup'] = df.loc[df['trial_start'].values, ['nTrial', 'r_trial', 'nr_trial', 'rpn', 'rpx', 'lpn', 'lpx', 'll', 'rl']]
        # #     setup_dfs['df_setup'].columns = ['nTrial', 'r_trial', 'nr_trial', 'rpn', 'rpx', 'lpn', 'lpx', 'll', 'rl']
        # #     setup_dfs['df_setup_sftd'] = df.loc[df['trial_start'].values, ['nTrial', 'r_trial', 'nr_trial', 'rpn', 'rpx', 'lpn', 'lpx', 'll', 'rl']]
        # #     setup_dfs['df_setup_sftd'].columns = ['nTrial', 'r_trial', 'nr_trial', 'rpn', 'rpx', 'lpn', 'lpx', 'll', 'rl']

        # #     return setup_dfs


        # def create_setup_dfs(dfrel, X_cols_sftd, y_col, drop_iti=False):
        #     '''
        #     Create setup dataframes
        #     Args:
        #         dfrel: full dataframe
        #         X_cols_sftd: list of post-shifted columns
        #         y_col: response column
        #         drop_iti: whether to drop iti columns
        #     Returns:
        #         X_setup: setup prediction dataframe
        #         y_setup: setup response column
        #         dfrel_setup: full setup dataframe
        #         wi_trial_keep: boolean array of which timesteps fall within the trial (True for within trial, False for ITI)
        #     '''
        #     wi_trial_keep = (dfrel['nTrial'] != dfrel['nEndTrial'])

        #     X_setup = dfrel[X_cols_sftd].copy()
        #     y_setup = dfrel[y_col].copy()

        #     if drop_iti:
        #         # Fit GLM only on the non-ITI data
        #         X_setup = X_setup[wi_trial_keep].copy()
        #         y_setup = y_setup[wi_trial_keep].copy()
        #         dfrel_setup = dfrel[wi_trial_keep].copy()
        #     else:
        #         # Fit GLM only on the non-ITI data
        #         X_setup = X_setup.copy()
        #         y_setup = y_setup.copy()
        #         dfrel_setup = dfrel.copy()

        #     return X_setup, y_setup, dfrel_setup, wi_trial_keep

        # def holdout_splits(X_setup, y_setup, dfrel_setup, id_cols=['nTrial'], perc_holdout=0.2, drop_iti_train=False, drop_iti_holdout=False, wi_trial_keep=None):
        #     '''
        #     Create holdout splits
        #     Args:
        #         X_setup: setup prediction dataframe
        #         y_setup: setup response column
        #         dfre_setup: full setup dataframe
        #         id_cols: list of columns to use as trial identifiers
        #         perc_holdout: percentage of data to holdout
        #         drop_iti_train: whether to drop iti columns in training data
        #         drop_iti_holdout: whether to drop iti columns in holdout data
        #         wi_trial_keep: boolean array of which timesteps fall within the trial (True for within trial, False for ITI)
        #     Returns:
        #         X_setup: setup prediction dataframe
        #         y_setup: setup response column
        #         X_holdout: holdout prediction dataframe
        #         y_holdout: holdout response column
        #         dfrel_setup: full setup dataframe
        #         dfrel_holdout: full holdout dataframe
        #     '''
        #     # Create holdout splits
        #     holdout = sglm_ez.holdout_split_by_trial_id(X_setup, y_setup, id_cols=id_cols, perc_holdout=perc_holdout)
        #     # holdout = sglm_ez.holdout_split_by_trial_id(X_setup, y_setup, id_cols=['nTrial'], strat_col='strat_id', strat_mode='stratify', perc_holdout=0.2)
        #     X_holdout = X_setup.loc[holdout]
        #     y_holdout = y_setup.loc[holdout]
        #     X_setup = X_setup.loc[~holdout]
        #     y_setup = y_setup.loc[~holdout]
        #     dfrel_holdout = dfrel_setup.loc[holdout]
        #     dfrel_setup = dfrel_setup.loc[~holdout]

        #     wi_trial_keep_tr = wi_trial_keep.loc[~holdout]
        #     wi_trial_keep_holdout = wi_trial_keep.loc[holdout]

        #     if drop_iti_train:
        #         X_setup = X_setup[wi_trial_keep_tr]
        #         y_setup = y_setup[wi_trial_keep_tr]
        #         dfrel_setup = dfrel_setup[wi_trial_keep_tr]
        #     if drop_iti_holdout:
        #         X_holdout = X_holdout[wi_trial_keep_holdout]
        #         y_holdout = y_holdout[wi_trial_keep_holdout]
        #         dfrel_holdout = dfrel_holdout[wi_trial_keep_holdout]

        #     return X_setup, y_setup, X_holdout, y_holdout, dfrel_setup, dfrel_holdout, wi_trial_keep_tr, wi_trial_keep_holdout

        # def print_best_model_info(X_setup, best_score, best_params, best_model, start):
        #     """
        #     Print best model info
        #     Args:
        #         X_setup: setup prediction dataframe
        #         best_score: best score
        #         best_params: best parameters
        #         best_model: best model
        #         start: start time
        #     """

        #     # Print out all non-zero coefficients
        #     print('Non-Zero Coeffs:')
        #     epsilon = 1e-10
        #     for ic, coef in enumerate(best_model.coef_):
        #         if np.abs(coef) > epsilon:
        #             print(f'> {coef}: {X_setup.columns[ic]}')

        #     # Print out information related to the best model
        #     print(f'Best Score: {best_score}')
        #     print(f'Best Params: {best_params}')
        #     print(f'Best Model: {best_model}')
        #     print(f'Best Model — Intercept: {best_model.intercept_}')

        #     # Print out runtime information
        #     print(f'Overall RunTime: {time.time() - start}')

        #     print()
        #     return

        #     # # Select column names to use for GLM predictors
        #     # X_cols = [
        #     # 'nTrial',
        #     # # #    'index',
            
        #     # # # 'cpn', 'cpx',
            
        #     # # # 'lpn', 'rpn',
        #     # # 'lpnr', 'rpnr',
        #     # # 'lpnnr', 'rpnnr',

        #     # # 'lpx', 'rpx',
        #     # # # 'lpxr', 'rpxr',
        #     # # # 'lpxnr', 'rpxnr',
        #     # # 'll', 'rl',
        #     # # # 'nr', 'r',
        #     # # #'cpo',
        #     # # #'lpo',
        #     # # #'rpo',

        #     # 'cpn', 'cpx',
        #     # # 'spn', 'spx',
        #     # 'spnr', 'spxr',
        #     # 'spnnr', 'spxnr',
        #     # 'sl',
        #     # ]

        # def training_fit_holdout_score(X_setup, y_setup, X_holdout, y_holdout, best_params, wi_trial_keep_tr, wi_trial_keep_holdout):
        #     '''
        #     Fit GLM on training data, and score on holdout data
        #     Args:
        #         X_setup: X training data on which to fit model
        #         y_setup: response column for training data
        #         X_holdout: X holdout data on which to score model
        #         y_holdout: response column for holdout data
        #         best_params: dictionary of best parameters
        #     Returns:
        #         glm: Fitted model
        #         holdout_score: Score on holdout data
        #         holdout_neg_mse_score: Negative mean squared error on holdout data
        #     '''
        #     # Refit the best model on the full setup (training) data
        #     glm = sglm_ez.fit_GLM(X_setup, y_setup, **best_params)

        #     # Get the R^2 and MSE scores for the best model on the holdout (test) data
        #     holdout_score = glm.r2_score(X_holdout[X_setup.columns][wi_trial_keep_holdout], y_holdout[wi_trial_keep_holdout])
        #     holdout_neg_mse_score = glm.neg_mse_score(X_holdout[X_setup.columns][wi_trial_keep_holdout], y_holdout[wi_trial_keep_holdout])

        #     return glm, holdout_score, holdout_neg_mse_score

        # def get_first_entry_time(tmp, _=None):
        #     '''
        #     Get first entry time
        #     Args:
        #         tmp: dataframe with ITI removed, and first_time (ft_rpn / ft_lpn / ft_cpn) columns defined
        #     Returns:
        #         dataframe with added time_adjusted columns releatvive to first entry
        #     '''
        #     # Get first entry time

        #     tmp['1'] = 1
        #     tmp['tim'] = tmp.groupby('nTrial')['1'].cumsum()
        #     # tmp['pred'] = glm.predict(tmp[X_setup.columns])

        #     # print(tmp)

        #     entry_timing_r = tmp.groupby('nTrial')['ft_rpn'].transform(lambda x: x.argmax()).astype(int)
        #     entry_timing_l = tmp.groupby('nTrial')['ft_lpn'].transform(lambda x: x.argmax()).astype(int)
        #     entry_timing = (entry_timing_r > entry_timing_l)*entry_timing_r + (entry_timing_r < entry_timing_l)*entry_timing_l

        #     adjusted_time = (tmp['tim'] - entry_timing)
        #     # print(adjusted_time)
        #     tmp['adjusted_time'] = adjusted_time
        #     adjusted_time.index = tmp.index

        #     entry_timing_c = tmp.groupby('nTrial')['ft_cpn'].transform(lambda x: x.argmax()).astype(int)
        #     adjusted_time_c = (tmp['tim'] - entry_timing_c)
        #     adjusted_time_c.index = tmp.index
        #     tmp['cpn_adjusted_time'] = adjusted_time_c
        #     return tmp


        # def to_profile():

        #     start = time.time()

        #     # # List of files in directory "dir_path" to be processed
        #     # files_list = [
        #     #     'dlight_only_WT36L_12172020.csv',
        #     #     'dlight_only_WT36L_12212020.csv',
        #     #     'dlight_only_WT36L_12242020.csv',
        #     #     'dlight_only_WT36L_12292020.csv',

        #     #     # 'Ach_only_WT53L_08262021xlsx.csv',
        #     #     # 'Ach_only_WT53L_09012021xlsx.csv',
        #     #     # 'Ach_only_WT53L_09032021xlsx.csv',
        #     #     # 'Ach_only_WT53L_09062021xlsx.csv',
        #     # ]

        #     files_list = [  #'dlight_only_WT63_11082021.csv',
        #                     #'dlight_only_WT63_11162021.csv',
        #                     #'dlight_only_WT63_11182021.csv',

        #                     'Ach_rDAh_WT63_11082021.csv',
        #                     'Ach_rDAh_WT63_11162021.csv',
        #                     'Ach_rDAh_WT63_11182021.csv'
        #                     ]

        #     res = {}

        #     for y_col in ['zsrdFF', 'zsgdFF']:

        #         # Loop through files to be processed
        #         for filename in files_list:

        #             # Select column name to use for outcome variable
        #             # y_col = 'zsrdFF'
                    
        #             # Load file
        #             df = pd.read_csv(f'{dir_path}/../{filename}')
        #             dfrel = preprocess_lynne(df)
                    
        #             # Select column names to use for GLM predictors
        #             X_cols = [
        #             'nTrial',
        #             'cpn', 'cpx',
        #             # 'spn', 'spx',
        #             'spnr', 'spxr',
        #             'spnnr', 'spxnr',
        #             'sl',
        #             ]

        #             # Timeshift X_cols forward by pos_order times and backward by neg_order times
        #             # dfrel, X_cols_sftd = timeshift_vals(dfrel, X_cols, neg_order=-7, pos_order=20)
        #             dfrel, X_cols_sftd = timeshift_vals(dfrel, X_cols, neg_order=-14, pos_order=14)
        #             # dfrel, X_cols_sftd = timeshift_vals(dfrel, X_cols, neg_order=-7, pos_order=7)

        #             # Drop NAs for non-existant timeshifts
        #             dfrel = dfrel.dropna()

        #             # Create setup dfs by dropping the inter-trial intervals (if drop_iti=False)
        #             # X_setup, y_setup, dfrel_setup = create_setup_dfs(dfrel, X_cols_sftd, y_col)
        #             X_setup, y_setup, dfrel_setup, wi_trial_keep = create_setup_dfs(dfrel, X_cols_sftd, y_col, drop_iti=False)

        #             # Split data into setup (training) and holdout (test) sets
        #             np.random.seed(30186)
        #             random.seed(30186)
        #             X_setup, y_setup, X_holdout, y_holdout, dfrel_setup, dfrel_holdout, wi_trial_keep_tr, wi_trial_keep_holdout = holdout_splits(X_setup, y_setup, dfrel_setup, id_cols=['nTrial'], perc_holdout=0.2, drop_iti_train=False, drop_iti_holdout=False, wi_trial_keep=wi_trial_keep)

        #             #######################
        #             #######################
        #             #######################

        #             # Generate cross-validation (technically, group / shuffle split) sets for training / model selection
        #             kfold_cv_idx = sglm_ez.cv_idx_by_trial_id(X_setup, y=y_setup, trial_id_columns=['nTrial'], num_folds=50, test_size=0.2)

        #             # Drop nTrial column from X_setup. (It is only used for group identification in group/shuffle/split)
        #             X_setup = X_setup[[_ for _ in X_setup.columns if _ not in ['nTrial']]]

                    
        #             score_method = 'r2'        

        #             # Select hyper parameters for GLM to use for model selection
        #             # Step 1: Create a dictionary of lists for these relevant keywords...
        #             kwargs_iterations = {
        #                 # # 'alpha': [0.0, 1.0],
        #                 # # 'l1_ratio': [0.0, 0.001],

        #                 # 'alpha': [0.001, 0.01, 0.1, 0.5, 0.9, 1.0],
        #                 # 'l1_ratio': [0.0, 0.001, 0.01],

        #                 'alpha': [0.001, 0.01, 0.1, 0.5, 0.9, 1.0],
        #                 'l1_ratio': [0.0, 0.001,],
        #             }

        #             # Step 2: Create a dictionary for the fixed keyword arguments that do not require iteration...
        #             kwargs_fixed = {
        #                 'max_iter': 1000,
        #                 'fit_intercept': True
        #             }

        #             # Step 3: Generate iterable list of keyword sets for possible combinations
        #             glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)
        #             best_score, best_score_std, best_params, best_model, cv_results = sglm_ez.simple_cv_fit(X_setup, y_setup, kfold_cv_idx, glm_kwarg_lst, model_type='Normal', verbose=0, score_method=score_method)

        #             print()
        #             print('---')
        #             print()

        #             print_best_model_info(X_setup, best_score, best_params, best_model, start)


        #             glm, holdout_score, holdout_neg_mse_score = training_fit_holdout_score(X_setup, y_setup, X_holdout, y_holdout, best_params, wi_trial_keep_tr, wi_trial_keep_holdout)

        #             # Collect
        #             res[f'{filename}_{y_col}'] = {'holdout_score':holdout_score,
        #                             'holdout_neg_mse_score':holdout_neg_mse_score,
        #                             'best_score':best_score,
        #                             'best_params':best_params,
        #                             'all_models':sorted([(_['cv_R2_score'],
        #                                                 _['cv_mse_score'],
        #                                                 sglm_ez.calc_l1(_['cv_coefs']),
        #                                                 sglm_ez.calc_l2(_['cv_coefs']),
        #                                                 _['glm_kwargs']) for _ in cv_results['full_cv_results']], key=lambda x: -x[0])
        #                             }
        #             print(f'Holdout Score: {holdout_score}')
        #             # print('Keys:',)

        #             # Generate and save plots of the beta coefficients
        #             X_cols_plot = [_ for _ in X_cols if _ in X_setup.columns]
        #             X_cols_sftd_plot = [_ for _ in X_cols_sftd if _ in X_setup.columns]

        #             fn = filename.split(".")[0]
        #             splt.plot_all_beta_coefs(glm.coef_, X_cols_plot,
        #                                             X_cols_sftd_plot,
        #                                             plot_width=2,
        #                                             y_lims=(-2.0, 2.0),
        #                                             # filename=f'{fn}_coeffs.png',
        #                                             binsize=50,
        #                                             filename=f'model_outputs/best_coeffs/77_{fn}_{y_col}_coeffs_R2_{np.round(holdout_score, 4)}.png',
        #                                             plot_name=f'{fn} — {y_col} — {best_params}'
        #                                             )
                    

        #             # dfrel_holdout
        #             # dfrel_setup


        #             # print('dfrel', list(X_setup.columns))

                    
        #             dfrel_holdout['nn'] = dfrel_holdout[['lpn', 'rpn']].sum(axis=1)
        #             dfrel_holdout['xx'] = dfrel_holdout[['lpx', 'rpx']].sum(axis=1)

        #             first_trans = dfrel_holdout.groupby('nTrial')[['nn', 'xx', 'lpn', 'rpn', 'lpx', 'rpx', 'cpn']].cumsum()
        #             first_trans = ((first_trans == 1)*1).diff()
        #             first_trans *= first_trans >= 0
        #             first_trans['lpn'] = dfrel_holdout['nn']*dfrel_holdout['lpn']
        #             first_trans['rpn'] = dfrel_holdout['nn']*dfrel_holdout['rpn']
        #             first_trans['lpx'] = dfrel_holdout['xx']*dfrel_holdout['lpx']
        #             first_trans['rpx'] = dfrel_holdout['xx']*dfrel_holdout['rpx']

        #             first_trans = first_trans.rename({_k:f'ft_{_k}' for _k in first_trans.columns}, axis=1)
        #             dfrel_holdout[first_trans.columns] = first_trans



        #             dfrel_holdout['ft_r_rpn'] = dfrel_holdout['ft_rpn'] * dfrel_holdout['r']
        #             dfrel_holdout['ft_r_lpn'] = dfrel_holdout['ft_lpn'] * dfrel_holdout['r']
        #             dfrel_holdout['ft_nr_rpn'] = dfrel_holdout['ft_rpn'] * dfrel_holdout['nr']
        #             dfrel_holdout['ft_nr_lpn'] = dfrel_holdout['ft_lpn'] * dfrel_holdout['nr']


        #             tmp = dfrel_holdout.set_index('nTrial').copy()
        #             tmp['pred'] = glm.predict(tmp[X_setup.columns])
        #             tmp = get_first_entry_time(tmp, X_setup)

        #             splt.plot_avg_reconstructions(tmp,
        #                                         y_col=y_col,
        #                                         binsize = 50,
        #                                         min_time = -20,
        #                                         max_time = 30,
        #                                         min_signal = -3.0,
        #                                         max_signal = 3.0,
        #                                         file_name=f'model_outputs/best_reconstructions/77_{fn}_{y_col}_arr_R2_{np.round(holdout_score, 4)}.png')
                



        #             for fitted_model_dict in cv_results['full_cv_results']:
        #                 fitted_model = fitted_model_dict['model']
        #                 kwarg_info = "_".join([f"{_k}_{fitted_model_dict['glm_kwargs'][_k]}" for _k in fitted_model_dict["glm_kwargs"]])



        #                 model_coef = fitted_model.coef_
        #                 model_intercept = fitted_model.intercept_

        #                 np.save(f'model_outputs/all_models/77_coeffs_{filename[:-4]}_{y_col}_{kwarg_info}.npy', model_coef)
        #                 np.save(f'model_outputs/all_models/77_intercept_{filename[:-4]}_{y_col}_{kwarg_info}.npy', model_intercept)
                        
        #                 # tmp_holdout_score = fitted_model.r2_score(X_holdout[X_setup.columns], y_holdout)
        #                 tmp_holdout_score = fitted_model.r2_score(dfrel_holdout[X_setup.columns][wi_trial_keep_holdout], y_holdout[wi_trial_keep_holdout])

        #                 # print("fitted_model_dict['glm_kwargs']",fitted_model_dict['glm_kwargs'])

        #                 tmp = dfrel_holdout.set_index('nTrial').copy()
        #                 tmp['pred'] = fitted_model.predict(tmp[X_setup.columns])
        #                 tmp = get_first_entry_time(tmp, X_setup)
                        
        #                 # tmp = X_holdout.set_index('nTrial').copy()
        #                 tmp_y = y_holdout.copy()
        #                 tmp_y.index = tmp.index
        #                 tmp[y_holdout.name] = tmp_y

        #                 tmp.to_csv(f'model_outputs/all_data/77_tmp_data_{filename[:-4]}_{y_col}_{kwarg_info}.csv')

        #                 splt.plot_avg_reconstructions(tmp,
        #                                             y_col=y_col,
        #                                             binsize = 50,
        #                                             min_time = -20,
        #                                             max_time = 30,
        #                                             min_signal = -3.0,
        #                                             max_signal = 3.0,
        #                                             file_name=f'model_outputs/all_reconstructions/77_avg_resp_reconstruction_{filename[:-4]}_{y_col}_{kwarg_info}_R2_{np.round(tmp_holdout_score, 4)}.png')
        #                 splt.plot_all_beta_coefs(fitted_model.coef_, X_cols_plot,
        #                                                 X_cols_sftd_plot,
        #                                                 plot_width=2,
        #                                                 y_lims=(-2.0, 2.0),
        #                                                 # filename=f'{fn}_coeffs.png',
        #                                                 binsize=50,
        #                                                 filename=f'model_outputs/all_coeffs/77_betas_{filename[:-4]}_{y_col}_{kwarg_info}_coeffs_R2_{np.round(tmp_holdout_score, 4)}.png',
        #                                                 plot_name=f'{fn} — {y_col} — {kwarg_info}'
        #                                                 )
                        


        #                 # print('entry_timing_r',entry_timing_r)
        #                 # print('entry_timing_l',entry_timing_l)
        #                 # print('entry_timing_c',entry_timing_c)
                        
                        
                

        #     # For every file iterated, for every result value, for every model fitted, print the reslts
        #     print(f'Final Results:')
        #     for k in res:
        #         print(f'> {k}')
        #         for k_ in res[k]:
        #             if type(res[k][k_]) != list:
        #                 print(f'>> {k_}: {res[k][k_]}')
        #             else:
        #                 lst_str_setup = f'>> {k_}: ['
        #                 lss_spc = ' '*(len(lst_str_setup)-1)
        #                 print(lst_str_setup)
        #                 for v_ in res[k][k_]:
        #                     print((f'{lss_spc} R^2: {np.round(v_[0], 5)} — MSE: {np.round(v_[1], 5)} —'+
        #                           f' L1: {np.round(v_[2], 5)} — L2: {np.round(v_[3], 5)} — '+
        #                           f'Params: {v_[4]}'))
        #                 print(lss_spc + ']')


        #     print('X_cols_plot:', X_cols_plot)
        #     print('X_cols_sftd_plot:', X_cols_sftd_plot)

