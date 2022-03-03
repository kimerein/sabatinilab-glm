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

import glob
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

def to_profile():
    start = time.time()


    ssave_folder = 'model_outputs/ssave'
    all_models_folder = 'model_outputs/all_models'
    all_data_folder = 'model_outputs/all_data'
    all_reconstruct_folder = 'model_outputs/all_reconstructions'
    all_coeffs_folder = 'model_outputs/all_coeffs'
    best_reconstruct_folder = 'model_outputs/best_reconstructions'
    best_coeffs_folder = 'model_outputs/best_coeffs'

    # prefix = 'w_unrewarded_cvsize=.01'
    # prefix = 'new_lynne2'
    # prefix = 'new_lynne_mn2l2_only_refac'
    # prefix = 'investig_resolved'
    # prefix = 'r_trial-r-refit'
    # prefix = 'allsess_fit_multi_alpha'
    prefix = '1diff_tbnds_manysess_individ'
    avg_reconstruct_basename = 'arr'
    all_betas_basename = 'betas'
    model_c_basename = 'coeffs'
    model_i_basename = 'intercept'
    tmp_data_basename = 'tmp_data'

    # Bigger files list
    files_list = glob.glob(f'{dir_path}/../GLM_SIGNALS_WT61_*') + \
                 glob.glob(f'{dir_path}/../GLM_SIGNALS_WT63_*') + \
                 glob.glob(f'{dir_path}/../GLM_SIGNALS_WT64_*') + \
                 glob.glob(f'{dir_path}/../GLM_SIGNALS_WT43_*') + \
                 glob.glob(f'{dir_path}/../GLM_SIGNALS_WT44_*')
    
    files_list = [_.split('/')[-1] for _ in files_list]
    print(files_list)

    y_col_lst = ['Ch1', 'Ch2', 'Ch5', 'Ch6']

    # Select column names to use for GLM predictors
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
        'alpha': [0.0],
        'l1_ratio': [0],

        # 'alpha': [0.0, 0.001, 0.01, 0.1, 1.0],
        # 'l1_ratio': [0.0, 0.001],
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

    results_dict = {}

    leave_one_out_list = [[]]
    # leave_one_out_list = [[]] + [[_] for _ in X_cols_all if _ != 'nTrial'] # Excluding column for groupby, 'nTrial'
    full_df_set = []

    # Loop through files to be processed
    for ifn,filename in enumerate(tqdm(files_list, 'filename')):
        fn = filename.split('.')[0].split('/')[-1]

        glmsave = ssave.GLM_data(ssave_folder, f'{prefix}_{fn}.pkl')
        glmsave.set_uid(prefix)
        glmsave.set_filename(filename)
        glmsave.set_timeshifts(neg_order, pos_order)
        glmsave.set_X_cols(X_cols_all)
        glmsave.set_gss_info(folds, pholdout, pgss, gssid=None)


        # Load file
        df = pd.read_csv(f'{dir_path}/../{filename}')
        df = lpp.preprocess_lynne(df, trial_shift_bounds=1)
        df['wi_trial_keep'] = lpp.get_is_not_iti(df)

        df['nTrial'] += ifn * 100000

        for y_col in y_col_lst:
            if 'SGP_' == y_col[:len('SGP_')]:
                df[y_col] = df[y_col].replace(0, np.nan)
            if df[y_col].std() >= 90:
                df[y_col] /= 100
        
        # full_df_set.append(df)
    
    # df = pd.concat(full_df_set)

    # glmsave.set_basedata(df)



    # Unindent from here to line 573 to revert to using the full dataframe and uncomment previous 3 lines.
        glmsave.set_basedata(df)
        for y_col in tqdm(y_col_lst, 'ycol'):
        # for y_col in tqdm(['zsrdFF', 'zsgdFF'], 'ycol'):

            # df = lpp.detrend(df, y_col)

            for left_out in tqdm(leave_one_out_list, 'left_out'):
                
                X_cols = [_ for _ in X_cols_all if _ not in left_out]

                if len(leave_one_out_list) > 1:
                    run_id = f'{prefix}_{fn}_{y_col}_drop={"_".join(left_out)}'
                else:
                    run_id = f'{prefix}_{fn}_{y_col}'

                print("Run ID:", run_id)
                dfrel = df.copy()




                # if 'SGP_' == y_col[:len('SGP_')]:
                #     dfrel[y_col] = dfrel[y_col].replace(0, np.nan)
                # if dfrel[y_col].std() >= 90:
                #     dfrel[y_col] /= 100



                # Timeshift X_cols forward by pos_order times and backward by neg_order times
                dfrel, X_cols_sftd = lpp.timeshift_vals(dfrel, X_cols, neg_order=neg_order, pos_order=pos_order)
                dfrel = dfrel.dropna()
                dfrel_setup, dfrel_holdout = holdout_splits(dfrel,
                                                            id_cols=['nTrial'],
                                                            perc_holdout=pholdout)

                # Generate cross-validation (technically, group / shuffle split) sets for training / model selection
                kfold_cv_idx = sglm_ez.cv_idx_by_trial_id(dfrel_setup,
                                                        trial_id_columns=['nTrial'],
                                                        num_folds=folds,
                                                        test_size=pgss)

                print([(len(_[0]), len(_[1])) for _ in kfold_cv_idx])

                prediction_X_cols = [_ for _ in X_cols if _ not in ['nTrial']]
                prediction_X_cols_sftd = [_ for _ in X_cols_sftd if _ not in ['nTrial']]

                X_setup = get_x(dfrel_setup, prediction_X_cols_sftd, keep_rows=None)
                y_setup = get_y(dfrel_setup, y_col, keep_rows=None)
                X_setup_noiti = get_x(dfrel_setup, prediction_X_cols_sftd, keep_rows=dfrel_setup['wi_trial_keep'])
                y_setup_noiti = get_y(dfrel_setup, y_col, keep_rows=dfrel_setup['wi_trial_keep'])
                best_score, best_score_std, best_params, best_model, cv_results = sglm_ez.simple_cv_fit(X_setup, y_setup, kfold_cv_idx, glm_kwarg_lst, model_type='Normal', verbose=0, score_method=score_method)
                
                sglm_ez.print_best_model_info(X_setup, best_score, best_params, best_model, start)
                
                X_holdout_witi = get_x(dfrel_holdout, prediction_X_cols_sftd, keep_rows=None)
                y_holdout_witi = get_y(dfrel_holdout, y_col, keep_rows=None)
                X_holdout_noiti = get_x(dfrel_holdout, prediction_X_cols_sftd, keep_rows=dfrel_holdout['wi_trial_keep'])
                y_holdout_noiti = get_y(dfrel_holdout, y_col, keep_rows=dfrel_holdout['wi_trial_keep'])
                glm, holdout_score, holdout_neg_mse_score = sglm_ez.training_fit_holdout_score(X_setup, y_setup, X_holdout_noiti, y_holdout_noiti, best_params)

                # Collect
                results_dict[f'{run_id}'] = {'holdout_score':holdout_score,
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

                # print('X_setup.columns', list(X_setup.columns), len(list(X_setup.columns)))
                # print('X_setup_noiti.columns', list(X_setup_noiti.columns), len(list(X_setup_noiti.columns)))
                # print('X_holdout_witi.columns', list(X_holdout_witi.columns), len(list(X_holdout_witi.columns)))
                # print('X_holdout_noiti.columns', list(X_holdout_noiti.columns), len(list(X_holdout_noiti.columns)))


                holdout_score_rnd = np.round(holdout_score, 4)
                best_beta_fn = f'{best_coeffs_folder}/{run_id}_best_{all_betas_basename}_R2_{holdout_score_rnd}.png'
                splt.plot_all_beta_coefs(glm.coef_, X_cols_plot,
                                                X_cols_sftd_plot,
                                                plot_width=2,
                                                y_lims=(-2.0, 2.0),
                                                # filename=f'{fn}_coeffs.png',
                                                binsize=54,
                                                filename=best_beta_fn,
                                                plot_name=f'Best Coeffs - {run_id} — {best_params}'
                                                )

                tmp = dfrel_holdout.set_index('nTrial').copy()
                tmp['pred'] = glm.predict(get_x(dfrel_holdout, prediction_X_cols_sftd))
                tmp = lpp.get_first_entry_time(tmp)


                best_beta_fn = f'{best_reconstruct_folder}/{run_id}_best_{avg_reconstruct_basename}_R2_{holdout_score_rnd}.png'
                splt.plot_avg_reconstructions(tmp,
                                            y_col=y_col,
                                            binsize = 54,
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
                                                'tr_witi':fitted_model.r2_score(X_setup, y_setup),
                                                'tr_noiti':fitted_model.r2_score(X_setup_noiti, y_setup_noiti),
                                                'gss_witi':fitted_model_dict['cv_R2_score'],
                                                'gss_noiti':None,
                                                'holdout_witi':fitted_model.r2_score(X_holdout_witi, y_holdout_witi),
                                                'holdout_noiti':fitted_model.r2_score(X_holdout_noiti, y_holdout_noiti)
                                            },
                                            gssids=kfold_cv_idx)

                    tmp = dfrel_holdout.set_index('nTrial').copy()
                    tmp['pred'] = fitted_model.predict(get_x(dfrel_holdout, prediction_X_cols_sftd, keep_rows=None))
                    tmp = lpp.get_first_entry_time(tmp)
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
                                                    binsize=54,
                                                    filename=f'{all_coeffs_folder}/{std_name}_{all_betas_basename}_R2_{holdout_score_rnd}.png',
                                                    plot_name=f'Coeffs by Timeshift - {run_id} — {kwarg_info}'
                                                    # plot_name=f'{fn} — {y_col} — {kwarg_info}'
                                                    )
                    
                    plt.close('all')
                plt.close('all')
                

        glmsave.save()

    # For every file iterated, for every result value, for every model fitted, print the reslts
    print(f'Final Results:')
    for run_id in results_dict:
        print(f'> {run_id}')
        single_run_results = results_dict[run_id]
        for run_info_key in single_run_results:
            if type(single_run_results[run_info_key]) != list:
                print(f'>> {run_info_key}: {single_run_results[run_info_key]}')
            else:
                lst_str_setup = f'>> {run_info_key}: ['
                lss_spc = ' '*(len(lst_str_setup)-1)
                print(lst_str_setup)
                for single_hyperparam_result in single_run_results[run_info_key]:
                    print((f'{lss_spc} R^2: {np.round(single_hyperparam_result[0], 5)} — MSE: {np.round(single_hyperparam_result[1], 5)} —'+
                        f' L1: {np.round(single_hyperparam_result[2], 5)} — L2: {np.round(single_hyperparam_result[3], 5)} — '+
                        f'Params: {single_hyperparam_result[4]}'))
                print(lss_spc + ']')


    print('X_cols_plot:', X_cols_plot)
    print('X_cols_sftd_plot:', X_cols_sftd_plot)

    end = time.time()
    print('Runtime:',end-start)

if __name__ == '__main__':
    profile = cProfile.run('to_profile()', filename='./profile_val.prof', sort='cumtime')
