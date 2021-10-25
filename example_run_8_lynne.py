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

    # filename = 'dlight_only_WT35_12212020.csv'
    # filename = 'dlight_only_WT36_12212020.csv'
    # filename = 'Ach_only_WT53L_09032021xlsx.csv'
    # filename = 'Ach_only_WT60R_10042021xlsx.csv'



    # Try dropping lick time shifts (or only forward a little bit)
    # Get rid of left / right port exits






    # filename = 'dlight_only_WT36L_12172020.csv'
    # filename = 'dlight_only_WT36L_12212020.csv'
    # filename = 'dlight_only_WT36L_12242020.csv'
    # filename = 'dlight_only_WT36L_12292020.csv'

    # filename = 'Ach_only_WT53L_08262021xlsx.csv'
    # filename = 'Ach_only_WT53L_09012021xlsx.csv'
    # filename = 'Ach_only_WT53L_09032021xlsx.csv'
    # filename = 'Ach_only_WT53L_09062021xlsx.csv'

    # files_list = [
    #     'dlight_only_WT36L_12172020.csv',
    #     'dlight_only_WT36L_12212020.csv',
    #     'dlight_only_WT36L_12242020.csv',
    #     'dlight_only_WT36L_12292020.csv',

    #     'Ach_only_WT53L_08262021xlsx.csv',
    #     'Ach_only_WT53L_09012021xlsx.csv',
    #     'Ach_only_WT53L_09032021xlsx.csv',
    #     'Ach_only_WT53L_09062021xlsx.csv',
    # ]

    files_list = [
        'dlight_only_WT36L_12242020.csv',
        'Ach_only_WT53L_09062021xlsx.csv'
    ]

    for filename in files_list:


        df = pd.read_csv(f'{dir_path}/../{filename}')
        
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


                        'dF/F green (Ach3.0)': 'gdFF',
                        'zscored green (Ach3.0)': 'zsgdFF',

                        'dF/F green (dLight1.1)': 'gdFF',
                        'zscored green (dLight1.1)': 'zsgdFF',

                        'dF/F green (dlight1.1)': 'gdFF',
                        'zscored green (dlight1.1)': 'zsgdFF'
                        }, axis=1)
        
        df['event_col_a'] = ((df['cpo'].diff() > 0)*1).replace(0, np.nan) * 1.0
        df['event_col_b'] = df['nr'].replace(0, np.nan) * 2.0
        df['event_col_c'] = df['r'].replace(0, np.nan) * 3.0

        df['event_col'] = df['event_col_a'].combine_first(df['event_col_b']).combine_first(df['event_col_c'])


        df['event_col'] = df['event_col'].bfill()
        
        df['trial_start_flag'] = ((df['event_col'] == 1.0)&(df['event_col'].shift(-1) != df['event_col']) * 1.0).shift(-5) * 1.0
        df['nTrial'] = df['trial_start_flag'].cumsum()



        # df['trial_end_flag'] = ((df['event_col'] != 1.0)&(df['event_col'].shift(-1) == 1.0)&(df['event_col'].shift(-1) != df['event_col']) * 1.0).shift(10) * 1.0
        # df['nEndTrial'] = df['trial_end_flag'].cumsum()

        df['event_col_d'] = ((df['lpx'] > 0)*1.0).replace(0, np.nan) * 1.0
        df['event_col_e'] = ((df['rpx'] > 0)*1.0).replace(0, np.nan) * 1.0
        # df['event_col_end'] = df['event_col_d'].combine_first(df['event_col_e']).combine_first(df['event_col_b']).combine_first(df['event_col_c']).ffill()
        # df['trial_end_flag'] = ((df['event_col_end'] == 1.0)&(df['event_col_end'].shift(1) != 1.0)&(df['event_col_end'].shift(1) != df['event_col_end'])&(df['nTrial'] > 0) * 1.0).shift(5) * 1.0
        # df['nEndTrial'] = df['trial_end_flag'].cumsum()


        df['event_col_end'] = df['event_col_d'].combine_first(df['event_col_e']).combine_first(df['trial_start_flag'].replace(0.0, np.nan)*2.0)
        df['event_col_end'] = df['event_col_end'].ffill()
        # df['trial_end_flag'] = ((df['event_col_start'] != 1.0)&(df['event_col_start'].shift(-1) == 1.0)&(df['event_col_start'].shift(-1) != df['event_col_start']) * 1.0).shift(5) * 1.0
        df['trial_end_flag'] = ((df['event_col_end'] == 1.0)&(df['event_col_end'].shift(1) == 2.0)&(df['event_col_end'].shift(1) != df['event_col_end'])&(df['nTrial'] > 0) * 1.0).shift(5) * 1.0
        df['nEndTrial'] = df['trial_end_flag'].cumsum()


        df = df.drop(['event_col_a', 'event_col_b', 'event_col_c', 'event_col_d', 'event_col_e'], axis=1)


        wi_trial_keep = (df['nTrial'] != df['nEndTrial'])

        if 'index' in df.columns:
            df = df.drop('index', axis=1)
        
        # y_setup_col = 'gdFF' # photometry response
        # df = sglm_ez.diff_cols(df, ['gdFF'])

        df['r_trial'] = df.groupby('nTrial')['r'].transform(np.sum)
        df['nr_trial'] = df.groupby('nTrial')['nr'].transform(np.sum)

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
        # 'nr', 'r',
        #'cpo',
        #'lpo',
        #'rpo',
        ]

        y_col = 'zsgdFF'

        dfrel = df[X_cols + [y_col]].copy()
        dfrel = dfrel.replace('False', 0).astype(float)
        dfrel = dfrel*1
        
        neg_order = -20
        pos_order = 20

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
            # 'alpha': reversed([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]),
            'alpha': [1.0],
            'l1_ratio': [1.0]
        }

        # Step 2: Create a dictionary for the fixed keyword arguments that do not require iteration...
        kwargs_fixed = {
            'max_iter': 1000,
            'fit_intercept': True
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

        X_cols_plot = [_ for _ in X_cols if _ in X_setup.columns]
        X_cols_sftd_plot = [_ for _ in X_cols_sftd if _ in X_setup.columns]

        fn = filename.split(".")[0]

        sglm_ez.plot_all_beta_coefs(glm, X_cols_plot,
                                        X_cols_sftd_plot,
                                        plot_width=2,
                                        y_lims=(-2.0, 2.0),
                                        # filename=f'{fn}_coeffs.png',
                                        filename=f'{fn}_coeffs_R2_{np.round(holdout_score, 4)}.png',
                                        plot_name=f'{fn} — {best_params}'
                                        )

to_profile()
