
from sglm.models import sglm_cv
import itertools
import time
import glob
import numpy as np
import pandas as pd
import numpy as np
from sglm.features import gen_signal_df as gsd
from sglm.features import build_features as bf
from sglm.features import setup_model_fit as smf
from sglm.models import sglm_cv
from sglm import models
from sglm.visualization import visualize
from sglm.models import train_model
from sglm.models import eval
from sglm import features

neg_order = -20
pos_order = 20

fix_training = True
ft_str = '-ft' if fix_training else ''

# multifile_fit = 'single' #False
# multifile_fit = 'all' #True
# multifile_fit = 'by_mouse'
# multifile_fit_list = ['all']
multifile_fit_list = ['by_mouse']
# multifile_fit_list = ['by_mouse', 'all']
# multifile_fit_list = ['single']

# base_prefix = 'trial_abm_slOff'
# base_prefix = 'bidir_pred_bidir'
# base_prefix = 'dh'
# base_prefix = 'gr-dual-n62'
# base_prefix = 'bidir-ctrl-nslO'
# base_prefix = 'DA-DA-pred'
# base_prefix = 'ACH-to-DA'
# base_prefix = 'DA-to-ACH__'

# base_prefix = 'ind'
# base_prefix = 'resid'
# base_prefix = 'resid-chk'
# base_prefix = 'resid-chk5'
# base_prefix = 'test-res_trk'


# base_prefix = 'f1-bkwd-sel'
# base_prefix = 'f1-bkwd-sel-gACH2-rl'
# base_prefix = 'f1-bkwd-sel-rDA2'
# base_prefix = 'f1-bkwd-sel-gDA2'

# base_prefix = 'f1-bkwd-sel-hypPrm-rDA'
# base_prefix = 'f1-bkwd-sel-hypPrm-gACH'

tmp_y_col_setup = 'gACH'
# base_prefix = f'f1-bkwd-sel-{tmp_y_col_setup}3'
# base_prefix = 'f1-bkwd-sel-rDA3'
# base_prefix = 'f1-bkwd-sel-gDA3'

base_prefix = f'mse-tr-te-invstg2-{tmp_y_col_setup}'

data_folder = 'fig1'
# data_folder = 'fig2'
# data_folder = 'fig2-dualhem'
# data_folder = 'fig3/g1'
# data_folder = 'fig3/g2'
# data_folder = 'fig3/g3'
# data_folder = 'fig3/g4'
# data_folder = 'fig3/g5'

# data_folder = 'fig2-dualhem'

if data_folder == 'fig1':
    wt_used = [#'WT62',
               'WT63', 'WT64', 'WT65', 'WT66', 'WT67', 'WT68', 'WT69', # DA
               'WT57', 'WT58', 'WT59', 'WT60', 'WT61', 'WT53', 'WT55', 'WT56' # ACH
               ]
elif data_folder == 'fig2':
    wt_used = ['WT61', 'WT63', 'WT64', 'WT44', 'WT51']
elif data_folder == 'fig2-dualhem':
    wt_used = ['WT63', 'WT64', 'WT65']
elif data_folder == 'fig3/g1': # Drd2f/f control: S1417, 1419, 1421
    wt_used = ['S1417', 'S1419', 'S1421']
elif data_folder == 'fig3/g2': # Chat Cre X Drd2f/f : S1416, 1418, 1420, 1422
    wt_used = ['S1416', 'S1418', 'S1420', 'S1422']
elif data_folder == 'fig3/g3': # Chat Cre control: S1355-1358, S1374, S1376
    wt_used = ['S1355', 'S1356', 'S1357', 'S1358', 'S1374', 'S1376']
elif data_folder == 'fig3/g4': # Chat Cre control: S1399-1401
    wt_used = ['S1399', 'S1400', 'S1401']
elif data_folder == 'fig3/g5':
    wt_used = ['S1355', 'S1356', 'S1357', 'S1358', 'S1374', 'S1376', 'S1399', 'S1400', 'S1401']
elif data_folder == 'fig3/g6':
    wt_used = []
else:
    raise ValueError('Unimplemented figure values.')

data_folder_join = '_'.join(data_folder.split('/'))

# ### Dual recording analsysis — predictive
# X_y_pairings = [
#     {'X_cols': {

#                 'photometryCenterInIndex':(0,0), #'photometryCenterOutIndex':(0,0),
#                 'sl': (0,0),
#                 # 'photometrySideInIndexr':(0,0), 'photometrySideInIndexnr':(0,0),
                
#                 'photometrySideInIndexAA':(0,0), 'photometrySideInIndexAa':(0,0),
#                 'photometrySideInIndexaA':(0,0), 'photometrySideInIndexaa':(0,0),
#                 'photometrySideInIndexAB':(0,0), 'photometrySideInIndexAb':(0,0),
#                 'photometrySideInIndexaB':(0,0), 'photometrySideInIndexab':(0,0),

#                 'photometrySideOutIndex':(0,0),
#                 'spnnrOff': (0,0),
#                 'rDA': (0,0),
#                },
#      'y_col': 'gACH'},
#     {'X_cols': {

#                 'photometryCenterInIndex':(0,0),
#                 'sl': (0,0),

#                 'photometrySideInIndexAA':(0,0), 'photometrySideInIndexAa':(0,0),
#                 'photometrySideInIndexaA':(0,0), 'photometrySideInIndexaa':(0,0),
#                 'photometrySideInIndexAB':(0,0), 'photometrySideInIndexAb':(0,0),
#                 'photometrySideInIndexaB':(0,0), 'photometrySideInIndexab':(0,0),

#                 'photometrySideOutIndex':(0,0),
#                 'spnnrOff': (0,0),
#                 # 'slOff': (0,0),
#                 # 'rDA': (0,0),
#                },
#      'y_col': 'gACH'},
#     {'X_cols': {

#                 'photometryCenterInIndex':(0,0), #'photometryCenterOutIndex':(0,0),
#                 'sl': (0,0),
#                 'photometrySideInIndexr':(0,0), 'photometrySideInIndexnr':(0,0),
                
#                 'photometrySideOutIndex':(0,0),
#                 'spnnrOff': (0,0),
#                 'rDA': (0,0),
#                },
#      'y_col': 'gACH'},
#     {'X_cols': {

#                 'photometryCenterInIndex':(0,0), #'photometryCenterOutIndex':(0,0),
#                 'sl': (0,0),
#                 'photometrySideInIndexr':(0,0), 'photometrySideInIndexnr':(0,0),
                
#                 'photometrySideOutIndex':(0,0),
#                 'spnnrOff': (0,0),
#                },
#      'y_col': 'gACH'},
#     {'X_cols': {
#                 'rDA': (0,0),
#                },
#      'y_col': 'gACH'},

# ]


### Backwards Selection
X_y_pairings = [
    {'X_cols': {
                'photometryCenterInIndex':(0,0),
                'photometryCenterOutIndex':(0,0),
                'photometrySideInIndex':(0,0),
                'photometrySideInIndexr':(0,0),
                'photometrySideOutIndex':(0,0),
                'sl': (0,0),
                'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
    {'X_cols': {
                # 'photometryCenterInIndex':(0,0),
                'photometryCenterOutIndex':(0,0),
                'photometrySideInIndex':(0,0),
                'photometrySideInIndexr':(0,0),
                'photometrySideOutIndex':(0,0),
                'sl': (0,0),
                'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
    {'X_cols': {
                'photometryCenterInIndex':(0,0),
                # 'photometryCenterOutIndex':(0,0),
                'photometrySideInIndex':(0,0),
                'photometrySideInIndexr':(0,0),
                'photometrySideOutIndex':(0,0),
                'sl': (0,0),
                'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
    {'X_cols': {
                'photometryCenterInIndex':(0,0),
                'photometryCenterOutIndex':(0,0),
                # 'photometrySideInIndex':(0,0),
                'photometrySideInIndexr':(0,0),
                'photometrySideOutIndex':(0,0),
                'sl': (0,0),
                'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
    {'X_cols': {
                'photometryCenterInIndex':(0,0),
                'photometryCenterOutIndex':(0,0),
                'photometrySideInIndex':(0,0),
                # 'photometrySideInIndexr':(0,0),
                'photometrySideOutIndex':(0,0),
                'sl': (0,0),
                'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
    {'X_cols': {
                'photometryCenterInIndex':(0,0),
                'photometryCenterOutIndex':(0,0),
                'photometrySideInIndex':(0,0),
                'photometrySideInIndexr':(0,0),
                # 'photometrySideOutIndex':(0,0),
                'sl': (0,0),
                'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
     {'X_cols': {
                'photometryCenterInIndex':(0,0),
                'photometryCenterOutIndex':(0,0),
                
                'photometrySideInIndex':(0,0),
                'photometrySideInIndexr':(0,0),
                
                'photometrySideOutIndex':(0,0),
                # 'sl': (0,0),
                
                'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
     {'X_cols': {
                'photometryCenterInIndex':(0,0),
                'photometryCenterOutIndex':(0,0),
                
                'photometrySideInIndex':(0,0),
                'photometrySideInIndexr':(0,0),
                
                'photometrySideOutIndex':(0,0),
                'sl': (0,0),
                
                # 'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
    {'X_cols': {
                # 'photometryCenterInIndex':(0,0),
                # 'photometryCenterOutIndex':(0,0),
                'photometrySideInIndex':(0,0),
                'photometrySideInIndexr':(0,0),
                'photometrySideOutIndex':(0,0),
                'sl': (0,0),
                'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
    {'X_cols': {
                'photometryCenterInIndex':(0,0),
                'photometryCenterOutIndex':(0,0),
                # 'photometrySideInIndex':(0,0),
                # 'photometrySideInIndexr':(0,0),
                'photometrySideOutIndex':(0,0),
                'sl': (0,0),
                'spnnrOff': (0,0),
               },
     'y_col': tmp_y_col_setup},
]

# 'gACH_flx_drd', 'rDA_flx_drd', 'gDA_flx_drd',
# 'gACH_flx_cht', 'rDA_flx_cht', 'gDA_flx_cht',
# 'gACH_flx_drdcht',
# 'rDA_flx_drdcht',
# 'gDA_flx_drdcht',

plot_width = 2
max_cols_len = max([len(_['X_cols']) for _ in X_y_pairings])
plot_rows = max_cols_len//plot_width + (max_cols_len%plot_width > 0)*1

drop_cols_basis = [
    'nTrial',
    'cpn', 'cpx',
    'spnnr',
    'spxnr',
    'spnr', 'spxr',

    'photometryCenterInIndex', 'photometryCenterOutIndex',
    'photometrySideInIndexr', 'photometrySideInIndexnr',
    'photometrySideOutIndex', 'spnnrOff',

    'photometrySideInIndexAA', 'photometrySideInIndexAa',
    'photometrySideInIndexaA', 'photometrySideInIndexaa',
    'photometrySideInIndexAB', 'photometrySideInIndexAb',
    'photometrySideInIndexaB', 'photometrySideInIndexab',

    'photometrySideOutIndexAA', 'photometrySideOutIndexAa',
    'photometrySideOutIndexaA', 'photometrySideOutIndexaa',
    'photometrySideOutIndexAB', 'photometrySideOutIndexAb',
    'photometrySideOutIndexaB', 'photometrySideOutIndexab',
    
    'sl',
    'slOff'
]

ssave_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/models/ssave'
all_models_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/models/all_models'
all_data_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/models/all_data'
all_reconstruct_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/reports/figures/all_reconstruct'
all_coeffs_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/reports/figures/all_coeffs'
best_reconstruct_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/reports/figures/best_reconstruct'
best_coeffs_npy_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/reports/coeffs/best_coeffs'
best_resids_npy_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/reports/residuals/best_resids'
best_combined_coeffs_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/reports/coeffs/best_combined_coeffs'
best_coeffs_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/reports/figures/best_coeffs'

all_models_folder = '/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/models'

avg_reconstruct_basename = 'arr'
all_betas_basename = 'betas'
model_c_basename = 'coeffs'
model_i_basename = 'intercept'
tmp_data_basename = 'tmp_data'



score_method = 'r2'

# Select hyper parameters for GLM to use for model selection
# Step 1: Create a dictionary of lists for these relevant keywords...
kwargs_iterations = {
    'alpha': [0],
    'l1_ratio': [0],

    # 'alpha': [0.0, 0.001, 0.01, 0.1, 1.0],
    # 'l1_ratio': [0.0, 0.0001, 0.001, 0.01],
}

# Step 2: Create a dictionary for the fixed keyword arguments that do not require iteration...
kwargs_fixed = {
    'max_iter': 1000,
    'fit_intercept': False
}

# neg_order, pos_order = -14, 14
# folds = 50
# folds = 10
folds = 1
pholdout = 0.2
pgss = 0.2

# Step 3: Generate iterable list of keyword sets for possible combinations
glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)

widest_orders = [{'X_cols': smf.X_cols_dict_to_default(_['X_cols'], neg_order, pos_order),
                  'y_col': _['y_col']} for _ in X_y_pairings]
widest_orders = smf.xy_pairs_to_widest_orders(widest_orders)

all_coeff_dfs = []

multi_start = time.time()

for multifile_fit in multifile_fit_list:
    data_folder_join = '_'.join(data_folder.split('/'))

    # prefix = f'{data_folder}-{multifile_fit}-words-recons'
    prefix = f'{data_folder_join}-{multifile_fit}-{base_prefix}{ft_str}'

    # Load Signal Data
    signal_files = []
    mouse_names = []
    for wt in wt_used:
        addl_sig_files = glob.glob(f'/Users/josh/Documents/Harvard/GLM/sabatinilab-glm/sglm/data/interim/{data_folder}/GLM_SIGNALS_INTERIM_{wt}_*')
        signal_files += addl_sig_files
        mouse_names += [wt] * len(addl_sig_files)
        
    if multifile_fit == 'all':
        combo_dfs, X_cols_sftd, _ = smf.multi_file_analysis_prep(signal_files, widest_orders,)
        combo_fns = ['_'.join(wt_used).replace('WT', '').replace('S', '')]
        mouse_names = combo_fns
    elif multifile_fit == 'by_mouse':
        combo_dfs = []
        X_cols_sftd_lst = []
        combo_fns = []
        mouse_names_2 = []
        for mouse_id in wt_used:
            mouse_id_files = [_ for _ in signal_files if mouse_id in _]
            assert len(mouse_id_files) > 0, f'No file with mouse_id {mouse_id} found in {signal_files}'
            mouse_names_2 += [mouse_id]
            # print('mouse_id', mouse_id)
            combo_dfs_tmp, X_cols_sftd_tmp, _ = smf.multi_file_analysis_prep(mouse_id_files, widest_orders,)
            combo_dfs += combo_dfs_tmp
            X_cols_sftd_lst.append(X_cols_sftd_tmp)
            combo_fns.append(mouse_id)
            
        for xcsl in X_cols_sftd_lst:
            if xcsl != X_cols_sftd_lst[0]:
                raise ValueError('X_cols_sftd_lst should contain the same elements for every entry')
        mouse_names = mouse_names_2
        
        X_cols_sftd = X_cols_sftd_lst[0]

    elif multifile_fit == 'single':
        combo_dfs, X_cols_sftd, combo_fns = smf.single_file_analysis_prep(signal_files, widest_orders,)
        mouse_names = mouse_names
    else:
        raise ValueError('multifile_fit must be "all", "single", or "by_mouse"')

    # print(combo_dfs)

    start = time.time()

    results_dict = {}

    for file_num in range(len(combo_dfs)):


        # Load Table Data
        signal_df = combo_dfs[file_num].copy()
        signal_fn = combo_fns[file_num]
        mouse_id = mouse_names[file_num]

        fn = signal_fn.split('.')[0].split('/')[-1]

        print(mouse_names, file_num, mouse_id)

        dfrel_basis = signal_df.copy()
        print('dfrel_basis', dfrel_basis.shape)

        signal_filename_out = 'FINAL_' + signal_fn.split('/')[-1].replace('GLM_SIGNALS_INTERIM_', '').replace('txt', 'csv') + '.csv'
        print(signal_filename_out)

        dfrel_basis['mouse_id'] = mouse_id
        dfrel_basis.set_index(['file_num'], append=True, inplace=True)


        dfrr_cols = ['nTrial', 'wi_trial_keep', 'gDA', 'gACH', 'rDA',
        'photometryCenterInIndex', 'photometryCenterOutIndex',
        'photometrySideInIndexr', 'photometrySideInIndexnr',
        'photometrySideOutIndex', 'spnnrOff', 'sl',

        'photometrySideInIndexAA', 'photometrySideInIndexAa',
        'photometrySideInIndexaA','photometrySideInIndexaa',
        'photometrySideInIndexAB', 'photometrySideInIndexAb',
        'photometrySideInIndexaB','photometrySideInIndexab',

        ]
        

        # dfrel_resids = dfrel_basis[['nTrial', 'wi_trial_keep', 'gDA', 'gACH', 'rDA']].copy()
        dfrel_resids = dfrel_basis[dfrr_cols].copy()

        holdout_score_rnd = None

        dfrel_resids_setup = pd.DataFrame()
        dfrel_resids_holdout = pd.DataFrame()

        if fix_training:
            full_drop_basis = []
            y_col_lst = []
            for X_y_dct in X_y_pairings:
                X_cols_basis = X_y_dct['X_cols']
                y_col = X_y_dct['y_col']
                X_cols_sftd_basis = bf.col_shift_bounds_dict_to_col_list(X_cols_basis, X_cols_sftd)
                full_drop_basis += X_cols_sftd_basis
                
                if y_col[-len('_resid'):] != '_resid':
                    y_col_lst += [y_col]
            full_drop_basis = sorted(list(set(full_drop_basis)))
            y_col_drop_basis = sorted(list(set(y_col_lst)))
            full_drop_basis = sorted(list(set(drop_cols_basis + full_drop_basis + y_col_drop_basis)))

            print('-',y_col_lst)

            dfrel_ft = dfrel_basis.copy()
            # print('full_drop_basis', full_drop_basis)
            # print('y_col_drop_basis', y_col_drop_basis)
            # print('dfrel_ft', len(dfrel_ft))
            srs_a = (dfrel_ft[full_drop_basis].isna().sum(axis=1))
            srs_b = (dfrel_ft[y_col_drop_basis] == 0).sum(axis=1)
            # print('srs_a', srs_a[srs_a > 0])
            # print('srs_b', srs_b[srs_b > 0])
            
            dfrel_ft = dfrel_ft[(dfrel_ft[full_drop_basis].isna().sum(axis=1) == 0)&((dfrel_ft[y_col_drop_basis] == 0).sum(axis=1) == 0)]
            if dfrel_ft.shape[0] == 0:
                print(f'No datapoints found for non-NaN dropcols & non-zero ycols for fixed_training: {prefix}_{fn}')
                continue
            
            dfrel_ft_setup, dfrel_ft_holdout, holdout = models.split_data.holdout_splits(dfrel_ft, id_cols=['nTrial'], perc_holdout=pholdout)
            dfrel_ft_setup, dfrel_ft_holdout = dfrel_ft_setup.copy(), dfrel_ft_holdout.copy()

            dfrel_resids_setup = dfrel_ft_setup[dfrr_cols].copy()
            dfrel_resids_holdout = dfrel_ft_holdout[dfrr_cols].copy()


        
        for iXyd, X_y_dct in enumerate(X_y_pairings):
            dfrel = dfrel_basis.copy()
            X_cols_basis = X_y_dct['X_cols']
            y_col = X_y_dct['y_col']
            X_cols_sftd_basis = bf.col_shift_bounds_dict_to_col_list(X_cols_basis, X_cols_sftd)

            run_id = f'{prefix}_{fn}_{y_col}_{iXyd}'

            # print('dfrel.columns', list(dfrel.columns))
            #### Revise line to reduce length / number of checks

            dfrel = dfrel[(dfrel[drop_cols_basis + X_cols_sftd_basis + [y_col]].isna().sum(axis=1) == 0)&(dfrel[y_col] != 0)]
            if dfrel.shape[0] == 0:
                print(f'No datapoints found for non-NaN dropcols & non-zero ycols for run id: {run_id}.')
                continue
            
            if fix_training:
                assert np.all(dfrel_ft.fillna(0) == dfrel.fillna(0))
                dfrel = dfrel_ft.copy()
                dfrel_setup, dfrel_holdout = dfrel_ft_setup.copy(), dfrel_ft_holdout.copy()
            else:
                dfrel_setup, dfrel_holdout, holdout = models.split_data.holdout_splits(dfrel, id_cols=['nTrial'], perc_holdout=pholdout)
                dfrel_setup, dfrel_holdout = dfrel_setup.copy(), dfrel_holdout.copy()

            
            print('dfr.shape', dfrel.shape)

            print(f'> Included file_nums for y_col {y_col}:', list(dfrel.reset_index()['file_num'].unique()))
            

            # # kfold_cv_idx = models.split_data.cv_idx_by_trial_id(dfrel_setup, trial_id_columns=['nTrial'], num_folds=folds, test_size=pgss)
            # if len(glm_kwarg_lst) > 1:
            #     # Generate cross-validation (technically, group / shuffle split) sets for training / model selection
            #     kfold_cv_idx = models.split_data.cv_idx_by_trial_id(dfrel_setup, trial_id_columns=['nTrial'], num_folds=folds, test_size=pgss)
            #     print('size>1:',kfold_cv_idx)
            # else:
            #     kfold_cv_idx = models.split_data.cv_idx_by_trial_id(dfrel_setup, trial_id_columns=['nTrial'], num_folds=1, test_size=pgss)
            #     print('size=1:',kfold_cv_idx)
            # kfold_cv_idx = models.split_data.cv_idx_by_trial_id(dfrel_setup, trial_id_columns=['nTrial'], num_folds=folds, test_size=pgss)



            prediction_X_cols = [_ for _ in X_cols_basis if _ not in ['nTrial']]
            prediction_X_cols_sftd = [_ for _ in X_cols_sftd_basis if _ not in ['nTrial']]
            
            X_setup_witi, y_setup_witi, X_setup_noiti, y_setup_noiti = train_model.get_xy_all_noniti(dfrel_setup, prediction_X_cols_sftd, y_col, noniticol='wi_trial_keep')
            X_holdout_witi, y_holdout_witi, X_holdout_noiti, y_holdout_noiti = train_model.get_xy_all_noniti(dfrel_holdout, prediction_X_cols_sftd,
                                                                                                             y_col, noniticol='wi_trial_keep')


            dfrel_setup_witi, _, dfrel_setup_noiti, _ = train_model.get_xy_all_noniti(dfrel_setup, list(dfrel_setup.columns), y_col, noniticol='wi_trial_keep')
            

            # Generate cross-validation (technically, group / shuffle split) sets for training / model selection
            kfold_cv_idx = models.split_data.cv_idx_by_trial_id(dfrel_setup_noiti, trial_id_columns=['nTrial'], num_folds=folds, test_size=pgss)

            print('X_setup.columns:', X_setup_witi.columns)

            best_score, best_score_std, best_params, best_model, cv_results = models.sglm_cv.simple_cv_fit(X_setup_noiti, y_setup_noiti, kfold_cv_idx, glm_kwarg_lst, model_type='Normal',
                                                                                                           verbose=0, score_method=score_method)
            
            
            
            
            
            
            
            eval.print_best_model_info(X_setup_witi, best_score, best_params, best_model, start)
            glm, holdout_score, holdout_neg_mse_score = eval.training_fit_holdout_score(X_setup_noiti, y_setup_noiti, X_holdout_noiti, y_holdout_noiti, best_params)


            dfrel['pred'] = glm.predict(dfrel[prediction_X_cols_sftd])
            dfrel_setup['pred'] = glm.predict(dfrel_setup[prediction_X_cols_sftd])
            dfrel_holdout['pred'] = glm.predict(dfrel_holdout[prediction_X_cols_sftd])

            # Collect
            results_dict[f'{run_id}'] = {'holdout_score':holdout_score,
                                        'holdout_neg_mse_score':holdout_neg_mse_score,
                                        'best_score':best_score,
                                        'best_params':best_params,
                                        'all_models':sorted([(_['cv_R2_score'],
                                                                _['cv_mse_score'],
                                                                eval.calc_l1(_['cv_coefs']),
                                                                eval.calc_l2(_['cv_coefs']),
                                                                _['glm_kwargs']) for _ in cv_results['full_cv_results']], key=lambda x: -x[0])
                                        }
            print(f'Holdout Score: {holdout_score}')

            # Generate and save plots of the beta coefficients
            X_cols_plot = prediction_X_cols
            X_cols_sftd_plot = prediction_X_cols_sftd
            
            holdout_score_rnd = np.round(holdout_score, 4)
            best_beta_fn = f'{best_coeffs_folder}/{run_id}_best_{all_betas_basename}_R2_{holdout_score_rnd}.png'

            tr_score = dfrel_setup

            setup_df = pd.DataFrame(np.concatenate([np.array([glm.intercept_]), glm.coef_], axis=0).reshape(1,-1),
                        index=[run_id], columns=['int']+prediction_X_cols_sftd)
            setup_df['mouse_id'] = mouse_id
            setup_df['channel_name'] = y_col + '_' + str(iXyd)

            print(len(cv_results), cv_results.keys())

            # Only get R^2 values if only a single model fit
            if len(cv_results['full_cv_results']) == 1:
                setup_df['mse_tr'] = -glm.neg_mse_score(X_setup_noiti, y_setup_noiti)
                setup_df['mse_cv'] = cv_results['full_cv_results'][0]['cv_mse_score']
                setup_df['mse_te'] = -glm.neg_mse_score(X_holdout_noiti, y_holdout_noiti) #-holdout_neg_mse_score

                # setup_df['tr_num_r'] = dfrel_setup
                # setup_df['tr_num_nr'] = dfrel_setup
                # setup_df['cv_num_r'] = dfrel_setup
                # setup_df['cv_num_nr'] = dfrel_setup
                # setup_df['te_num_r'] = dfrel_setup
                # setup_df['te_num_nr'] = dfrel_setup
                print('Here')

            else: 
                setup_df['mse_tr'] = 0.0
                setup_df['mse_cv'] = 0.0
                setup_df['mse_te'] = 0.0
            
            multi_end = time.time()
            setup_df['timestamp'] = str(multi_end - multi_start) + ' s'

            setup_df = setup_df.set_index(['mouse_id', 'channel_name', 'mse_tr', 'mse_cv', 'mse_te', 'timestamp'], append=True)

            setup_df.to_csv(f'{best_coeffs_npy_folder}/{run_id}_best_coeffs_R2_{holdout_score_rnd}.csv', index=True, header=True)
            all_coeff_dfs.append(setup_df)


            # visualize.plot_all_beta_coefs(glm.coef_, X_cols_plot,
            #                                 X_cols_sftd_plot,
            #                                 # plot_width=4,
            #                                 plot_width=plot_width,
            #                                 plot_rows=plot_rows,
            #                                 y_lims=(-3.0, 3.0),
            #                                 binsize=54,
            #                                 filename=best_beta_fn,
            #                                 plot_name=f'Best Coeffs - {run_id} — {best_params}'
            #                                 )
            
            # best_rcnstrct_fn = f'{best_reconstruct_folder}/{run_id}_best_{avg_reconstruct_basename}_R2_{holdout_score_rnd}.png'


            # visualize.plot_avg_reconstructions_v2(dfrel_holdout,
            #                             alignment_col_lst=[ #'cpn', 'spnr', 'spnnr',
            #                                                 # 'photometryCenterInIndex', #'photometryCenterOutIndex',
            #                                                 'photometrySideInIndexr', 'photometrySideInIndexnr',
            #                                                 'photometrySideOutIndexr', 'photometrySideOutIndexnr',

            #                                                 'photometrySideInIndexAA', 'photometrySideInIndexAa',
            #                                                 'photometrySideInIndexaA','photometrySideInIndexaa',
            #                                                 'photometrySideInIndexAB', 'photometrySideInIndexAb',
            #                                                 'photometrySideInIndexaB','photometrySideInIndexab',

            #                                                 'photometryCenterInIndex', #'photometryCenterOutIndex',

            #                                                 'photometrySideInIndex',

            #                                                 # 'photometrySideOutIndexAA', 'photometrySideOutIndexAa',
            #                                                 # 'photometrySideOutIndexaA', 'photometrySideOutIndexaa',
            #                                                 # 'photometrySideOutIndexAB', 'photometrySideOutIndexAb',
            #                                                 # 'photometrySideOutIndexaB', 'photometrySideOutIndexab',
            #                                         ],
                                    
            #                             channel=y_col,
            #                             binsize = 54,
            #                             # plot_width=4,
            #                             plot_width=2,
            #                             min_time = -20,
            #                             max_time = 30,
            #                             min_signal = -3.0,
            #                             max_signal = 3.0,
            #                             file_name=best_rcnstrct_fn,
            #                             title=f'Best Average Reconstruction - {run_id} — {best_params}'
            #                             )

            # best_rcnstrct_fn = f'{best_reconstruct_folder}/{run_id}_tr+ho_best_{avg_reconstruct_basename}_R2_{holdout_score_rnd}.png'

            # visualize.plot_avg_reconstructions_v2(dfrel,
            #                             alignment_col_lst=[ #'cpn', 'spnr', 'spnnr',
            #                                                 # 'photometryCenterInIndex', #'photometryCenterOutIndex',
            #                                                 'photometrySideInIndexr', 'photometrySideInIndexnr',
            #                                                 'photometrySideOutIndexr', 'photometrySideOutIndexnr',

            #                                                 'photometrySideInIndexAA', 'photometrySideInIndexAa',
            #                                                 'photometrySideInIndexaA','photometrySideInIndexaa',
            #                                                 'photometrySideInIndexAB', 'photometrySideInIndexAb',
            #                                                 'photometrySideInIndexaB','photometrySideInIndexab',

            #                                                 'photometryCenterInIndex', #'photometryCenterOutIndex',

            #                                                 'photometrySideInIndex',

            #                                                 # 'photometrySideOutIndexAA', 'photometrySideOutIndexAa',
            #                                                 # 'photometrySideOutIndexaA', 'photometrySideOutIndexaa',
            #                                                 # 'photometrySideOutIndexAB', 'photometrySideOutIndexAb',
            #                                                 # 'photometrySideOutIndexaB', 'photometrySideOutIndexab',
            #                                         ],
                                    
            #                             channel=y_col,
            #                             binsize = 54,
            #                             # plot_width=4,
            #                             plot_width=2,
            #                             min_time = -20,
            #                             max_time = 30,
            #                             min_signal = -3.0,
            #                             max_signal = 3.0,
            #                             file_name=best_rcnstrct_fn,
            #                             title=f'Best Average Reconstruction — Training + Holdout - {run_id} — {best_params}'
            #                             )


            # dfrel_resids = dfrel[['file_num', 'nTrial', 'wi_trial_keep', 'gDA', 'gACH', 'rDA', 'pred']].copy()
            # dfrel_resids['mouse_id'] = mouse_id
            # dfrel_resids['channel_name'] = y_col

            # dfrel_resids.set_index(['mouse_id', 'channel_name', 'file_num', 'nTrial'], inplace=True)
            # dfrel_resids.to_csv(f'{best_resids_npy_folder}/{run_id}_best_resids_R2_{holdout_score_rnd}.csv', index=True, header=True)

            dfrel_basis[y_col + '_resid'] = dfrel_basis[y_col] - dfrel['pred']
            if fix_training:
                dfrel_ft[y_col + '_resid'] = dfrel_ft[y_col] - dfrel['pred']
                dfrel_ft_setup[y_col + '_resid'] = dfrel_ft_setup[y_col] - dfrel_setup['pred']
                dfrel_ft_holdout[y_col + '_resid'] = dfrel_ft_holdout[y_col] - dfrel_holdout['pred']
                
            # print(dfrel_resids)
            # print(dfrel[['pred']])
            # dfrel.set_index(['file_num_inx'], append=True, inplace=True)

            dfrel_resids[f'pred_paramsNum={iXyd}_{y_col}'] = glm.predict(dfrel_basis[prediction_X_cols_sftd])
            
            if len(dfrel_resids_setup):
                dfrel_resids_setup[f'pred_paramsNum={iXyd}_{y_col}'] = glm.predict(dfrel_ft[~holdout][prediction_X_cols_sftd])
            if len(dfrel_resids_holdout):
                dfrel_resids_holdout[f'pred_paramsNum={iXyd}_{y_col}'] = glm.predict(dfrel_ft[holdout][prediction_X_cols_sftd])


            # for fitted_model_dict in (cv_results['full_cv_results']):
            #     fitted_model = fitted_model_dict['model']
            #     kwarg_info = "_".join([f"{_k}_{fitted_model_dict['glm_kwargs'][_k]}" for _k in fitted_model_dict["glm_kwargs"]])

            #     model_coef = fitted_model.coef_
            #     model_intercept = fitted_model.intercept_

            #     std_name = f'{run_id}_{kwarg_info}'
            #     np.save(f'{all_models_folder}/coeffs/{std_name}_{model_c_basename}.npy', model_coef)
            #     np.save(f'{all_models_folder}/intercepts/{std_name}_{model_i_basename}.npy', model_intercept)
                
            #     tmp_holdout_score = fitted_model.r2_score(X_holdout_noiti, y_holdout_noiti)
            #     holdout_score_rnd = np.round(tmp_holdout_score, 4)
                

            #     visualize.plot_all_beta_coefs(fitted_model.coef_, X_cols_plot,
            #                                     X_cols_sftd_plot,
            #                                     plot_width=4,
            #                                     y_lims=(-3.0, 3.0),
            #                                     # filename=f'{fn}_coeffs.png',
            #                                     binsize=54,
            #                                     filename=f'{all_coeffs_folder}/{std_name}_{all_betas_basename}_R2_{holdout_score_rnd}.png',
            #                                     plot_name=f'Coeffs by Timeshift - {run_id} — {kwarg_info}'
            #                                     # plot_name=f'{fn} — {y_col} — {kwarg_info}'
            #                                     )


            #     visualize.plot_avg_reconstructions_v2(dfrel_holdout,
            #     # visualize.plot_avg_reconstructions_v2(dfrel,
            #                                     channel=y_col,
            #                                     plot_width=4,
            #                                     binsize = 54,
            #                                     min_time = -20,
            #                                     max_time = 30,
            #                                     min_signal = -2.5,
            #                                     max_signal = 2.5,
            #                                     file_name=f'{all_reconstruct_folder}/{std_name}_{avg_reconstruct_basename}_R2_{holdout_score_rnd}.png',
            #                                     title=f'Average Reconstruction - {run_id} — {kwarg_info}'
            #                                 )
        
        if len(dfrel_resids) != 0 and holdout_score_rnd is not None:
            dfrel_resids.set_index(['nTrial'], append=True, inplace=True)
            dfrel_resids.to_csv(f'{best_resids_npy_folder}/{prefix}_{fn}_best_resids_R2_{holdout_score_rnd}_all.csv', index=True, header=True)
        if len(dfrel_resids_setup) != 0 and holdout_score_rnd is not None:
            dfrel_resids_setup.set_index(['nTrial'], append=True, inplace=True)
            dfrel_resids_setup.to_csv(f'{best_resids_npy_folder}/{prefix}_{fn}_best_resids_R2_{holdout_score_rnd}_stp.csv', index=True, header=True)
        if len(dfrel_resids_holdout) != 0 and holdout_score_rnd is not None:
            dfrel_resids_holdout.set_index(['nTrial'], append=True, inplace=True)
            dfrel_resids_holdout.to_csv(f'{best_resids_npy_folder}/{prefix}_{fn}_best_resids_R2_{holdout_score_rnd}_ho.csv', index=True, header=True)

combined_best_coeffs = pd.concat(all_coeff_dfs, axis=0)
combined_best_coeffs.to_csv(f'{best_combined_coeffs_folder}/{prefix}.csv', index=True, header=True)
