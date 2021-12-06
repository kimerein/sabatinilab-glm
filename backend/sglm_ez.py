import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import sglm
import sglm_pp
import sglm_cv
import matplotlib.pyplot as plt

# def setup_autoregression(X, response_cols, order):
#     col_nums = sglm_pp.get_column_nums(X, response_cols)
#     return sglm_pp.timeshift_multiple(X, shift_inx=col_nums, shift_amt_list=list(range(order + 1)))

def timeshift_cols_by_signal_length(X, cols_to_shift, neg_order=0, pos_order=1, trial_id='nTrial', dummy_col='nothing', shift_amt_ratio=2.0):
    """
    Shift the columns of X by fractional amounts of the minimum non-zero signal length (in order to reduce multicollinearity).

    JZ 2021

    Args:
        X : pd.DataFrame
            Underlying pandas DataFrame data to be shifted
        cols_to_shift : list(str)
            Column names in pandas DataFrame to shift
        neg_order : int
            Negative order i.e. number of shifts to perform backwards (i.e. max number of timesteps backwards to be
            shifted regardless of the length of the signal itself)
        pos_order : int
            Positive order i.e. number of shifts to perform forwards (i.e. max number of timesteps forwards to be
            shifted regardless of the length of the signal itself)
        trial_id : str
            Column name that identifies event lengths on a per-trial basis
        dummy_col : str
            Dummy column name to be used for counting the number of entries, which are non-zero in the DataFrame (a new
            column is created with this name and dropped afterwards if it does not exist at the start)
        shift_amt_ratio : float
            The factor of a signal length to shift forward / backward (as calculated from the min signal length).
            (e.g. if the shortest 'Cue' to which a mouse is exposed is 20 timesteps and we run this function on 'Cue'
            with a shift_amt_ratio of 2, timeshifts will be performed in incraments of 10 timesteps.)
    
    Returns: (Timeshifted DataFrame, List of Timeshift orders used)
    """

    X = X.copy()
    if dummy_col not in X.columns:
        X[dummy_col] = 1
        created = True
    else:
        created = False

    min_num_ts = {}
    sft_orders = {}
    for col in cols_to_shift:
        min_num_ts[col] = X.query(f'{col} > 0').groupby([trial_id, col])[dummy_col].count().min()

        col_nums = sglm_pp.get_column_nums(X, [col])

        shift_amt = min_num_ts[col] // shift_amt_ratio
        shift_amt = max(shift_amt, 1)

        print(f'mnts: {min_num_ts[col]}, sar: {shift_amt_ratio}')
        

        neg_order_lst = list(np.arange(neg_order, 0, shift_amt))
        pos_order_lst = list(np.arange(shift_amt, pos_order + 1, shift_amt))

        sft_orders[col] = (neg_order_lst, pos_order_lst)

        X = sglm_pp.timeshift_multiple(X, shift_inx=col_nums, shift_amt_list= [0] + neg_order_lst + pos_order_lst)

    if created:
        X = X.drop(dummy_col, axis=1)

    return X, sft_orders


def add_timeshifts_by_sl_to_col_list(all_cols, shifted_cols, sft_orders):
    """
    Add the number of timeshifts to the shifted_cols list provided for every column used. 

    JZ 2021
    
    Args:
        all_cols : list(str)
            All column names prior to the addition of shifted column names
        shifted_cols : list(str)
            The list of columns that have been timeshifted by the sft_orders
        sft_orders : list(int)
            A list of the timeshifts used for all of the columns provided in "shifted_columns"
    
    Returns: List of all column names remaining after shifts in question
    """ 
    out_col_list = []
    for col in shifted_cols:
        neg_order_lst = sft_orders[col][0]
        pos_order_lst = sft_orders[col][1]
        out_col_list.extend([col + f'_{_}' for _ in neg_order_lst + pos_order_lst])

    return all_cols + out_col_list


def timeshift_cols(X, cols_to_shift, neg_order=0, pos_order=1):
    """
    Shift the columns of X forward by all timesteups up to pos_order (inclusive)
    and backward by all timesteps down to neg_roder (inclusive)

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Underlying pandas DataFrame data to be shifted
        cols_to_shift : list(str)
            Column names in pandas DataFrame to shift
        neg_order : int
            Negative order i.e. number of shifts to perform backwards (should be in range -inf to 0 (incl.))
        pos_order : int
            Positive order i.e. number of shifts to perform forwards (should be in range 0 (incl.) to inf)
    
    Returns: New DataFrame with all shifted cols included in output
    """    
    col_nums = sglm_pp.get_column_nums(X, cols_to_shift)
    return sglm_pp.timeshift_multiple(X, shift_inx=col_nums, shift_amt_list=[0]+list(range(neg_order, 0))+list(range(1, pos_order + 1)))


def add_timeshifts_to_col_list(all_cols, shifted_cols, neg_order=0, pos_order=1):
    """
    Add a number of timeshifts to the shifted_cols name list provided for every column used. 

    JZ 2021
    
    Args:
        all_cols : list(str)
            All column names prior to the addition of shifted column names
        shifted_cols : list(str)
            The list of columns that have been timeshifted
        neg_order : int
            Negative order i.e. number of shifts performed backwards (should be in range -inf to 0 (incl.))
        pos_order : int
            Positive order i.e. number of shifts performed forwards (should be in range 0 (incl.) to inf)
    
    Returns: List of all column names remaining after shifts in question
    """ 
    out_col_list = []
    for shift_amt in list(range(neg_order, 0))+list(range(1, pos_order + 1)):
        out_col_list.extend([_ + f'_{shift_amt}' for _ in shifted_cols])
    return all_cols + out_col_list

def fit_GLM(X, y, model_name='Gaussian', *args, **kwargs):
    """
    Fit GLM on training dataset of predictor columns of X and response y

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Predictor DataFrame from which to predict the response
        y : pd.Series
            Response to be predicted
        model_name : str
            Type of GLM to build (e.g. Gaussian, Poisson, Logistic, etc.)
        *args : *iterable
            Positional arguments to be passed to GLM model
        **kwargs : **dict
            Keyword arguments to be passed to GLM model
    
    Returns: Fitted GLM model
    """
    glm = sglm.GLM(model_name, *args, **kwargs)
    glm.fit(X.values, y.values)
    return glm

def diff_cols(X, cols, append_to_base=True):
    """
    Take differentials along columns col of DataFrame X

    JZ 2021
    
    Args:
        X : pd.DataFrame
            DataFrame of data of which to take the differential
        cols : list(str)
            Names of specific columns along which to take the differential
        append_to_base : bool
            Whether or not those columns should be returned as columns added to the original DataFrame
    
    Returns: DataFrame X with expected diffs performed
    """
    col_nums = sglm_pp.get_column_nums(X, cols)
    X = sglm_pp.diff(X, col_nums, append_to_base=append_to_base)
    return X

def cv_idx_by_timeframe(X, y=None, timesteps_per_bucket=20, num_folds=10, test_size=None):
    """
    Generate Cross Validation indices by keeping together bucketed timesteps
    (bucketing together timesteps between intervals of timesteps_per_bucket).

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Prediction DataFrame from which to bucket
        y : pd.Series
            Response Series
        timesteps_per_bucket : int
            Number of timesteps (i.e. rows in the DataFrame) that should be kept together as buckets
        num_folds : int
            Number of Cross Validation segmentations that should be used for k-fold Cross Validation
        test_size : float
            Percentage of datapoints to use in each GroupShuffleSplit fold for validation
    
    Returns: List of tuples of indices to be used for validation / hyperparameter selection
    """
    bucket_ids = sglm_pp.bucket_ids_by_timeframe(X.shape[0], timesteps_per_bucket=timesteps_per_bucket)
    cv_idx = sglm_pp.cv_idx_from_bucket_ids(bucket_ids, X, y=y, num_folds=num_folds, test_size=test_size)
    return cv_idx


def holdout_split_by_trial_id(X, y=None, id_cols=['nTrial', 'iBlock'], perc_holdout=0.2):
    """
    Create a True/False pd.Series using Group ID columns to identify the holdout data to
    be used via GroupShuffleSplit.

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Prediction DataFrame from which to bucket
        y : pd.Series
            Response Series
        id_cols : list(str)
            Columns to use to identify bucketing identifiers
        perc_holdout : int
            Percentage of group identifiers to holdout as test set
    
    Returns: pd.Series of True values if it should be heldout, False if it should be part of training/validation
    """
    
    for i, idc in enumerate(id_cols):
        if i == 0:
            bucket_ids = X[idc].astype(str).str.len().astype(str) + ':' + X[idc].astype(str)
        else:
            bucket_ids = bucket_ids + '_' + X[idc].astype(str)
    bucket_ids = bucket_ids.astype("category").cat.codes

    num_bucket_ids = int(bucket_ids.max() + 1)
    num_buckets_for_test = int(num_bucket_ids * perc_holdout)

    test_ids = np.random.choice(num_bucket_ids, size=num_buckets_for_test)
    holdout = bucket_ids.isin(test_ids)

    return holdout


def cv_idx_by_trial_id(X, y=None, trial_id_columns=[], num_folds=5, test_size=None):
    """
    Generate Cross Validation indices by keeping together trial id columns
    (bucketing together by trial_id_columns) via GroupShuffleSplit.

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Prediction DataFrame from which to bucket
        y : pd.Series
            Response Series
        trial_id_columns : list(str)
            Columns to use to identify bucketing identifiers
        num_folds : int
            Number of Cross Validation segmentations that should be used for GroupShuffleSplit fold Cross Validation
        test_size : float
            Percentage of datapoints to use in each GroupShuffleSplit fold for validation (Defaults to 1/num_folds if None)

    Returns: List of tuple of indices to be used for validation / hyperparameter selection
    """
    X = pd.DataFrame(X)

    for i, idc in enumerate(trial_id_columns):
        if i == 0:
            bucket_ids = X[idc].astype(str).str.len().astype(str) + ':' + X[idc].astype(str)
        else:
            bucket_ids = bucket_ids + '_' + X[idc].astype(str)
    
    bucket_ids = bucket_ids.astype("category").cat.codes
    
    cv_idx = sglm_pp.cv_idx_from_bucket_ids(bucket_ids, X, y=y, num_folds=num_folds, test_size=test_size)
    return cv_idx


# Trial-based splitting (remove inter-trial information?)
def simple_cv_fit(X, y, cv_idx, glm_kwarg_lst, model_type='Normal', verbose=0, score_method='mse'):
    """
    Fit the desired model using the list of keyword arguments provided in
    glm_kwarg_lst, identify the best model, and return the associated
    score, parameters, and the model itself.

    JZ 2021
    
    Args:
        X : pd.DataFrame
            Predictor DataFrame to fit
        y : pd.Series
            Response Series to fit
        cv_idx : list(tuple(tuple(int)))
            List of list of indices to use for fold Cross Validation —
            k-folds list [ ( training tuple(indices), testing tuple(indices) ) ]
        glm_kwarg_lst : list(dict)
            List of dictionaries of keyword arguments to try for validation parameter search
        model_type : str
            Keyword arguments to be passed to GLM model
        verbose ; int
            Amount of information to print out during model fitting / validation (larger numbers print more)
        score_method : str
            Either 'mse' or 'r2' to base cross-validation selection on Mean Squared Error or R^2

    Returns: From the model with the best (largest) score value, return the...
             Best Score Value, Best Score Standard Deviation, Best Params, Best Model
    """
    # Step 4: Fit GLM models for all possible sets of values
    cv_results = sglm_cv.cv_glm_mult_params(X.values,
                                            y.values,
                                            cv_idx,
                                            model_type,
                                            glm_kwarg_lst,
                                            verbose=verbose,
                                            score_method=score_method
                                            # [tuple([glm_kwarg[_] for _ in []]) for glm_kwarg in glm_kwarg_lst]
                                            )
    best_score = cv_results['best_score']
    best_score_std = cv_results['best_score_std']
    best_params = cv_results['best_params']
    best_model = cv_results['best_model']
    return best_score, best_score_std, best_params, best_model, cv_results





def get_coef_name_sets(coef_names, sftd_coef_names):    
    coef_cols = {}

    for coef_name in coef_names:
        if coef_name in ['nTrial', 'nEndTrial']:
            continue
        lst = [_ for _ in sftd_coef_names if coef_name in _.split('_')]
        lst = [_ if _ != coef_name else coef_name+'_0' for _ in lst]
        lst = sorted(lst, key=lambda x: int(x.split('_')[-1]))
        # lst = [_.replace('_0', '') for _ in lst]
        coef_cols[coef_name] = lst
    return coef_cols

def get_single_coef_set(names, lookup):
    return [int(_.split('_')[-1]) for _ in names], [lookup[_.replace('_0', '')] for _ in names]


def calc_l1(coeffs):
    return np.sum(np.abs(coeffs))

def calc_l2(coeffs):
    return np.sum(np.square(coeffs))


def get_trial_timestamp(df, trial_id_col='nTrial'):
    dummy_col_name = '1'
    timestamp_name = 'tim'

    df_tmp = df.copy()
    df_tmp[dummy_col_name] = 1
    df_tmp[timestamp_name] = df_tmp.groupby(trial_id_col)[dummy_col_name].cumsum()
    return df_tmp[timestamp_name]

def get_first_timestamp_abv_threshold(df, thresh_col, trial_id_col='nTrial', threshold=0.0):
    tmp = df.copy()
    tmp['above_flag'] = (tmp[thresh_col] > threshold)*1.0
    above_loc = tmp.groupby('nTrial')['above_flag'].transform(lambda x: x.argmax()).astype(int)
    return above_loc

def get_is_trial(X, gb_name=['nTrial'], col_names=['r']):
    for icol, col in enumerate(col_names):
        if icol == 0:
            check_list = (X.groupby(gb_name)[col].transform(np.max) == 1)
        else:
            check_list = check_list&(X.groupby(gb_name)[col].transform(np.max) == 1)
    return check_list

def get_sem(df, filt, gb, col, mult=1.96):
    interim = df[filt].groupby(gb)[col].agg([np.mean, np.std, np.size])
    interim['sem'] = interim['std'] / np.sqrt(interim['size'])
    interim['ub'] = interim['mean'] + 1.96*interim['sem']
    interim['lb'] = interim['mean'] - 1.96*interim['sem']
    return interim[['lb', 'mean', 'ub', 'sem', 'size', 'std']]


# # def 

# tmp = dfrel[holdout].copy()
# tmp['tim'] = get_trial_timestamp(tmp, (tmp['nTrial'] != tmp['nEndTrial']), trial_id_col='nTrial')

# # tmp = dfrel[holdout]
# # tmp = tmp[(tmp['nTrial'] != tmp['nEndTrial'])].copy()
# # tmp['1'] = 1
# # tmp['tim'] = tmp.groupby('nTrial')['1'].cumsum()
# tmp['pred'] = glm.predict(tmp[X_cols_sftd])

# entry_timing_r = tmp.groupby('nTrial')['rpn'].agg(lambda x: (x).argmax()).astype(int)
# entry_timing_l = tmp.groupby('nTrial')['lpn'].agg(lambda x: (x).argmax()).astype(int)
# entry_timing = (entry_timing_r > entry_timing_l)*entry_timing_r + (entry_timing_r < entry_timing_l)*entry_timing_l

# adjusted_time = (tmp.set_index('nTrial')['tim'] - entry_timing)
# adjusted_time.index = tmp.index
# tmp['adjusted_time'] = adjusted_time

# def get_is_trial(X, gb_name=['nTrial'], col_names=['r']):
#     for icol, col in enumerate(col_names):
#         if icol == 0:
#             check_list = (X.groupby(gb_name)[col].transform(np.max) == 1)
#         else:
#             check_list = check_list&(X.groupby(gb_name)[col].transform(np.max) == 1)
#     return check_list

# def get_sem(df, filt, gb, col, mult=1.96):
#     interim = df[filt].groupby(gb)[col].agg([np.mean, np.std, np.size])
#     interim['sem'] = interim['std'] / np.sqrt(interim['size'])
#     interim['ub'] = interim['mean'] + 1.96*interim['sem']
#     interim['lb'] = interim['mean'] - 1.96*interim['sem']
#     return interim[['lb', 'mean', 'ub', 'sem', 'size', 'std']]

# tmp_backup = tmp

# binsize = 50

# min_time = -20
# max_time = 30
# min_signal = -3.0
# max_signal = 3.0

# x_label = 'Timesteps __ from Event'
# y_label = 'Response'

# if binsize is not None:
#     min_time *= binsize
#     max_time *= binsize
#     x_label = x_label.replace(' __', ' (ms)')
#     tmp_backup['plot_time'] = tmp_backup['adjusted_time'] * binsize
# else:
#     x_label = x_label.replace(' __', '')
#     tmp_backup['plot_time'] = tmp_backup['adjusted_time']



# tmp = tmp_backup[tmp_backup['plot_time'].between(min_time, max_time)].copy()



# fig, ax = plt.subplots(2,2)
# fig.suptitle('Average Photometry Response Aligned to Side Port Entry — Holdout Data Only')
# fig.set_figheight(20)
# fig.set_figwidth(40)

# tmp['is_rlpn_trial'] = get_is_trial(tmp, ['nTrial'], ['r', 'lpn'])
# tmp['is_rrpn_trial'] = get_is_trial(tmp, ['nTrial'], ['r', 'rpn'])
# tmp['is_nrlpn_trial'] = get_is_trial(tmp, ['nTrial'], ['nr', 'lpn'])
# tmp['is_nrrpn_trial'] = get_is_trial(tmp, ['nTrial'], ['nr', 'rpn'])

# ci_setup = get_sem(tmp, tmp['is_rlpn_trial'], 'plot_time', 'zsgdFF')
# ax[0,0].plot(ci_setup['mean'], color='b')
# ax[0,0].fill_between(ci_setup.index, ci_setup['lb'], ci_setup['ub'], color='b', alpha=.2)

# ci_setup = get_sem(tmp, tmp['is_rlpn_trial'], 'plot_time', 'pred')
# ax[0,0].plot(ci_setup['mean'], color='r')


# ax[0,0].set_xlim((min_time, max_time))
# ax[0,0].set_ylim((min_signal, max_signal))
# ax[0,0].title.set_text('Rewarded, Left Port Entry')
# ax[0,0].set_xlabel(x_label)
# ax[0,0].set_ylabel(y_label)
# ax[0,0].grid()

# ci_setup = get_sem(tmp, tmp['is_rrpn_trial'], 'plot_time', 'zsgdFF')
# ax[0,1].plot(ci_setup['mean'])
# ax[0,1].fill_between(ci_setup.index, ci_setup['lb'], ci_setup['ub'], color='b', alpha=.2)

# ci_setup = get_sem(tmp, tmp['is_rrpn_trial'], 'plot_time', 'pred')
# ax[0,1].plot(ci_setup['mean'], color='r')



# ax[0,1].set_xlim((min_time, max_time))
# ax[0,1].set_ylim((min_signal, max_signal))
# ax[0,1].title.set_text('Rewarded, Right Port Entry')
# ax[0,1].set_xlabel(x_label)
# ax[0,1].set_ylabel(y_label)
# ax[0,1].grid()

# ci_setup = get_sem(tmp, tmp['is_nrlpn_trial'], 'plot_time', 'zsgdFF')
# ax[1,0].plot(ci_setup['mean'])
# ax[1,0].fill_between(ci_setup.index, ci_setup['lb'], ci_setup['ub'], color='b', alpha=.2)

# ci_setup = get_sem(tmp, tmp['is_nrlpn_trial'], 'plot_time', 'pred')
# ax[1,0].plot(ci_setup['mean'], color='r')



# ax[1,0].set_xlim((min_time, max_time))
# ax[1,0].set_ylim((min_signal, max_signal))
# ax[1,0].title.set_text('Unrewarded, Left Port Entry')
# ax[1,0].set_xlabel(x_label)
# ax[1,0].set_ylabel(y_label)
# ax[1,0].grid()

# ci_setup = get_sem(tmp, tmp['is_nrrpn_trial'], 'plot_time', 'zsgdFF')
# ax[1,1].plot(ci_setup['mean'])
# ax[1,1].fill_between(ci_setup.index, ci_setup['lb'], ci_setup['ub'], color='b', alpha=.2)

# ci_setup = get_sem(tmp, tmp['is_nrrpn_trial'], 'plot_time', 'pred')
# ax[1,1].plot(ci_setup['mean'], color='r')



# ax[1,1].set_xlim((min_time, max_time))
# ax[1,1].set_ylim((min_signal, max_signal))
# ax[1,1].title.set_text('Unrewarded, Right Port Entry')
# ax[1,1].set_xlabel(x_label)
# ax[1,1].set_ylabel(y_label)
# ax[1,1].grid()

# ax[1,1].legend(['Mean Photometry Response',
#                 'Predicted Photometry Response',
#                 '95% SEM Confidence Interval'])

# fig.savefig('figure_outputs/average_response_reconstruction.png')




# sglm_ez.plot_all_beta_coefs(glm, X_cols, X_cols_sftd, plot_width=2)














if __name__ == '__main__':
    X_tmp = pd.DataFrame(np.arange(20).reshape((10, 2)), columns=['A', 'B'])
    X_tmp['B'] = (X_tmp['B']-1) * 2 + 1
    print()
    print(X_tmp)

    X_tmp = timeshift_cols(X_tmp, ['A'], 0, 2)
    print(X_tmp)

    print('HERE')

    X_tmp = diff_cols(X_tmp, ['B'])
    print()
    print(X_tmp)

    X_tmp = X_tmp.dropna()
    print()
    print(X_tmp)
    
    # glm = fit_GLM(X_tmp[['A', 'B_1', 'B_2', 'B_3', 'B_4', 'A_1', 'A_2', 'B_1_diff']], X_tmp['B'], reg_lambda=0.1)
    glm = fit_GLM(X_tmp[['A', 'A_1', 'A_2']], X_tmp['B'], alpha=0.1)
    print(glm.coef_, glm.intercept_)

    # Step 1: Create a dictionary of lists for these relevant keywords...
    kwargs_iterations = {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'fit_intercept': [True, False]
    }

    # Step 2: Create a dictionary for the fixed keyword arguments that do not require iteration...
    kwargs_fixed = {
        'link': 'auto',
        'max_iter': 1000
    }

    # Step 3: Generate iterable list of keyword sets for possible combinations
    glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)
