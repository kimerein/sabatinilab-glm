import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import sglm
import sglm_pp
import sglm_cv

# def setup_autoregression(X, response_cols, order):
#     col_nums = sglm_pp.get_column_nums(X, response_cols)
#     return sglm_pp.timeshift_multiple(X, shift_inx=col_nums, shift_amt_list=list(range(order + 1)))


# Update K_folds to be num_folds
# Rather than trial_split name use group_split


def timeshift_cols_by_signal_length(X, cols_to_shift, neg_order=0, pos_order=1, trial_id='nTrial', dummy_col='nothing', shift_amt_ratio=2.0):
    """
    Shift the columns of X by a fractional amounts of the minimum non-zero signal length (in order to reduce multicollinearity).
    neg_order and pos_order . shift_amt_ratio 

    JZ 2021

    Parameters
    ----------
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
        Column name identifying in which trial the event is currently occurring
    dummy_col : str
        Dummy column to be used for counting the number of entries, which are non-zero in the DataFrame (a new
        column is created with this name and dropped afterwards if it does not exist at the start)
    shift_amt_ratio : float
        The factor of a signal length to shift forward / backward (as calculated from the min signal length).
        (e.g. if the shortest 'Cue' to which a mouse is exposed is 20 timesteps and we run this function on 'Cue'
        with a shift_amt_ratio of 2, timeshifts will be performed in incraments of 10 timesteps.)
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
    
    Parameters
    ----------
    all_cols : list(str)
        All column names prior to the addition of shifted column names
    shifted_cols : list(str)
        The list of columns that have been timeshifted by the sft_orders
    sft_orders : list(int)
        A list of the timeshifts used for all of the columns provided in "shifted_columns"
    """ 
    out_col_list = []
    for col in shifted_cols:
        neg_order_lst = sft_orders[col][0]
        pos_order_lst = sft_orders[col][1]
        out_col_list.extend([col + f'_{_}' for _ in neg_order_lst + pos_order_lst])

    return all_cols + out_col_list


def timeshift_cols(X, cols_to_shift, neg_order=0, pos_order=1):
    """
    Shift the columns of X forward by all timesteups up to pos_order (inclusive) and backward by all timesteps down to neg_roder (inclusive)

    JZ 2021
    
    Parameters
    ----------
    X : pd.DataFrame
        Underlying pandas DataFrame data to be shifted
    cols_to_shift : list(str)
        Column names in pandas DataFrame to shift
    neg_order : int
        Negative order i.e. number of shifts to perform backwards
    pos_order : int
        Positive order i.e. number of shifts to perform forwards
    """    
    col_nums = sglm_pp.get_column_nums(X, cols_to_shift)
    return sglm_pp.timeshift_multiple(X, shift_inx=col_nums, shift_amt_list=[0]+list(range(neg_order, 0))+list(range(1, pos_order + 1)))


def add_timeshifts_to_col_list(all_cols, shifted_cols, neg_order=0, pos_order=1):
    """
    Add a number of timeshifts to the shifted_cols list provided for every column used. 

    JZ 2021
    
    Parameters
    ----------
    all_cols : list(str)
        All column names prior to the addition of shifted column names
    shifted_cols : list(str)
        The list of columns that have been timeshifted
    neg_order : int
        Negative order i.e. number of shifts performed backwards
    pos_order : int
        Positive order i.e. number of shifts performed forwards
    """ 
    out_col_list = []
    for shift_amt in list(range(neg_order, 0))+list(range(1, pos_order + 1)):
        out_col_list.extend([_ + f'_{shift_amt}' for _ in shifted_cols])
    return all_cols + out_col_list

def fit_GLM(X, y, model_name='Gaussian', *args, **kwargs):
    """
    Fit GLM on training dataset of predictor columns of X and response y

    JZ 2021
    
    Parameters
    ----------
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
    """
    glm = sglm.GLM(model_name, *args, **kwargs)
    glm.fit(X.values, y.values)
    return glm

def diff_cols(X, cols, append_to_base=True):
    """
    Take differentials along columns col of DataFrame X

    JZ 2021
    
    Parameters
    ----------
    X : pd.DataFrame
        DataFrame of data of which to take the differential
    cols : list(str)
        Names of specific columns along which to take the differential
    append_to_base : bool
        Whether or not those columns should be returned as columns added to the original DataFrame
    """
    col_nums = sglm_pp.get_column_nums(X, cols)
    X = sglm_pp.diff(X, col_nums, append_to_base=append_to_base)
    return X

def cv_idx_by_timeframe(X, y=None, timesteps_per_bucket=20, num_folds=10, test_size=None):
    """
    Generate Cross Validation indices by keeping together bucketed timesteps
    (bucketing together timesteps between intervals of timesteps_per_bucket).

    JZ 2021
    
    Parameters
    ----------
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
    """
    bucket_ids = sglm_pp.bucket_ids_by_timeframe(X.shape[0], timesteps_per_bucket=timesteps_per_bucket)
    cv_idx = sglm_pp.cv_idx_from_bucket_ids(bucket_ids, X, y=y, num_folds=num_folds, test_size=test_size)
    return cv_idx


def holdout_split_by_trial_id(X, y=None, trial_id_columns=[], num_folds=5, test_size=None):
    """
    Generate Cross Validation indices by keeping together trial id columns
    (bucketing together by trial_id_columns).

    JZ 2021
    
    Parameters
    ----------
    X : pd.DataFrame
        Prediction DataFrame from which to bucket
    y : pd.Series
        Response Series
    trial_id_columns : list(str)
        Columns to use to identify bucketing identifiers
    num_folds : int
        Number of Cross Validation segmentations that should be used for GroupShuffleSplit fold Cross Validation
    test_size : float
        Percentage of datapoints to use in each GroupShuffleSplit fold for validation
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


def cv_idx_by_trial_id(X, y=None, trial_id_columns=[], num_folds=5, test_size=None):
    """
    Generate Cross Validation indices by keeping together trial id columns
    (bucketing together by trial_id_columns).

    JZ 2021
    
    Parameters
    ----------
    X : pd.DataFrame
        Prediction DataFrame from which to bucket
    y : pd.Series
        Response Series
    trial_id_columns : list(str)
        Columns to use to identify bucketing identifiers
    num_folds : int
        Number of Cross Validation segmentations that should be used for GroupShuffleSplit fold Cross Validation
    test_size : float
        Percentage of datapoints to use in each GroupShuffleSplit fold for validation
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

# Trial-based splitting (remove inter-trial information)

def simple_cv_fit(X, y, cv_idx, glm_kwarg_lst, model_type='Normal', verbose=0):
    """
    Fit the desired model using the list of keyword arguments provided in
    glm_kwarg_lst, identify the best model, and return the associated
    score, parameters, and the model itself.

    JZ 2021
    
    Parameters
    ----------
    X : pd.DataFrame
        Predictor DataFrame to fit
    y : pd.Series
        Response Series to fit
    cv_idx : list(tuple(tuple(int)))
        List of list of indices to use for k-fold Cross Validation
        - k-folds list [ ( training tuple(indices), testing tuple(indices) ) ]
    glm_kwarg_lst : list(dict)
        List of dictionaries of keyword arguments to try for Cross Validation parameter search
    model_type : str
        Keyword arguments to be passed to GLM model
    """
    # Step 4: Fit GLM models for all possible sets of values
    cv_results = sglm_cv.cv_glm_mult_params(X.values,
                                            y.values,
                                            cv_idx,
                                            model_type,
                                            glm_kwarg_lst,
                                            verbose=verbose
                                            # [tuple([glm_kwarg[_] for _ in []]) for glm_kwarg in glm_kwarg_lst]
                                            )
    best_score = cv_results['best_score']
    best_params = cv_results['best_params']
    best_model = cv_results['best_model']
    return best_score, best_params, best_model

























if __name__ == '__main__':
    X_tmp = pd.DataFrame(np.arange(20).reshape((10, 2)), columns=['A', 'B'])
    X_tmp['B'] = (X_tmp['B']-1) * 2 + 1
    print()
    print(X_tmp)
    # X_tmp = setup_autoregression(X_tmp, ['B'], 4)
    # print()
    # print(X_tmp)

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
