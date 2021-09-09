import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import sglm
import sglm_pp
import sglm_cv

# def setup_autoregression(X, response_cols, order):
#     col_nums = sglm_pp.get_column_nums(X, response_cols)
#     return sglm_pp.timeshift_multiple(X, shift_inx=col_nums, shift_amt_list=list(range(order + 1)))

def timeshift_cols(X, cols_to_shift, neg_order=0, pos_order=1):
    """
    Shift the columns of X forward by all timesteups up to pos_order and backward by all timesteps down to neg_roder

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
    return sglm_pp.timeshift_multiple(X, shift_inx=col_nums, shift_amt_list=list(range(neg_order, pos_order + 1)))

def fit_GLM(X, y, model_name='Gaussian', *args, **kwargs):
    """
    Fit GLM on training dataset of predictor columns of X and response y

    Parameters
    ----------
    X : pd.DataFrame
        Predictor DataFrame from which to predict the response
    y : pd.Series
        Response to be predicted
    model_name : str
        Type of GLM to build (e.g. Gaussian, Poisson, Logistic, etc.)
    args : iterable
        Positional arguments to be passed to GLM model
    kwargs : dict
        Keyword arguments to be passed to GLM model
    """
    glm = sglm.GLM(model_name, *args, **kwargs)
    glm.fit(X.values, y.values)
    return glm

def diff_cols(X, cols, append_to_base=True):
    """
    Take differentials along columns col of DataFrame X

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame of data to take the differential along column
    cols : list
        Names of the columns along which to take the differential
    append_to_base : bool
        Whether or not those columns should be returned as columns added to the original DataFrame
    """
    col_nums = sglm_pp.get_column_nums(X, cols)
    X = sglm_pp.diff(X, col_nums, append_to_base=append_to_base)
    return X

def cv_idx_by_timeframe(X, y=None, timesteps_per_bucket=20, k_folds=10):
    """
    Generate Cross Validation indices by keeping together bucketed timesteps
    (bucketing together by sets of timesteps_per_bucket).

    Parameters
    ----------
    X : pd.DataFrame
        Prediction DataFrame from which to bucket
    y : pd.Series
        Response Series
    timesteps_per_bucket : int
        Number of timesteps (i.e. rows in the DataFrame) that should be kept together as buckets
    k_folds : int
        Number of Cross Validation segmentations that should be used for k-fold Cross Validation
    """
    bucket_ids = sglm_pp.bucket_ids_by_timeframe(X.shape[0], timesteps_per_bucket=20)
    cv_idx = sglm_pp.cv_idx_from_bucket_ids(bucket_ids, X, y=y, k_folds=k_folds)
    return cv_idx


def cv_idx_by_trial_id(X, y=None, trial_id_columns=[], k_folds=10):
    """
    Generate Cross Validation indices by keeping together trial id columns
    (bucketing together by trial_id_columns).

    Parameters
    ----------
    X : pd.DataFrame
        Prediction DataFrame from which to bucket
    y : list
        Response Series
    trial_id_columns : int
        Columns to use to identify bucketing identifiers
    k_folds : int
        Number of Cross Validation segmentations that should be used for k-fold Cross Validation
    """
    X = pd.DataFrame(X)

    for i, idc in enumerate(trial_id_columns):
        if i == 0:
            bucket_ids = X[idc].astype(str).str.len().astype(str) + ':' + X[idc].astype(str)
        else:
            bucket_ids = bucket_ids + '_' + X[idc].astype(str)
    
    bucket_ids = bucket_ids.astype("category").cat.codes
    
    cv_idx = sglm_pp.cv_idx_from_bucket_ids(bucket_ids, X, y=y, k_folds=k_folds)
    return cv_idx

# Trial-based splitting (remove inter-trial information)

def simple_cv_fit(X, y, cv_idx, glm_kwarg_lst, model_type='Normal'):
    """
    Fit the desired model using the list of keyword arguments provided in
    glm_kwarg_lst, identify the best model, and return the associated
    score, parameters, and the model itself.

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
                                            glm_kwarg_lst)
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

    X_tmp = diff_cols(X_tmp, ['B_1', 'B'])
    print()
    print(X_tmp)

    X_tmp = X_tmp.dropna()
    print()
    print(X_tmp)
    
    glm = fit_GLM(X_tmp[['A', 'B_1', 'B_2', 'B_3', 'B_4', 'A_1', 'A_2', 'B_1_diff']], X_tmp['B'], reg_lambda=0.1)
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
