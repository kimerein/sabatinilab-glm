import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import sglm
import sglm_pp
import sglm_cv

def setup_autoregression(X, response_cols, order):
    col_nums = sglm_pp.get_column_nums(X, response_cols)
    return sglm_pp.timeshift_multiple(X, shift_inx=col_nums, shift_amt_list=list(range(order + 1)))

def timeshift_cols(X, cols_to_shift, neg_order=0, pos_order=1):
    col_nums = sglm_pp.get_column_nums(X, cols_to_shift)
    return sglm_pp.timeshift_multiple(X, shift_inx=col_nums, shift_amt_list=list(range(neg_order, pos_order + 1)))

def fit_GLM(X, y, model_name='Gaussian', *args, **kwargs):
    glm = sglm.GLM(model_name, *args, **kwargs)
    glm.fit(X.values, y.values)
    return glm

def diff_cols(X, cols, append_to_base=True):
    col_nums = sglm_pp.get_column_nums(X, cols)
    X = sglm_pp.diff(X, col_nums, append_to_base=append_to_base)
    return X

def cv_idx_by_timeframe(X, y=None, k_folds=None, timesteps_per_bucket=20):
    bucket_ids = sglm_pp.bucket_ids_by_timeframe(X.shape[0], timesteps_per_bucket=20)
    cv_idx = sglm_pp.cv_idx_from_bucket_ids(bucket_ids, X, y=y, k_folds=k_folds)
    return cv_idx

def simple_cv_fit(X, y, cv_idx, glm_kwarg_lst, model_type='Normal'):
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
    X_tmp = setup_autoregression(X_tmp, ['B'], 4)
    print()
    print(X_tmp)

    X_tmp = timeshift_cols(X_tmp, ['A'], 2)
    print()
    print(X_tmp)

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
