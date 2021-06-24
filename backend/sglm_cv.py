import numpy as np
import sglm
import itertools

def cv_glm_single_params(X, y, cv_idx, model_name, glm_kwargs):
    """
    Runs cross-validation on GLM for a single set of parameters.

    Parameters
    ----------
    X : np.ndarray
        The full set of available predictor data.
    y : np.ndarray
        The full set of corresponding available response data.
    cv_idx : list of pairs of lists
        Cross-validation indices. Each entry in the outer list is
        for a different run. Each entry in the outer list should 
        contain 2 lists, the first one containing the training 
        indices, and the second one containing the test indices.
    model_name : str
        The type of GLM to construct ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
    glm_kwargs : dict
        Keyword arguments to pass to the GLM constructor
    """

    n_coefs = X.shape[1]
    n_idx = len(cv_idx)

    cv_coefs = np.zeros((n_coefs, n_idx))
    cv_intercepts = np.zeros(n_idx)
    cv_scores_train = np.zeros(n_idx)
    cv_scores_test = np.zeros(n_idx)

    for iter_cv, (idx_train, idx_test) in enumerate(cv_idx):
        X_train = X[idx_train,:]
        y_train = y[idx_train]
        X_test = X[idx_test,:]
        y_test = y[idx_test]

        glm = sglm.GLM(model_name, **glm_kwargs)
        glm.fit(X_train, y_train)

        cv_coefs[:, iter_cv] = glm.model.coef_
        cv_intercepts[iter_cv] = glm.model.intercept_
        cv_scores_train[iter_cv] = glm.model.score(X_train, y_train)
        cv_scores_test[iter_cv] = glm.model.score(X_test, y_test)

    return cv_coefs, cv_intercepts, cv_scores_train, cv_scores_test



def cv_glm_mult_params(X, y, cv_idx, glm_kwarg_lst):
    resp = list()
    for kwarg in glm_kwarg_lst:
        
        model_name = kwarg['model_name'] if 'model_name' in kwarg else 'Gaussian'
        roll = kwarg['roll'] if 'roll' in kwarg else 0
        y_rolled = np.roll(y.reshape(-1), roll)

        resp.append(cv_glm_single_params(X, y_rolled, cv_idx, model_name, glm_kwarg_lst))
    
    return resp


def generate_mult_params(kwargs, kwarg_lists):
    fliped_dict_list = [kwargs] + [[{key:_} for _ in kwarg_lists[key]] for key in kwarg_lists]
    cart_prod = list(itertools.product(*fliped_dict_list))
    return [{_key:dct[_key] for dct in cart_prod[i] for _key in dct} for i in range(len(cart_prod))]

