import numpy as np
import sglm

def cv_glm_single_params(X, y, cv_idx, model_name, glm_kwargs, fit_kwargs):
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
        f
    fit_kwargs : dict
        f
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
        glm.fit(X_train, y_train, **fit_kwargs)

        cv_coefs[:, iter_cv] = glm.model.coef_
        cv_intercepts[iter_cv] = glm.model.intercept_
        cv_scores_train[iter_cv] = glm.model.score(X_train, y_train)
        cv_scores_test[iter_cv] = glm.model.score(X_test, y_test)

    return cv_coefs, cv_intercepts, cv_scores_train, cv_scores_test



def cv_glm_mult_params(X, y, cv_idx, glm_kwargs, fit_kwargs, cv_kwargs, rolls=[0]):
    
    for kwarg in cv_kwargs:
        # if key[:4] == 'glm_':
        #     key = key[4:]
        #     glm_params[key] = cv_params[key]
        # elif key[:4] == 'fit_':
        #     key = key[4:]
        #     fit_params[key] = cv_params[key]
        # else:
        #     ext_params[key] = cv_params[key]
        pass

    return





