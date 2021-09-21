import pandas as pd
import numpy as np
import sglm
import itertools
import threading
import time

# TODO: Multidimensional Array -- paramgrid and output grid
# TODO: Add an OrderedDict implementation for the generate_mult_params version

# TODO: Add a Feature Selection methodology -- to adjust feature selection based on cross-validation 

def cv_glm_single_params(X, y, cv_idx, model_name, glm_kwargs, GLM_CLS=None, verbose=0, resp_list=[], beta_=None, beta0_=None):
    """
    Runs cross-validation on GLM for a single set of parameters.

    JZ 2021
    
    Parameters
    ----------
    X : np.ndarray
        The full set of available predictor data (columns should be features, rows should be timesteps).
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

    threads = []

    n_coefs = X.shape[1]
    n_idx = len(cv_idx)

    roll = glm_kwargs.pop('roll', 0)
    y_rolled = np.roll(y.reshape(-1), roll)

    cv_coefs = np.zeros((n_coefs, n_idx))
    cv_intercepts = np.zeros(n_idx)
    cv_scores_train = np.zeros(n_idx)
    cv_scores_test = np.zeros(n_idx)

    for iter_cv, (idx_train, idx_test) in enumerate(cv_idx):
        X_train = X[idx_train,:]
        y_train = y_rolled[idx_train]
        X_test = X[idx_test,:]
        y_test = y_rolled[idx_test]

        # if iter_cv == 0 and not isinstance(beta_, np.ndarray) and beta0_ is None:
        #     start = time.time()

        #     pca_glm = sglm.GLM(model_name, **glm_kwargs)
        #     pca_glm.pca_fit(X_train, y_train)
        #     print(f'> PCA GLM Built in {time.time() - start} seconds')

        #     beta0_ = pca_glm.beta0_
        #     beta_ = pca_glm.beta_.copy()

        if GLM_CLS:
            glm = GLM_CLS(model_name, beta0_=beta0_, beta_=beta_, **glm_kwargs)
            # glm = GLM_CLS(model_name, **glm_kwargs)
        else:
            glm = sglm.GLM(model_name, beta0_=beta0_, beta_=beta_, **glm_kwargs)
            # glm = sglm.GLM(model_name, **glm_kwargs)


        threads.append(threading.Thread(target=glm.fit_set, args=(X_train, y_train, X_test, y_test,
                                                                  cv_coefs, cv_intercepts, cv_scores_train, cv_scores_test,
                                                                  iter_cv,), kwargs={'id_fit': iter_cv, 'verbose': verbose}))
        threads[-1].start()

        
        # start = time.time()
        # glm.fit(X_train, y_train)
        # print(f'GLM fit in {time.time() - start} seconds')

        
        # cv_coefs[:, iter_cv] = glm.coef_
        # cv_intercepts[iter_cv] = glm.intercept_
        # cv_scores_train[iter_cv] = glm.score(X_train, y_train)
        # cv_scores_test[iter_cv] = glm.score(X_test, y_test)

    for thread in threads:
        thread.join()

    if verbose > 0:
        print('Completing arguments:', glm_kwargs)

    ret_dict = {
        'cv_coefs': cv_coefs,
        'cv_intercepts': cv_intercepts,
        'cv_scores_train': cv_scores_train,
        'cv_scores_test': cv_scores_test,
        'cv_mean_score': np.mean(cv_scores_test),
        'cv_std_score': np.std(cv_scores_test),
        'glm_kwargs': glm_kwargs,
        'model': glm
    }

    resp_list.append(ret_dict)

    return ret_dict



def cv_glm_mult_params(X, y, cv_idx, model_name, glm_kwarg_lst, GLM_CLS=None, verbose=0):
    """
    Runs cross-validation on GLM over a list of possible parameters.

    JZ 2021
    
    Parameters
    ----------
    X : np.ndarray
        The full set of available predictor data (columns should be features, rows should be timesteps).
    y : np.ndarray
        The full set of corresponding available response data (1D).
    cv_idx : list of pairs of lists
        Cross-validation indices. Each entry in the outer list is
        for a different run. Each entry in the outer list should 
        contain 2 lists, the first one containing the training 
        indices, and the second one containing the test indices.
    model_name : str
        The type of GLM to construct ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
    glm_kwarg_lst : list(dict(GLM parameters))
        A list of all kwarg parameters for the GLM through which
        the crossvalidation function should iterate. ('roll' should be
        specified here if desired where 'roll' corresponds to the amount
        of 1D index shifts that should be applied)
    """

    # score_metric = glm_kwarg_lst[0]['score_metric'] if 'score_metric' in glm_kwarg_lst else 'pseudo_R2'

    final_results = {}
    best_score = -np.inf #if score_metric == 'pseudo_R2' else np.inf
    best_params = None
    # best_model = None

    threads = []
    resp = list()


    start = time.time()

    pca_glm = sglm.GLM(model_name)
    pca_glm.pca_fit(X, y)
    print(f'> PCA GLM Built in {time.time() - start} seconds')

    beta0_ = pca_glm.beta0_
    beta_ = pca_glm.beta_.copy()



    for i, glm_kwargs in enumerate(glm_kwarg_lst):
        print(glm_kwargs)

        model_name = glm_kwargs.pop('model_name', 'Gaussian')
        # cv_result = cv_glm_single_params(X, y, cv_idx, model_name, glm_kwargs, GLM_CLS=GLM_CLS, verbose=verbose, resp_list=resp)
        # resp.append(cv_result)

        threads.append(threading.Thread(target=cv_glm_single_params, args=(X, y, cv_idx, model_name, glm_kwargs,),
                                                                     kwargs={'GLM_CLS': GLM_CLS,
                                                                             'verbose': verbose,
                                                                             'resp_list': resp,
                                                                             'beta0_':beta0_,
                                                                             'beta_':beta_
                                                                             }))
        threads[-1].name = str(glm_kwargs)
        threads[-1].start()

        if i % 5 == 4:
            for thread in threads:
                thread.join()
            threads = []


    for thread in threads:
        thread.join()

    for cv_result in resp:
        # if ((cv_result['model'].score_metric == 'pseudo_R2' and cv_result['cv_mean_score'] > best_score)): # or
            # (cv_result['model'].score_metric == 'deviance' and cv_result['cv_mean_score'] < best_score)):
        if (cv_result['cv_mean_score'] > best_score): # or
            best_score = cv_result['cv_mean_score']
            best_score_std = cv_result['cv_std_score']
            best_params = cv_result['glm_kwargs']
            best_model = cv_result['model']
    

    final_results = {
        'best_score': best_score,
        'best_score_std': best_score_std,
        'best_params': best_params,
        'best_model': best_model,
        'full_cv_results': resp,
    }

    return final_results


    
    # out_dict = {}
    # threads = []
    
    # resp = list()
    # for i, glm_kwargs in enumerate(glm_kwarg_lst):

    #     model_name = glm_kwargs.pop('model_name', 'Gaussian')

    #     threads.append(threading.Thread(target=cv_glm_single_params, args=(X, y, cv_idx, model_name, glm_kwargs),
    #                                                                  kwargs={'GLM_CLS': GLM_CLS,
    #                                                                          'verbose': verbose,
    #                                                                          'dct': out_dict
    #                                                                          }))

    #     # cv_result = cv_glm_single_params(X, y, cv_idx, model_name, glm_kwargs, GLM_CLS=GLM_CLS, verbose=verbose)
    #     # resp.append(cv_result)
        
    #     threads[i].start()

    
    # for thread in threads:
    #     thread.join()
    
    # shifted_list = [out_dict[tuple(_)] for _ in glm_kwarg_lst]

    

    # for cv_result in shifted_list:
    #     if ((cv_result['model'].score_metric == 'pseudo_R2' and cv_result['cv_mean_score'] > best_score)): # or
    #         # (cv_result['model'].score_metric == 'deviance' and cv_result['cv_mean_score'] < best_score)):
    #         best_score = cv_result['cv_mean_score']
    #         best_params = cv_result['params']
    #         best_model = cv_result['model']
    
    # final_results = {
    #     'best_score': best_score,
    #     'best_params': best_params,
    #     'best_model': best_model,
    #     'full_cv_results': resp,
    # }

    # return final_results


def generate_mult_params(kwarg_lists, kwargs=None):
    """
    Generates a list of dictionaries of all possible parameter combinations
    from a dictionary of lists.

    JZ 2021
    
    Parameters
    ----------
    kwarg_lists : dict(list(keywords))
        Dictionary where each key is associated with a list of possible parameters to consider.
    kwargs : Optional[dict(keywords)]
        Dictionary of fixed keyword arguments that should remain the same across all CV trials.
    """

    base_list = [[kwargs]] if kwargs else []
    flipped_dict_list = base_list + [[{key:_} for _ in kwarg_lists[key]] for key in kwarg_lists]
    cart_prod = list(itertools.product(*flipped_dict_list))

    return [{_key:dct[_key] for dct in cart_prod[i] for _key in dct} for i in range(len(cart_prod))]


# def execute_backward_selection(X, y, *args, criterion='AIC', **kwargs):

#     """
#     Generates a list of dictionaries of all possible parameter combinations
#     from a dictionary of lists.

#     Parameters
#     ----------
#     X : np.ndarray
#         The full set of available predictor data (columns should be features, rows should be timesteps).
#     y : np.ndarray
#         The full set of corresponding available response data (1D).
#     GLM : sglm.GLM or sglm.SKGLM class
#         GLM model to fit
#     args : Optional[dict(keywords)]
#         Positional arguments to GLM that should remain the same across all CV trials.
#     criterion : str — 'AIC' or 'BIC'
#         The criterion to use for backward selection (note that these are pseudo-AIC / BIC for now since likelihood varies by link function)
#     kwargs : Optional[dict(keywords)]
#         Dictionary of fixed keyword arguments to GLM that should remain the same across all CV trials.
#     """

#     def calc_criterion(X, y, model, criterion):
#         score = model.score(X, y)
#         neg_score = 1 - score
#         params = 

