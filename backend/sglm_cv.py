import pandas as pd
import numpy as np
import sglm_
import itertools
import threading, queue
from multiprocessing import Process, Pool
import time

# TODO: Multidimensional Array -- paramgrid and output grid
# TODO: Add an OrderedDict implementation for the generate_mult_params version

# TODO: Add a Feature Selection methodology -- to adjust feature selection based on cross-validation


class SGLM_worker():
    def __init__(self, queue, verbose=0):
        if verbose > 1:
            print('Worker created')
        self.queue = queue
        self.verbose = verbose

    def run_single(self):
        while True:
            if self.verbose > 1:
                print('Running single')
            glm, args, kwargs = self.queue.get()
            glm.fit_set(*args, **kwargs)
            self.queue.task_done()
            if self.queue.empty():
                return
    
    def run_multi(self):
        while True:
            if self.verbose > 1:
                print('Running multi')
            args, kwargs = self.queue.get()
            cv_glm_single_params(*args, **kwargs)
            self.queue.task_done()
            if self.queue.empty():
                return

def cv_glm_single_params(X, y, cv_idx, model_name, glm_kwargs, verbose=0, resp_list=[], beta_=None, beta0_=None, score_method='mse'):
    """
    Runs cross-validation on GLM for a single set of parameters.

    JZ 2021
    
    Args:
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
        verbose : int
            How much information to print during the fititng process
        resp_list : list
            list for in place appending of fitting results (for use in multi-threading)
        beta_ : np.ndarray
            Pre-initialized coefficient values for warm-start-based fitting
        beta0_ : np.ndarray
            Pre-initialized intercept value for warm-start-based fitting
        score_method : str
            'mse' for Mean Squared Error-based scoring, 'r2' for R^2 based scoring
    
    Returns: dict of information relevant to fitted validation model based on single set of GLM parameters
    """

    
    q = queue.Queue()
    # def sglm_worker():
    #     while True:
    #         # try:
    #         glm, args, kwargs = q.get()
    #         # print(f'Working on {kwargs}')
    #         glm.fit_set(*args, **kwargs)
    #         q.task_done()
    #         # print(f'Finished {kwargs}')
    #         # except queue.Empty:
    #         #     break


    threads = []

    n_coefs = X.shape[1]
    n_idx = len(cv_idx)

    roll = glm_kwargs.pop('roll', 0)
    y_rolled = np.roll(y.reshape(-1), roll)

    cv_coefs = np.zeros((n_coefs, n_idx))
    cv_intercepts = np.zeros(n_idx)
    cv_scores_train = np.zeros(n_idx)
    cv_scores_test = np.zeros(n_idx)

    resids = []
    mean_resids = []

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
    
        glm = sglm_.GLM(model_name, beta0_=beta0_, beta_=beta_, **glm_kwargs, score_method=score_method)
        # glm = sglm.GLM(model_name, **glm_kwargs)
        args = (X_train, y_train, X_test, y_test,
                cv_coefs, cv_intercepts, cv_scores_train, cv_scores_test,
                iter_cv,)
        kwargs = {'resids': resids, 'mean_resids': mean_resids,
                  'id_fit': iter_cv, 'verbose': verbose
                 }

        q.put((glm, args, kwargs))
        # threads.append(threading.Thread(target=glm.fit_set, args=args, kwargs=kwargs))

        # threads.append(Process(target=glm.fit_set, args=(X_train, y_train, X_test, y_test,
        #                                                           cv_coefs, cv_intercepts, cv_scores_train, cv_scores_test,
        #                                                           iter_cv,), kwargs={'resids': resids, 'mean_resids': mean_resids,
        #                                                                              'id_fit': iter_cv, 'verbose': verbose
        #                                                                              }))

        # threads[-1].name = str(glm_kwargs) + f' - {iter_cv}'
        # threads[-1].start()

        # # if iter_cv % 5 == 4:
        # #     for thread in threads:
        # #         thread.join()
        # #     threads = []


        
        # # cv_coefs[:, iter_cv] = glm.coef_
        # # cv_intercepts[iter_cv] = glm.intercept_
        # # cv_scores_train[iter_cv] = glm.score(X_train, y_train)
        # # cv_scores_test[iter_cv] = glm.score(X_test, y_test)


    # threading.Thread(target=sglm_worker, daemon=True).start()
    # threading.Thread(target=sglm_worker, daemon=True).start()
    # threading.Thread(target=sglm_worker, daemon=True).start()
    # threading.Thread(target=sglm_worker, daemon=True).start()
    # # threading.Thread(target=sglm_worker, daemon=True).start()

    num_workers = 4
    workers = [SGLM_worker(q) for _ in range(num_workers)]
    threads = [threading.Thread(target=worker.run_single, daemon=True) for worker in workers]
    for thread in threads:
        thread.start()
    
    q.join()
    for thread in threads:
        thread.join()


    # for thread in threads:
    #     thread.join()
    # threads = []


    #####################

    glm = sglm_.GLM(model_name, beta0_=beta0_, beta_=beta_, **glm_kwargs)
    glm.fit(X, y)

    #####################

    if verbose > 0:
        print('Completing arguments:', glm_kwargs)

    ret_dict = {
        'cv_coefs': cv_coefs,
        'cv_intercepts': cv_intercepts,
        'cv_scores_train': cv_scores_train,
        'cv_scores_test': cv_scores_test,
        'cv_mean_score_train': np.mean(cv_scores_train),
        'cv_mean_score': np.mean(cv_scores_test),
        'cv_std_score': np.std(cv_scores_test),
        'cv_R2_score': sglm_.calc_R2(np.concatenate(resids), np.concatenate(mean_resids)),
        'cv_mse_score': np.mean(np.square(np.concatenate(resids))),
        'glm_kwargs': glm_kwargs,
        'model': glm
    }

    print(f"{glm_kwargs}\n> cv_mean_score_train: {ret_dict['cv_mean_score_train']}\n> cv_R2_score: {ret_dict['cv_R2_score']}\n> cv_mean_score: {ret_dict['cv_mean_score']}")

    resp_list.append(ret_dict)

    return ret_dict



def cv_glm_mult_params(X, y, cv_idx, model_name, glm_kwarg_lst, verbose=0, score_method='mse'):
    """
    Runs cross-validation on GLM over a list of possible parameters.

    JZ 2021
    
    Args:
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
        verbose : int
            How much information to print during the fititng process
        score_method : str
            'mse' for Mean Squared Error-based scoring, 'r2' for R^2 based scoring
    
    Returns: dict of information relevant to the best model identified and overall fit information
    """
    




    q2 = queue.Queue()

    
    # def sglm_worker():
    #     while True:
    #         # try:
    #         args, kwargs = q2.get()
    #         # print(f'Working on {kwargs}')
    #         # glm.fit_set(*args, **kwargs)
    #         cv_glm_single_params(*args, **kwargs)
    #         q2.task_done()
    #         # print(f'Finished {kwargs}')
    #         # except queue.Empty:
    #         #     break



    # score_metric = glm_kwarg_lst[0]['score_metric'] if 'score_metric' in glm_kwarg_lst else 'pseudo_R2'

    final_results = {}
    best_score = -np.inf #if score_metric == 'pseudo_R2' else np.inf
    best_params = None
    # best_model = None

    threads = []
    resp = list()


    start = time.time()

    pca_glm = sglm_.GLM('PCA Normal') if model_name in {'Normal', 'Gaussian'} else sglm_.GLM(model_name)
    pca_glm.pca_fit(X, y)
    print(f'> PCA GLM Built in {time.time() - start} seconds')

    # beta0_ = pca_glm.beta0_
    # beta_ = pca_glm.beta_
    beta0_ = None
    beta_ = None


    for i, glm_kwargs in enumerate(glm_kwarg_lst):
        print(glm_kwargs)

        model_name = glm_kwargs.pop('model_name', 'Gaussian')

        args = (X, y, cv_idx, model_name, glm_kwargs,)
        kwargs = {'verbose': verbose,
                  'resp_list': resp,
                  'beta0_':beta0_,
                  'beta_':beta_,
                  'score_method':score_method
                 }

        # threads.append(threading.Thread(target=cv_glm_single_params, args=args,
        #                                                              kwargs=kwargs))
        # threads.append(Process(target=cv_glm_single_params, args=(X, y, cv_idx, model_name, glm_kwargs,),
        #                                                              kwargs={'verbose': verbose,
        #                                                                      'resp_list': resp,
        #                                                                      'beta0_':beta0_,
        #                                                                      'beta_':beta_,
        #                                                                      'score_method':score_method
        #                                                                      }))
        
        q2.put((args, kwargs))
        
        # threads[-1].name = str(glm_kwargs)
        # threads[-1].start()
        
    #     if i % 4 == 3:
    #         for thread in threads:
    #             thread.join()
    #         threads = []


    # for thread in threads:
    #     thread.join()
    # threads = []



    # threading.Thread(target=sglm_worker, daemon=True).start()
    # # threading.Thread(target=sglm_worker, daemon=True).start()
    # # threading.Thread(target=sglm_worker, daemon=True).start()
    # # threading.Thread(target=sglm_worker, daemon=True).start()

    num_workers = 1
    workers = [SGLM_worker(q2) for _ in range(num_workers)]
    threads = [threading.Thread(target=worker.run_multi, daemon=True) for worker in workers]
    for thread in threads:
        thread.start()
    q2.join()
    for thread in threads:
        thread.join()




    # # score_metric = glm_kwarg_lst[0]['score_metric'] if 'score_metric' in glm_kwarg_lst else 'pseudo_R2'

    # final_results = {}
    # best_score = -np.inf #if score_metric == 'pseudo_R2' else np.inf
    # best_params = None
    # # best_model = None

    # threads = []
    # resp = list()


    # start = time.time()

    # pca_glm = sglm.GLM('PCA Normal') if model_name in {'Normal', 'Gaussian'} else sglm.GLM(model_name)
    # pca_glm.pca_fit(X, y)
    # print(f'> PCA GLM Built in {time.time() - start} seconds')

    # # beta0_ = pca_glm.beta0_
    # # beta_ = pca_glm.beta_
    # beta0_ = None
    # beta_ = None


    # for i, glm_kwargs in enumerate(glm_kwarg_lst):
    #     print(glm_kwargs)

    #     model_name = glm_kwargs.pop('model_name', 'Gaussian')

    #     threads.append(threading.Thread(target=cv_glm_single_params, args=(X, y, cv_idx, model_name, glm_kwargs,),
    #                                                                  kwargs={'verbose': verbose,
    #                                                                          'resp_list': resp,
    #                                                                          'beta0_':beta0_,
    #                                                                          'beta_':beta_,
    #                                                                          'score_method':score_method
    #                                                                          }))
    #     # threads.append(Process(target=cv_glm_single_params, args=(X, y, cv_idx, model_name, glm_kwargs,),
    #     #                                                              kwargs={'verbose': verbose,
    #     #                                                                      'resp_list': resp,
    #     #                                                                      'beta0_':beta0_,
    #     #                                                                      'beta_':beta_,
    #     #                                                                      'score_method':score_method
    #     #                                                                      }))
        
    #     threads[-1].name = str(glm_kwargs)
    #     threads[-1].start()
        
    #     if i % 1 == 0:
    #         for thread in threads:
    #             thread.join()
    #         threads = []


    # for thread in threads:
    #     thread.join()
    # threads = []

    if len(resp) == 0:
        print('len resp', len(resp))
        print('resp', resp)
    
    for cv_result in resp:
        # if ((cv_result['model'].score_metric == 'pseudo_R2' and cv_result['cv_mean_score'] > best_score)): # or
            # (cv_result['model'].score_metric == 'deviance' and cv_result['cv_mean_score'] < best_score)):
        
        if (score_method == 'r2' and cv_result['cv_R2_score'] > best_score):
            best_score = cv_result['cv_R2_score']
            best_score_std = cv_result['cv_std_score']
            best_params = cv_result['glm_kwargs']
            best_model = cv_result['model']
        elif (score_method == 'mse' and cv_result['cv_mean_score'] > best_score):
            best_score = cv_result['cv_mean_score']
            best_score_std = cv_result['cv_std_score']
            best_params = cv_result['glm_kwargs']
            best_model = cv_result['model']
            
    
    # print(score_method, cv_result['cv_R2_score'], best_score)

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
    
    Args:
        kwarg_lists : dict(list(keywords))
            Dictionary where each key is associated with a list of possible parameters to consider.
        kwargs : Optional[dict(keywords)]
            Dictionary of fixed keyword arguments that should remain the same across all CV trials.
    
    Returns: list of dicts of keyword arguments for GLM model fitting
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











    # q2 = queue.Queue()
    # def sglm_worker():
    #     while True:
    #         # try:
    #         args, kwargs = q2.get()
    #         # print(f'Working on {kwargs}')
    #         # glm.fit_set(*args, **kwargs)
    #         cv_glm_single_params(*args, **kwargs)
    #         q2.task_done()
    #         # print(f'Finished {kwargs}')
    #         # except queue.Empty:
    #         #     break



    # # score_metric = glm_kwarg_lst[0]['score_metric'] if 'score_metric' in glm_kwarg_lst else 'pseudo_R2'

    # final_results = {}
    # best_score = -np.inf #if score_metric == 'pseudo_R2' else np.inf
    # best_params = None
    # # best_model = None

    # threads = []
    # resp = list()


    # start = time.time()

    # pca_glm = sglm.GLM('PCA Normal') if model_name in {'Normal', 'Gaussian'} else sglm.GLM(model_name)
    # pca_glm.pca_fit(X, y)
    # print(f'> PCA GLM Built in {time.time() - start} seconds')

    # # beta0_ = pca_glm.beta0_
    # # beta_ = pca_glm.beta_
    # beta0_ = None
    # beta_ = None


    # for i, glm_kwargs in enumerate(glm_kwarg_lst):
    #     print(glm_kwargs)

    #     model_name = glm_kwargs.pop('model_name', 'Gaussian')

    #     args = (X, y, cv_idx, model_name, glm_kwargs,)
    #     kwargs = {'verbose': verbose,
    #               'resp_list': resp,
    #               'beta0_':beta0_,
    #               'beta_':beta_,
    #               'score_method':score_method
    #              }

    #     # threads.append(threading.Thread(target=cv_glm_single_params, args=args,
    #     #                                                              kwargs=kwargs))
    #     # threads.append(Process(target=cv_glm_single_params, args=(X, y, cv_idx, model_name, glm_kwargs,),
    #     #                                                              kwargs={'verbose': verbose,
    #     #                                                                      'resp_list': resp,
    #     #                                                                      'beta0_':beta0_,
    #     #                                                                      'beta_':beta_,
    #     #                                                                      'score_method':score_method
    #     #                                                                      }))
        
    #     q2.put((args, kwargs))
        
    #     # threads[-1].name = str(glm_kwargs)
    #     # threads[-1].start()
        
    # #     if i % 4 == 3:
    # #         for thread in threads:
    # #             thread.join()
    # #         threads = []


    # # for thread in threads:
    # #     thread.join()
    # # threads = []



    # threading.Thread(target=sglm_worker, daemon=True).start()
    # # threading.Thread(target=sglm_worker, daemon=True).start()
    # # threading.Thread(target=sglm_worker, daemon=True).start()
    # # threading.Thread(target=sglm_worker, daemon=True).start()

    # q2.join()

