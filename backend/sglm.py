from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, TweedieRegressor, LogisticRegression, PoissonRegressor
import sklearn.metrics
import pyglmnet
import scipy.stats
import numpy as np
import pandas as pd
import time

from sklearn.decomposition import PCA

from numba import njit, jit, prange

# TODO: Potentially add additional alternatives for different GLM API implementations (SKLearn, etc.)
# TODO: Potentially add switching it to also allowing pandas DataFrames as the fitting function

# import seaborn as sns
# sns.set(style='white', palette='colorblind', context='poster')



# Try batch gradient descent with alpha = 0.5 & with orthogonal matrix

# class GLM(pyglmnet.GLM):
#     """
#     Generalized Linear Model class built on pyglmnet's underlying regression models.

#     Attributes
#     ----------
#     model_name : str
#         GLM distribution name to create ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
#     model : sklearn.linear_model.TweedieRegressor or sklearn.linear_model.LogisticRegression
#         Underlying pyglmnet model that is built
#     coef_ : 
#         Coefficients (parameters) of the GLM predictors
#     intercept_ : 
#         GLM linear intercept (i.e. bias coefficient)

#     Methods
#     -------
#     fit(X, y, *args, **kwargs):
#         Fits the GLM to the provided X predictors and y responses.
#         (Calls the underlying sklearn fit method for the associated model.)
#     """

#     model = None
#     model_name_options = {'Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', 'Binomial', 'Multinomial'}
#     tweedie_lookup = {'Normal': 0, 'Gaussian':0, 'Poisson': 1, 'Gamma': 2}

#     def __init__(self, model_name, beta0_=None, beta_=None, *args, **kwargs):
#         """
#         Create the GLM model.

#         model_name : str
#             GLM distribution name to create ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
#         *args : positional arguments
#             See https://glm-tools.github.io/pyglmnet/api.html for relevant arguments.
#         **kwargs : keyword arguments
#             Note: Implementation overrides default solver to be cdfast (i.e. Newton Conjugate Gradient) instead of batch
#             gradient descent
#             See https://glm-tools.github.io/pyglmnet/api.html for relevant arguments.
#         """

#         base_kwargs = {
#             'solver':'cdfast',
#             'score_metric':'pseudo_R2'
#         }
#         base_kwargs.update(kwargs)

#         self.model_name = model_name
#         if model_name in self.model_name_options: #{'Normal', 'Gaussian', 'Poisson', 'Gamma'}:
#             model_name = 'Gaussian' if model_name in {'Normal'} else model_name
#             model_name = 'Binomial' if model_name in {'Logistic', 'Multinomial'} else model_name
#             base_kwargs['distr'] = model_name.lower()
#             # mdl = pyglmnet.GLM(*args, **base_kwargs)
#             super().__init__(*args, **base_kwargs)

#             self.beta0_ = beta0_ if beta0_ is not None else self.beta0_
#             self.beta_ = beta_ if isinstance(beta_, np.ndarray) else self.beta_
#         else:
#             print('Distribution not yet implemented.')
#             raise NotYetImplementedError()
        
#         # self.model = mdl
    
#     def pca_fit(self, X, y):
#         self.alpha = 0
#         self.reg_lambda = 0

#         start = time.time()
#         self.pca = PCA().fit(X)
#         print(f'PCA fit in {time.time() - start} seconds')
#         X_trans = self.pca.transform(X)[:, :(X.shape[1]-10)]

#         start = time.time()
#         self.fit(X_trans, y)
#         print(f'PCA-based Model fit in {time.time() - start} seconds')


#         W = self.pca.components_.T[:, :(X.shape[1]-10)]
#         B = self.beta_.reshape([-1, 1])
#         B0 = np.array(self.beta0_).reshape(-1)
#         X_bar = self.pca.mean_

#         self.beta_ = (W @ B).reshape(-1)
#         self.beta0_ = (B0 - X_bar @ W @ B).reshape(-1)

#     def fit(self, X, y, *args):
#         """
#         Fits the GLM to the provided X predictors and y responses.

#         Parameters
#         ----------
#         X : np.ndarray or pd.DataFrame
#             Array of predictor variables on which to fit the model
#         y : np.ndarray or pd.Series
#             Array of response variables on which to fit the model
#         """
#         super().fit(X, y, *args)
#         self.coef_ = self.beta_
#         self.intercept_ = self.beta0_
    
    
#     def fit_set(self, X, y, X_test, y_test, cv_coefs, cv_intercepts, cv_scores_train, cv_scores_test, iter_cv, *args):
#         """
#         Fits the GLM to the provided X predictors and y responses.

#         Parameters
#         ----------
#         X : np.ndarray or pd.DataFrame
#             Array of predictor variables on which to fit the model
#         y : np.ndarray or pd.Series
#             Array of response variables on which to fit the model
#         """
#         self.fit(X, y, *args)

#         cv_coefs[:, iter_cv] = self.coef_
#         cv_intercepts[iter_cv] = self.intercept_
#         cv_scores_train[iter_cv] = self.score(X, y)
#         cv_scores_test[iter_cv] = self.score(X_test, y_test)


#     def predict(self, X):
#         if type(X) == pd.DataFrame:
#             X = X.values
#         return super().predict(X)

#     def log_likelihood(self, prediction, truth):

#         if self.model_name in {'Normal', 'Gaussian'}:
#             resid = truth - prediction
#             std = np.std(resid)
#             loglik = scipy.stats.norm.logpdf(resid, loc=0, scale=std)
#             log_likelihood = np.sum(loglik)

#         elif self.model_name in {'Poisson'}:
#             pass 
#             raise NotYetImplementedError()

#         elif self.model_name in {'Logistic'}:
#             log_likelihood = -sklearn.metrics.log_loss(truth, prediction)

#         elif self.model_name in {'Multinomial'}:
#             pass
#             raise NotYetImplementedError()
            
#         else:
#             pass
#             raise NotYetImplementedError()
        
#         return log_likelihood



class GLM():
    """
        Generalized Linear Model class built on scikit-learn's underlying regression models.
        power : float
            Only specify with a 'Tweedie' model_name in order to use fractional powers for Tweedie distribution
        *args : positional arguments
            See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for Logistic / Multinomial
            See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html otherwise
        **kwargs : keyword arguments
            See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for Logistic / Multinomial
            See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html otherwise
    """

    model = None
    model_name_options = {'Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', 'Binomial', 'Multinomial'}
    tweedie_lookup = {'Normal': 0, 'Gaussian':0, 'Poisson': 1, 'Gamma': 2}

    def __init__(self, model_name, beta0_=None, beta_=None, *args, **kwargs):
        """
        Create the GLM model.
        model_name : str
            GLM distribution name to create ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
        *args : positional arguments
            See links in SKGLM documentation for arguments
        **kwargs : keyword arguments
            See links in SKGLM documentation for arguments
        """

        self.model_name = model_name
        if model_name in {'Normal', 'Gaussian'}:
            Base = ElasticNet
        elif model_name in {'Poisson', 'Gamma'}:
            power = self.tweedie_lookup[model_name]
            kwargs['power'] = power
            Base = TweedieRegressor
        elif model_name in {'Tweedie'}:
            Base = TweedieRegressor
        elif model_name in {'Logistic', 'Multinomial'}:
            kwargs['multi_class'] = 'multinomial' if model_name == 'Multinomial' else 'auto'
            kwargs['n_jobs'] = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
            Base = LogisticRegression
        else:
            print('Distribution not yet implemented.')
            raise NotYetImplementedError()
        
        self.Base = Base
        # self.Base.__init__(self, *args, **kwargs)
        self.kwargs = kwargs
        self.model = self.Base(*args, **kwargs)
        self.score = self.model.score
    
    def pca_fit(self, X, y):
        self.alpha = 0
        self.reg_lambda = 0

        start = time.time()
        self.pca = PCA().fit(X)
        print(f'PCA fit in {time.time() - start} seconds')
        X_trans = self.pca.transform(X)

        start = time.time()
        self.fit(X_trans, y)
        print(f'PCA-based Model fit in {time.time() - start} seconds')

        W = self.pca.components_.T
        B = self.beta_.reshape([-1, 1])
        B0 = np.array(self.beta0_).reshape(-1)
        X_bar = self.pca.mean_

        self.beta_ = (W @ B).reshape(-1)
        self.coef_ = self.beta_
        self.beta0_ = (B0 - X_bar @ W @ B).reshape(-1)
        self.intercept_ = self.beta0_

    def fit(self, X, y, *args):
        """
        Fits the GLM to the provided X predictors and y responses.
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Array of predictor variables on which to fit the model
        y : np.ndarray or pd.Series
            Array of response variables on which to fit the model
        """

        self.model.fit(X, y, *args)

        self.coef_ = self.model.coef_ if self.model_name in {'Logistic', 'Multinomial', 'Gaussian', 'Normal'} else self.model.beta_
        self.beta_ = self.coef_
        self.intercept_ = self.model.intercept_ if self.model_name in {'Logistic', 'Multinomial', 'Gaussian', 'Normal'} else self.model.beta0_
        self.beta0_ = self.intercept_


    def fit_set(self, X, y, X_test, y_test, cv_coefs, cv_intercepts, cv_scores_train, cv_scores_test, iter_cv, *args):
        """
        Fits the GLM to the provided X predictors and y responses.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Array of predictor variables on which to fit the model
        y : np.ndarray or pd.Series
            Array of response variables on which to fit the model
        """
        start = time.time()
        print(f'Fitting: {self.kwargs}')
        self.fit(X, y, *args)
        print(f'Done with: {self.kwargs} — in {time.time() - start}')

        cv_coefs[:, iter_cv] = self.coef_
        cv_intercepts[iter_cv] = self.intercept_
        cv_scores_train[iter_cv] = self.score(X, y)
        cv_scores_test[iter_cv] = self.score(X_test, y_test)

    
    def predict(self, X):
        if type(X) == pd.DataFrame:
            X = X.values
        return self.model.predict(X)

    def log_likelihood(self, prediction, truth):

        if self.model_name in {'Normal', 'Gaussian'}:
            resid = truth - prediction
            std = np.std(resid)
            loglik = scipy.stats.norm.logpdf(resid, loc=0, scale=std)
            log_likelihood = np.sum(loglik)

        elif self.model_name in {'Poisson'}:
            pass 
            raise NotYetImplementedError()

        elif self.model_name in {'Logistic'}:
            log_likelihood = -sklearn.metrics.log_loss(truth, prediction)

        elif self.model_name in {'Multinomial'}:
            pass
            raise NotYetImplementedError()
            
        else:
            pass
            raise NotYetImplementedError()
        
        return log_likelihood

