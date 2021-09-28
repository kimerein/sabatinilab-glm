from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet, TweedieRegressor, LogisticRegression, PoissonRegressor, LinearRegression
import sklearn.metrics
import pyglmnet
import scipy.stats
import numpy as np
import pandas as pd
import time

from typing import Union, List, Tuple, Optional

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

    JZ 2021

    Attributes:
        model_name (str) : Type of GLM built (e.g. Normal, Poisson, Logistic, Multinomial, etc.)
        Base (class in sklearn.linear_models.*) : SKLearn class associated with model_name
        kwargs (dict) : Keyword arguments specified to build Base model object
        model (object of Base) : Object of class Base with arguments kwargs for fitting
        intercept_ (float) : Bias term to be fitted in model
        coef_ (np.ndarray) : Weights by which predictors in X are multiplied for linear prediction
        beta0_ (float) : Another name for intercept_
        beta_ (np.ndarray) : Another name for coef_
        pca (object of sklearn.decomposition.PCA) : PCA object for use in PCA pre-procesisng of data

    Methods:
        __init__ : Create the GLM model
        score : Call underlying score function of model
        pca_fit : Automatically apply PCA to predictors prior to fitting and use inverse transform to
                  identify associated coefficients in the non-PCA space. (Should be a faster fit &
                  can be used to seed other fits.)
        fit : Fits the GLM model
        fit_set : Fits the GLM model while setting variables for in-place calculations for CV
        predict : Run prediction through the GLM model
        log_likelihood : Calculate log_likelihood of the model for given datset (requires dsitributional
                  assumptions of residuals)

    """

    model = None
    model_name_options = {'Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', 'Binomial', 'Multinomial'}
    tweedie_lookup = {'Normal': 0, 'Gaussian':0, 'Poisson': 1, 'Gamma': 2}

    def __init__(self, model_name, beta0_=None, beta_=None, score_method='mse', *args, **kwargs):
        """
        Create the GLM model.

        JZ 2021

        Args:
            model_name : str
                GLM distribution name to create ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', 'Multinomial',
                'PCA Normal', or 'PCA Gaussian'). Use 'PCA Normal' or 'PCA Gaussian' to fit a PCA model for Normal / Gaussian warm
                start bases.
            beta0_ : int
                Pre-initialized intercept value for warm-start-based fitting
            beta_ : np.ndarray
                Pre-initialized coefficient values for warm-start-based fitting
            score_method : str
                'mse' for Mean Squared Error-based scoring, 'r2' for R^2 based scoring
            power : float
                Only specify with a 'Tweedie' model_name in order to use fractional powers for Tweedie distribution
            *args : positional arguments
                See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for Normal / Gaussian
                See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for Logistic / Multinomial
                See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html otherwise
            **kwargs : keyword arguments
                See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for Normal / Gaussian
                See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for Logistic / Multinomial
                See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html otherwise
        
        Returns: N/A
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
        elif model_name in {'PCA Normal', 'PCA Gaussian'}:
            Base = LinearRegression
        else:
            print('Distribution not yet implemented.')
            raise NotYetImplementedError()
        
        self.Base = Base
        # self.Base.__init__(self, *args, **kwargs)
        self.kwargs = kwargs
        self.model = self.Base(*args, **kwargs)

        # print(self.model.get_params())

        if beta0_ is not None:
            self.model.intercept_ = beta0_
            self.beta0_ = beta0_
        if isinstance(beta_, np.ndarray):
            self.model.coef_ = beta_
            self.beta_ = beta_
        
        if score_method == 'r2':
            self.score = self.r2_score
        elif score_method == 'mse':
            self.score = self.neg_mse_score
        else:
            self.score = self.neg_mse_score

    
    def neg_mse_score(self, X, y):
        """
        Score function based on the negative of the Mean Squared Error for use in model selection.
        (i.e. greater / less negative values represent a more accurate model)

        JZ 2021

        Args:
            X: pd.DataFrame
                Data from which to run the prediction model
            y: pd.Series
                True y response values against which to calculate the MSE
        
        Returns: Negative of the MSE between X-based prediction and true response y
        """
        pred = self.predict(X)
        resid = (y - pred)
        return -np.mean(resid**2)

    def r2_score(self, X, y):
        """
        Score function for use with out-of-sample R^2 for use in model selection.
        (i.e. greater values represent a more accurate model)

        JZ 2021

        Args:
            X: pd.DataFrame
                Input data for the prediction model
            y: pd.Series
                True y response values against which to evaluate R^2
        
        Returns: The calculated R^2 value
        """
        return self.model.score(X, y)

    def pca_fit(self, X, y):
        """
        Apply PCA to X (without dimensionality reduction), fit the model, and inverse the transform
        to more quickly generate coefficients in the original basis of X. (Best used as a setup process
        to speed up the fitting of other models.)

        JZ 2021

        Args:
            X : np.ndarray or pd.DataFrame
                Array of predictor variables on which to fit the model
            y : np.ndarray or pd.Series
                Array of response variables on which to fit the model
        
        Returns: N/A
        """
        if self.model_name in {'Normal', 'Gaussian'}:
            self.model.alpha = 0.1 if 'alpha' not in self.kwargs else self.kwargs['alpha']
            self.model.l1_ratio = 0.5 if 'l1_ratio' not in self.kwargs else self.kwargs['l1_ratio']

        start = time.time()
        self.pca = PCA().fit(X)
        print(f'PCA fit in {time.time() - start} seconds')
        X_trans = self.pca.transform(X)

        start = time.time()
        self.fit(X_trans, y)
        print(f'> PCA-based Model fit in {time.time() - start} seconds')

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

        JZ 2021
    
        Args:
            X : np.ndarray or pd.DataFrame
                Array of predictor variables on which to fit the model
            y : np.ndarray or pd.Series
                Array of response variables on which to fit the model
            *args : any
                Additional arguments to pass into the GLM fit function
        
        Returns: N/A
        """

        self.model.fit(X, y, *args)

        self.coef_ = self.model.coef_ if self.model_name in {'Logistic', 'Multinomial', 'Gaussian', 'Normal',
                                                             'PCA Gaussian', 'PCA Normal'} else self.model.beta_
        self.beta_ = self.coef_
        self.intercept_ = self.model.intercept_ if self.model_name in {'Logistic', 'Multinomial', 'Gaussian', 'Normal',
                                                                       'PCA Gaussian', 'PCA Normal'} else self.model.beta0_
        self.beta0_ = self.intercept_


    def fit_set(self, X, y, X_test, y_test, cv_coefs,
                cv_intercepts, cv_scores_train, cv_scores_test,
                iter_cv, *args, resids=[], mean_resids=[], id_fit='None', verbose=0):
        """
        Fits the GLM to the provided X predictors and y responses and sets associated values
        in place in output variables (for use in multi-threading applications).

        JZ 2021
    
        Args:
            X : np.ndarray or pd.DataFrame
                Array of predictor variables on which to fit the model
            y : np.ndarray or pd.Series
                Array of response variables on which to fit the model
            X_test : np.ndarray or pd.DataFrame
                Array of predictor variables on which to evaluate the model
            y_test : np.ndarray or pd.Series
                Array of response variables on which to evaluate the model
            cv_coefs : np.ndarray
                array for in place setting of fitted coefficients
            cv_intercepts : np.ndarray
                array for in place setting of fitted intercepts
            cv_scores_train : np.ndarray
                array for in place setting of validation scores on training set
            cv_scores_test : np.ndarray
                array for in place setting of validation scores on testing set
            iter_cv : list(tuple(np.ndarray))
                List of tuples of validation indices to be used for validation / hyperparameter selection
            *args : any
                Additional arguments to pass into the GLM fit function
            resids : list(np.ndarray)
                list for in place appending of deltas (resids) between test responses and predictions
            mean_resids : list(np.ndarray)
                list for in place appending of deltas (resids) between test responses and mean of test responses
            id_fit : str
                Identifier for verbose printing threads
            verbose : int
                How much information to print during the fititng process
        
        Returns: N/A
        """
        if verbose > 1:
            start = time.time()
            print(f'Fitting: {self.kwargs} — {id_fit}')
        
        self.fit(X, y, *args)

        if verbose > 1:
            print(f'Done with: {self.kwargs} — {id_fit} — in {time.time() - start}')
        
        cv_coefs[:, iter_cv] = self.coef_
        cv_intercepts[iter_cv] = self.intercept_
        cv_scores_train[iter_cv] = self.score(X, y)
        cv_scores_test[iter_cv] = self.score(X_test, y_test)

        residuals, mean_residuals = self.get_residuals(X_test, y_test)
        resids.append(residuals)
        mean_resids.append(mean_residuals)

    def get_residuals(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the prediction of the responses and returns the residuals and mean_residuals (for
        downstream calculations of RSS and TSS).

        JZ 2021
    
        Args:
            X : np.ndarray or pd.DataFrame
                Array of predictor variables with which to calculate R^2
            y : np.ndarray or pd.Series
                Array of response variables with which to calculate R^2
        
        Returns: Iterable of residuals, Iterable of delta between y and mean
        """
        residuals = (y - self.predict(X))
        mean_residuals = (y - np.mean(y))
        return residuals, mean_residuals
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Run a prediction of features, X, through the model.

        JZ 2021
    
        Args:
            X : np.ndarray or pd.DataFrame
                Array of predictor variables from which to run model predictions

        Returns: np.ndarray of predicted responses
        """
        if type(X) == pd.DataFrame:
            X = X.values
        return self.model.predict(X)

    def log_likelihood(self, prediction: Union[np.ndarray, pd.Series], truth: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate the log likelihood of the predictions generated by the model in comparison with the truth
        (IN PROGRESS)

        JZ 2021
    
        Args:
            prediction : np.ndarray or pd.Series
                Predictions generated by the model
            truth : np.ndarray or pd.Series
                True values to compare to the prediction
        
        Returns: float of log_likelihood values
        """
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


def calc_R2(residuals: np.ndarray, mean_residuals: np.ndarray) -> float:
    """
    Calculates the R^2 value given the residuals (for RSS) and residuals from the mean (for TSS)

    JZ 2021

    Args:
        residuals : np.ndarray
            Array of all residual values observed during model fitting (e.g. concatenated validaiton residuals)
        mean_residuals : np.ndarray
            Array of all deltas between observed values and response means (e.g. concatenated y - np.mean(y) values)
    
    Returns: R^2 Value
    """
    rss = np.sum(residuals**2)
    tss = np.sum(mean_residuals**2)
    r2 = 1 - rss/tss
    return r2

# def calc_R2(residuals, y):
#     rss = np.sum(residuals**2)
#     tss = np.sum((y - np.mean(y))**2)
#     print(f'rss: {rss}')
#     print(f'tss: {tss}')
#     r2 = 1 - rss/tss
#     return r2
