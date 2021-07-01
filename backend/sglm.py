import sklearn.linear_model
import pyglmnet

# TODO: Add additional alternatives for different GLM API implementations (SKLearn, etc.)

class GLM():
    """
    Generalized Linear Model class built on scikit-learn's underlying regression models.

    Attributes
    ----------
    model_name : str
        GLM distribution name to create ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
    *args : positional arguments
        See https://glm-tools.github.io/pyglmnet/api.html for relevant arguments.
    **kwargs : keyword arguments
        See https://glm-tools.github.io/pyglmnet/api.html for relevant arguments.
    model : sklearn.linear_model.TweedieRegressor or sklearn.linear_model.LogisticRegression
        Underlying pyglmnet model that is built

    Methods
    -------
    fit(X, y, *args, **kwargs):
        Fits the GLM to the provided X predictors and y responses.
        (Calls the underlying sklearn fit method for the associated model.)
    """

    model = None
    model_name_options = {'Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', 'Binomial', 'Multinomial'}
    tweedie_lookup = {'Normal': 0, 'Gaussian':0, 'Poisson': 1, 'Gamma': 2}

    def __init__(self, model_name, *args, **kwargs):
        """
        Create the GLM model.
        """

        self.model_name = model_name
        if model_name in self.model_name_options: #{'Normal', 'Gaussian', 'Poisson', 'Gamma'}:
            model_name = 'Gaussian' if model_name in {'Normal'} else model_name
            model_name = 'Binomial' if model_name in {'Logistic', 'Multinomial'} else model_name
        #     power = self.tweedie_lookup[model_name]
        #     kwargs['power'] = power
        #     mdl = sklearn.linear_model.TweedieRegressor(*args, **kwargs)

            kwargs['distr'] = model_name.lower()
            mdl = pyglmnet.GLM(*args, **kwargs)

        # elif model_name in {'Tweedie'}:
        #     mdl = sklearn.linear_model.TweedieRegressor(*args, **kwargs)
        # elif model_name in {'Logistic', 'Multinomial'}:
        #     kwargs['multi_class'] = 'multinomial' if model_name == 'Multinomial' else 'auto'
        #     kwargs['n_jobs'] = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
        #     mdl = sklearn.linear_model.LogisticRegression(*args, **kwargs)
        else:
            print('Distribution not yet implemented.')
            raise NotYetImplementedError()
        
        self.model = mdl
    
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
        # self.coef_ = self.model.coef_ if self.model_name in {'Logistic', 'Multinomial'} else self.model.beta_
        # self.intercept_ = self.model.intercept_ if self.model_name in {'Logistic', 'Multinomial'} else self.model.beta0_

        # self.coef_ = self.model.coef_
        # self.intercept_ = self.model.intercept_

        self.coef_ = self.model.beta_
        self.intercept_ = self.model.beta0_




### Original SKLearn Implementation-related Documentation
"""
    power : float
        Only specify with a 'Tweedie' model_name in order to use fractional powers for Tweedie distribution

    *args : positional arguments
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for Logistic / Multinomial
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html otherwise
    **kwargs : keyword arguments
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for Logistic / Multinomial
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html otherwise
"""