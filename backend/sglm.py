import sklearn.linear_model

# TODO:
# Include train/test split - by 2 min segmentation

class GLM():
    """
    Generalized Linear Model class built on scikit-learn's underlying regression models.

    Attributes
    ----------
    model_name : str
        GLM distribution name to create ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
    power : float
        Only specify with a 'Tweedie' model_name in order to use fractional powers for Tweedie distribution
    *args : positional arguments
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for Logistic / Multinomial
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html otherwise
    **kwargs : keyword arguments
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for Logistic / Multinomial
        See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html otherwise
    model : sklearn.linear_model.TweedieRegressor or sklearn.linear_model.LogisticRegression
        Underlying sklearn model that is built

    Methods
    -------
    fit(X, y, *args, **kwargs):
        Fits the GLM to the provided X predictors and y responses.
        (Calls the underlying sklearn fit method for the associated model.)
    """

    model = None
    model_name_options = {'Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', 'Multinomial'}
    tweedie_lookup = {'Normal': 0, 'Gaussian':0, 'Poisson': 1, 'Gamma': 2}

    def __init__(self, model_name, *args, **kwargs):
        """
        Create the GLM model.
        """

        self.model_name = model_name
        if model_name in {'Normal', 'Gaussian', 'Poisson', 'Gamma'}:
            power = self.tweedie_lookup[model_name]
            kwargs['power'] = power
            mdl = sklearn.linear_model.TweedieRegressor(*args, **kwargs)
        elif model_name in {'Tweedie'}:
            mdl = sklearn.linear_model.TweedieRegressor(*args, **kwargs)
        elif model_name in {'Logistic', 'Multinomial'}:
            kwargs['multi_class'] = 'multinomial' if model_name == 'Multinomial' else 'auto'
            kwargs['n_jobs'] = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
            mdl = sklearn.linear_model.LogisticRegression(*args, **kwargs)
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
