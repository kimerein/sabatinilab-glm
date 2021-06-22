import sklearn.linear_model

class GLM():
    """
    Generalized Linear Model class built on scikit-learn's underlying regression models.

    Attributes
    ----------
    model_name : str
        GLM distribution name to create ('Normal', 'Gaussian', 'Poisson', 'Tweedie', 'Gamma', 'Logistic', or 'Multinomial')
    power : float
        Only specify with a 'Tweedie' model_name in order to use fractional powers for Tweedie distribution

    Methods
    -------
    fit(X, y, *args, **kwargs):
        the GLM to
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
            mdl = sklearn.linear_model.LogisticRegression(*args, **kwargs)
        else:
            print('Distribution not yet implemented.')
            raise NotYetImplementedError()
        
        self.model = mdl
    
    def fit(self, X, y, *args, **kwargs):
        """
        
        Parameters
        ----------
        X : np.ndarray
            Array of predictor variables on which to fit the model
        y : np.ndarray
            Array of response variables on which to fit the model
        """

        self.model.fit(X, y, *args, **kwargs)
    
    