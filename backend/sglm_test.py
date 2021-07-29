import sys
sys.path.append('./backend')

import sglm
import sglm_cv
import sklearn.linear_model
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

import sklearn.model_selection

# Normal (OLS)
def test_normal_ols():
    norm = stats.norm()

    true_x = norm.rvs(size=1000)
    true_y = true_x * 0.5
    obs_y = (true_y + norm.rvs(size=1000)*0.2)

    x = true_x[:,None]
    y = obs_y

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, alpha = 0.25)

    glm = sglm.GLM('Normal', alpha=0)
    glm.fit(x, y)
    coef, intercept = glm.coef_, glm.intercept_

    view_x = np.linspace(x.min(), x.max(), num=100)
    view_y = view_x*coef + intercept
    obs_y = (true_y + norm.rvs(size=1000)*0.)

    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0], y, alpha = 0.25)
    plt.plot(view_x, np.squeeze(view_y), color='g')

    return

# Poisson GLM
def test_poisson_glm():
    norm = stats.norm()

    true_x = np.array(sorted(norm.rvs(size=1000)*.75))
    true_y = np.exp(true_x)
    obs_y = np.array([stats.poisson(mu=np.exp(_)).rvs(1) for _ in true_x]).reshape(-1)

    x = true_x[:,None]
    y = obs_y

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, alpha = 0.25)

    # glm = sglm.GLM('Poisson', alpha=0, link='identity')
    glm = sglm.GLM('Poisson', reg_lambda=0)
    glm.fit(x, y)
    coef, intercept = glm.coef_, glm.intercept_
    
    view_x = np.linspace(x.min(), x.max(), num=100)
    view_y = np.exp(view_x*coef + intercept)
    # view_y = view_x*coef + intercept

    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0], y, alpha = 0.25)
    plt.plot(view_x, np.squeeze(view_y), color='g')
    plt.plot(true_x, np.squeeze(true_y), color='r')
    return

# Logistic GLM
def test_logistic_glm():
    norm = stats.norm()

    true_x = norm.rvs(size=1000)
    true_y = true_x * 0.5
    obs_y = ((true_y + norm.rvs(size=1000)*0.1) > 0)*1.0

    x = true_x[:,None]
    y = obs_y

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, alpha = 0.25)

    glm = sglm.GLM('Logistic', reg_lambda=0)
    glm.fit(x, y)
    coef, intercept = glm.coef_, glm.intercept_
    
    view_x = np.linspace(x.min(), x.max(), num=100)
    view_y = 1/(1+np.exp(-(view_x*coef + intercept)))

    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0], y, alpha = 0.25)
    plt.plot(view_x, np.squeeze(view_y), color='g')

    return

# Normal (OLS) CV Test
def test_normal_ols_cv():
    norm = stats.norm()

    true_x = norm.rvs(size=1000)
    true_y = true_x * 0.5
    obs_y = (true_y + norm.rvs(size=1000)*0.2)

    x = true_x[:,None]
    y = obs_y

    plt.figure(figsize=(5,5))
    plt.scatter(x, y, alpha = 0.25)
    
    ss = sklearn.model_selection.ShuffleSplit()
    inx = list(ss.split(x, y))

    sglm_cv.cv_glm_single_params(x, y, inx, 'Gaussian', {'alpha': 0})


    param_list = sglm_cv.generate_mult_params(
                                            {
                                            'alpha': [0,0.01,0.1,1],
                                            'roll': [0,1,2,3,4]
                                            },
                                            kwargs={'fit_intercept': True}
                                            )

    sglm_cv.cv_glm_mult_params(x, y, inx, 'Gaussian', param_list)

    return



if __name__ == '__main__':
    test_normal_ols()
    test_poisson_glm()
    test_logistic_glm()
    test_normal_ols_cv()