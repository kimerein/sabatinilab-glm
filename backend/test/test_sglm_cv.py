# import pytest

# import sys
# import os 
# dir_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
# sys.path.append(f'{dir_path}/..')
# sys.path.append('{dir_path}/backend')
# sys.path.append('{dir_path}/../backend')

# import sglm
# import sglm_cv
# import sglm_pp
# import sklearn.linear_model
# from scipy import stats
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# import sklearn.model_selection



# # Normal (OLS) CV Test
# def test_normal_ols_cv():
#     np.random.seed(117)
#     norm = stats.norm()

#     true_x = norm.rvs(size=1000)
#     true_y = true_x * 0.5
#     obs_y = (true_y + norm.rvs(size=1000)*0.2)

#     x = true_x[:,None]
#     y = obs_y

#     plt.figure(figsize=(5,5))
#     plt.scatter(x, y, alpha = 0.25)
    
#     ss = sklearn.model_selection.ShuffleSplit()
#     inx = list(ss.split(x, y))

#     sglm_cv.cv_glm_single_params(x, y, inx, 'Gaussian', {'alpha': 0})


#     param_list = sglm_cv.generate_mult_params(
#                                             {
#                                             'alpha': [0,0.01,0.1,1],
#                                             'roll': [0,1,2,3,4]
#                                             },
#                                             kwargs={'fit_intercept': True}
#                                             )

#     sglm_cv.cv_glm_mult_params(x, y, inx, 'Gaussian', param_list)

#     return

