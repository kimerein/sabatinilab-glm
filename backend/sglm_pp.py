import numpy as np
import pandas as pd
import scipy.signal


### Still very much in Alpha Phase (uncommented / undocumented / untested for now)


def timeshift(X, y_name=None, time_step_begin=1, time_step_end=2):

    df_raw = pd.DataFrame(X)

    if y_name:
        df_raw = df_raw.drop(y_name, axis=1)
    
    df = df_raw.copy()

    for shift_size in range(time_step_begin, time_step_end):
        df[[_+'_f'+str(shift_size) for _ in df_raw.columns]] = df_raw[df_raw.columns].shift(shift_size)
        df[[_+'_b'+str(shift_size) for _ in df_raw.columns]] = df_raw[df_raw.columns].shift(-shift_size)

    return df


def zscore(X):
    return (X - X.mean(axis=0))/X.std(axis=0)


def deconvolve(X, divisor):
    return scipy.signal.deconvolve(X, divisor)

