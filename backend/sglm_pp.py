import numpy as np
import pandas as pd
import scipy.signal

# TODO: Write testcases & check validity

def timeshift(X, shift_names=[], ignore_names=[], max_bwd_shift=1, max_fwd_shift=1):
    """
    Shift the "shift_names" columns of X backward by 1-max_bwd_shift steps and forward by 1-max_fwd_shift steps

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Array of all variables
    shift_names : list(str) or list(int)
        Column names to shift forward/backward (if X is np.ndarray, should be column number integers)
    ignore_names : list(str) or list(int)
        Column names to not shift forward/backward (if X is np.ndarray, should be column number integers)
    max_bwd_shift : int
        Maximum number of shifts backward to include in the data
    max_fwd_shift : int
        Maximum number of shifts forward to include in the data
    """

    df_raw = pd.DataFrame(X)
    df = df_raw.copy()

    if not shift_names:
        shift_names = [_ for _ in df_raw.columns if _ not in ignore_names]
    else:
        shift_names = [_ for _ in shift_names if _ not in ignore_names]

    for shift_size in range(-max_bwd_shift, max_fwd_shift+1):
        if shift_size == 0:
            continue
        dir_str = 'b' if shift_size < 0 else 'f'
        df[[_ + '_' + dir_str + str(shift_size) for _ in shift_names]] = df_raw[shift_names].shift(shift_size)
    
    return df


def zscore(X):
    """
    Convert X values to z-scores along the 0th axis

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Array of variables to zscore
    """
    return (X - X.mean(axis=0))/X.std(axis=0)


def deconvolve(X, divisor):
    """
    Deconvolve X with the divisor.

    Parameters
    ----------
    X : np.ndarray or list-like
        Array of signal to be deconvolved
    divisor : np.ndarray or list-like
        Convolution that was used to generate the signal in X
    """

    return scipy.signal.deconvolve(X, divisor)




