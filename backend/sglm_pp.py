import numpy as np
import pandas as pd
import scipy.signal

# TODO: Write testcases & check validity

# TODO: Revise pandas implementation to make it more generalizable (to ndarrays)
# TODO: np.concatenate(np.zeros((n, x.shape[1]), x[n:], axis=1)
# TODO: np.ascontiguousarray

# TODO: Include train/test split - by 2 min segmentation
# TODO: Include diff

# Replace deconvolve with https://github.com/agiovann/constrained_foopsi_python


def timeshift(X, shift_inx=[], shift_amt=1, keep_non_inx=False):
    """
    Shift the column indicies "shift_inx" forward by shift_amt steps (backward if shift_amt < 0)

    Parameters
    ----------
    X : np.ndarray (preferably contiguous array)
        Array of all variables
    shift_inx : list(int)
        Column indices to shift forward/backward
    shift_amt : int
        Amount by which to shift forward the columns in question (backward if shift_amt < 0)
    keep_non_inx : bool
        If True, data from all columns (shifted or not) will be returned from the function. If False,
        only shifted columns are returned.
    """

    if type(X) == pd.DataFrame:
        npX = X.values
    else:
        npX = X
    
    shift_inx = shift_inx if shift_inx else range(npX.shape[1])
    X_to_shift = npX[:, shift_inx]

    append_vals = np.zeros((np.abs(shift_amt), X_to_shift.shape[1]))
    if shift_amt > 0:
        shifted_X = np.concatenate([append_vals, X_to_shift], axis=0)
        shifted_X = shifted_X[:-np.abs(shift_amt), :]
    elif shift_amt < 0:
        shifted_X = np.concatenate([X_to_shift, append_vals], axis=0)
        shifted_X = shifted_X[np.abs(shift_amt):, :]
    else:
        shifted_X = X_to_shift
    
    if type(X) == pd.DataFrame:
        return_setup = X.copy()
        return_setup.iloc[:, shift_inx] = shifted_X
        if not keep_non_inx:
            return_setup = return_setup.iloc[:, shift_inx]
    else:
        if keep_non_inx:
            return_setup = npX.copy()
            return_setup[:, shift_inx] = shifted_X
        else:
            return_setup = shifted_X.copy()

    return return_setup

def timeshift_multiple(X, shift_inx=[], shift_amt_list=[-1,0,1], unshifted_keep_all=True):
    """
    Collect all forward/backward shifts of columns shift_inx as columns in the returned array

    Parameters
    ----------
    X : np.ndarray (preferably contiguous array)
        Array of all variables
    shift_inx : list(int)
        Column indices to shift forward/backward
    shift_amt_list : list(int)
        List of amounts by which to shift forward the columns in question (backward where list elements are < 0)
    unshifted_keep_all : bool
        Whether or not to keep all unshifted columns in the returned array
    """

    shifted_list = []
    for shift_amt in shift_amt_list:
        shifted = timeshift(X, shift_inx=shift_inx, shift_amt=shift_amt, keep_non_inx=(shift_amt == 0 and unshifted_keep_all))
        shifted_list.append(shifted)
    
    return np.concatenate(shifted_list, axis=1)


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



def cvt_to_contiguous(x):
    return x


# """
# Credit: Rich Hakim
# """
# @njit(parallel=True)
# def var_numba(X):
#     Y = np.zeros(X.shape[0], dtype=X.dtype)
#     for ii in prange(X.shape[0]):
#         Y[ii] = np.var(X[ii,:])
#     return Y


# @njit(parallel=True)
# def min_numba(X):
#     output = np.zeros(X.shape[0])
#     for ii in prange(X.shape[0]):
#         output[ii] = np.min(X[ii])
#     return output


# @njit(parallel=True)
# def max_numba(X):
#     output = np.zeros(X.shape[0])
#     for ii in prange(X.shape[0]):
#         output[ii] = np.max(X[ii])
#     return output

