import numpy as np
import pandas as pd
import scipy.signal
from numba import njit, jit, prange

import sys
import threading

import caiman 
# If this causes an error, navigate to backend/lib/CaImAn and run:
#     pip install -r requirements.txt
#     pip install .
### Suite 2p's regularization / deconvolution


# TODO: Write testcases & check validity
# TODO: Include train/test split - by 2 min segmentation
# TODO: Switch to suite2p's convolution
# TODO: Numba impolementations

def timeshift(X, shift_inx=[], shift_amt=1, keep_non_inx=False, dct=None):
    """
    Shift the column indicies "shift_inx" forward by shift_amt steps (backward if shift_amt < 0)

    Parameters
    ----------
    X : np.ndarray (preferably contiguous array)
        Array of all variables (columns should be features, rows should be timesteps)
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

    append_vals = np.zeros((np.abs(shift_amt), X_to_shift.shape[1])) * np.nan
    if shift_amt > 0:
        shifted_X = np.concatenate((append_vals, X_to_shift), axis=0)
        shifted_X = shifted_X[:-shift_amt, :]
    elif shift_amt < 0:
        shifted_X = np.concatenate((X_to_shift, append_vals), axis=0)
        shifted_X = shifted_X[-shift_amt:, :]
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

    if dct is None:
        return return_setup
    else:
        dct[shift_amt] = return_setup
        return


def timeshift_multiple(X, shift_inx=[], shift_amt_list=[-1,0,1], unshifted_keep_all=True):
    """
    Collect all forward/backward shifts of columns shift_inx as columns in the returned array

    Parameters
    ----------
    X : np.ndarray (preferably contiguous array)
        Array of all variables (columns should be features, rows should be timesteps)
    shift_inx : list(int)
        Column indices to shift forward/backward
    shift_amt_list : list(int)
        List of amounts by which to shift forward the columns in question (backward where list elements are < 0)
    unshifted_keep_all : bool
        Whether or not to keep all unshifted columns in the returned array
    """

    shifted_dict = {}
    threads = []
    for i,shift_amt in enumerate(shift_amt_list):
        # shifted = timeshift(X, shift_inx=shift_inx, shift_amt=shift_amt, keep_non_inx=(shift_amt == 0 and unshifted_keep_all))
        # shifted_dict[shift_amt] = shifted

        threads.append(threading.Thread(target=timeshift, args=(X,), kwargs={'shift_inx':shift_inx,
                                                                 'shift_amt':shift_amt,
                                                                 'keep_non_inx':(shift_amt == 0 and unshifted_keep_all),
                                                                 'dct':shifted_dict
                                                                 }))
        threads[i].start()

    for i,shift_amt in enumerate(shift_amt_list):
        threads[i].join()

    shifted_list = [shifted_dict[_] for _ in shift_amt_list]

    
    # print(shifted_list)

    if type(X) == pd.DataFrame:
        ret = pd.DataFrame()
        for isa, shift_amt in enumerate(shift_amt_list):
            col_names = shifted_list[isa].columns
            sft_col_names = [f"{_}_{shift_amt}" for _ in col_names] if shift_amt != 0 else col_names
            ret[sft_col_names] = shifted_list[isa][col_names]
        return ret
    else:
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


def diff(X, diff_inx=[], n=1, axis=0, prepend=0):
    """
    Return the differential between each timestep and the previous timestep, n times.
    
    Documentation adopted from numpy diff.

    ---

    Parameters
    ----------
    X : np.ndarray (preferably contiguous array)
        Array of all variables (columns should be features, rows should be timesteps)
    n : int, optional
        The number of times values are differenced. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend, append : array_like, optional
        Values to prepend or append to `a` along axis prior to
        performing the difference.  Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes.  Otherwise the
        dimension and shape must match `a` except along axis.
    """

    if type(X) == pd.DataFrame or type(X) == pd.Series:
        X_val = X.values
    else:
        X_val = X
    
    if len(X.shape) == 1:
        X_val = X_val.reshape((-1,1))
    
    diff_inx = diff_inx if diff_inx else list(range(X_val.shape[1]))
    ret = np.diff(X_val[:,diff_inx], n=n, axis=axis, prepend=prepend)

    if type(X) == pd.DataFrame:
        ret = pd.DataFrame(ret, columns=[f'{_}_diff' for _ in X.columns], index=X.index)
    elif type(X) == pd.Series:
        ret = pd.Series(ret.reshape(-1), name=f'{X.name}_diff', index=X.index)
    
    return ret

# Replaced scipy deconvolve with https://github.com/agiovann/constrained_foopsi_python
def deconvolve(*args, **kwargs):
    """
    Deconvolve using CaImAn implementation of constrained_foopsi.

    To install, navigate to backend/lib/CaImAn and run:
        pip install .
        pip install -r requirements.txt

    ---

    Infer the most likely discretized spike train underlying a fluorescence trace

    It relies on a noise constrained deconvolution approach
    
    Parameters
    ----------
        fluor: np.ndarray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.

        bl: [optional] float
            Fluorescence baseline value. If no value is given, then bl is estimated
            from the data.

        c1: [optional] float
            value of calcium at time 0

        g: [optional] list,float
            Parameters of the AR process that models the fluorescence impulse response.
            Estimated from the data if no value is given

        sn: float, optional
            Standard deviation of the noise distribution.  If no value is given,
            then sn is estimated from the data.

        p: int
            order of the autoregression model

        method_deconvolution: [optional] string
            solution method for basis projection pursuit 'cvx' or 'cvxpy' or 'oasis'

        bas_nonneg: bool
            baseline strictly non-negative

        noise_range:  list of two elms
            frequency range for averaging noise PSD

        noise_method: string
            method of averaging noise PSD

        lags: int
            number of lags for estimating time constants

        fudge_factor: float
            fudge factor for reducing time constant bias

        verbosity: bool
             display optimization details

        solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers
            to be used with cvxpy, default is ['ECOS','SCS']

        optimize_g : [optional] int, only applies to method 'oasis'
            Number of large, isolated events to consider for optimizing g.
            If optimize_g=0 (default) the provided or estimated g is not further optimized.

        s_min : float, optional, only applies to method 'oasis'
            Minimal non-zero activity within each bin (minimal 'spike size').
            For negative values the threshold is abs(s_min) * sn * sqrt(1-g)
            If None (default) the standard L1 penalty is used
            If 0 the threshold is determined automatically such that RSS <= sn^2 T

    Returns:
        c: np.ndarray float
            The inferred denoised fluorescence signal at each time-bin.

        bl, c1, g, sn : As explained above

        sp: ndarray of float
            Discretized deconvolved neural activity (spikes)

        lam: float
            Regularization parameter
    Raises:
        Exception("You must specify the value of p")

        Exception('OASIS is currently only implemented for p=1 and p=2')

        Exception('Undefined Deconvolution Method')

    References:
        * Pnevmatikakis et al. 2016. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037
        * Machado et al. 2015. Cell 162(2):338-350

    \image: docs/img/deconvolution.png
    \image: docs/img/evaluationcomponent.png
    """

    return caiman.source_extraction.cnmf.deconvolution.constrained_foopsi(*args, **kwargs)

# https://github.com/MouseLand/suite2p/blob/main/suite2p/extraction/dcnv.py

# # compute deconvolution
# from suite2p.extraction import dcnv
# import numpy as np
# tau = 1.0 # timescale of indicator
# fs = 30.0 # sampling rate in Hz
# neucoeff = 0.7 # neuropil coefficient
# # for computing and subtracting baseline
# baseline = 'maximin' # take the running max of the running min after smoothing with gaussian
# sig_baseline = 10.0 # in bins, standard deviation of gaussian with which to smooth
# win_baseline = 60.0 # in seconds, window in which to compute max/min filters
# ops = {'tau': tau, 'fs': fs, 'neucoeff': neucoeff,
#        'baseline': baseline, 'sig_baseline': sig_baseline, 'win_baseline': win_baseline}
# # load traces and subtract neuropil
# F = np.load('F.npy')
# Fneu = np.load('Fneu.npy')
# Fc = F - ops['neucoeff'] * Fneu
# # baseline operation
# Fc = dcnv.preprocess(
#      F=Fc,
#      baseline=ops['baseline'],
#      win_baseline=ops['win_baseline'],
#      sig_baseline=ops['sig_baseline'],
#      fs=ops['fs'],
#      prctile_baseline=ops['prctile_baseline']
#  )
# # get spikes
# spks = dcnv.oasis(F=Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])



@jit(parallel=True)
def zscore_numba(array):
    '''
    Parallel (multicore) Z-Score. Uses numba.
    Computes along second dimension (axis=1) for speed
    Best to input a contiguous array.
    RH 2021
    Args:
        array (ndarray):
            2-D array. Percentile will be calculated
            along first dimension (columns)
    
    Returns:
        output_array (ndarray):
            2-D array. Z-Scored array
    '''

    output_array = np.zeros_like(array)
    for ii in prange(array.shape[0]):
        array_tmp = array[ii,:]
        output_array[ii,:] = (array_tmp - np.mean(array_tmp)) / np.std(array_tmp)
    return output_array


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

