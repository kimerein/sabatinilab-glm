import os
import sys
dir_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.append(f'{dir_path}/sabatinilab-glm/backend')
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/backend')
sys.path.append(f'{dir_path}/../backend')
# sys.path.append('./backend')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
import time

from scipy.fft import fft, ifft
from scipy.signal import welch

import sglm
import sglm_cv
import sglm_pp
import sglm_ez


def reconstruct_signal(glm, X, y_true=-1):
    """
    Plot a graph of the predicted signal overlaid with the truly observed signal

    Args:
        glm : sglm.GLM
            Model with which to predict X
        X : np.ndarray or pd.DataFrame
            Predictors to use for signal reconstruction
        y : np.ndarray or pd.DataFrame
            True signal for comparison

    Returns: Tuple[y_true, y_hat]
    """
    import seaborn as sns
    sns.set(style='white', palette='colorblind', context='poster')

    pred = glm.predict(X)

    plt.figure(figsize=(20,10))
    plt.plot(X.index, pred, label='Predicted Signal', alpha=0.5)
    
    if type(y_true) != int:
        plt.plot(X.index, y_true, label='True Signal', alpha=0.5)
    
    plt.xlabel('Data Point Index')
    plt.ylabel(f'{"True vs. " if type(y_true) != int else ""}Prediction Reconstructed Signals')
    plt.legend()

    return y_true, pred


def plot_single_coef_set(name, timeshifts, coefs, ax=None, y_lims=None, binsize=None):
    # Binsize in Milliseconds
    x_vals = np.array(timeshifts)
    x_label = 'Timeshifts'

    if binsize is not None:
        x_vals *= binsize
        x_label += ' (ms)'

    ax.plot(x_vals, coefs)

    ax.set_ylim(y_lims)
    ax.set_title(name)

    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel('Coefficient Value', fontsize=15)
    ax.grid()
    return

def plot_all_beta_coefs(glm, coef_names, sftd_coef_names, plot_width=4, y_lims=None, filename='', plot_name='', binsize=None):
    """
    Plot all beta coefficients for a given model
    Args:
        glm : sglm.GLM
            Model with which to predict X
        coef_names : List[str]
            List of names of beta coefficients to plot
        sftd_coef_names : List[str]
            List of names of sftd coefficients to plot
        plot_width : int
            Width of plot in inches
        y_lims : List[float]
            Limits of y-axis
        filename : str
            Name of file to save plot to
        plot_name : str
            Name of plot
        binsize : int
            Binsize in milliseconds
    """
    if y_lims is None:
        y_lims = (glm.coef_.min(), glm.coef_.max())
    print(y_lims)

    coef_lookup = {sftd_coef_names[i]:glm.coef_[i] for i in range(len(sftd_coef_names))}
    coef_cols = sglm_ez.get_coef_name_sets(coef_names, sftd_coef_names)
    
    fig, axs = plt.subplots(len(coef_cols)//plot_width + (len(coef_cols)%plot_width > 0)*1, plot_width)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    addl_plot_name = ' — ' + plot_name if plot_name else ''
    fig.suptitle(f'Feature Coefficients by Timeshift{addl_plot_name}', fontsize=20)

    for icn, coef_name in enumerate(coef_cols):
        print(icn)
        timeshifts, coefs = sglm_ez.get_single_coef_set(coef_cols[coef_name], coef_lookup)
        
        if len(axs.shape) > 1:
            axs_a = axs[icn//plot_width]
        else:
            axs_a = axs
        
        axs_tmp = axs_a[icn%plot_width]

        plot_single_coef_set(coef_name, timeshifts, coefs, axs_tmp, y_lims, binsize=binsize)
    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    return


def plot_power_spectra(y_true_full, y_hat_full):
    """
    Plot the power spectra of the true and predicted signals
    Args:
        y_true_full : np.ndarray
            True signal
        y_hat_full : np.ndarray
            Predicted signal
    """

    fft_ytrue = np.abs(fft(y_true_full.values))
    fft_yhat = np.abs(fft(y_hat_full))

    f_true, Pxx_den_true = welch(y_true_full.values, fs=20)
    f_hat, Pxx_den_hat = welch(y_hat_full, fs=20)
    f_resid, Pxx_den_resid = welch(y_true_full - y_hat_full, fs=20)

    y_lim_min = np.min([*Pxx_den_true, *Pxx_den_hat, *Pxx_den_resid])
    y_lim_max = np.max([*Pxx_den_true, *Pxx_den_hat, *Pxx_den_resid])
    y_lims = (y_lim_min, y_lim_max)

    x_lim_min = np.min([*f_true, *f_hat, *f_resid])
    x_lim_max = np.max([*f_true, *f_hat, *f_resid])
    x_lims = (x_lim_min, x_lim_max)

    fig, ax = plt.subplots(1,3)
    fig.suptitle('Welch Power Spectra of Response / Reconstruction — All Data, Excluding ITI')
    fig.set_figheight(20)
    fig.set_figwidth(40)


    ax[0].semilogy(f_true, Pxx_den_true)
    # plt.ylim([0.5e-3, 1])
    ax[0].set_xlabel('frequency [Hz]')
    ax[0].set_ylabel('PSD [V**2/Hz]')
    ax[0].set_ylim(y_lims)
    ax[0].set_title('Power Spectra of Raw Response')
    ax[0].grid()

    ax[1].semilogy(f_hat, Pxx_den_hat)
    ax[1].set_xlabel('frequency [Hz]')
    ax[1].set_ylabel('PSD [V**2/Hz]')
    ax[1].set_ylim(y_lims)
    ax[1].set_title('Power Spectra of Reconstructed Response')
    ax[1].grid()


    ax[2].semilogy(f_resid, Pxx_den_resid)
    ax[2].set_xlabel('frequency [Hz]')
    ax[2].set_ylabel('PSD [V**2/Hz]')
    ax[2].set_ylim(y_lims)
    ax[2].set_title('Power Spectra of (Raw - Reconstructed)')
    ax[2].grid()

    plt.show()

    fig.savefig('figure_outputs/spectral_out.png')

    return

def plot_single_avg_reconstruction(ci_setup_true, ci_setup_pred, ax, min_time, max_time, min_signal, max_signal, x_label, y_label, title):

    ax.plot(ci_setup_true['mean'], color='b')
    ax.fill_between(ci_setup_true.index, ci_setup_true['lb'], ci_setup_true['ub'], color='b', alpha=.2)

    ax.plot(ci_setup_pred['mean'], color='r')
    ax.set_xlim((min_time, max_time))
    ax.set_ylim((min_signal, max_signal))
    ax.title.set_text(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()

    return

def plot_avg_reconstructions(tmp_backup,
                             binsize = 50,
                             min_time = -20, max_time = 30,
                             min_signal = -3.0, max_signal = 3.0,
                             title='Average Photometry Response Aligned to Side Port Entry — Holdout Data Only',
                             file_name='figure_outputs/average_response_reconstruction.png'):
    """
    Plot the average reconstruction of the data
    Args:
        tmp_backup : pd.DataFrame
            Dataframe of the data
        binsize : int
            Size of the bins to average over
        min_time : int
            Minimum time to plot
        max_time : int
            Maximum time to plot
        min_signal : float
            Minimum signal to plot
        max_signal : float
            Maximum signal to plot
    """

    x_label = 'Timesteps __ from Event'
    y_label = 'Response'

    if binsize is not None:
        min_time *= binsize
        max_time *= binsize
        x_label = x_label.replace(' __', ' (ms)')
        tmp_backup['plot_time'] = tmp_backup['adjusted_time'] * binsize
    else:
        x_label = x_label.replace(' __', '')
        tmp_backup['plot_time'] = tmp_backup['adjusted_time']

    tmp = tmp_backup[tmp_backup['plot_time'].between(min_time, max_time)].copy()

    # plt.figure(figsize=(10,5))
    # fig, ax = plt.subplots(2,2)
    fig, ax = plt.subplots(3,2)
    fig.suptitle(title)
    fig.set_figheight(20)
    fig.set_figwidth(40)

    tmp['is_r_lpn_trial'] = sglm_ez.get_is_trial(tmp, ['nTrial'], ['r', 'lpn'])
    ci_setup_true = sglm_ez.get_sem(tmp, tmp['is_r_lpn_trial'], 'plot_time', 'zsgdFF')
    ci_setup_pred = sglm_ez.get_sem(tmp, tmp['is_r_lpn_trial'], 'plot_time', 'pred')
    plot_single_avg_reconstruction(ci_setup_true, ci_setup_pred, ax[0,0], min_time, max_time, min_signal, max_signal, x_label, y_label, 'Rewarded, Left Port Entry')


    tmp['is_r_rpn_trial'] = sglm_ez.get_is_trial(tmp, ['nTrial'], ['r', 'rpn'])
    ci_setup_true = sglm_ez.get_sem(tmp, tmp['is_r_rpn_trial'], 'plot_time', 'zsgdFF')
    ci_setup_pred = sglm_ez.get_sem(tmp, tmp['is_r_rpn_trial'], 'plot_time', 'pred')
    plot_single_avg_reconstruction(ci_setup_true, ci_setup_pred, ax[0,1], min_time, max_time, min_signal, max_signal, x_label, y_label, 'Rewarded, Right Port Entry')


    tmp['is_nr_lpn_trial'] = sglm_ez.get_is_trial(tmp, ['nTrial'], ['nr', 'lpn'])
    ci_setup_true = sglm_ez.get_sem(tmp, tmp['is_nr_lpn_trial'], 'plot_time', 'zsgdFF')
    ci_setup_pred = sglm_ez.get_sem(tmp, tmp['is_nr_lpn_trial'], 'plot_time', 'pred')
    plot_single_avg_reconstruction(ci_setup_true, ci_setup_pred, ax[1,0], min_time, max_time, min_signal, max_signal, x_label, y_label, 'Unrewarded, Left Port Entry')


    tmp['is_nr_rpn_trial'] = sglm_ez.get_is_trial(tmp, ['nTrial'], ['nr', 'rpn'])
    ci_setup_true = sglm_ez.get_sem(tmp, tmp['is_nr_rpn_trial'], 'plot_time', 'zsgdFF')
    ci_setup_pred = sglm_ez.get_sem(tmp, tmp['is_nr_rpn_trial'], 'plot_time', 'pred')
    plot_single_avg_reconstruction(ci_setup_true, ci_setup_pred, ax[1,1], min_time, max_time, min_signal, max_signal, x_label, y_label, 'Unrewarded, Right Port Entry')


    tmp['is_cpn_trial'] = sglm_ez.get_is_trial(tmp, ['nTrial'], ['cpn'])
    ci_setup_true = sglm_ez.get_sem(tmp, tmp['is_cpn_trial'], 'plot_time', 'zsgdFF')
    ci_setup_pred = sglm_ez.get_sem(tmp, tmp['is_cpn_trial'], 'plot_time', 'pred')
    plot_single_avg_reconstruction(ci_setup_true, ci_setup_pred, ax[2,0], min_time, max_time, min_signal, max_signal, x_label, y_label, 'Center Port Entry')

    ax[2,0].legend(['Mean Photometry Response',
                    'Predicted Photometry Response',
                    '95% SEM Confidence Interval'])
    
    plt.tight_layout()

    fig.savefig(file_name)

    return

