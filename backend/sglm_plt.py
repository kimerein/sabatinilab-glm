


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
    if y_lims is None:
        y_lims = (glm.coef_.min(), glm.coef_.max())
    print(y_lims)

    coef_lookup = {sftd_coef_names[i]:glm.coef_[i] for i in range(len(sftd_coef_names))}
    coef_cols = get_coef_name_sets(coef_names, sftd_coef_names)
    
    fig, axs = plt.subplots(len(coef_cols)//plot_width + (len(coef_cols)%plot_width > 0)*1, plot_width)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    addl_plot_name = ' — ' + plot_name if plot_name else ''
    fig.suptitle(f'Feature Coefficients by Timeshift{addl_plot_name}', fontsize=20)

    for icn, coef_name in enumerate(coef_cols):
        print(icn)
        timeshifts, coefs = get_single_coef_set(coef_cols[coef_name], coef_lookup)
        
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


def plot_power_spectra():
    
    return


def plot_avg_reconstruction():
    
    return