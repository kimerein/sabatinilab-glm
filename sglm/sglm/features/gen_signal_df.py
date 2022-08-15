import glob
import numpy as np
import pandas as pd
import numpy as np
from sglm.models import sglm
from sglm.features import table_file as tbf

# Break down Preprocess Lynne into component parts


# Generate AB Labels
def set_first_prv_trial_letter(prv_wasRewarded_series, label_series, loc=0):
    """
    Sets the first letter of the rewarded / unrewarded (Aa) labels from the previous trial.

    Parameters
    ----------
    prv_wasRewarded_series : Series
        Boolean series representing whether or not previous trial was rewarded.
    label_series : Series
        Series of string labels in which to replace the previous trial letter.
    loc : int
        Location of the first letter of the label.
    
    Returns
    -------
    label : Series
        Revised series of labels with the first letter of the label set based on the previous trial.
    """
    label = label_series.copy()
    label.loc[prv_wasRewarded_series] = label.loc[prv_wasRewarded_series].str.slice_replace(loc, loc+1, 'A')
    label.loc[~prv_wasRewarded_series] = label.loc[~prv_wasRewarded_series].str.slice_replace(loc, loc+1, 'a')
    return label

def set_current_trial_letter_switch(sameSide_series, wasRewarded_series, label_series, loc=1):
    """
    Sets a subsequent letter of the rewarded / unrewarded (Xx) / same / change side (Ab) labels from the previous trial.

    Parameters
    ----------
    sameSide_series : Series
        Boolean series representing whether or not the current trial is on the same side as the previous.
    wasRewarded_series : Series
        Boolean series representing whether or not current trial is rewarded.
    label_series : Series
        Series of string labels in which to replace the current trial letter.
    loc : int
        Location of the current letter of the label.

    Returns
    -------
    label : Series
        Revised series of labels with a later letter of the label set based on the trial.
    """
    label = label_series.copy()

    label.loc[(sameSide_series&wasRewarded_series)] = label.loc[(sameSide_series&wasRewarded_series)].str.slice_replace(loc, loc+1, 'A')
    label.loc[(~sameSide_series&wasRewarded_series)] = label.loc[(~sameSide_series&wasRewarded_series)].str.slice_replace(loc, loc+1, 'B')
    label.loc[(sameSide_series&~wasRewarded_series)] = label.loc[(sameSide_series&~wasRewarded_series)].str.slice_replace(loc, loc+1, 'a')
    label.loc[(~sameSide_series&~wasRewarded_series)] = label.loc[(~sameSide_series&~wasRewarded_series)].str.slice_replace(loc, loc+1, 'b')
    
    return label

def set_current_trial_letter_side(choseRight, wasRewarded_series, label_series, loc=2):
    """
    Sets a letter of the right / left (Rl) labels from a trial.

    Parameters
    ----------
    choseRight : Series
        Boolean series representing whether or not the current trial is a right sided selection.
    wasRewarded_series : Series
        Boolean series representing whether or not current trial is rewarded.
    label_series : Series
        Series of string labels in which to replace the current trial letter.
    loc : int
        Location of the current letter of the label.

    Returns
    -------
    label : Series
        Revised series of labels with a later letter of the label set based on the trial.
    """
    label = label_series.copy()

    label.loc[(choseRight&wasRewarded_series)] = label.loc[(choseRight&wasRewarded_series)].str.slice_replace(loc, loc+1, 'R')
    label.loc[(~choseRight&wasRewarded_series)] = label.loc[(~choseRight&wasRewarded_series)].str.slice_replace(loc, loc+1, 'L')
    label.loc[(choseRight&~wasRewarded_series)] = label.loc[(choseRight&~wasRewarded_series)].str.slice_replace(loc, loc+1, 'r')
    label.loc[(~choseRight&~wasRewarded_series)] = label.loc[(~choseRight&~wasRewarded_series)].str.slice_replace(loc, loc+1, 'l')
    
    return label

def check_Ab_labels(df_t):
    """
    Sanity checks the Ab labels generated are in the correct format.
    
    Parameters
    ----------
    df_t : DataFrame
        DataFrame that includes Ab values in label column. And values for 'prv_wasRewarded', 'wasRewarded', and 'sameSide'.
    """

    df_t['pwR'] = df_t['prv_wasRewarded'].astype(int)
    df_t['wR'] = df_t['wasRewarded'].astype(int)
    df_t['sS'] = df_t['sameSide'].astype(int)
    check = df_t[['pwR', 'wR', 'sS', 'label']].copy()
    check['val_a'] = df_t['label'].str.slice(0, 1).replace('a', 0).replace('A', 1) + df_t['label'].str.slice(1, 2).replace('b', 0).replace('B', 2).replace('a', 4).replace('A', 6)
    check['val_b'] = check['pwR'] + check['wR']*2 + check['sS']*4
    assert (check['val_a'] == check['val_b']).all()
    return

def generate_Ab_labels(df_t):
    """
    Generates the Ab labels for a given DataFrame.

    Parameters
    ----------
    df_t : DataFrame
        DataFrame that includes values for 'wasRewarded', 'choseRight', 'choseLeft'.
    
    Returns
    -------
    df_t : DataFrame
        DataFrame with Ab labels added to the label column.
    """
    df_t = df_t.copy()
    
    df_t['wasRewarded'] = df_t['wasRewarded'].astype(bool)
    df_t['prv_wasRewarded'] = df_t['wasRewarded'].shift(1).astype(bool)
    df_t['prv_choseLeft'] = df_t['choseLeft'].shift(1).astype(bool)
    df_t['prv_choseRight'] = df_t['choseRight'].shift(1).astype(bool)
    
    df_t['sameSide'] = ((df_t['choseLeft'] == df_t['prv_choseLeft'])&(df_t['choseRight'] == df_t['prv_choseRight'])).astype(bool).fillna(False)
    df_t['label'] = '  '

    df_t['label'] = set_first_prv_trial_letter(df_t['prv_wasRewarded'], df_t['label'], loc=0)
    df_t['label'] = set_current_trial_letter_switch(df_t['sameSide'], df_t['wasRewarded'], df_t['label'], loc=1)

    df_t['label_side'] = '  '
    df_t['label_side'] = set_current_trial_letter_side(df_t['prv_choseRight'], df_t['prv_wasRewarded'], df_t['label_side'], loc=0)
    df_t['label_side'] = set_current_trial_letter_side(df_t['choseRight'], df_t['wasRewarded'], df_t['label_side'], loc=1)

    df_t['label_rewarded'] = ' '
    df_t['label_rewarded'] = set_first_prv_trial_letter(df_t['wasRewarded'], df_t['label_rewarded'], loc=0)

    df_t['wasRewarded'] = df_t['wasRewarded'].fillna(False).astype(int)
    df_t['prv_wasRewarded'] = df_t['prv_wasRewarded'].fillna(False).astype(int)
    df_t['prv_choseLeft'] = df_t['prv_choseLeft'].astype(int)
    df_t['prv_choseRight'] = df_t['prv_choseRight'].astype(int)

    df_t.loc[df_t['prv_wasRewarded'].isna(), 'label'] = np.nan
    df_t.loc[df_t['prv_wasRewarded'].isna(), 'label'] = np.nan
    df_t = df_t.dropna()

    check_Ab_labels(df_t)

    return df_t

def replace_missed_center_out_indexes(df_t, max_num_duplications=None, verbose=0):
    """
    Replaces the indexes of trials where the detector missed the center out behavior with the center in behavior.

    Parameters
    ----------
    df_t : DataFrame
        DataFrame that includes values for 'hasAllPhotometryData', 'photometryCenterInIndex', 'photometryCenterOutIndex'.
    max_num_duplications : int
        Maximum number of times to check duplications after iterating back CenterInIndices = CenterOutIndices. None means no limit.
    verbose : int
        Verbosity level.

    Returns
    -------
    df_t : DataFrame
        DataFrame with the center out behavior indices replaced.
    """
    df_t = df_t.copy()

    i = 0

    while True:
        
        num_inx_vals = df_t[df_t['photometryCenterOutIndex'] > 0].groupby('photometryCenterOutIndex')['hasAllPhotometryData'].count()
        # print(_, num_inx_vals.max())

        duplicated_CO_inx = (df_t['photometryCenterOutIndex'] == df_t['photometryCenterOutIndex'].shift(-1))&(df_t['photometryCenterOutIndex'].shift(-1) > 0)&(df_t['photometryCenterOutIndex'] > 0)
        duplicated_CO_inx2 = (df_t['photometryCenterOutIndex'] > df_t['photometryCenterOutIndex'].shift(-1))&(df_t['photometryCenterOutIndex'].shift(-1) > 0)&(df_t['photometryCenterOutIndex'] > 0)
        
        # if (duplicated_CO_inx|duplicated_CO_inx2).sum() > 0:
        #     key = df_t.loc[duplicated_CO_inx|duplicated_CO_inx2].index[0]
        #     print(key)
        #     lb, ub = key-5, key+5
        #     display(df_t.loc[lb:ub])
        
        #     if i > 10:
        #         raise ValueError('test')

        df_t.loc[duplicated_CO_inx, 'photometryCenterOutIndex'] = df_t.loc[duplicated_CO_inx, 'photometryCenterInIndex']
        df_t.loc[duplicated_CO_inx2, 'photometryCenterOutIndex'] = df_t.loc[duplicated_CO_inx2, 'photometryCenterInIndex']
        
        if num_inx_vals.max() == 1 and (duplicated_CO_inx|duplicated_CO_inx2).sum() == 0:
            break
        if max_num_duplications and i > max_num_duplications:
            break
        i += 1

    if verbose > 0:
        print('# of iterations', i,'— Final max amount of duplicated Center Out Indices:', num_inx_vals.max())
    
    return df_t

def get_is_relevant_trial(hasAllData_srs, index_event_srs):
    """
    Returns a Series of booleans indicating whether a trial is relevant (i.e. whether it has all data and has a non-zero-value).

    Parameters
    ----------
    hasAllData_srs : Series
        Series of booleans indicating whether a trial has all data.
    index_event_srs : Series
        Series of booleans indicating whether a trial has a non-zero-value for the index event.
    """
    return (hasAllData_srs > 0)&(index_event_srs >= 0)

def matlab_indexing_to_python(index_event_srs):
    """
    Converts a Series of matlab-indexed values to python-indexed values.

    Parameters
    ----------
    index_event_srs : Series
        Series of matlab-indexed (i.e. 1-indexed) values.

    Returns
    -------
    index_event_srs : Series
        Series of python-indexed values.
    """
    return index_event_srs - 1

def get_is_not_iti(df):
    """
    Returns a boolean array of whether the trial is not ITI
    Args:
        df: dataframe with entry, exit, lick, reward, and dFF columns
    Returns:
        boolean array of whether the trial is not ITI
    """
    return df['nTrial'] != df['nEndTrial']

def get_trial_start(center_in_srs):
    """
    Returns the index of the first center in event.

    Parameters
    ----------
    center_in_srs : Series
        Series of center in events.

    Returns
    -------
    trial_start : int
        Index of the first center in event where it is non-nan.
    """
    return (((~center_in_srs.isna())&(center_in_srs==1))*1)
    
def get_trial_end(center_out_srs):
    """
    Returns the index of the last center out event.

    Parameters
    ----------
    center_out_srs : Series
        Series of center out events.

    Returns
    -------
    trial_end : int
        Index of the last center out event where it is non-nan.
    """
    return (((~center_out_srs.isna())&(center_out_srs==1))*1)

# Rename Columns
## * File-Specific Columns
## * General Columns

# # Add Missing Y-Columns to File
# for y_col in y_col_lst_all:
#             if y_col not in df.columns:
#                 df[y_col] = np.nan
#                 continue
#             if 'SGP_' == y_col[:len('SGP_')]:
#                 df[y_col] = df[y_col].replace(0, np.nan)
#             if df[y_col].std() >= 90:
#                 df[y_col] /= 100


# Combine Signal & Table Files
## * Define trial starts / ends
## * Get is not ITI
## * Define rewarded/unrewarded trials (across entire trial)
## * Define side-agnostic events
## * Get first-time events
## * Generate AB Labels

# Detrend Data


# Timeshift value predictors


# Split Data


# Fit / cross-validate


# Plot Results


# Evaluate Results


# Save Results


def generate_signal_df(signal_filename, table_filename,
                        #   signal_filename_out=None, table_filename_out=None, 
                        table_index_columns = ['photometryCenterInIndex',
                                                'photometryCenterOutIndex',
                                                'photometrySideInIndex',
                                                'photometrySideOutIndex',
                                                'photometryFirstLickIndex'
                                                ],
                        basis_Aa_cols = ['AA', 'Aa', 'aA', 'aa', 'AB', 'Ab', 'aB', 'ab'],
                        trial_bounds_before_center_in = -20,
                        trial_bounds_after_side_out = 20,

                  ):
    """
    Generates a DataFrame from a signal and table file.

    Parameters
    ----------
    signal_filename : str
        Path to the signal file.
    table_filename : str
        Path to the table file.
    signal_filename_out : str
        Path to the interim output signal file.
    table_filename_out : str
        Path to the interim output table file.
    table_index_columns : list of str
        Index column names in the table file that are used to index the signal file and create the Ab columns.
    basis_Aa_cols : list of str
        Suffixes in the table file that are combined with the table_index_columns used to index the signal file.
    trial_bounds_before_center_in : int
        Number of trial timesteps to include before the center in event as within trial.
    trial_bounds_after_side_out : int
        Number of trial timesteps to include after the side out event as within trial.
    
    Returns
    -------
    signal_df : DataFrame
        DataFrame with the signal data.
    table_df : DataFrame
        DataFrame with the table data.
    """
    # Load Signal Data
    signal_df = pd.read_csv(signal_filename)

    # Load Table Data
    table_df = pd.read_csv(table_filename)

    # Generate Ab Labels
    df_t = generate_Ab_labels(table_df)
    assert np.all(df_t['label'].dropna() == df_t['word'].dropna())
    # print(df_t[['label', 'word']])

    # Convert Ab Labels to Indicator Variables
    ab_dummies = pd.get_dummies(df_t['label'])
    for basis_col in basis_Aa_cols:
        if basis_col not in ab_dummies.columns:
            df_t[basis_col] = 0
    df_t[ab_dummies.columns] = ab_dummies

    # Convert MATLAB 1 indexing to python 0 indexing
    df_t[table_index_columns] = matlab_indexing_to_python(df_t[table_index_columns])

    # Replace center outs that got transferred to the next timestep to be equal to center in value
    df_t = replace_missed_center_out_indexes(df_t, verbose=1)

    # 
    for col in table_index_columns:
        df_t_tmp = df_t[get_is_relevant_trial(df_t['hasAllPhotometryData'], df_t[col])].copy()
        assert len([_ for _ in df_t_tmp.columns if col == _]) == 1, f"Error: Duplicate entries for {col}"
        assert signal_df.index.nunique() == len(signal_df.index), f"Error: Duplicate entries in signal_df.index"
        # assert signal_df.index.nunique() == len(signal_df.index), f"Error: Duplicate entries in signal_df.index"

        # print(col)
        # print('>', (df_t_tmp.set_index(col)['wasRewarded'] == df_t_tmp.set_index(col)['wasRewarded']*1).index)
        # print('> len', len(signal_df.index))
        # print('> nunique', signal_df.index.nunique())
        # print('> len b', len((df_t_tmp.set_index(col)['wasRewarded'] == df_t_tmp.set_index(col)['wasRewarded']*1).index))
        # print('> nu b', )

        # stop_flg = False

        # eq = (df_t_tmp.set_index(col)['wasRewarded'] == df_t_tmp.set_index(col)['wasRewarded']*1)
        # for e in eq.index:
        #     if len(eq.loc[[e]]) > 1:
        #         display(eq.loc[e])
        #         stop_flg = True

        # If it's not nan and it shows up in the table set it to 1 in signal df, else 0.
        signal_df[col] = (df_t_tmp.set_index(col)['wasRewarded'] == df_t_tmp.set_index(col)['wasRewarded'])*1

        # Set r to if it was rewarded.
        signal_df[f'{col}r'] = df_t_tmp.set_index(col)['wasRewarded']

        # Set nr to 1 - r
        signal_df[f'{col}nr'] = (1 - signal_df[f'{col}r'])

        # Set side in / side out word values in signal to associated values in table. Else 0 if not in table.
        if col in ['photometrySideInIndex', 'photometrySideOutIndex']: #, 'photometryCenterInIndex']:
            for basis in basis_Aa_cols:
                signal_df[col+basis] = df_t_tmp.set_index(col)[basis].fillna(0)
    
    # Trial number is the number of Center In values that have been observed shifted backwards by trial_bounds_before_center_in
    signal_df['nTrial'] = get_trial_start(signal_df['photometryCenterInIndex']).cumsum().shift(trial_bounds_before_center_in)
    # Trial end number is the number of Side Out values that have been observed shifted forwards by trial_bounds_after_side_out
    signal_df['nEndTrial'] = get_trial_end(signal_df['photometrySideOutIndex']).cumsum().shift(trial_bounds_after_side_out)
    # Difference between these two is used to indicate trial overlaps by time shifts
    signal_df['diffTrialNums'] = signal_df['nTrial'] - signal_df['nEndTrial']
    signal_df['dupe'] = False

    signal_df_with_dupes = []

    for nTrial_num in signal_df['nTrial'].unique():
        setup_df_nT = signal_df[signal_df['nTrial'] == nTrial_num]
        dupe_data = setup_df_nT[setup_df_nT['diffTrialNums'] > 1]

        if len(dupe_data) > 0:
            dupe_data = dupe_data.copy()
            dupe_data['nTrial'] = dupe_data['nTrial'] - 1
            dupe_data['dupe'] = True
            signal_df_with_dupes.append(dupe_data)

        signal_df_with_dupes.append(setup_df_nT)


        # if len(dupe_data) > 0:
        #     with pd.option_context('display.max_rows', 1000):
        #         tsd = pd.concat(signal_df_with_dupes, axis=0)
        #         display(tsd[tsd['diffTrialNums'] >= 1].tail(1000))
        #     return None

    signal_df = pd.concat(signal_df_with_dupes, axis=0)



    signal_df['wi_trial_keep'] = get_is_not_iti(signal_df)

    # signal_df = signal_df[signal_df['nTrial'] > 0].fillna(0)
    # if signal_filename_out:
    #     signal_df.to_csv(signal_filename_out)
    # if table_filename_out:
    #     table_df.to_csv(table_filename_out)

    return signal_df, table_df

# # signal_df['spnr'] = ((df['spnr'] == 1)&(df['photometrySideInIndex'] != 1)).astype(int)
# signal_df['spnnr'] = ((signal_df['spnnr'] == 1)&(signal_df['photometrySideInIndex'] != 1)).astype(int)
# # signal_df['spxr'] = ((df['spxr'] == 1)&(df['photometrySideOutIndex'] != 1)).astype(int)
# signal_df['spxnr'] = ((signal_df['spxnr'] == 1)&(signal_df['photometrySideOutIndex'] != 1)).astype(int)

