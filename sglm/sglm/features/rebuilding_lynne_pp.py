import glob
import numpy as np
import pandas as pd
import numpy as np
from sglm.models import sglm
from sglm.features import table_file as tbf

# Break down Preprocess Lynne into component parts


# Generate AB Labels
def set_first_prv_trial_letter(prv_wasRewarded_series, label_series, loc=0):
    label = label_series.copy()
    label.loc[prv_wasRewarded_series] = label.loc[prv_wasRewarded_series].str.slice_replace(loc, loc+1, 'A')
    label.loc[~prv_wasRewarded_series] = label.loc[~prv_wasRewarded_series].str.slice_replace(loc, loc+1, 'a')
    return label

def set_current_trial_letter_switch(sameSide_series, wasRewarded_series, label_series, loc=1):
    label = label_series.copy()

    label.loc[(sameSide_series&wasRewarded_series)] = label.loc[(sameSide_series&wasRewarded_series)].str.slice_replace(loc, loc+1, 'A')
    label.loc[(~sameSide_series&wasRewarded_series)] = label.loc[(~sameSide_series&wasRewarded_series)].str.slice_replace(loc, loc+1, 'B')
    label.loc[(sameSide_series&~wasRewarded_series)] = label.loc[(sameSide_series&~wasRewarded_series)].str.slice_replace(loc, loc+1, 'a')
    label.loc[(~sameSide_series&~wasRewarded_series)] = label.loc[(~sameSide_series&~wasRewarded_series)].str.slice_replace(loc, loc+1, 'b')
    
    return label

def set_current_trial_letter_side(choseRight, wasRewarded_series, label_series, loc=2):
    label = label_series.copy()

    label.loc[(choseRight&wasRewarded_series)] = label.loc[(choseRight&wasRewarded_series)].str.slice_replace(loc, loc+1, 'R')
    label.loc[(~choseRight&wasRewarded_series)] = label.loc[(~choseRight&wasRewarded_series)].str.slice_replace(loc, loc+1, 'L')
    label.loc[(choseRight&~wasRewarded_series)] = label.loc[(choseRight&~wasRewarded_series)].str.slice_replace(loc, loc+1, 'r')
    label.loc[(~choseRight&~wasRewarded_series)] = label.loc[(~choseRight&~wasRewarded_series)].str.slice_replace(loc, loc+1, 'l')
    
    return label

def check_Ab_labels(df_t):
    df_t['pwR'] = df_t['prv_wasRewarded'].astype(int)
    df_t['wR'] = df_t['wasRewarded'].astype(int)
    df_t['sS'] = df_t['sameSide'].astype(int)
    check = df_t[['pwR', 'wR', 'sS', 'label']].copy()
    check['val_a'] = df_t['label'].str.slice(0, 1).replace('a', 0).replace('A', 1) + df_t['label'].str.slice(1, 2).replace('b', 0).replace('B', 2).replace('a', 4).replace('A', 6)
    check['val_b'] = check['pwR'] + check['wR']*2 + check['sS']*4
    assert (check['val_a'] == check['val_b']).all()
    return

def generate_Ab_labels(df_t):
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

    df_t['wasRewarded'] = df_t['wasRewarded'].fillna(False)
    df_t['prv_wasRewarded'] = df_t['prv_wasRewarded'].fillna(False)

    df_t.loc[df_t['prv_wasRewarded'].isna(), 'label'] = np.nan
    df_t.loc[df_t['prv_wasRewarded'].isna(), 'label'] = np.nan
    df_t = df_t.dropna()

    check_Ab_labels(df_t)

    return df_t

def replace_missed_center_out_indexes(df_t, max_num_duplications=None, verbose=0):
    df_t = df_t.copy()

    # num_inx_vals = df_t.groupby('photometryCenterOutIndex')['hasAllPhotometryData'].count()
    # if len(num_inx_vals) == 0:
    #     return
    # num_inx_vals = df_t.groupby('photometryCenterOutIndex')['hasAllPhotometryData'].transform(np.size)
    # reps = df_t[num_inx_vals > 1].copy()
    # reps['cin_out_delta'] = reps['photometryCenterOutIndex'] - reps['photometryCenterInIndex']
    # overwrite_inx = reps[reps['cin_out_delta'] != reps.groupby('photometryCenterOutIndex')['cin_out_delta'].transform(lambda x: np.min(np.abs(x)))].index
    # df_t.loc[overwrite_inx, 'photometryCenterOutIndex'] = df_t.loc[overwrite_inx, 'photometryCenterInIndex']

    i = 0

    while True:
        
        num_inx_vals = df_t[df_t['photometryCenterOutIndex'] > 0].groupby('photometryCenterOutIndex')['hasAllPhotometryData'].count()
        # print(_, num_inx_vals.max())
        if num_inx_vals.max() == 1:
            break
        duplicated_CO_inx = df_t['photometryCenterOutIndex'] == df_t['photometryCenterOutIndex'].shift(-1)
        df_t.loc[duplicated_CO_inx, 'photometryCenterOutIndex'] = df_t.loc[duplicated_CO_inx, 'photometryCenterInIndex']

        if max_num_duplications and i > max_num_duplications:
            break
        i += 1

    if verbose > 0:
        print('# of iterations', i,'— Final max amount of duplicated Center Out Indices:', num_inx_vals.max())
    
    return df_t

def get_is_relevant_trial(hasAllData_srs, index_event_srs):
    return (hasAllData_srs > 0)&(index_event_srs >= 0)

def matlab_indexing_to_python(index_event_srs):
    return index_event_srs - 1



# review = df_t[((df_t['photometryCenterOutIndex'].shift(1) == df_t['photometryCenterOutIndex'])|(df_t['photometryCenterOutIndex'].shift(-1) == df_t['photometryCenterOutIndex']))&(df_t['hasAllPhotometryData'] > 0)]
# review
# review_2 = df_t[((df_t['photometryCenterOutIndex'].shift(1) == df_t['photometryCenterOutIndex'])|(df_t['photometryCenterOutIndex'].shift(-1) == df_t['photometryCenterOutIndex']))&(df_t['hasAllPhotometryData'] > 0)]
# review
# (df_t.groupby('photometryCenterOutIndex')['hasAllPhotometryData'].count() >= 2).sum()
# # tmp_a = df_t[((df_t['photometryCenterOutIndex'].shift(1) == df_t['photometryCenterOutIndex'])|(df_t['photometryCenterOutIndex'].shift(-1) == df_t['photometryCenterOutIndex']))&(df_t['hasAllPhotometryData'] > 0)]
# tmp_inx = df_t[df_t['photometryCenterOutIndex'] > -1].copy()
# num_inx_vals = tmp_inx.groupby('photometryCenterOutIndex')['hasAllPhotometryData'].transform(np.size)
# resp = tmp_inx[num_inx_vals > 1].copy()
# tmp_inx['cin_out_delta'] = tmp_inx['photometryCenterOutIndex'] - tmp_inx['photometryCenterInIndex']
# overwrite_inx = tmp_inx[tmp_inx['cin_out_delta'] != tmp_inx.groupby('photometryCenterOutIndex')['cin_out_delta'].transform(np.min)].index
# df_t.loc[overwrite_inx, 'photometryCenterOutIndex'] = df_t.loc[overwrite_inx, 'photometryCenterInIndex']
# df_t[((df_t['photometryCenterOutIndex'].shift(1) == df_t['photometryCenterOutIndex'])|(df_t['photometryCenterOutIndex'].shift(-1) == df_t['photometryCenterOutIndex']))&(df_t['hasAllPhotometryData'] > 0)]
# # tmp = df_t[(df_t['photometryCenterOutIndex'].shift(1) == df_t['photometryCenterOutIndex'])|(df_t['photometryCenterOutIndex'].shift(-1) == df_t['photometryCenterOutIndex'])]
# tmp = df_t[df_t['photometryCenterOutIndex'] > -1].copy()

# display(tmp)

# num_inx_vals = tmp.groupby('photometryCenterOutIndex')['hasAllPhotometryData'].count()
# print((num_inx_vals > 1).sum())
# num_inx_vals = tmp.groupby('photometryCenterOutIndex')['hasAllPhotometryData'].transform(np.size)
# reps = tmp[num_inx_vals > 1].copy()
# reps['cin_out_delta'] = reps['photometryCenterOutIndex'] - reps['photometryCenterInIndex']
# overwrite_inx = reps[reps['cin_out_delta'] != reps.groupby('photometryCenterOutIndex')['cin_out_delta'].transform(np.min)].index
# tmp.loc[overwrite_inx, 'photometryCenterOutIndex'] = tmp.loc[overwrite_inx, 'photometryCenterInIndex']

# num_inx_vals = tmp.groupby('photometryCenterOutIndex')['hasAllPhotometryData'].count()
# print((num_inx_vals > 1).sum())

# # display(tmp)

# tmp[(tmp['photometryCenterOutIndex'].shift(1) == tmp['photometryCenterOutIndex'])|(tmp['photometryCenterOutIndex'].shift(-1) == tmp['photometryCenterOutIndex'])]
# reps[reps['cin_out_delta'] != reps.groupby('photometryCenterOutIndex')['cin_out_delta'].transform(np.min)]
# # tmp = df_t[(df_t['photometryCenterOutIndex'].shift(1) == df_t['photometryCenterOutIndex'])|(df_t['photometryCenterOutIndex'].shift(-1) == df_t['photometryCenterOutIndex'])]
# tmp = tmp[tmp['photometryCenterOutIndex'] > -1]
# tmp
# # def check_srs_val_is_nan(srs):
    # return srs.isna().any()

# signal_df[col] = (df_t_tmp[df_t_tmp[col].isin(single_inx_vals)].set_index(col)['wasRewarded'] == df_t_tmp[df_t_tmp[col].isin(single_inx_vals)].set_index(col)['wasRewarded'])*1

# for col in table_index_columns:
#     print(((df_t.set_index(col)['wasRewarded'] == df_t.set_index(col)['wasRewarded'])*1).sum())
# # df_t_tmp.columns
# # df_t_tmp.set_index(col)['wasRewarded'] == df_t_tmp.set_index(col)['wasRewarded']
# len(np.unique(signal_df.columns)), len(signal_df.columns)
# (df_t_tmp.set_index(col)['wasRewarded'] == df_t_tmp.set_index(col)['wasRewarded'])*1
# # len(np.unique(df_t_tmp.set_index('photometryCenterOutIndex')['wasRewarded'].index)), len((df_t_tmp.set_index('photometryCenterOutIndex')['wasRewarded'].index))
# # df_t_tmp[(df_t_tmp['photometryCenterOutIndex'] == df_t_tmp['photometryCenterOutIndex'].shift(1))|(df_t_tmp['photometryCenterOutIndex'] == df_t_tmp['photometryCenterOutIndex'].shift(-1))]

def get_is_not_iti(df):
    '''
    Returns a boolean array of whether the trial is not ITI
    Args:
        df: dataframe with entry, exit, lick, reward, and dFF columns
    Returns:
        boolean array of whether the trial is not ITI
    '''
    return df['nTrial'] != df['nEndTrial']

def get_trial_start(center_in_srs):
    return (((~center_in_srs.isna())&(center_in_srs==1))*1)
    
def get_trial_end(center_out_srs):
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


def get_signal_df(signal_filename, table_filename,
                  table_index_columns = ['photometryCenterInIndex',
                                        'photometryCenterOutIndex',
                                        'photometrySideInIndex',
                                        'photometrySideOutIndex',
                                        'photometryFirstLickIndex',],
                  basis_Aa_cols = ['AA', 'Aa', 'aA', 'aa', 'AB', 'Ab', 'aB', 'ab'],

                  ):
    # # Load Signal Data
    # signal_file = glob.glob(f'../data/raw/GLM_SIGNALS_WT61_*')[0]
    # signal_df = pd.read_csv(signal_file)

    # # Load Table Data
    # table_file = glob.glob(f'../data/raw/GLM_SIGNALS_WT61_*')[0].replace('GLM_SIGNALS', 'GLM_TABLE')
    # table_df = pd.read_csv(table_file)

    # Load Signal Data
    signal_df = pd.read_csv(signal_filename)

    # Load Table Data
    table_df = pd.read_csv(table_filename)

    
    df_t = generate_Ab_labels(table_df)
    ab_dummies = pd.get_dummies(df_t['label'])
    for basis_col in basis_Aa_cols:
        if basis_col not in ab_dummies.columns:
            df_t[basis_col] = 0
    df_t[ab_dummies.columns] = ab_dummies


    df_t = replace_missed_center_out_indexes(df_t, verbose=1)
    df_t[table_index_columns] = matlab_indexing_to_python(df_t[table_index_columns])

    for col in table_index_columns:
        df_t_tmp = df_t[get_is_relevant_trial(df_t['hasAllPhotometryData'], df_t[col])].copy()
        # relevant_inx = df_t_tmp[col]
        
        # single_inx_vals = num_inx_vals[num_inx_vals == 1].index
        # signal_df[col] = (df_t_tmp[df_t_tmp[col].isin(single_inx_vals)].set_index(col)['wasRewarded'] == df_t_tmp[df_t_tmp[col].isin(single_inx_vals)].set_index(col)['wasRewarded'])*1
        
        signal_df[col] = (df_t_tmp.set_index(col)['wasRewarded'] == df_t_tmp.set_index(col)['wasRewarded'])*1
        signal_df[f'{col}r'] = df_t_tmp.set_index(col)['wasRewarded']
        signal_df[f'{col}nr'] = (1 - signal_df[f'{col}r'])

        if col in ['photometrySideInIndex', 'photometrySideOutIndex']: #, 'photometryCenterInIndex']:
            for basis in basis_Aa_cols:
                signal_df[col+basis] = df_t_tmp.set_index(col)[basis].fillna(0)

    signal_df['nTrial'] = get_trial_start(signal_df['photometryCenterInIndex']).cumsum().shift(-5)
    signal_df['nEndTrial'] = get_trial_end(signal_df['photometrySideOutIndex']).cumsum().shift(5)
    signal_df['wi_trial_keep'] = get_is_not_iti(signal_df)

    signal_df = signal_df[signal_df['nTrial'] > 0].fillna(0)
    # signal_df

    return signal_df, table_df

# # signal_df['spnr'] = ((df['spnr'] == 1)&(df['photometrySideInIndex'] != 1)).astype(int)
# signal_df['spnnr'] = ((signal_df['spnnr'] == 1)&(signal_df['photometrySideInIndex'] != 1)).astype(int)
# # signal_df['spxr'] = ((df['spxr'] == 1)&(df['photometrySideOutIndex'] != 1)).astype(int)
# signal_df['spxnr'] = ((signal_df['spxnr'] == 1)&(signal_df['photometrySideOutIndex'] != 1)).astype(int)

