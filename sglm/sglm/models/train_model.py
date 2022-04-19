def get_x(df, x_cols, keep_rows=None):
    '''
    Get x values for fitting/scoring
    Args:
        df: dataframe that includes x_cols
        x_cols: list of column names to include in prediction
        keep_rows: boolean array of which rows to keep
    Returns:
        df[x_cols]: dataframe only including prediction columns and keep_rows
    '''
    if type(keep_rows) != type(None):
        df = df[keep_rows]
    return df[x_cols]

def get_y(df, y_col, keep_rows=None):
    '''
    Get y values for fitting/scoring
    Args:
        df: dataframe that includes x_cols
        y_cols: column name to use for response
        keep_rows: boolean array of which rows to keep
    Returns:
        df[y_col]: dataframe only including response column and keep_rows
    '''
    if type(keep_rows) != type(None):
        df = df[keep_rows]
    return df[y_col]

def holdout_splits(dfrel_setup, id_cols=['nTrial'], perc_holdout=0.2):
    '''
    Create holdout splits
    Args:
        dfrel_setup: full setup dataframe
        id_cols: list of columns to use as trial identifiers
        perc_holdout: percentage of data to holdout
    Returns:
        dfrel_setup: full setup dataframe
        dfrel_holdout: full holdout dataframe
    '''
    # Create holdout splits
    holdout = sglm_ez.holdout_split_by_trial_id(dfrel_setup, id_cols=id_cols, perc_holdout=perc_holdout)
    dfrel_holdout = dfrel_setup.loc[holdout]
    dfrel_setup = dfrel_setup.loc[~holdout]
    return dfrel_setup, dfrel_holdout

def setup_glmsave(glmsave, prefix, filename, neg_order, pos_order, X_cols_all, folds, pholdout, pgss, gssid=None):
    glmsave.set_uid(prefix)
    glmsave.set_filename(filename)
    glmsave.set_timeshifts(neg_order, pos_order)
    glmsave.set_X_cols(X_cols_all)
    glmsave.set_gss_info(folds, pholdout, pgss, gssid=None)
    return

def generate_Ab_labels(df_t):
    df_t = df_t.copy()
    df_t['prv_choseLeft'] = df_t['choseLeft'].shift(1)
    df_t['prv_choseRight'] = df_t['choseRight'].shift(1)
    df_t['prv_wasRewarded'] = df_t['wasRewarded'].shift(1)

    df_t['label_1Aa'] = df_t['prv_wasRewarded'].astype(bool).fillna(False)
    df_t['label_2AB'] = ((df_t['choseLeft'] == df_t['prv_choseLeft']) & (df_t['choseRight'] == df_t['prv_choseRight'])).astype(bool).fillna(False)
    df_t['label_2Aa'] = df_t['wasRewarded'].astype(bool).fillna(False)

    df_t['label'] = '  '

    df_t.loc[df_t['label_1Aa'], 'label'] = df_t.loc[df_t['label_1Aa'], 'label'].str.slice_replace(0, 1, 'A')
    df_t.loc[~df_t['label_1Aa'], 'label'] = df_t.loc[~df_t['label_1Aa'], 'label'].str.slice_replace(0, 1, 'a')

    df_t.loc[(df_t['label_2AB']&df_t['label_2Aa']), 'label'] = df_t.loc[(df_t['label_2AB']&df_t['label_2Aa']), 'label'].str.slice_replace(1, 2, 'A')
    df_t.loc[(~df_t['label_2AB']&df_t['label_2Aa']), 'label'] = df_t.loc[(~df_t['label_2AB']&df_t['label_2Aa']), 'label'].str.slice_replace(1, 2, 'B')
    df_t.loc[(df_t['label_2AB']&~df_t['label_2Aa']), 'label'] = df_t.loc[(df_t['label_2AB']&~df_t['label_2Aa']), 'label'].str.slice_replace(1, 2, 'a')
    df_t.loc[(~df_t['label_2AB']&~df_t['label_2Aa']), 'label'] = df_t.loc[(~df_t['label_2AB']&~df_t['label_2Aa']), 'label'].str.slice_replace(1, 2, 'b')

    df_t.loc[df_t['prv_wasRewarded'].isna(), 'label'] = np.nan

    return df_t
