import pytest

import sys
import os 
dir_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/backend')
sys.path.append(f'{dir_path}/../backend')

import sglm_pp
import numpy as np
import pandas as pd

def get_dummy_np():
    return np.arange(20).reshape(5,4)

def get_dummy_pd():
    return pd.DataFrame(np.arange(20).reshape(5,4), columns=['A','B','C','D'])

def test_unshifted():
    dummy_np_data = get_dummy_np()
    comparison_1 = dummy_np_data
    comparison_2 = dummy_np_data[:, [0, 3]]

    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_amt=0) == comparison_1))
    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,3],
                                    shift_amt=0) == comparison_2))

    dummy_pd_data = get_dummy_pd()
    comparison_1_pd = pd.DataFrame(comparison_1, columns=['A','B','C','D'])
    comparison_2_pd = pd.DataFrame(comparison_2, columns=['A','D'])
    
    inx_list = sglm_pp.get_column_nums(dummy_pd_data, ['A', 'D'])
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_amt=0) == comparison_1_pd))
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list,
                                    shift_amt=0) == comparison_2_pd))
    return

def test_forward_shift():
    dummy_np_data = get_dummy_np()
    empty_data = np.zeros((1, dummy_np_data.shape[-1]))
    fwd_np_data = np.concatenate([empty_data, dummy_np_data], axis=0)[:-1]

    comparison_1 = fwd_np_data
    comparison_2 = fwd_np_data[:, [0, 3]]
    
    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_amt=1, fill_value=0) == comparison_1))
    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,3], shift_amt=1,
                                    fill_value=0) == comparison_2))

    dummy_pd_data = get_dummy_pd()
    comparison_1_pd = pd.DataFrame(comparison_1, columns=['A','B','C','D'])
    comparison_2_pd = pd.DataFrame(comparison_2, columns=['A','D'])
    
    inx_list = sglm_pp.get_column_nums(dummy_pd_data, ['A', 'D'])
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_amt=1,
                                    fill_value=0) == comparison_1_pd))
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list, shift_amt=1,
                                    fill_value=0) == comparison_2_pd))
    return

def test_backward_shift():
    dummy_np_data = get_dummy_np()
    empty_data = np.zeros((1, dummy_np_data.shape[-1]))
    bwd_np_data = np.concatenate([dummy_np_data, empty_data], axis=0)[1:]

    comparison_1 = bwd_np_data
    comparison_2 = bwd_np_data[:, [0, 3]]

    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_amt=-1,
                                    fill_value=0) == comparison_1))
    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,3], shift_amt=-1,
                                    fill_value=0) == comparison_2))

    dummy_pd_data = get_dummy_pd()
    comparison_1_pd = pd.DataFrame(comparison_1, columns=['A','B','C','D'])
    comparison_2_pd = pd.DataFrame(comparison_2, columns=['A','D'])
    
    inx_list = sglm_pp.get_column_nums(dummy_pd_data, ['A', 'D'])
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_amt=-1,
                                    fill_value=0) == comparison_1_pd))
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list, shift_amt=-1,
                                    fill_value=0) == comparison_2_pd))
    return

def test_shift_keep_all_cols():
    dummy_np_data = get_dummy_np()
    empty_data = np.zeros((1, dummy_np_data.shape[-1]))[:, [0, 1]]

    fwd_np_data = np.concatenate([empty_data, dummy_np_data[:, [0, 1]]], axis=0)[:-1]
    fwd_dummy_np_data_overwrite = dummy_np_data.copy()
    fwd_dummy_np_data_overwrite[:, [0, 1]] = fwd_np_data

    bwd_np_data = np.concatenate([dummy_np_data[:, [0, 1]], empty_data], axis=0)[1:]
    bwd_dummy_np_data_overwrite = dummy_np_data.copy()
    bwd_dummy_np_data_overwrite[:, [0, 1]] = bwd_np_data

    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,1], shift_amt=1,
                                    fill_value=0, keep_non_inx=True) == fwd_dummy_np_data_overwrite))
    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,1], shift_amt=-1,
                                    fill_value=0, keep_non_inx=True) == bwd_dummy_np_data_overwrite))


    dummy_pd_data = get_dummy_pd()
    comparison_1_pd = pd.DataFrame(fwd_dummy_np_data_overwrite, columns=['A','B','C','D'])
    comparison_2_pd = pd.DataFrame(bwd_dummy_np_data_overwrite, columns=['A','B','C','D'])
    
    inx_list = sglm_pp.get_column_nums(dummy_pd_data, ['A', 'B'])
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list, shift_amt=1,
                                    fill_value=0, keep_non_inx=True) == comparison_1_pd))
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list, shift_amt=-1,
                                    fill_value=0, keep_non_inx=True) == comparison_2_pd))
    return

def test_timeshift_multiple():
    # timeshift_multiple

    dummy_np_data = get_dummy_np()

    comp_n1 = sglm_pp.timeshift(dummy_np_data, shift_amt=-1, fill_value=0)
    comp_0 = sglm_pp.timeshift(dummy_np_data, shift_amt=0, fill_value=0)
    comp_1 = sglm_pp.timeshift(dummy_np_data, shift_amt=1, fill_value=0)

    comparison_1 = np.concatenate([comp_n1, comp_0, comp_1], axis=-1)
    assert(np.all(comparison_1 == sglm_pp.timeshift_multiple(dummy_np_data, shift_amt_list=[-1,0,1],
                                                             unshifted_keep_all=True, fill_value=0)))

    comp2_n1 = sglm_pp.timeshift(dummy_np_data, shift_amt=-1, fill_value=0, shift_inx=[0, 3])
    comp2_0 = sglm_pp.timeshift(dummy_np_data, shift_amt=0, fill_value=0)
    comp2_1 = sglm_pp.timeshift(dummy_np_data, shift_amt=1, fill_value=0, shift_inx=[0, 3])

    comparison_2 = np.concatenate([comp2_n1, comp2_0, comp2_1], axis=-1)
    assert(np.all(comparison_2 == sglm_pp.timeshift_multiple(dummy_np_data, shift_inx=[0, 3],
                                                             shift_amt_list=[-1,0,1],
                                                             unshifted_keep_all=True, fill_value=0)))

    dummy_pd_data = get_dummy_pd()
    comparison_1_pd = pd.DataFrame(comparison_1, columns=['A_-1','B_-1','C_-1','D_-1',
                                                          'A','B','C','D',
                                                          'A_1','B_1','C_1','D_1'])
    assert(np.all(comparison_1_pd == sglm_pp.timeshift_multiple(dummy_pd_data, shift_amt_list=[-1,0,1],
                                                                unshifted_keep_all=True, fill_value=0)))

    comparison_2_pd = pd.DataFrame(comparison_2, columns=['A_-1','D_-1', 'A','B','C','D', 'A_1','D_1'])
    assert(np.all(comparison_2_pd == sglm_pp.timeshift_multiple(dummy_pd_data, shift_inx=[0, 3],
                                                                shift_amt_list=[-1,0,1],
                                                                unshifted_keep_all=True,
                                                                fill_value=0)))

    
    return

def test_zscore():
    X = np.array([[0, -1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [2, 3, 4]])
    mn = np.mean(X, 0)
    sd = np.std(X, 0)
    assert(np.all(sglm_pp.zscore(X) == (X - mn)/sd))


def test_diff():
    X = np.array([[0, -1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [2, 3, 4]])
    dff = np.array([[1, 2, 0],
                    [-1, 0, 0],
                    [2, 2, 4]])
    assert(np.all(sglm_pp.diff(X) == dff))


    X = pd.DataFrame([[0, -1, 0],
                      [1, 1, 0],
                      [0, 1, 0],
                      [2, 3, 4]], columns=['A', 'B', 'C'])
    dff = pd.DataFrame([[1, 2, 0],
                    [-1, 0, 0],
                    [2, 2, 4]], columns=['A_diff', 'B_diff', 'C_diff'], index=[1, 2, 3])
    basis = dff.rename({_:_+'_diff' for _ in X.columns}, axis=1)
    assert(np.all(sglm_pp.diff(X) == basis))

if __name__ == '__main__':
    test_unshifted()
    test_forward_shift()
    test_backward_shift()
    test_shift_keep_all_cols()
    test_timeshift_multiple()
    test_zscore()
    test_diff()

    # print()
    # print(X_tmp)
    # X_tmp = diff_cols(X_tmp, ['B'])

    # print(sglm_pp.diff(X_tmp['A'], append_to_base=True))
    # print(sglm_pp.diff(X_tmp[['A', 'B']], append_to_base=True))

    # print(diff_cols(X_tmp, ['A', 'B_1']))
