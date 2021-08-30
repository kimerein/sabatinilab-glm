import pytest

import sys
sys.path.append('..')
sys.path.append('../backend')
sys.path.append('./backend')

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
    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,3], shift_amt=0) == comparison_2))

    dummy_pd_data = get_dummy_pd()
    comparison_1_pd = pd.DataFrame(comparison_1, columns=['A','B','C','D'])
    comparison_2_pd = pd.DataFrame(comparison_2, columns=['A','D'])
    
    inx_list = sglm_pp.get_column_names(dummy_pd_data, ['A', 'D'])
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_amt=0) == comparison_1_pd))
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list, shift_amt=0) == comparison_2_pd))
    return

def test_forward_shift():
    dummy_np_data = get_dummy_np()
    empty_data = np.zeros((1, dummy_np_data.shape[-1]))
    fwd_np_data = np.concatenate([empty_data, dummy_np_data], axis=0)[:-1]

    comparison_1 = fwd_np_data
    comparison_2 = fwd_np_data[:, [0, 3]]

    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_amt=1, fill_value=0) == comparison_1))
    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,3], shift_amt=1, fill_value=0) == comparison_2))

    dummy_pd_data = get_dummy_pd()
    comparison_1_pd = pd.DataFrame(comparison_1, columns=['A','B','C','D'])
    comparison_2_pd = pd.DataFrame(comparison_2, columns=['A','D'])
    
    inx_list = sglm_pp.get_column_names(dummy_pd_data, ['A', 'D'])
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_amt=1, fill_value=0) == comparison_1_pd))
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list, shift_amt=1, fill_value=0) == comparison_2_pd))
    return

def test_backward_shift():
    dummy_np_data = get_dummy_np()
    empty_data = np.zeros((1, dummy_np_data.shape[-1]))
    bwd_np_data = np.concatenate([dummy_np_data, empty_data], axis=0)[1:]

    comparison_1 = bwd_np_data
    comparison_2 = bwd_np_data[:, [0, 3]]

    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_amt=-1, fill_value=0) == comparison_1))
    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,3], shift_amt=-1, fill_value=0) == comparison_2))

    dummy_pd_data = get_dummy_pd()
    comparison_1_pd = pd.DataFrame(comparison_1, columns=['A','B','C','D'])
    comparison_2_pd = pd.DataFrame(comparison_2, columns=['A','D'])
    
    inx_list = sglm_pp.get_column_names(dummy_pd_data, ['A', 'D'])
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_amt=-1, fill_value=0) == comparison_1_pd))
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list, shift_amt=-1, fill_value=0) == comparison_2_pd))
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

    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,1], shift_amt=1, fill_value=0, keep_non_inx=True) == fwd_dummy_np_data_overwrite))
    assert(np.all(sglm_pp.timeshift(dummy_np_data, shift_inx=[0,1], shift_amt=-1, fill_value=0, keep_non_inx=True) == bwd_dummy_np_data_overwrite))


    dummy_pd_data = get_dummy_pd()
    comparison_1_pd = pd.DataFrame(fwd_dummy_np_data_overwrite, columns=['A','B','C','D'])
    comparison_2_pd = pd.DataFrame(bwd_dummy_np_data_overwrite, columns=['A','B','C','D'])
    
    inx_list = sglm_pp.get_column_names(dummy_pd_data, ['A', 'B'])
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list, shift_amt=1, fill_value=0, keep_non_inx=True) == comparison_1_pd))
    assert(np.all(sglm_pp.timeshift(dummy_pd_data, shift_inx=inx_list, shift_amt=-1, fill_value=0, keep_non_inx=True) == comparison_2_pd))
    return

if __name__ == '__main__':
    test_unshifted()
    test_forward_shift()
    test_backward_shift()
    test_shift_keep_all_cols()
    

# data = np.arange(100).reshape(10,-1)
# data = np.concatenate([data, np.ones((10,1))], axis=-1)
# data[5:,-1] = 0

# print('Original Data:\n', data)
# print('Unshifted:\n', sglm_pp.timeshift(data, shift_inx=[0,1,4,5,7], shift_amt=0))
# print('Forward Shifted:\n', sglm_pp.timeshift(data, shift_amt=3))
# print('Backward Shifted:\n', sglm_pp.timeshift(data, shift_amt=-3))
# print('Forward Shifted—Specific Columns:\n', sglm_pp.timeshift(data, shift_inx=[0,1,2,9], shift_amt=3))
# print('Backward Shifted—Specific Columns:\n', sglm_pp.timeshift(data, shift_inx=[0,1,2,9], shift_amt=-3))

# print('Forward Shifted—Specific Columns (Keep All):\n', sglm_pp.timeshift(data, shift_inx=[0,1,2,9], shift_amt=3, keep_non_inx=True))