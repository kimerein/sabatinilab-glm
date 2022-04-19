import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
import time
import random

import glob
from sglm.features import sglm_pp
import freely_moving_helpers as lpp
from tqdm import tqdm, trange

def rename_columns_by_file(files_list, channel_definitions, verbose=0):
    """
    Rename channels in a file.
    Args:
        files_list : list(str)
            List of filenames with which to associate different names for channels
        channel_definitions : dict(dict)
            Dictionary of Keys: tuple of filename identifiers (all must be matched) to Vals: dictionary mapping initial channel names to final channel names
        verbose : int
            Verbosity level
    Returns:
        channel_assignments : dict
            Dictionary of Keys: filenames with required renamings to Vals: 

    Example:
        files_list = glob.glob(f'{dir_path}/../GLM_SIGNALS_WT61_*') + \
                    glob.glob(f'{dir_path}/../GLM_SIGNALS_WT63_*') + \
                    glob.glob(f'{dir_path}/../GLM_SIGNALS_WT64_*')
        channel_definitions = {
            ('WT61',): {'Ch1': 'gACH', 'Ch2': 'rDA'},
            ('WT64',): {'Ch1': 'gACH', 'Ch2': 'empty'},
            ('WT63',): {'Ch1': 'gDA', 'Ch2': 'empty'},
        }
        channel_assignments = rename_channels(files_list, channel_definitions)

        (channel_assignments will map each individual filename to a renaming dictionary of columns)
    """
    channel_assignments = {}
    for file_lookup in channel_definitions:
        print(file_lookup)
        channel_renamings = channel_definitions[file_lookup]
        relevant_files = [f for f in files_list if all(x in f for x in file_lookup)]
        for relevant_file in relevant_files:
            relevant_file = relevant_file.split('/')[-1]
            print('>', relevant_file)
            channel_assignments[relevant_file] = channel_renamings
    
    return channel_assignments

def rename_consistent_columns(files_list, channel_definitions, verbose=0):
    """
    Rename channels in a file.
    Args:
        files_list : list(str)
            List of filenames with which to associate different names for channels
        channel_definitions : dict(dict)
            Dictionary of Keys: tuple of filename identifiers (all must be matched) to Vals: dictionary mapping initial channel names to final channel names
        verbose : int
            Verbosity level
    Returns:
        channel_assignments : dict
            Dictionary of Keys: filenames with required renamings to Vals: 

    Example:
        files_list = glob.glob(f'{dir_path}/../GLM_SIGNALS_WT61_*') + \
                    glob.glob(f'{dir_path}/../GLM_SIGNALS_WT63_*') + \
                    glob.glob(f'{dir_path}/../GLM_SIGNALS_WT64_*')
        channel_definitions = {
            ('WT61',): {'Ch1': 'gACH', 'Ch2': 'rDA'},
            ('WT64',): {'Ch1': 'gACH', 'Ch2': 'empty'},
            ('WT63',): {'Ch1': 'gDA', 'Ch2': 'empty'},
        }
        channel_assignments = rename_channels(files_list, channel_definitions)

        (channel_assignments will map each individual filename to a renaming dictionary of columns)
    """
    channel_assignments = {}
    for file_lookup in channel_definitions:
        print(file_lookup)
        channel_renamings = channel_definitions[file_lookup]
        relevant_files = [f for f in files_list if all(x in f for x in file_lookup)]
        for relevant_file in relevant_files:
            relevant_file = relevant_file.split('/')[-1]
            print('>', relevant_file)
            channel_assignments[relevant_file] = channel_renamings
    
    return channel_assignments












def generate_toy_data():
    pass