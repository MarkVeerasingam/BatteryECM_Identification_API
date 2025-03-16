"""
data_loader.py 
loads in battery data from ..//Data//Input//
"""
import os
import scipy.io as sio
import numpy as np
import pandas as pd

# Path setup (hardcoded to LGM50 cuz its the only thing im using)
battery_data = "LGM50"
data_input_dir = os.path.join("Data", "Input", battery_data)

def load_LGM50_data(test_data, battery_label):
    """
    Load data for a specific battery label (e.g., 'G1') from either capacity or HPPC test data.
    
    :param test_data: Type of test data to load ('capacity_test' or 'HPPC_test')
    :param battery_label: The battery label to filter data by (e.g., 'G1', 'W3', etc.)
    :return: A dictionary containing the 'vcell', 'current', and 'cap' for HPPC test or 'vcell' for capacity test.
    """
    # Define file paths
    # again this is only A or B, and they have the same header, keeping it simple
    capacity_test_data = os.path.join(data_input_dir, "capacity_test.mat")
    hppc_test_data = os.path.join(data_input_dir, "HPPC_test.mat")
    
    # load the hppc or capacity test matlab data
    if test_data == "capacity_test":
        mat = sio.loadmat(capacity_test_data)
    elif test_data == "HPPC_test":
        mat = sio.loadmat(hppc_test_data)

    # both tests share the same headers and we need to index into the battery labels.
    try:
        col_index = np.where([label[0] == battery_label for label in mat['col_cell_label'][0]])[0][0]
    except IndexError:
        raise ValueError(f"Battery label '{battery_label}' not found in the file. Must be 'W3', 'W4', 'W5', 'W7', 'W8', 'W9', 'W10', 'G1', 'V4', 'V5'")


    # since the share the same header, we can extract the same keys
    vcell = mat['vcell'][:, col_index]
    current = mat['curr'][:, col_index]
    cap = mat['cap'][:, col_index]

    return vcell, current, cap

def load_soc_ocv_data(battery_label):
        """
        Load the SOC-OCV lookup table from CSV file generated from running the capacity test..
        """
        file_path = f"Data/Output/{battery_data}/Capacity_Test/{battery_label}/{battery_label}_soc_ocv.csv"
        soc_ocv_data = pd.read_csv(file_path)
        print(f"Loaded SOC-OCV data from {file_path}")
        return soc_ocv_data