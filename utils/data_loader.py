"""
data_loader.py 
loads in battery data from ..//Data//Input//
"""
import os
import scipy.io as sio
import numpy as np

# Path setup (hardcoded battery data and input directory)
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
    col_index = np.where([label[0] == battery_label for label in mat['col_cell_label'][0]])[0][0]

    # since the share the same header, we can extract the same keys
    vcell = mat['vcell'][:, col_index]
    current = mat['curr'][:, col_index]
    cap = mat['cap'][:, col_index]

    return vcell, current, cap