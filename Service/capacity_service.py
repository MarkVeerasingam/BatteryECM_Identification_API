import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  
from utils.data_loader import load_LGM50_data

def extract_soc_ocv_LGM50(battery_label):
    """
    Extracts SOC (State of Charge) and OCV (Open Circuit Voltage) for each cycle from the capacity test data.
    
    :param battery_label: The battery label to filter data by (e.g., 'G1', 'W3', etc.)
    :return: SOC and OCV data for each cycle
    """
    vcell, current, cap = load_LGM50_data(battery_label=battery_label, test_data="capacity_test")

    # Initialize lists to store SOC and OCV data
    SOC = []
    OCV = []

    # Get the minimum number of cycles 
    num_cycles = min(len(cap), len(vcell)) 

    for i in range(num_cycles):
        capacity = cap[i]

        # this is to ensure that we are going to use only valid data an drmeove NaNs
        if capacity.size > 1 and not np.isnan(capacity).all():
            cap_cycle = capacity[~np.isnan(capacity)].reshape(-1)
            Q_end = cap_cycle[-1] # Take the last capacity value as the full capacity

            # Calculate SOC (State of Charge)
            soc_cycle = (cap_cycle / Q_end) * 100 # SOC from 0% to 100%
            soc_cycle = 100 - soc_cycle.flatten() # Invert SOC (100% to 0%)

            SOC.append(soc_cycle)
            # Extract the corresponding OCV (Open Circuit Voltage)
            vcell_cycle = vcell[i].flatten()
            OCV.append(vcell_cycle)

    return SOC, OCV

