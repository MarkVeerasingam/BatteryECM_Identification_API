import matplotlib.pyplot as plt
import numpy as np
from utils.data_loader import load_LGM50_data
from Service.capacity_service import extract_soc_ocv_LGM50, fit_soc_ocv_polynomial

# Function to analyze and plot the capacity test data
def plot_LGM50_SoC_OCV_capacity_test(battery_label):
    """
    Analyzes the capacity test data for a given battery label, calculates SOC, and plots OCV vs SOC for each cycle.
    
    :param battery_label: The battery label to filter data by (e.g., 'G1', 'W3', etc.)
    """
    # Extract SOC and OCV data for the given battery label
    SOC, OCV = extract_soc_ocv_LGM50(battery_label)

    # Plot OCV vs SOC for each cycle
    plt.figure(figsize=(10, 6))
    for i in range(len(SOC)):
        plt.plot(SOC[i], OCV[i], label=f"Cycle {i+1}")

    # Adding labels, title, and grid for better visualization
    plt.xlabel("State of Charge (SOC, %)")
    plt.ylabel("Open Circuit Voltage (OCV, V)")
    plt.title(f"OCV vs SOC for All Cycles - Battery {battery_label}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_raw_LGM50_capacity_test(battery_label):
    vcell, current, cap = load_LGM50_data(battery_label=battery_label, test_data="capacity_test")

    plt.figure(figsize=(10, 6))
    for i, vcell_cycle in enumerate(vcell):
        if vcell_cycle.size > 1 and not np.isnan(vcell_cycle).all():  # Skip empty or NaN rows
            vcell = vcell_cycle[~np.isnan(vcell_cycle)].reshape(-1)  # Remove NaN and flatten
            plt.plot(vcell, label=f"Cycle {i+1}")

    # Plot formatting
    plt.xlabel("Index (Data Points)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Voltage Profile for All Cycles - Battery {battery_label}")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()

def plot_soc_ocv_from_capacity_test(battery_label, degree):
    """Plots the SOC-OCV data along with the fitted polynomial curve."""
    SOC_fitting, OCV_fitting, SOC_flat, OCV_flat, SOC_flat_scaled = fit_soc_ocv_polynomial(battery_label=battery_label, degree=degree)

    plt.figure(figsize=(10, 6))
    plt.scatter(SOC_flat_scaled, OCV_flat, label="Measured Data", color='blue', alpha=0.6)
    plt.plot(SOC_fitting, OCV_fitting, label=f"Fitted Polynomial (Degree {degree})", color='red', linewidth=2)
    plt.xlabel("State of Charge (SOC, %)")
    plt.ylabel("Open Circuit Voltage (OCV, V)")
    plt.title(f"OCV vs SOC - Battery {battery_label}")
    plt.legend()
    plt.grid(True)
    plt.show()

