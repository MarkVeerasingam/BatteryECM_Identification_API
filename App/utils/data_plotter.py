import matplotlib.pyplot as plt
import numpy as np
from App.Service.CapacityTest import CapacityTest

# Function to analyze and plot the capacity test data
def plot_LGM50_SoC_OCV_capacity_test(battery_label):
    """
    Analyzes the capacity test data for a given battery label, calculates SOC, and plots OCV vs SOC for each cycle.
    
    :param battery_label: The battery label to filter data by (e.g., 'G1', 'W3', etc.)
    """
    # Create a BatteryTest instance
    battery_test = CapacityTest(battery_label=battery_label, test_type="capacity_test")

    # Use the BatteryTest class method to extract SOC and OCV data
    battery_test.extract_soc_ocv()

    # Plot OCV vs SOC for each cycle
    plt.figure(figsize=(10, 6))
    for i in range(len(battery_test.SOC)):
        plt.plot(battery_test.SOC[i], battery_test.OCV[i], label=f"Cycle {i+1}")

    # Adding labels, title, and grid for better visualization
    plt.xlabel("State of Charge (SOC, %)") 
    plt.ylabel("Open Circuit Voltage (OCV, V)")
    plt.title(f"OCV vs SOC for All Cycles - Battery {battery_label}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_voltage_response(battery_label, test_type):
    # Create a BatteryTest instance
    battery_test = CapacityTest(battery_label=battery_label, test_type=test_type)

    # Extract the raw voltage data (vcell, current, capacity) directly from the BatteryTest class
    vcell, current, cap = battery_test.vcell, battery_test.current, battery_test.cap

    plt.figure(figsize=(10, 6))
    for i, vcell_cycle in enumerate(vcell):
        if vcell_cycle.size > 1 and not np.isnan(vcell_cycle).all():  # Skip empty or NaN rows
            vcell_cycle = vcell_cycle[~np.isnan(vcell_cycle)].reshape(-1)  # Remove NaN and flatten
            plt.plot(vcell_cycle, label=f"Cycle {i+1}")

    # Plot formatting
    plt.xlabel("Index (Data Points)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Voltage Profile for All Cycles - Battery {battery_label}")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()
