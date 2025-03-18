import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from App.utils.data_loader import load_LGM50_data

class CapacityTest:
    def __init__(self, battery_label):
        """
        Initialize the CapacityTest class.
        
        :param battery_label: The battery label to filter data by (e.g., 'G1', 'W3', etc.)
        :param degree: The degree of the polynomial fit (default is 11)
        """
        self.battery_label = battery_label
        self.test_type = "capacity_test" 
        self.degree = None  
        # an array to hold the results value of soc and ocv so i can save them to a csv.
        self.SOC = []
        self.OCV = []

        # Load the data based on the capacity test data
        self.vcell, self.current, self.cap = load_LGM50_data(battery_label=self.battery_label, test_data=self.test_type)

        # Analysis results storage
        self.results_data = None

    def extract_soc_ocv(self):
        """
        Extracts State of Charge (SOC) and Open Circuit Voltage (OCV) for each cycle.
        """
        num_cycles = min(len(self.vcell), len(self.current))

        for i in range(num_cycles):
            if self.cap is not None:
                capacity = self.cap[i]

                if capacity.size > 1 and not np.isnan(capacity).all():
                    cap_cycle = capacity[~np.isnan(capacity)].reshape(-1)
                    Q_end = cap_cycle[-1]

                    # Calculate SOC (State of Charge)
                    soc_cycle = (cap_cycle / Q_end) * 100
                    soc_cycle = 100 - soc_cycle.flatten()  # Invert SOC (100% to 0%)

                    self.SOC.append(soc_cycle)
                    # Extract corresponding OCV (Open Circuit Voltage)
                    vcell_cycle = self.vcell[i].flatten()
                    self.OCV.append(vcell_cycle)

    def fit_soc_ocv_polynomial(self, degree):
        """
        Fits a polynomial to the SOC and OCV data.
        """
        self.degree = degree
        self.extract_soc_ocv()

        if self.cap is not None:  # If capacity is available, fit SOC-OCV relationship
            SOC_flat = np.concatenate(self.SOC)
            OCV_flat = np.concatenate(self.OCV)

            # Scale SOC to the 0-1 range
            SOC_flat_scaled = SOC_flat / 100  # Normalize SOC to the range [0, 1]

            # Fit polynomial to the scaled SOC and OCV
            coeffs = np.polyfit(SOC_flat_scaled, OCV_flat, degree)  # Use the degree from class
            poly_fit = np.poly1d(coeffs)

            # Generate the fitted and smoothed  SOC values for plotting the fit curve
            SOC_Fitted = np.linspace(min(SOC_flat_scaled), max(SOC_flat_scaled), 100)
            OCV_Fitted = poly_fit(SOC_Fitted)

            self.results_data = {
                "SOC_Fitted": SOC_Fitted,
                "OCV_Fitted": OCV_Fitted
            }

            return self.results_data

    def plot_capacity_test(self):
        plt.figure(figsize=(10, 6))
        for i in range(len(self.SOC)):
            plt.plot(self.SOC[i], self.OCV[i], label=f"Cycle {i+1}")

        plt.xlabel("State of Charge (SOC, %)")
        plt.ylabel("Open Circuit Voltage (OCV, V)")
        plt.title(f"OCV vs SOC for All Cycles - Battery {self.battery_label}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_ocv_soc_fitting(self):
        """
        Plots the polynomial fitting results for SOC vs OCV for the capacity test
        """
        if self.results_data is None:
            raise ValueError("No fitting has been run. Call fit_soc_ocv_polynomial()")

        SOC_Fitted = self.results_data["SOC_Fitted"]
        OCV_Fitted = self.results_data["OCV_Fitted"]

        plt.figure(figsize=(10, 6))

        if self.cap is not None:  # Capacity test - Plot SOC vs OCV
            # Concatenate the original data
            SOC_flat = np.concatenate(self.SOC)
            OCV_flat = np.concatenate(self.OCV)

            # Scale SOC to the 0-1 range
            SOC_flat_scaled = SOC_flat / 100  # Normalize SOC to the range [0, 1]

            # Plot the original data as a scatter plot
            plt.scatter(SOC_flat_scaled, OCV_flat, label="Measured Data", color='blue', alpha=0.6)

            # Plot the fitted polynomial curve
        plt.plot(SOC_Fitted, OCV_Fitted, label=f"Fitted Polynomial (Degree {self.degree})", color='red', linewidth=2)

        plt.xlabel("State of Charge (SOC, %) ")
        plt.ylabel("Open Circuit Voltage (OCV, V) ")
        plt.title(f"OCV vs SOC - Battery {self.battery_label}")

        # Final plot adjustments
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_to_csv(self, output_path=None):
        """
        Save the fitted SOC and OCV data to a CSV file.
        
        :param SOC_Fitted: The fitted SOC data 
        :param OCV_Fitted: The fitted OCV data
        :param filename: Optional filename for the CSV file. If not provided, defaults to 'soc_ocv_data.csv'.
        """
        if self.results_data is None:
            raise ValueError("No fitting has been run. Call fit_soc_ocv_polynomial()")

        SOC_Fitted = self.results_data["SOC_Fitted"]
        OCV_Fitted = self.results_data["OCV_Fitted"]

        df = pd.DataFrame({
            'SOC': SOC_Fitted,
            'OCV': OCV_Fitted,
        })

         # Default directory if no output path provided
        if output_path is None:
            output_path = os.path.join("Data", "Output", "LGM50", "Capacity_Test", self.battery_label)

        # Ensure output_path is a directory
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        file_name = f"{self.battery_label}_soc_ocv.csv"
        full_path = os.path.join(output_path, file_name)

        # Save DataFrame to CSV
        df.to_csv(full_path, index=False)
        print(f"Pulse data saved to: {full_path}")

        return full_path
