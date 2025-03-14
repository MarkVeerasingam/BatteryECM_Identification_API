import csv
import numpy as np
from App.utils.data_loader import load_LGM50_data

class CapacityTest:
    def __init__(self, battery_label, degree=11):
        """
        Initialize the CapacityTest class.
        
        :param battery_label: The battery label to filter data by (e.g., 'G1', 'W3', etc.)
        :param degree: The degree of the polynomial fit (default is 11)
        """
        self.battery_label = battery_label
        self.test_type = "capacity_test" 
        self.degree = degree  
        self.SOC = []
        self.OCV = []

        # Load the data based on the capacity test
        self.vcell, self.current, self.cap = load_LGM50_data(battery_label=self.battery_label, test_data=self.test_type)

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

    def fit_soc_ocv_polynomial(self):
        """
        Fits a polynomial to the SOC and OCV data.
        """
        self.extract_soc_ocv()

        if self.cap is not None:  # If capacity is available, fit SOC-OCV relationship
            SOC_flat = np.concatenate(self.SOC)
            OCV_flat = np.concatenate(self.OCV)

            # Scale SOC to the 0-1 range
            SOC_flat_scaled = SOC_flat / 100  # Normalize SOC to the range [0, 1]

            # Fit polynomial to the scaled SOC and OCV
            coeffs = np.polyfit(SOC_flat_scaled, OCV_flat, self.degree)  # Use the degree from class
            poly_fit = np.poly1d(coeffs)

            # Generate the fitted and smoothed  SOC values for plotting the fit curve
            SOC_Fitted = np.linspace(min(SOC_flat_scaled), max(SOC_flat_scaled), 100)
            OCV_Fitted = poly_fit(SOC_Fitted)

            return SOC_Fitted, OCV_Fitted

    def show_ocv_soc_plot(self, SOC_Fitted, OCV_Fitted):
        """
        Plots the polynomial fitting results for SOC vs OCV for the capacity test
        """
        import matplotlib.pyplot as plt

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

    def save_to_csv(self, SOC_Fitted, OCV_Fitted, filename=None):
        """
        Save the fitted SOC and OCV data to a CSV file.
        
        :param SOC_Fitted: The fitted SOC data 
        :param OCV_Fitted: The fitted OCV data
        :param filename: Optional filename for the CSV file. If not provided, defaults to 'soc_ocv_data.csv'.
        """
        if filename is None:
            filename = f"battery_{self.battery_label}_soc_ocv_fitted.csv"

        # Prepare data for CSV 
        data = list(zip(SOC_Fitted, OCV_Fitted))

        # Write to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SOC', 'OCV'])  # Write the header
            writer.writerows(data)  # Write the fitted SOC and OCV data

        print(f"Fitted data saved to {filename}")
