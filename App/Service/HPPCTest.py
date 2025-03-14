import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from App.utils.data_loader import load_LGM50_data

class HPPCTest:
    def __init__(self, battery_label, cycle_number):
        """
        Initialize the HPPCTest class.
        
        :param battery_label: The battery label to filter data by (e.g., 'G1', 'W3', etc.)
        :param cycle_number: The cycle number to analyze (default is 0)
        """
        self.battery_label = battery_label
        self.test_type = "HPPC_test"
        self.cycle_number = cycle_number
        
        # Load the data based on the HPPC test
        self.vcell, self.current, self.cap = load_LGM50_data(test_data=self.test_type, battery_label=self.battery_label)
        
        # Convert loaded data to the right format
        # Taking the specified cycle and flattening the array
        self.vcell_cycle = np.array(self.vcell[self.cycle_number]).flatten()
        self.current_cycle = np.array(self.current[self.cycle_number]).flatten() * -1  # Ensure discharge is positive
        
        # Remove NaN values
        valid_indices = ~np.isnan(self.vcell_cycle) & ~np.isnan(self.current_cycle)
        self.vcell_cycle = self.vcell_cycle[valid_indices]
        self.current_cycle = self.current_cycle[valid_indices]
        self.time_vector = np.arange(len(self.vcell_cycle))
        
        # Load OCV-to-SOC LUT from the capacity test
        self.ocv_lut_file = f"Data/Output/LGM50/Capacity_Test/{battery_label}/{battery_label}_soc_ocv.csv"
        self.ocv_lut = pd.read_csv(self.ocv_lut_file)
        self.ocv_lut = self.ocv_lut.sort_values(by="OCV")
        self.soc_values = self.ocv_lut["SOC"].values
        self.ocv_values = self.ocv_lut["OCV"].values
        
        # Calculate SOC for the cycle
        self.soc_cycle = self.estimate_soc_from_ocv(self.vcell_cycle)
        
        # Find main pulse sequences
        self.pulse_starts = self.find_main_pulses(self.current_cycle)
        
        # Analysis results storage
        self.selected_pulse_data = None
        self.pulse_characteristics = None
    
    def estimate_soc_from_ocv(self, voltage):
        """
        Estimate SOC from voltage using the OCV lookup table.
        
        :param voltage: Battery voltage values
        :return: Estimated SOC values
        """
        return np.interp(voltage, self.ocv_values, self.soc_values)
    
    def find_main_pulses(self, current, min_distance=1000):
        """
        Find the main pulse starts in the current data.
        
        :param current: Current data array
        :param min_distance: Minimum distance between pulses
        :return: Array of pulse start indices
        """
        current_threshold = 0.1  # Detect significant changes in current
        all_changes = np.where(np.abs(np.diff(current)) > current_threshold)[0]
        
        main_pulses = []
        last_pulse = -min_distance  # Initialize with negative distance
        
        for idx in all_changes:
            if idx - last_pulse >= min_distance:
                main_pulses.append(idx)
                last_pulse = idx
        return np.array(main_pulses)
    
    def get_pulse_count(self):
        """
        Get the total number of pulses detected in the data.
        
        :return: Integer count of detected pulses
        """
        return len(self.pulse_starts)
    
    def extract_pulse(self, start_idx, window_size=1000):
        """
        Extract pulse data for a given start index.
        
        :param start_idx: Start index of the pulse
        :param window_size: Window size for the pulse
        :return: Tuple of (time, current, voltage, soc) for the pulse
        """
        end_idx = min(start_idx + window_size, len(self.current_cycle))
        return (
            self.time_vector[start_idx:end_idx],
            self.current_cycle[start_idx:end_idx],
            self.vcell_cycle[start_idx:end_idx],
            self.soc_cycle[start_idx:end_idx]
        )
    
    def run_analysis(self, pulse_number=0, window_size=1000):
        """
        Run the HPPC analysis for a specific pulse.
        
        :param pulse_number: Index of the pulse to analyze
        :param window_size: Window size for analysis
        :return: Dictionary of pulse characteristics
        """
        pulse_count = self.get_pulse_count()
        if pulse_count == 0:
            raise ValueError("No pulses detected in the data.")
        
        if pulse_number >= pulse_count:
            raise ValueError(f"Pulse number {pulse_number} exceeds available pulses ({pulse_count}).")
        
        selected_start = self.pulse_starts[pulse_number]
        time_pulse, current_pulse, voltage_pulse, soc_pulse = self.extract_pulse(
            selected_start, window_size
        )

        # Store pulse data for plotting
        self.selected_pulse_data = {
            'start_idx': selected_start,
            'time': time_pulse,
            'current': current_pulse,
            'voltage': voltage_pulse,
            'soc': soc_pulse,
            'window_size': window_size,
            'pulse_number': pulse_number
        }
        
        self.pulse_characteristics = {
            'start_index': selected_start,
            'pulse_duration': len(time_pulse),
            'voltage_drop': voltage_pulse.max() - voltage_pulse.min(),
            'peak_current': np.abs(current_pulse).max(),
            'initial_soc': soc_pulse[0],
            'final_soc': soc_pulse[-1],
            'cycle_number': self.cycle_number
        }
        
        print(f"\nCycle {self.cycle_number} - Pulse {pulse_number} characteristics:")
        print(f"Start index: {selected_start}")
        print(f"Pulse duration: {len(time_pulse)} points")
        print(f"Voltage drop: {voltage_pulse.max() - voltage_pulse.min():.3f}V")
        print(f"Peak current: {np.abs(current_pulse).max():.2f}A")
        print(f"Initial SoC: {soc_pulse[0]:.2f}%")
        print(f"Final SoC: {soc_pulse[-1]:.2f}%")
        
        return self.pulse_characteristics

    def plot_hppc_analysis(self):
        """
        Plot the HPPC analysis results.
        """
        if self.selected_pulse_data is None:
            raise ValueError("No analysis has been run. Call run_analysis() first.")
        
        # Extract data for plotting
        selected_start = self.selected_pulse_data['start_idx']
        time_pulse = self.selected_pulse_data['time']
        current_pulse = self.selected_pulse_data['current']
        voltage_pulse = self.selected_pulse_data['voltage']
        soc_pulse = self.selected_pulse_data['soc']
        window_size = self.selected_pulse_data['window_size']
        pulse_number = self.selected_pulse_data['pulse_number']
        
        # Create figure with 4 subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
        
        # Plot full cycle voltage with selected pulse highlighted
        ax1.plot(self.time_vector, self.vcell_cycle, 'b-', label='Voltage')
        ax1.axvspan(selected_start, selected_start + window_size, color='red', alpha=0.2, label='Selected Pulse')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title(f'Full HPPC Cycle - Battery {self.battery_label} - Selected Pulse {pulse_number}')
        ax1.grid(True)
        ax1.legend()
        
        # Plot full cycle current with selected pulse highlighted
        ax2.plot(self.time_vector, self.current_cycle, 'r-', label='Current')
        ax2.axvspan(selected_start, selected_start + window_size, color='red', alpha=0.2, label='Selected Pulse')
        ax2.set_ylabel('Current (A)')
        ax2.grid(True)
        ax2.legend()
        
        # Plot extracted pulse for voltage and current
        ax3.plot(time_pulse, voltage_pulse, 'b-', label='Voltage')
        ax3.plot(time_pulse, current_pulse, 'r-', label='Current')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Voltage (V) / Current (A)')
        ax3.set_title(f'Extracted Pulse {pulse_number}')
        ax3.grid(True)
        ax3.legend()
        
        # Plot SOC over the full cycle (using OCV LUT)
        ax4.plot(self.time_vector, self.soc_cycle, 'g-', label='State of Charge')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('SOC (%)')
        ax4.set_title('State of Charge over Cycle Test (OCV LUT-based)')
        ax4.axvline(x=selected_start, color='red', linestyle='--', label='Pulse Start')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_to_csv(self, output_path=None):
        """
        Save the current pulse data to a CSV file.

        :param output_path: Path or directory to save the CSV file.
        :return: Path to the saved CSV file
        """
        if self.selected_pulse_data is None:
            raise ValueError("No analysis has been run. Call run_analysis() first.")

        # Extract data
        time_pulse = self.selected_pulse_data['time']
        voltage_pulse = self.selected_pulse_data['voltage']
        current_pulse = self.selected_pulse_data['current']
        soc_pulse = self.selected_pulse_data['soc']
        pulse_number = self.selected_pulse_data['pulse_number']

        # Create DataFrame
        pulse_df = pd.DataFrame({
            'Time': time_pulse,
            'Voltage': voltage_pulse,
            'Current': current_pulse,
            'SoC': soc_pulse
        })

        # Default directory if no output path provided
        if output_path is None:
            output_path = os.path.join("Data", "Output", "LGM50", "HPPC_Test", self.battery_label, f"Cycle_{self.cycle_number}",)
        
        # Ensure output_path is a directory
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        file_name = f"{self.battery_label}_cycle_{self.cycle_number}_pulse_{pulse_number}_hppc.csv"
        full_path = os.path.join(output_path, file_name)

        # Save DataFrame to CSV
        pulse_df.to_csv(full_path, index=False)
        print(f"Pulse data saved to: {full_path}")

        return full_path
