import json
import os

file_loc = "Data\\Output\\LGM50\\Optmization_Results\\G1\\1"
file_pattern = "G1_cycle_1_pulse_{}_ecm_parameters.json"
num_files = 10  # From 0 to 9

# Lists to store extracted values
ocv_list = []
r0_list = []
r1_list = []
c1_list = []
r2_list = []
c2_list = []

# Dictionary to store common parameters
common_params = {
    "Ideal gas constant [J.K-1.mol-1]": 8.314462618,
    "Faraday constant [C.mol-1]": 96485.33212,
    "Boltzmann constant [J.K-1]": 1.380649e-23,
    "Electron charge [C]": 1.602176634e-19,
    "Initial SoC": 1.0,
    "Initial temperature [K]": 298.15,
    "Cell capacity [A.h]": 4.85,
    "Nominal cell capacity [A.h]": 4.85,
    "Ambient temperature [K]": 298.15,
    "Current function [A]": 100,
    "Upper voltage cut-off [V]": 4.2,
    "Lower voltage cut-off [V]": 2.5,
    "Cell thermal mass [J/K]": 1000,
    "Cell-jig heat transfer coefficient [W/K]": 10,
    "Jig thermal mass [J/K]": 500,
    "Jig-air heat transfer coefficient [W/K]": 10,
    "Element-1 initial overpotential [V]": 0,
    "RCR lookup limit [A]": 340,
    "Element-2 initial overpotential [V]": 0
}

# Read and extract data from JSON files
for i in range(num_files):
    filename = os.path.join(file_loc, file_pattern.format(i))
    if os.path.exists(filename):
        with open(filename, "r") as file:
            data = json.load(file)
            
            # Extracting RC and OCV values
            ocv_list.append(data["Open-circuit voltage [V]"])
            r0_list.append(data["R0 [Ohm]"])
            r1_list.append(data["R1 [Ohm]"])
            c1_list.append(data["C1 [F]"])
            r2_list.append(data["R2 [Ohm]"])
            c2_list.append(data["C2 [F]"])
    else:
        print(f"Warning: {filename} not found.")

# Create final ECM JSON structure
final_ecm_data = {
    **common_params,
    "Open-circuit voltage [V]": ocv_list,
    "R0 [Ohm]": r0_list,
    "R1 [Ohm]": r1_list,
    "C1 [F]": c1_list,
    "R2 [Ohm]": r2_list,
    "C2 [F]": c2_list
}

# Write to final ECM JSON file
final_filename = os.path.join(file_loc, "final_ecm_parameters.json")
with open(final_filename, "w") as outfile:
    json.dump(final_ecm_data, outfile, indent=4)

print(f"Final ECM file saved as {final_filename}")
