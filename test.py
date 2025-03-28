import pybamm
import numpy as np
import pandas as pd
import os

battery_label = "G1"
cycle_number = 1

file_loc = f"Data\\Output\\LGM50\\Optimization_Results\\{battery_label}\\{cycle_number}\\"
loaded_data = pd.read_csv(file_loc + "ecm_lut_table.csv")

# Extract data from CSV
soc_values = np.array(loaded_data["SoC"])
voltage_values = np.array(loaded_data["voltage"])
current_values = np.array(loaded_data["current"])
temperature_values = np.array(loaded_data["temperature"])
R0_values = np.array(loaded_data["r0"])  
R1_values = np.array(loaded_data["r1"])  
C1_values = np.array(loaded_data["c1"]) 
R2_values = np.array(loaded_data["r2"])  
C2_values = np.array(loaded_data["c2"])

# Create interpolation functions for each parameter
def ocv(soc):
    return pybamm.Interpolant(soc_values, voltage_values, soc, name="OCV", extrapolate=True)

# Since your data only has one current and temperature, we'll interpolate only in SOC
def r0(current, temperature, soc):
    return pybamm.Interpolant(soc_values, R0_values, soc, name="R0", extrapolate=True)

def r1(current, temperature, soc):
    return pybamm.Interpolant(soc_values, R1_values, soc, name="R1", extrapolate=True)

def c1(current, temperature, soc):
    return pybamm.Interpolant(soc_values, C1_values, soc, name="C1", extrapolate=True)

def r2(current, temperature, soc):
    return pybamm.Interpolant(soc_values, R2_values, soc, name="R2", extrapolate=True)

def c2(current, temperature, soc):
    return pybamm.Interpolant(soc_values, C2_values, soc, name="C2", extrapolate=True)

# Create a Thevenin model with 2RC elements
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})

# Define parameter values
parameter_values = pybamm.ParameterValues("ECM_Example")

# Update with our custom parameters
updated_data = {
    "Open-circuit voltage [V]": ocv,
    "R0 [Ohm]": r0,
    "R1 [Ohm]": r1,
    "C1 [F]": c1,
    "R2 [Ohm]": r2,
    "C2 [F]": c2,
    "Element-1 initial overpotential [V]": 0,
    "Element-2 initial overpotential [V]": 0,
    "Initial SoC": 1.0,  # Set initial SOC to 1 (fully charged)
    # "Initial temperature [K]": 298.15,  # 25°C
    # "Current function [A]": 1.0,  
}

parameter_values.update(updated_data, check_already_exists=False)

# Define experiment
experiment = pybamm.Experiment(
    [
        # "Discharge at 1 A until 3 V",
        # "Rest for 2 hours",
        "Charge at 1 A until 4.2 V",
        # "Hold at 4.2 V until 1 A",
        # "Rest for 2 hours",
        # "Discharge at 0.5 A until 3.6V",
        # "Charge at C/5 for 1 hour"
    ]
)

# Create solver
solver = pybamm.CasadiSolver(mode="safe", rtol=1e-6, atol=1e-6)

# Create and solve simulation
sim = pybamm.Simulation(
    model, 
    parameter_values=parameter_values, 
    solver=solver, 
    experiment=experiment
)

# Solve and plot
solution = sim.solve()
sim.plot()