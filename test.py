import pybamm
import numpy as np
import pandas as pd
import os

battery_label = "G1"
cycle_number = 1

file_loc = f"Data\\Output\\LGM50\\Optimization_Results\\{battery_label}\\"
loaded_data = pd.read_csv(file_loc + "G1_aggregated_ecm_results.csv")
loaded_data = loaded_data.drop_duplicates(subset=['soc'], keep='first')
loaded_data = loaded_data.sort_values('soc')

# Extract data from CSV
soc_values = np.array(loaded_data["soc"])
current_values = np.array(loaded_data["current"])
temperature_values = np.array(loaded_data["temperature"])
R0_values = np.array(loaded_data["r0"])  
R1_values = np.array(loaded_data["r1"])  
C1_values = np.array(loaded_data["c1"]) 
R2_values = np.array(loaded_data["r2"])  
C2_values = np.array(loaded_data["c2"])
voltage_values = np.array(loaded_data["voltage"])

# Create interpolation functions for each parameter
def ocv(soc):
    return pybamm.Interpolant(soc_values, voltage_values, soc, name="OCV", interpolator="linear", extrapolate=True)

# Since the data only has one current and temperature, we'll interpolate only in SOC
def r0(current, temperature, soc):
    return pybamm.Interpolant(soc_values, R0_values, soc, name="R0", interpolator="linear", extrapolate=True)

def r1(current, temperature, soc):
    return pybamm.Interpolant(soc_values, R1_values, soc, name="R1", interpolator="linear", extrapolate=True)

def c1(current, temperature, soc):
    return pybamm.Interpolant(soc_values, C1_values, soc, name="C1", interpolator="linear", extrapolate=True)

def r2(current, temperature, soc):
    return pybamm.Interpolant(soc_values, R2_values, soc, name="R2", interpolator="linear", extrapolate=True)

def c2(current, temperature, soc):
    return pybamm.Interpolant(soc_values, C2_values, soc, name="C2", interpolator="linear", extrapolate=True)

# Create a Thevenin model with 2RC elements
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})

# Define parameter values
parameter_values = pybamm.ParameterValues("ECM_Example")

# Update with custom parameters
updated_data = {
    "Open-circuit voltage [V]": ocv,
    "R0 [Ohm]": r0,
    "R1 [Ohm]": r1,
    "C1 [F]": c1,
    "R2 [Ohm]": r2,
    "C2 [F]": c2,
    "Element-1 initial overpotential [V]": 0,
    "Element-2 initial overpotential [V]": 0,
    "Initial SoC": 1.0, 
}

parameter_values.update(updated_data, check_already_exists=False)

# Define experiment
experiment = pybamm.Experiment(
    [
        "Discharge at 4 A until 2.5 V",
        "Rest for 2 hours",
        "Charge at 5 A until 4.2 V",
        "Hold at 4.2 V until 3 A",
        "Discharge at 5 A for 13 hours",
        "Rest for 6 hours"
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