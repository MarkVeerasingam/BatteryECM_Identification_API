import pybamm
import json
import numpy as np
import pandas as pd

battery_label = "G1"
cycle_number = 1

file_loc = f"Data\\Output\\LGM50\\Optmization_Results\\{battery_label}\\{cycle_number}\\"
with open(file_loc + "final_ecm_parameters.json", "r") as f:
    loaded_data = json.load(f)

# ocv_values = np.array(loaded_data["Open-circuit voltage [V]"])
R0_values = np.array(loaded_data["R0 [Ohm]"])
R1_values = np.array(loaded_data["R1 [Ohm]"])
C1_values = np.array(loaded_data["C1 [F]"])
R2_values = np.array(loaded_data["R2 [Ohm]"])
C2_values = np.array(loaded_data["C2 [F]"])

soc_values = np.linspace(-0.1, 1.1, len(R0_values)) 

soc_ocv_data = pd.read_csv("Data\Output\LGM50\Capacity_Test\G1\G1_soc_ocv.csv")
soc_data = np.array(soc_ocv_data['SOC'])
ocv_data = np.array(soc_ocv_data['OCV'])

ocv_interpolant = pybamm.Interpolant(
    soc_data,  
    ocv_data,
    pybamm.StateVector(slice(0, 1)), 
    interpolator="linear",
    extrapolate=False
)

R0_interpolant = pybamm.Interpolant(
    soc_values,  
    R0_values,
    pybamm.StateVector(slice(0, 1)),  
    interpolator="linear",
    extrapolate=True
)

R1_interpolant = pybamm.Interpolant(
    soc_values,  
    R1_values,
    pybamm.StateVector(slice(0, 1)),  
    interpolator="linear",
    extrapolate=True
)

C1_interpolant = pybamm.Interpolant(
    soc_values,  
    C1_values,
    pybamm.StateVector(slice(0, 1)),  
    interpolator="linear",
    extrapolate=True
)

R2_interpolant = pybamm.Interpolant(
    soc_values,  
    R2_values,
    pybamm.StateVector(slice(0, 1)),  
    interpolator="linear",
    extrapolate=True
)

C2_interpolant = pybamm.Interpolant(
    soc_values,  
    C2_values,
    pybamm.StateVector(slice(0, 1)),  
    interpolator="linear",
    extrapolate=True
)

updated_data = {
    "Open-circuit voltage [V]": ocv_interpolant,
    "R0 [Ohm]": R0_interpolant,
    "R1 [Ohm]": R1_interpolant,
    "C1 [F]": C1_interpolant,
    "R2 [Ohm]": R2_interpolant,
    "C2 [F]": C2_interpolant,
    "Element-2 initial overpotential [V]": 0,
    "Element-1 initial overpotential [V]": 0,
    "inital SoC": 1
}

# Create a Thevenin model 2RC 
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})

# Load ECM example parameters which have default values for all required parameters
parameter_values = pybamm.ParameterValues("ECM_Example")

# Update with your loaded parameters
parameter_values.update(updated_data, check_already_exists=False)

# Define experiment with simpler step first
experiment = pybamm.Experiment(
    [
        "Discharge at 3 A for 10 hours",
        "Rest for 2 hours",
        "Charge at 1 A until 4.2 V",
        # "Hold at 4.2 V until 1 A",
        # "Rest for 2 hours",
        # "Discharge at 0.5 A until 3.6V",
        # "Charge at C/5 for 1 hour"
    ]
)

# Create solver with relaxed tolerances
solver = pybamm.CasadiSolver(mode="fast")

# Create and solve simulation
sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver, experiment=experiment)
sim.solve()
sim.plot()