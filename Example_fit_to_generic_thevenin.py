import pybamm
import json
import numpy as np
import pandas as pd

battery_label = "G1"
cycle_number = 1

file_loc = f"Data\\Output\\LGM50\\Optmization_Results\\{battery_label}\\{cycle_number}\\"
with open(file_loc + "final_ecm_parameters_2.json", "r") as f:
    loaded_data = json.load(f)

# Create a Thevenin model 2RC 
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})

parameter_values = model.default_parameter_values

# Update with your loaded parameters
parameter_values.update(loaded_data, check_already_exists=False)

# Use default OCV curve if your loaded data doesn't have one
# We'll only use your custom OCV if needed
use_custom_ocv = 1
if use_custom_ocv:
    soc_ocv_data = pd.read_csv("Data\Output\LGM50\Capacity_Test\G1\G1_soc_ocv.csv")
    soc_values = np.array(soc_ocv_data['SOC'])
    ocv_values = np.array(soc_ocv_data['OCV'])

    # Create the interpolant using the proper variables
    ocv_interpolant = pybamm.Interpolant(
        soc_values,  
        ocv_values,
        pybamm.StateVector(slice(0, 1)),  # This represents the SOC state variable
        interpolator="linear",
        extrapolate=True
    )
    parameter_values.update({"Open-circuit voltage [V]": ocv_interpolant})


# Define experiment with simpler step first
experiment = pybamm.Experiment(
    [
        "Discharge at 1 A for 1 hours",
        # "Rest for 30 minutes",
        # "Charge at 1 A until 4.2 V",
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