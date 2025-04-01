import pybamm
import json
import pandas as pd
import numpy as np

battery_label = "G1"
cycle_number = 1

with open('final_ecm_parameters.json', 'r') as file:
    parmaeter_values_data = json.load(file)

model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})
parameter_values = pybamm.ParameterValues("ECM_Example")
parameter_values.update(parmaeter_values_data, check_already_exists=False)

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


experiment = pybamm.Experiment(
    [
        "Discharge at 0.5 A for 8 hours",
        # "Rest for 30 minutes",
        # "Charge at 1 A until 4.2 V",
        # "Hold at 4.2 V until 1 A",
        # "Rest for 2 hours",
        # "Discharge at 0.5 A until 3.6V",
        # "Charge at C/5 for 1 hour"
    ]
)
solver = pybamm.CasadiSolver(mode="fast")
sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver, experiment=experiment)
sim.solve()
sim.plot()