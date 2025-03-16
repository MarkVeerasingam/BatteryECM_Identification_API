import pybamm
import json

battery_label = "G1"
cycle_number = 1

file_loc = f"Data\Output\LGM50\Optmization_Results"
with open("generic_ecm_parameters.json", "r") as f:
    loaded_data = json.load(f)

# Create a Thevenin model with 2RC elements if your parameters specify that
model = pybamm.equivalent_circuit.Thevenin(options={"number of rc elements": 2})

# Load ECM example parameters which have default values for all required parameters
parameter_values = pybamm.ParameterValues("ECM_Example")

# Update with your loaded parameters
parameter_values.update(loaded_data, check_already_exists=False)

# Define experiment with simpler step first
experiment = pybamm.Experiment(
    [
        "Discharge at 0.5 A for 8 hours or until 3.2 V",
        "Rest for 30 minutes",
        "Charge at 1 A until 4.2 V",
        "Hold at 4.2 V until 1 A",
        "Rest for 2 hours",
        "Discharge at 0.5 A until 3.6V",
        "Charge at C/5 for 1 hour"
    ]
)

# Create solver with relaxed tolerances
solver = pybamm.CasadiSolver(mode="safe")

# Create and solve simulation
sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver, experiment=experiment)
sim.solve()
sim.plot()