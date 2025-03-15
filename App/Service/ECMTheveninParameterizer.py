import os
import pandas as pd
import pybamm
import pybop
import logging

pybamm.set_logging_level("INFO")

class ECMTheveninParameterizer:
    def __init__(self, battery_label, cycle_number, parameter_set_name="ECM_Example"):
        self.battery_label = battery_label
        self.cycle_number = cycle_number
        self.number_of_rc_pairs = None  # Will be set later to setup_model()
        
        self.parameter_set = pybop.ParameterSet(parameter_set=parameter_set_name)
        self.model = None
        self.problem = None
        self.optim = None
        self.results = None

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def update_parameters(self):
        self.logger.info("Updating parameter set with base parameters...")
        # Base parameters for all models
        self.parameter_set.update({
            "Initial SoC": 1.0,
            "Cell capacity [A.h]": 4.85,
            "Nominal cell capacity [A.h]": 4.85,
            "Element-1 initial overpotential [V]": 0,
            "Upper voltage cut-off [V]": 4.2,
            "Lower voltage cut-off [V]": 2.5,
            "R0 [Ohm]": 1e-3,
            "R1 [Ohm]": 2e-4,
            "C1 [F]": 1e4,
            "Open-circuit voltage [V]": pybop.empirical.Thevenin().default_parameter_values["Open-circuit voltage [V]"]
        })
        
        # Add parameters for the 2 RC pairs for the thevenin model
        if self.number_of_rc_pairs == 2:
            self.logger.info("Updating parameters for 2 RC pairs...")
            self.parameter_set.update({
                "R2 [Ohm]": 0.0003,
                "C2 [F]": 40000,
                "Element-2 initial overpotential [V]": 0,
            }, check_already_exists=False)

    def load_pulses(self, pulse_number):
        self.logger.info(f"Loading data for pulse {pulse_number}...")
        # Construct the file path dynamically
        file_name = f"{self.battery_label}_cycle_{self.cycle_number}_pulse_{pulse_number}_hppc.csv"
        file_path = os.path.join(
            "Data", "Output", "LGM50", "HPPC_Test", self.battery_label, f"Cycle_{self.cycle_number}", file_name
        )
        
        # Load the data
        df = pd.read_csv(file_path, index_col=None, na_values=["NA"])
        df = df.drop_duplicates(subset=["Time"], keep="first")
        
        # Prepare the dataset
        self.dataset = pybop.Dataset({
            "Time [s]": df["Time"].to_numpy(),
            "Current function [A]": df["Current"].to_numpy(),
            "Voltage [V]": df["Voltage"].to_numpy(),
        })
        self.initial_state_of_charge = df["SoC"].iloc[0]
        self.logger.info(f"Data loaded successfully. Initial SoC: {self.initial_state_of_charge}")

    def setup_model(self, number_of_rc_pairs=2):
        self.number_of_rc_pairs = number_of_rc_pairs

        self.logger.info(f"Setting up model with {number_of_rc_pairs} RC pairs...")
        
        # Need to update the inital base parameters. If the rc_pairs are 2, .update_parameters() accounts for that.
        self.update_parameters()
        
        # define the thevenin equivalent circuit model
        if self.number_of_rc_pairs == 1:
            self.model = pybop.empirical.Thevenin(
                parameter_set=self.parameter_set,
                options={"number of rc elements": 1},
                solver=pybamm.CasadiSolver(mode="safe", dt_max=10),
            )
        elif self.number_of_rc_pairs == 2:
            self.model = pybop.empirical.Thevenin(
                parameter_set=self.parameter_set,
                options={"number of rc elements": 2},
                solver=pybamm.CasadiSolver(mode="safe", dt_max=10),
            )
        
        # Build the model
        # "Inital SoC" is scaled from 0-1. By default the user must find and fit the polynomial soc-ocv relationship of their desired battery label ot use this function
        # FOR EXAMPLE:
        # capacity_test = CapacityTest(battery_label=battery_label)
        # capacity_test.fit_soc_ocv_polynomial(degree=11)
        # capacity_test.plot_ocv_soc_fitting()
        # capacity_test.save_to_csv() 
        self.model.build(initial_state={"Initial SoC": self.initial_state_of_charge})
        self.logger.info("Model built successfully.")

    def setup_problem(self, r_guess=0.005):
        self.logger.info("Setting up optimization problem...")
        # the bounds are hardcoded right now. I might want to pass them as r0_bounds, r1_bounds, c1_bounds... all expecting a range of [lower_bound, upper_bound]
        # if i do this i need to also let the .optimize

        # Define the optimization problem. We want to identify the RC pairs we want fit to the Thevenin Model (1rc or 2rc). 
        if self.number_of_rc_pairs == 1:
            self.parameters = pybop.Parameters(
                pybop.Parameter(
                    "R0 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),
                    bounds=[0, 0.5],
                ),
                pybop.Parameter(
                    "R1 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),
                    bounds=[0, 0.5],
                ),
                pybop.Parameter(
                    "C1 [F]",
                    prior=pybop.Gaussian(500, 100),
                    bounds=[1, 2000],
                ),
            )
        elif self.number_of_rc_pairs == 2:
            self.parameters = pybop.Parameters(
                pybop.Parameter(
                    "R0 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),
                    bounds=[0, 0.5],
                ),
                pybop.Parameter(
                    "R1 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),
                    bounds=[0, 0.5],
                ),
                pybop.Parameter(
                    "R2 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),
                    bounds=[0, 0.5],
                ),
                pybop.Parameter(
                    "C1 [F]",
                    prior=pybop.Gaussian(500, 100),
                    bounds=[1, 2000],
                ),
                pybop.Parameter(
                    "C2 [F]",
                    prior=pybop.Gaussian(2000, 500),
                    bounds=[0, 2000],
                ),
            )
        
        self.problem = pybop.FittingProblem(
            self.model,
            self.parameters,
            self.dataset,
        )

    def optimize(self, max_unchanged_iterations=30, max_iterations=100):
        self.logger.info("Starting optimization...")
        cost = pybop.SumSquaredError(self.problem)
        
        # sigma0 notation by need to be passable to make a customisable problem? its hardcoded for now anyway.
        if self.number_of_rc_pairs == 1:
            sigma0 = [1e-3, 1e-3, 50]
        else:
            sigma0 = [1e-3, 1e-3, 1e-3, 50, 500]
            
        self.optim = pybop.PSO(
            cost,
            sigma0=sigma0,
            max_unchanged_iterations=max_unchanged_iterations,
            max_iterations=max_iterations,
        )
        self.results = self.optim.run()
        self.logger.info("Optimization completed successfully.")

    def export_results(self, output_file=None):
        self.logger.info("Exporting results...")
         # Default file path
        if output_file is None:
            default_dir = os.path.join("Data", "Output", "LGM50", "Optmization_Results", self.battery_label)
            os.makedirs(default_dir, exist_ok=True)  # Create the directory if it doesn't exist
            output_file = os.path.join(default_dir, f"{self.battery_label}_cycle_{self.cycle_number}_ecm_parameters.json")
        
        # Export the parameters
        self.parameter_set.export_parameters(output_file, fit_params=self.parameters)
        self.logger.info(f"Parameters saved to {output_file}")

    def plot_results(self):
        self.logger.info("Plotting results...")
        pybop.plot.quick(self.problem, problem_inputs=self.results.x, title="Optimised Comparison")
        pybop.plot.convergence(self.optim)
        pybop.plot.parameters(self.optim)