import os
import pandas as pd
import pybamm
import pybop
import logging
import numpy as np
from App.utils.data_loader import load_soc_ocv_data 
from scipy.signal import savgol_filter

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
        self.pulse_number = None

        # Load SOC-OCV data - instead of using the emperical thevenin model for OCV, we will use the 
        # soc-ocv relationship fitted data from the capacity test
        self.soc_ocv_data = load_soc_ocv_data(self.battery_label)

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.results_lut = [None]

    def interpolate_ocv(self, soc):
        """
        Interpolate OCV value from the lookup table based on SOC.
        
        :param soc: The state of charge (SOC) to look up.
        :return: The corresponding OCV value.
        """
        # Interpolate using the lookup table
        if self.soc_ocv_data is not None:
            soc_values = self.soc_ocv_data['SOC']
            ocv_values = self.soc_ocv_data['OCV']
            
            return np.interp(soc, soc_values, ocv_values)
        else:
            raise ValueError("SOC-OCV data not loaded. Please load the data first.")

    def update_parameters(self, inital_soc=1.0, upper_voltage_cutoff=4.2, lower_voltage_cutoff=2.5, cell_capacity=4.85, 
                          R0_Ohm=1e-3, R1_Ohm=2e-4, C1_F=1e4, R2_Ohm=0.0003, C2_F=40000):
        self.logger.info("Updating parameter set with base parameters...")
        # Base parameters for all models
        self.parameter_set.update({
            "Initial SoC": inital_soc,
            "Cell capacity [A.h]": cell_capacity,
            "Nominal cell capacity [A.h]": cell_capacity,
            "Element-1 initial overpotential [V]": 0,
            "Upper voltage cut-off [V]": upper_voltage_cutoff,
            "Lower voltage cut-off [V]": lower_voltage_cutoff,
            "R0 [Ohm]": R0_Ohm,
            "R1 [Ohm]": R1_Ohm,
            "C1 [F]": C1_F,
            # "Open-circuit voltage [V]": self.interpolate_ocv(self.initial_state_of_charge)
            "Open-circuit voltage [V]": pybop.empirical.Thevenin().default_parameter_values["Open-circuit voltage [V]"]
        })
        
        # Add parameters for the 2 RC pairs for the thevenin model
        if self.number_of_rc_pairs == 2:
            self.logger.info("Updating parameters for 2 RC pairs...")
            self.parameter_set.update({
                "R2 [Ohm]": R2_Ohm,
                "C2 [F]": C2_F,
                "Element-2 initial overpotential [V]": 0,
            }, check_already_exists=False)

    def load_pulses(self, pulse_number):
        self.logger.info(f"Loading data for pulse {pulse_number}...")
        self.pulse_number = pulse_number

        # Construct the file path dynamically
        file_name = f"{self.battery_label}_cycle_{self.cycle_number}_pulse_{pulse_number}_hppc.csv"
        file_path = os.path.join(
            "Data", "Output", "LGM50", "HPPC_Test", self.battery_label, f"Cycle_{self.cycle_number}", file_name
        )
        
        # Load the data
        df = pd.read_csv(file_path, index_col=None, na_values=["NA"])
        df = df.drop_duplicates(subset=["Time"], keep="first")
        
        # df["Voltage"] = savgol_filter(df["Voltage"], window_length=7, polyorder=2)

        # Prepare the dataset
        self.dataset = pybop.Dataset({
            "Time [s]": df["Time"].to_numpy(),
            "Current function [A]": df["Current"].to_numpy(),
            "Voltage [V]": df["Voltage"].to_numpy(),
        })
        self.initial_state_of_charge = df["SoC"].iloc[0]
        self.logger.info(f"Data loaded successfully. Initial SoC: {self.initial_state_of_charge}")

        # pulse entry is the LUT .csv results with temperature, soc and rc values
        self.pulse_entry = {
            "current": df.loc[df["Current"] != 0, "Current"].iloc[0],  # First nonzero current
            "voltage": df.loc[df["Current"] == 0, "Voltage"].iloc[-1],  # Last rest voltage
            "temperature": 298.15,  # Fixed temperature (you can change this)
            "SoC": self.initial_state_of_charge
        }

    def setup_solver(self, dt_max=5, mode="safe"):
        self.solver = pybamm.CasadiSolver(mode=mode, dt_max=dt_max)

    def setup_thevenin_model(self, number_of_rc_pairs=2, dt_max=5):
        self.number_of_rc_pairs = number_of_rc_pairs

        self.logger.info(f"Setting up model with {number_of_rc_pairs} RC pairs...")
        
        # Need to update the inital base parameters. If the rc_pairs are 2, .update_parameters() accounts for that.
        self.update_parameters()
        
        # define the thevenin equivalent circuit model
        if self.number_of_rc_pairs == 1:
            self.model = pybop.empirical.Thevenin(
                parameter_set=self.parameter_set,
                options={"number of rc elements": 1},
                solver=self.solver,
            )
        elif self.number_of_rc_pairs == 2:
            self.model = pybop.empirical.Thevenin(
                parameter_set=self.parameter_set,
                options={"number of rc elements": 2},
                solver=self.solver,
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

    def setup_problem(self, r_guess=0.005, r0_bounds=[0, 0.5], r1_bounds=[0, 0.5], c1_bounds=[1, 2000], 
                        r2_bounds=[0, 0.5], c2_bounds=[0, 2000], c1_Gaussian=(500, 100), c2_Gaussian=(2000, 500)):
        self.logger.info("Setting up optimization problem...")
        # the bounds are hardcoded right now. I might want to pass them as r0_bounds, r1_bounds, c1_bounds... all expecting a range of [lower_bound, upper_bound]
        # if i do this i need to also let the .optimize

        # Define the optimization problem. We want to identify the RC pairs we want fit to the Thevenin Model (1rc or 2rc). 
        if self.number_of_rc_pairs == 1:
            self.parameters = pybop.Parameters(
                pybop.Parameter(
                    "R0 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),  # mean, sigma
                    bounds=r0_bounds
                ),
                pybop.Parameter(
                    "R1 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),  # mean, sigma
                    bounds=r1_bounds,
                ),
                pybop.Parameter(
                    "C1 [F]",
                    prior=pybop.Gaussian(c1_Gaussian[0], c1_Gaussian[1]),  # mean, sigma
                    bounds=c1_bounds,
                ),
            )
        elif self.number_of_rc_pairs == 2:
            self.parameters = pybop.Parameters(
                pybop.Parameter(
                    "R0 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),  # mean, sigma
                    bounds=r0_bounds,
                ),
                pybop.Parameter(
                    "R1 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),  # mean, sigma
                    bounds=r1_bounds,
                ),
                pybop.Parameter(
                    "R2 [Ohm]",
                    prior=pybop.Gaussian(r_guess, r_guess / 10),  # mean, sigma
                    bounds=r2_bounds,
                ),
                pybop.Parameter(
                    "C1 [F]",
                    prior=pybop.Gaussian(c1_Gaussian[0], c1_Gaussian[1]),  # mean, sigma
                    bounds=c1_bounds,
                ),
                pybop.Parameter(
                    "C2 [F]",
                    prior=pybop.Gaussian(c2_Gaussian[0], c2_Gaussian[1]),  # mean, sigma
                    bounds=c2_bounds,
                ),
            )

        self.problem = pybop.FittingProblem(
            self.model,
            self.parameters,
            self.dataset,
        )

    def optimize(self, max_unchanged_iterations=30, max_iterations=100, sigma0 = [1e-3, 1e-3, 1e-3, 50, 500],):
        self.logger.info("Starting optimization...")
        cost = pybop.SumSquaredError(self.problem)
        
        # sigma0 notation by need to be passable to make a customisable problem? its hardcoded for now anyway.
        if self.number_of_rc_pairs == 1:
            sigma0 = [1e-3, 1e-3, 50] # For R0, R1, C1
        else:
            sigma0 = [1e-3, 1e-3, 1e-3, 50, 500] # For R0, R1, R2, C1, C2
            
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
            default_dir = os.path.join("Data", "Output", "LGM50", "Optimization_Results", self.battery_label, str(self.cycle_number))
            os.makedirs(default_dir, exist_ok=True)  # Ensure directory exists
            output_file = os.path.join(default_dir, f"{self.battery_label}_cycle_{self.cycle_number}_pulse_{self.pulse_number}_ecm_parameters.json")

        # Export the parameters
        self.parameter_set.export_parameters(output_file, fit_params=self.parameters)
        self.logger.info(f"Parameters saved to {output_file}")

        # Extract optimized parameters
        if self.number_of_rc_pairs == 1:
            r0, r1, c1 = self.results.x
            r2, c2 = None, None  # Not used for 1 RC pair
        else:  # self.number_of_rc_pairs == 2
            r0, r1, r2, c1, c2 = self.results.x

        # Create a new pulse entry
        pulse_entry = {
            "battery_label": self.battery_label,
            "cycle": self.cycle_number,
            "pulse_number": self.pulse_number,
            "current": self.pulse_entry["current"],
            "voltage": self.pulse_entry["voltage"],
            "temperature": self.pulse_entry["temperature"],
            "r0": r0,
            "r1": r1,
            "c1": c1,
            "SoC": self.pulse_entry["SoC"],
        }

        # Include second RC pair if applicable
        if self.number_of_rc_pairs == 2:
            pulse_entry["r2"] = r2
            pulse_entry["c2"] = c2

        # Ensure results_lut is a DataFrame
        if not isinstance(self.results_lut, pd.DataFrame):
            self.results_lut = pd.DataFrame(columns=list(pulse_entry.keys()))

        # Append the new pulse entry
        self.results_lut = pd.concat([self.results_lut, pd.DataFrame([pulse_entry])], ignore_index=True)

        # Define CSV file path (only one file for all pulses)
        csv_filename = os.path.join(default_dir, f"{self.battery_label}_{self.cycle_number}_ecm_lut_table.csv")

        self.results_lut.to_csv(csv_filename, mode="w", index=False)

        self.logger.info(f"Results successfully stored in {csv_filename}.")



    def plot_parameter_convergence_results(self):
        pybop.plot.convergence(self.optim)
        pybop.plot.parameters(self.optim)

    def plot_voltage_model_reference(self):
        pybop.plot.quick(self.problem, problem_inputs=self.results.x, title="Optimised Comparison")
