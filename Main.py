from App.Service.CapacityTest import CapacityTest
from App.Service.HPPCTest import HPPCTest
from App.Service.ECMTheveninParameterizer import ECMTheveninParameterizer

if __name__ == "__main__":
    # Choose Battery Label:
    battery_label = "G1"
    cycle_number = 0

    """
    Capacity Test:
    """
    capacity_test = CapacityTest(battery_label=battery_label)
    capacity_test.fit_soc_ocv_polynomial(degree=13)
    capacity_test.plot_capacity_test()
    capacity_test.plot_ocv_soc_fitting()
    capacity_test.save_to_csv() 

    """
    Hybrid pulse power characterization (HPPC) Test:
    """
    hppc_test = HPPCTest(battery_label=battery_label, cycle_number=cycle_number)
    pulse_count = hppc_test.get_pulse_count()
    for pulse in range(pulse_count):
        hppc_test.run_analysis(pulse_number=pulse) 
        hppc_test.save_to_csv()
    hppc_test.plot_hppc_analysis()

    """
    ECM Thevenin Parameterization:
    """
    ecm_parameterizer = ECMTheveninParameterizer(battery_label=battery_label, cycle_number=cycle_number)
    ##############################
    # All Pulses Parameterization:
    ##############################
    for pulse_number in range(pulse_count):
            print(f"Processing pulse {pulse_number}")
            ecm_parameterizer.load_pulses(pulse_number)
            ecm_parameterizer.setup_solver(mode="fast", dt_max=10)
            ecm_parameterizer.setup_thevenin_model(number_of_rc_pairs=2)
            ecm_parameterizer.update_parameters(
                R0_Ohm=1e-3, 
                R1_Ohm=2e-4, 
                C1_F=1e4,
                R2_Ohm=2e-4, 
                C2_F=2e4
            )
            ecm_parameterizer.setup_problem(
                r_guess=0.005,
                r0_bounds=[0, 0.5],
                r1_bounds=[0, 0.5],
                c1_bounds=[50, 1000],
                r2_bounds=[0, 0.5],
                c2_bounds=[100, 10000],
                c1_Gaussian=(1000, 100),  
                c2_Gaussian=(10000, 500)  
            )
            ecm_parameterizer.optimize(sigma0=[1e-3, 2e-4, 2e-4, 100, 500]) # R0, R1, R2, C1, C2
            ecm_parameterizer.plot_voltage_model_reference()
            ecm_parameterizer.export_results()

    ################################
    # Single Pulse Parameterization:
    ###############################     
    # ecm_parameterizer.load_pulses(pulse_number=0)
    # ecm_parameterizer.setup_solver(mode="fast", dt_max=10)
    # ecm_parameterizer.setup_thevenin_model(number_of_rc_pairs=2)
    # ecm_parameterizer.update_parameters(
    #     R0_Ohm=1e-3, 
    #     R1_Ohm=2e-4, 
    #     C1_F=1e4,
    #     R2_Ohm=2e-4, 
    #     C2_F=2e4
    # )
    # ecm_parameterizer.setup_problem(
    #     r_guess=0.005,
    #     r0_bounds=[0, 0.5],
    #     r1_bounds=[0, 0.5],
    #     c1_bounds=[50, 1000],
    #     r2_bounds=[0, 0.5],
    #     c2_bounds=[100, 10000],
    #     c1_Gaussian=(1000, 100),  
    #     c2_Gaussian=(10000, 500)  
    # )
    # ecm_parameterizer.optimize(sigma0=[1e-3, 2e-4, 2e-4, 100, 500]) # R0, R1, R2, C1, C2
    # ecm_parameterizer.plot_results()
    # ecm_parameterizer.export_results()